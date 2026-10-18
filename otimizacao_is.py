"""
Distribuição de ticket entre coberturas e cálculo da Importância Segurada (IS).

Modelo (por linha = pacote x perfil):
    min  sum_i (t_i - t0_i)^2 / t0_i
    s.a. sum_i t_i = T              (T = soma dos tickets recomendados)
         L_i <= t_i <= U_i          (limites de ticket vindos de IS mín/máx)

Solução (KKT):  t_i = clip(s * t0_i, L_i, U_i), com um único fator de escala s
encontrado por bisseção.

Cobertura especial: ticket = IS * taxa(faixa) + agravo1(faixa) + agravo2.
Ela é resolvida enumerando as faixas: cada linha é multiplicada pelo número de
faixas do seu perfil (join), cada candidato é resolvido como problema comum e,
no fim, escolhe-se o melhor candidato por linha (menor sobra, depois menor distorção).

Convenção de colunas
--------------------
df_pacotes (uma linha por pacote x perfil):
    - colunas de perfil (chave do join com df_faixas)
    - para cada cobertura comum c:  ticket_{c}, taxa_{c}, is_min_{c}, is_max_{c}
    - para a cobertura especial e:  ticket_{e}, is_min_{e}, is_max_{e}
      (ticket_{e} já contempla IS*taxa + agravo1 + agravo2)
    - ticket ausente (NaN) ou 0  => cobertura fora do pacote, IS = 0

df_faixas (uma linha por perfil x faixa):
    - colunas de perfil
    - faixa, is_inf, is_sup, taxa, agravo1, agravo2
      (nomes configuráveis em `cols_faixa`)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COLS_FAIXA_PADRAO = {
    "faixa": "faixa",
    "is_inf": "is_inf",
    "is_sup": "is_sup",
    "taxa": "taxa",
    "agravo1": "agravo1",
    "agravo2": "agravo2",
}


# ---------------------------------------------------------------------------
# Núcleo: resolve o problema "comum" para N linhas de uma vez (vetorizado)
# ---------------------------------------------------------------------------
def resolver_distribuicao(t0: np.ndarray, L: np.ndarray, U: np.ndarray, n_iter: int = 200):
    """
    t0, L, U: matrizes (n_linhas x n_coberturas). Coberturas inativas devem ter
    t0 = L = U = 0.

    Retorna dict com:
        t         tickets finais (n x k)
        s         fator de escala (1 + lambda) por linha
        T         ticket total por linha
        sobra     T - sum(t)   (>0: não coube tudo; <0: mínimos excedem T)
        distorcao sum (t - t0)^2 / t0 nas coberturas ativas
        situacao  'ok' | 'teto' (sum U < T) | 'piso' (sum L > T)
    """
    t0 = np.asarray(t0, float)
    L = np.asarray(L, float)
    U = np.asarray(U, float)
    ativo = t0 > 0

    T = t0.sum(axis=1)
    soma_L = L.sum(axis=1)
    soma_U = U.sum(axis=1)

    # Intervalo de busca do fator s: em s = 0 tudo vai para L; em s_hi tudo vai para U.
    with np.errstate(divide="ignore", invalid="ignore"):
        razao = np.where(ativo, U / np.where(ativo, t0, 1.0), 0.0)
    s_lo = np.zeros(len(T))
    s_hi = np.maximum(razao.max(axis=1), 1.0) * 1.0001 + 1e-12

    def soma_t(s):
        return np.clip(s[:, None] * t0, L, U).sum(axis=1)

    # Bisseção vetorizada (soma_t é não decrescente em s)
    for _ in range(n_iter):
        s_mid = 0.5 * (s_lo + s_hi)
        abaixo = soma_t(s_mid) < T
        s_lo = np.where(abaixo, s_mid, s_lo)
        s_hi = np.where(abaixo, s_hi, s_mid)
    s = 0.5 * (s_lo + s_hi)

    t = np.clip(s[:, None] * t0, L, U)

    situacao = np.full(len(T), "ok", dtype=object)
    teto = soma_U < T
    piso = soma_L > T
    t[teto] = U[teto]
    t[piso] = L[piso]
    situacao[teto] = "teto"
    situacao[piso] = "piso"

    with np.errstate(divide="ignore", invalid="ignore"):
        dist = np.where(ativo, (t - t0) ** 2 / np.where(ativo, t0, 1.0), 0.0).sum(axis=1)

    return {
        "t": t,
        "s": s,
        "T": T,
        "sobra": T - t.sum(axis=1),
        "distorcao": dist,
        "situacao": situacao,
    }


# ---------------------------------------------------------------------------
# Função principal: pacotes x faixas -> IS por cobertura
# ---------------------------------------------------------------------------
def calcular_is(
    df_pacotes: pd.DataFrame,
    df_faixas: pd.DataFrame,
    coberturas: list[str],
    especial: str,
    chave_perfil: str | list[str] = "perfil",
    cols_faixa: dict | None = None,
    tol: float = 1e-6,
    n_iter: int = 200,
    desempate_is: str = "menor",
):
    """
    desempate_is: 'menor' ou 'maior'. Quando duas faixas empatam em sobra e
        distorção (o mesmo ticket corresponde a duas IS, porque a taxa cai nas
        faixas altas), escolhe a faixa que gera a menor (ou maior) IS da especial.

    Retorna (resultado, candidatos):
        resultado   1 linha por linha de df_pacotes, com a faixa escolhida,
                    tickets finais, IS, proporções, sobra, distorção e status.
        candidatos  1 linha por (linha de df_pacotes x faixa), útil para auditoria.
    """
    cf = {**COLS_FAIXA_PADRAO, **(cols_faixa or {})}
    chave = [chave_perfil] if isinstance(chave_perfil, str) else list(chave_perfil)
    todas = list(coberturas) + [especial]
    k = len(todas)

    base = df_pacotes.reset_index(drop=True).copy()
    base["_id_linha"] = np.arange(len(base))

    # --- 1) join: cada linha vira n_faixas candidatos ----------------------
    fx = df_faixas[chave + list(cf.values())].rename(
        columns={v: f"_fx_{k_}" for k_, v in cf.items()}
    )
    cand = base.merge(fx, on=chave, how="left", validate="many_to_many")
    sem_tabela = cand["_fx_taxa"].isna()

    # --- 2) matrizes t0, L, U ---------------------------------------------
    t0 = np.column_stack([cand[f"ticket_{c}"].fillna(0).to_numpy(float) for c in todas])
    t0 = np.where(t0 > 0, t0, 0.0)
    L = np.zeros_like(t0)
    U = np.zeros_like(t0)

    for j, c in enumerate(coberturas):
        taxa = cand[f"taxa_{c}"].to_numpy(float)
        L[:, j] = cand[f"is_min_{c}"].to_numpy(float) * taxa
        U[:, j] = cand[f"is_max_{c}"].to_numpy(float) * taxa

    # Especial: intervalo de IS = [IS mín, IS máx] ∩ [início da faixa, fim da faixa]
    je = k - 1
    taxa_e = cand["_fx_taxa"].to_numpy(float)
    ag1 = cand["_fx_agravo1"].fillna(0).to_numpy(float)
    ag2 = cand["_fx_agravo2"].fillna(0).to_numpy(float)
    is_lo = np.maximum(cand[f"is_min_{especial}"].to_numpy(float), cand["_fx_is_inf"].to_numpy(float))
    is_hi = np.minimum(cand[f"is_max_{especial}"].to_numpy(float), cand["_fx_is_sup"].to_numpy(float))
    especial_ativa = t0[:, je] > 0
    faixa_valida = (is_lo <= is_hi) & ~sem_tabela.to_numpy()

    L[:, je] = np.where(faixa_valida, is_lo * taxa_e + ag1 + ag2, 0.0)
    U[:, je] = np.where(faixa_valida, is_hi * taxa_e + ag1 + ag2, 0.0)

    # Cobertura fora do pacote: L = U = 0
    L = np.where(t0 > 0, L, 0.0)
    U = np.where(t0 > 0, U, 0.0)

    # Faixa sem interseção com [IS mín, IS máx]: candidato descartado
    # (se a especial não está no pacote, a faixa é irrelevante e fica "válida")
    candidato_valido = faixa_valida | ~especial_ativa

    # --- 3) resolve todos os candidatos de uma vez ------------------------
    r = resolver_distribuicao(t0, L, U, n_iter=n_iter)

    cand["_candidato_valido"] = candidato_valido
    cand["T"] = r["T"]
    cand["fator_escala"] = r["s"]
    cand["sobra"] = r["sobra"]
    cand["distorcao"] = r["distorcao"]
    cand["situacao"] = r["situacao"]
    for j, c in enumerate(todas):
        cand[f"ticket_final_{c}"] = r["t"][:, j]
    with np.errstate(divide="ignore", invalid="ignore"):
        cand[f"is_{especial}"] = np.where(
            especial_ativa & faixa_valida, (r["t"][:, je] - ag1 - ag2) / taxa_e, 0.0
        )

    # --- 4) escolhe a melhor faixa por linha ------------------------------
    # prioridade: candidato válido > não estoura mínimos > menor sobra > menor distorção
    # > desempate pela IS da especial (o mesmo ticket pode caber em 2 faixas)
    cand["_rk_invalido"] = (~cand["_candidato_valido"]).astype(int)
    cand["_rk_piso"] = (cand["situacao"] == "piso").astype(int)
    cand["_rk_sobra"] = cand["sobra"].abs().round(6)
    cand["_rk_dist"] = cand["distorcao"].round(6)
    sinal = 1.0 if desempate_is == "menor" else -1.0
    cand["_rk_is"] = sinal * cand[f"is_{especial}"]
    melhor = (
        cand.sort_values(["_id_linha", "_rk_invalido", "_rk_piso", "_rk_sobra", "_rk_dist", "_rk_is"])
        .drop_duplicates("_id_linha", keep="first")
        .sort_values("_id_linha")
        .reset_index(drop=True)
    )

    # --- 5) converte ticket -> IS -----------------------------------------
    for c in coberturas:
        t = melhor[f"ticket_final_{c}"]
        melhor[f"is_{c}"] = np.where(t > 0, t / melhor[f"taxa_{c}"], 0.0)

    t_e = melhor[f"ticket_final_{especial}"]
    ativa_e = melhor[f"ticket_{especial}"].fillna(0) > 0
    melhor[f"is_{especial}"] = np.where(
        ativa_e,
        (t_e - melhor["_fx_agravo1"].fillna(0) - melhor["_fx_agravo2"].fillna(0)) / melhor["_fx_taxa"],
        0.0,
    )
    melhor["faixa_escolhida"] = melhor["_fx_faixa"].where(ativa_e)
    melhor[f"taxa_{especial}"] = melhor["_fx_taxa"].where(ativa_e)
    melhor[f"agravo1_{especial}"] = melhor["_fx_agravo1"].where(ativa_e)
    melhor[f"agravo2_{especial}"] = melhor["_fx_agravo2"].where(ativa_e)

    # --- 6) proporções e status -------------------------------------------
    for c in todas:
        melhor[f"prop_ini_{c}"] = melhor[f"ticket_{c}"].fillna(0) / melhor["T"]
        melhor[f"prop_final_{c}"] = melhor[f"ticket_final_{c}"] / melhor["T"]

    status = np.where(melhor["sobra"].abs() <= tol, "ok", "sobra_nao_distribuida")
    status = np.where(melhor["situacao"] == "piso", "minimos_excedem_T", status)
    status = np.where(~melhor["_candidato_valido"], "sem_faixa_valida", status)
    melhor["status"] = status
    melhor.loc[melhor["status"] == "sem_faixa_valida", f"is_{especial}"] = np.nan

    # --- 7) monta saída ----------------------------------------------------
    cols_saida = (
        list(df_pacotes.columns)
        + ["faixa_escolhida", f"taxa_{especial}", f"agravo1_{especial}", f"agravo2_{especial}"]
        + ["T", "fator_escala", "sobra", "distorcao", "status"]
        + [f"ticket_final_{c}" for c in todas]
        + [f"is_{c}" for c in todas]
        + [f"prop_ini_{c}" for c in todas]
        + [f"prop_final_{c}" for c in todas]
    )
    resultado = melhor[cols_saida]

    cols_cand = (
        ["_id_linha"] + chave
        + ["_fx_faixa", "_candidato_valido", "situacao", "fator_escala", "sobra", "distorcao"]
        + [f"ticket_final_{c}" for c in todas]
        + [f"is_{especial}"]
    )
    candidatos = cand[cols_cand].rename(
        columns={"_id_linha": "id_linha", "_fx_faixa": "faixa", "_candidato_valido": "faixa_valida"}
    )
    return resultado, candidatos


# ---------------------------------------------------------------------------
# Exemplo (o mesmo da explicação didática)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)

    coberturas = ["danos_eletricos", "roubo"]
    especial = "incendio"

    df_pacotes = pd.DataFrame(
        {
            "pacote": ["A", "B", "C"],
            "perfil": ["P1", "P1", "P1"],
            # ticket recomendado (proporção x T)
            "ticket_incendio": [600, 700, 600],
            "ticket_danos_eletricos": [250, np.nan, 250],  # B não tem danos elétricos
            "ticket_roubo": [150, 300, 150],
            # IS mín / máx da especial
            "is_min_incendio": [50_000, 50_000, 50_000],
            "is_max_incendio": [1_000_000, 1_000_000, 300_000],  # C limita IS do incêndio
            # comuns
            "taxa_danos_eletricos": [0.005, 0.005, 0.005],
            "is_min_danos_eletricos": [5_000, 5_000, 5_000],
            "is_max_danos_eletricos": [30_000, 30_000, 30_000],
            "taxa_roubo": [0.01, 0.01, 0.01],
            "is_min_roubo": [2_000, 2_000, 2_000],
            "is_max_roubo": [50_000, 50_000, 20_000],
        }
    )

    df_faixas = pd.DataFrame(
        {
            "perfil": ["P1", "P1", "P1"],
            "faixa": ["F1", "F2", "F3"],
            "is_inf": [50_000, 200_000, 500_000],
            "is_sup": [200_000, 500_000, 1_000_000],
            "taxa": [0.0020, 0.0015, 0.0012],
            "agravo1": [20, 40, 60],
            "agravo2": [30, 30, 30],
        }
    )

    res, cand = calcular_is(df_pacotes, df_faixas, coberturas, especial)

    print("=== Candidatos (linha x faixa) ===")
    print(cand.round(2).to_string(index=False))
    print("\n=== Resultado ===")
    cols = ["pacote", "faixa_escolhida", "status", "sobra", "distorcao", "fator_escala",
            "ticket_final_incendio", "ticket_final_danos_eletricos", "ticket_final_roubo",
            "is_incendio", "is_danos_eletricos", "is_roubo"]
    print(res[cols].round(2).to_string(index=False))
