"""
Distribuição de ticket entre coberturas e cálculo da Importância Segurada (IS).

Layout de entrada (linhas JÁ explodidas por faixa da cobertura especial):

    colunas_de_perfil..., ticket_total,
    proporcao_<cob>..., t_<cob>...,            # t_<cob> = ticket recomendado
    ismin_<cob>..., ismax_<cob>...,            # na especial, variam por faixa
    taxa_<cob>...,                             # na especial, varia por faixa
    agravo_1, agravo_2                         # agravos da especial

Para as coberturas comuns, ismin/ismax/taxa apenas se repetem entre as linhas de
faixa de um mesmo pacote x perfil.

Saída: uma linha por pacote x perfil (a da faixa escolhida), com
    tf_<cob>   ticket final de cada cobertura
    isf_<cob>  IS final de cada cobertura
    + sobra, distorcao, fator_escala, status

Modelo (por linha = pacote x perfil x faixa):
    min  sum_i (t_i - t0_i)^2 / t0_i
    s.a. sum_i t_i = T              (T = soma dos tickets recomendados)
         L_i <= t_i <= U_i          (limites de ticket vindos de IS mín/máx)

Solução (KKT): t_i = clip(s * t0_i, L_i, U_i), com um único fator de escala s
por linha, encontrado por bisseção. Depois escolhe-se a melhor faixa de cada
pacote x perfil (menor sobra, depois menor distorção).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PREFIXOS_PADRAO = {
    "t": "t_",          # ticket recomendado (entrada)
    "ismin": "ismin_",
    "ismax": "ismax_",
    "taxa": "taxa_",
    "tf": "tf_",        # ticket final (saída)
    "isf": "isf_",      # IS final (saída)
}


# ---------------------------------------------------------------------------
# Núcleo: resolve o problema para N linhas de uma vez (vetorizado)
# ---------------------------------------------------------------------------
def resolver_distribuicao(t0: np.ndarray, L: np.ndarray, U: np.ndarray, n_iter: int = 200):
    """
    t0, L, U: matrizes (n_linhas x n_coberturas). Coberturas inativas devem ter
    t0 = L = U = 0.

    Retorna dict com t (tickets finais), s (fator de escala), T, sobra,
    distorcao e situacao ('ok' | 'teto' (soma U < T) | 'piso' (soma L > T)).
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

    return {"t": t, "s": s, "T": T, "sobra": T - t.sum(axis=1), "distorcao": dist, "situacao": situacao}


# ---------------------------------------------------------------------------
# Função principal
# ---------------------------------------------------------------------------
def calcular_is(
    df: pd.DataFrame,
    coberturas: list[str],
    especial: str,
    chave_grupo: list[str],
    col_agravo1: str = "agravo_1",
    col_agravo2: str = "agravo_2",
    prefixos: dict | None = None,
    desempate_is: str = "menor",
    tol: float = 1e-6,
    n_iter: int = 200,
    retornar_todas_faixas: bool = False,
):
    """
    df            dataframe com as linhas já explodidas por faixa.
    coberturas    TODAS as coberturas, incluindo a especial (ex.: ["cob_01", "cob_02", ...]).
    especial      nome da cobertura com agravos (ex.: "cob_01").
    chave_grupo   colunas que identificam um pacote x perfil (as linhas de faixa
                  de um mesmo grupo compartilham esses valores). Ex.: colunas de
                  perfil + colunas de proporção.
    desempate_is  'menor' ou 'maior': se duas faixas empatam em sobra e distorção,
                  fica a que gera a menor (ou maior) IS da especial.
    retornar_todas_faixas  se True, devolve também o resultado de cada faixa (auditoria).

    Retorna o df com uma linha por grupo (a da faixa escolhida) acrescido de
    tf_<cob>, isf_<cob>, sobra, distorcao, fator_escala e status.
    """
    px = {**PREFIXOS_PADRAO, **(prefixos or {})}
    k_esp = coberturas.index(especial)

    base = df.reset_index(drop=True).copy()
    grupo = base.groupby(chave_grupo, dropna=False, sort=False).ngroup().to_numpy()

    # 1) Matrizes t0, L, U (linhas x coberturas)
    def mat(prefixo):
        return np.column_stack([base[f"{prefixo}{c}"].to_numpy(float) for c in coberturas])

    t0 = np.nan_to_num(mat(px["t"]), nan=0.0).clip(min=0)
    taxa, ismin, ismax = mat(px["taxa"]), mat(px["ismin"]), mat(px["ismax"])
    agravos = base[col_agravo1].fillna(0).to_numpy(float) + base[col_agravo2].fillna(0).to_numpy(float)

    L = ismin * taxa
    U = ismax * taxa
    L[:, k_esp] += agravos          # especial: ticket = IS*taxa + agravo_1 + agravo_2
    U[:, k_esp] += agravos

    ativo = t0 > 0
    L = np.where(ativo, np.nan_to_num(L), 0.0)   # fora do pacote => ticket 0
    U = np.where(ativo, np.nan_to_num(U), 0.0)

    # Faixa sem intervalo de IS (ismin > ismax) é descartada
    esp_ativa = ativo[:, k_esp]
    faixa_valida = (ismin[:, k_esp] <= ismax[:, k_esp]) | ~esp_ativa
    U[:, k_esp] = np.where(faixa_valida, U[:, k_esp], L[:, k_esp])

    # 2) Resolve todas as linhas de uma vez
    r = resolver_distribuicao(t0, L, U, n_iter=n_iter)
    t = r["t"]

    # 3) Ticket -> IS
    with np.errstate(divide="ignore", invalid="ignore"):
        is_f = np.where(ativo, t / taxa, 0.0)
        is_f[:, k_esp] = np.where(esp_ativa, (t[:, k_esp] - agravos) / taxa[:, k_esp], 0.0)

    for j, c in enumerate(coberturas):
        base[f"{px['tf']}{c}"] = t[:, j]
        base[f"{px['isf']}{c}"] = is_f[:, j]
    base["sobra"] = r["sobra"]
    base["distorcao"] = r["distorcao"]
    base["fator_escala"] = r["s"]

    status = np.where(np.abs(r["sobra"]) <= tol, "ok", "sobra_nao_distribuida")
    status = np.where(r["situacao"] == "piso", "minimos_excedem_T", status)
    status = np.where(~faixa_valida, "faixa_invalida", status)
    base["status"] = status

    # 4) Melhor faixa por grupo:
    #    válida > não estoura mínimos > menor |sobra| > menor distorção > desempate de IS
    rank = pd.DataFrame({
        "grupo": grupo,
        "invalida": (~faixa_valida).astype(int),
        "piso": (r["situacao"] == "piso").astype(int),
        "sobra": np.abs(r["sobra"]).round(6),
        "dist": r["distorcao"].round(6),
        "is_esp": is_f[:, k_esp] * (1 if desempate_is == "menor" else -1),
    })
    idx = rank.sort_values(["grupo", "invalida", "piso", "sobra", "dist", "is_esp"]) \
              .drop_duplicates("grupo").index
    resultado = base.loc[idx].sort_index().reset_index(drop=True)
    resultado.loc[resultado["status"] == "faixa_invalida", "status"] = "sem_faixa_valida"

    if retornar_todas_faixas:
        return resultado, base.assign(grupo=grupo)
    return resultado


# ---------------------------------------------------------------------------
# Exemplo (o mesmo da explicação didática, no layout explodido)
# cob_01 = Incêndio (especial), cob_02 = Danos Elétricos, cob_03 = Roubo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    pacotes = [
        # pacote, t_cob_01, t_cob_02, t_cob_03, IS máx do Incêndio, IS máx do Roubo
        ("A", 600, 250, 150, 1_000_000, 50_000),
        ("B", 700, np.nan, 300, 1_000_000, 50_000),   # B não tem danos elétricos
        ("C", 600, 250, 150, 300_000, 20_000),        # C limita IS de incêndio e roubo
    ]
    faixas = [  # (início, fim, taxa, agravo_1)
        (50_000, 200_000, 0.0020, 20),
        (200_000, 500_000, 0.0015, 40),
        (500_000, 1_000_000, 0.0012, 60),
    ]

    linhas = []
    for pac, t1, t2, t3, ismax_inc, ismax_rou in pacotes:
        total = np.nansum([t1, t2, t3])
        for inicio, fim, taxa_f, ag1 in faixas:
            linhas.append({
                "perfil": "P1", "pacote": pac, "ticket_total": total,
                "proporcao_cob_01": t1 / total,
                "proporcao_cob_02": (t2 / total) if t2 == t2 else np.nan,
                "proporcao_cob_03": t3 / total,
                "t_cob_01": t1, "t_cob_02": t2, "t_cob_03": t3,
                # especial: limites da faixa ∩ limites da cobertura
                "ismin_cob_01": max(50_000, inicio), "ismax_cob_01": min(ismax_inc, fim),
                "ismin_cob_02": 5_000, "ismax_cob_02": 30_000,
                "ismin_cob_03": 2_000, "ismax_cob_03": ismax_rou,
                "taxa_cob_01": taxa_f, "taxa_cob_02": 0.005, "taxa_cob_03": 0.01,
                "agravo_1": ag1, "agravo_2": 30,
            })
    df = pd.DataFrame(linhas)

    res = calcular_is(
        df,
        coberturas=["cob_01", "cob_02", "cob_03"],
        especial="cob_01",
        chave_grupo=["perfil", "pacote"],
    )
    cols = ["pacote", "ismin_cob_01", "ismax_cob_01", "status", "sobra",
            "tf_cob_01", "tf_cob_02", "tf_cob_03", "isf_cob_01", "isf_cob_02", "isf_cob_03"]
    print(res[cols].round(2).to_string(index=False))
