"""
Distribuição de ticket entre coberturas e cálculo da Importância Segurada (IS).

Layout de entrada (linhas JÁ explodidas por faixa da cobertura especial):

    colunas_de_perfil..., ticket_total,
    <cob>_proporcao..., <cob>_t...,            # <cob>_t = ticket recomendado
    <cob>_ismin..., <cob>_ismax...,            # na especial, variam por faixa
    <cob>_taxa...,                             # na especial, varia por faixa
    agravo_1, agravo_2                         # agravos da especial

Os sufixos (_t, _ismin, _ismax, _taxa) são configuráveis pelo parâmetro `sufixos`,
para casar com os nomes reais da base (ex.: "cob_01_is_max_final").

Para as coberturas comuns, ismin/ismax/taxa apenas se repetem entre as linhas de
faixa de um mesmo pacote x perfil.

Saída: uma linha por pacote x perfil (a da faixa escolhida), com
    <cob>_tf   ticket final de cada cobertura (pode ter decimais)
    <cob>_isf  IS final de cada cobertura (sempre inteira, dentro de [ismin, ismax])
    + sobra, dif_arredondamento, distorcao, fator_escala, status

Modelo (por linha = pacote x perfil x faixa):
    min  sum_i (t_i - t0_i)^2 / t0_i
    s.a. sum_i t_i = T              (T = soma dos tickets recomendados)
         L_i <= t_i <= U_i          (limites de ticket vindos de IS mín/máx)

Solução (KKT): t_i = clip(s * t0_i, L_i, U_i), com um único fator de escala s
por linha, encontrado por bisseção. Depois escolhe-se a melhor faixa de cada
pacote x perfil (menor sobra, depois menor distorção).

IS inteira: os limites viram [ceil(ismin), floor(ismax)] antes da otimização; o
ótimo contínuo é então arredondado (floor + ajuste guloso) sem sair desses limites.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Nome da coluna = <cob> + sufixo. Ex.: "cob_01" + "_ismax" -> "cob_01_ismax"
SUFIXOS_PADRAO = {
    "t": "_t",            # ticket recomendado (entrada)
    "ismin": "_ismin",    # IS mínima (entrada)
    "ismax": "_ismax",    # IS máxima (entrada)
    "taxa": "_taxa",      # taxa (entrada)
    "tf": "_tf",          # ticket final (saída)
    "isf": "_isf",        # IS final (saída)
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
# IS inteira
# ---------------------------------------------------------------------------
CASAS_LIMPEZA = 6   # casas decimais usadas para limpar ruído de ponto flutuante


def limites_inteiros(ismin: np.ndarray, ismax: np.ndarray):
    """
    Limites de IS válidos em números inteiros: [ceil(ismin), floor(ismax)].
    Antes de ceil/floor, arredonda para CASAS_LIMPEZA casas, para que ruídos como
    75000.0000000001 não virem 75001 (nem 74999.9999999 vire 74999).
    """
    lo = np.ceil(np.round(ismin, CASAS_LIMPEZA))
    hi = np.floor(np.round(ismax, CASAS_LIMPEZA))
    return lo, hi


ULPS_TOLERANCIA = 64   # tolerância = 64 unidades da última casa binária do teto (~1e-11 p/ R$ 1.000)


def tickets(is_int, taxa0, fixo0, casas_ticket=None):
    """Tickets finais por cobertura, exatamente como saem nas colunas <cob>_tf."""
    t = is_int * taxa0 + fixo0
    return np.round(t, casas_ticket) if casas_ticket is not None else t


def passa_do_teto(soma, teto):
    """
    True se a soma passa do teto. Empatar com o teto é permitido. A única folga
    aceita é o ruído de representação binária do float (ULPS_TOLERANCIA unidades
    da última casa, cerca de 1e-11 para um teto de R$ 1.000): assim
    1000.0000000000002 conta como 1000, mas qualquer excesso real é bloqueado.
    """
    return soma > teto + ULPS_TOLERANCIA * np.spacing(np.abs(teto))


def arredondar_is(is_cont, lo, hi, taxa, fixo, ativo, alvo, teto, casas_ticket=None):
    """
    Converte a IS contínua (ótimo do modelo) em IS inteira, linha a linha,
    com a garantia de que a soma dos tickets finais NUNCA passa do teto.

    is_cont, lo, hi, taxa, fixo, ativo: matrizes (linhas x coberturas)
        ticket_i = IS_i * taxa_i + fixo_i   (fixo = agravos na especial, 0 nas demais)
    alvo: soma de tickets que o ótimo contínuo distribuiu (por linha)
    teto: ticket total máximo permitido (por linha)
    casas_ticket: se informado, os tickets são arredondados nessas casas e a
        garantia vale para os tickets já arredondados.

    1) começa com floor(IS), sempre dentro de [lo, hi];
    2) percorre as coberturas da maior para a menor parte fracionária e sobe a
       IS em +1 quando isso aproxima a soma do alvo, sem passar de hi e sem a
       soma passar do teto;
    3) trava final: recalcula a soma exatamente como as colunas <cob>_tf e, se alguma
       linha passar do teto, desce 1 unidade de IS da cobertura de maior taxa
       que ainda pode descer, até caber.
    """
    x = np.round(np.where(ativo, is_cont, 0.0), CASAS_LIMPEZA)
    base = np.where(ativo, np.clip(np.floor(x), lo, hi), 0.0)
    frac = np.where(ativo, x - base, -1.0)

    taxa0 = np.where(ativo, np.nan_to_num(taxa), 0.0)
    fixo0 = np.where(ativo, fixo, 0.0)

    def soma(b):
        return tickets(b, taxa0, fixo0, casas_ticket).sum(axis=1)

    linhas = np.arange(x.shape[0])
    ordem = np.argsort(-frac, axis=1)
    s_atual = soma(base)
    for pos in range(x.shape[1]):
        i = ordem[:, pos]
        pode = ativo[linhas, i] & (base[linhas, i] + 1 <= hi[linhas, i]) & (taxa0[linhas, i] > 0)
        teste = base.copy()
        teste[linhas[pode], i[pode]] += 1
        s_teste = soma(teste)
        melhora = np.abs(alvo - s_teste) < np.abs(alvo - s_atual) - 1e-12
        sobe = pode & melhora & ~passa_do_teto(s_teste, teto)
        base[linhas[sobe], i[sobe]] += 1
        s_atual = np.where(sobe, s_teste, s_atual)

    # Trava final, com a soma recalculada do zero
    for _ in range(10 * x.shape[1]):
        pode_descer = ativo & (base - 1 >= lo) & (taxa0 > 0)
        excede = passa_do_teto(soma(base), teto) & pode_descer.any(axis=1)
        if not excede.any():
            break
        i = np.argmax(np.where(pode_descer, taxa0, -np.inf), axis=1)
        base[linhas[excede], i[excede]] -= 1

    return base


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
    sufixos: dict | None = None,
    desempate_is: str = "menor",
    col_ticket_total: str | None = "ticket_total",
    casas_ticket: int | None = None,
    coberturas_valor_fixo: dict | None = None,
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
    sufixos       sufixos dos nomes de coluna (<cob> + sufixo). Só é preciso passar
                  os que diferem do padrão SUFIXOS_PADRAO. Ex.:
                  {"t": "_ticket", "ismin": "_is_min_final", "ismax": "_is_max_final"}
    desempate_is  'menor' ou 'maior': se duas faixas empatam em sobra e distorção,
                  fica a que gera a menor (ou maior) IS da especial.
    col_ticket_total  coluna com o ticket total do pacote. A soma dos <cob>_tf nunca
                  passa de min(ticket_total, soma dos <cob>_t). Use None para
                  usar só a soma dos <cob>_t.
    casas_ticket  se informado (ex.: 2), os <cob>_tf saem arredondados nessas casas e a
                  garantia de não passar do ticket total vale já com esse arredondamento.
    coberturas_valor_fixo  coberturas cujo ticket é  valor_fixo + IS * taxa / escala_taxa.
                  Dicionário {cobertura: {"valor": coluna_do_valor_fixo,
                                          "escala_taxa": 100}}   (escala padrão: 100).
                  - A coluna de taxa (<cob>_taxa) é opcional: se não existir, taxa = 0.
                  - Com taxa 0, o ticket final é exatamente o valor fixo e a IS final
                    sai 0. A diferença entre o ticket recomendado (<cob>_t) e o valor
                    fixo é redistribuída entre as demais coberturas (ou retirada
                    delas, se o valor fixo for maior que o recomendado).
                  - Com taxa > 0, a cobertura entra na otimização como qualquer
                    outra, com o valor fixo somado ao ticket.
                  As coberturas daqui também devem estar na lista `coberturas`.
    retornar_todas_faixas  se True, devolve também o resultado de cada faixa (auditoria).

    Retorna o df com uma linha por grupo (a da faixa escolhida) acrescido de
    <cob>_tf (ticket final, pode ter decimais), <cob>_isf (IS final, inteira),
    sobra, dif_arredondamento, distorcao, fator_escala e status.
    """
    sx = {**SUFIXOS_PADRAO, **(sufixos or {})}
    k_esp = coberturas.index(especial)
    cvf = {c: {"escala_taxa": 100, **cfg} for c, cfg in (coberturas_valor_fixo or {}).items()}
    fora_da_lista = [c for c in cvf if c not in coberturas]
    if fora_da_lista:
        raise ValueError(f"Coberturas de valor fixo fora da lista `coberturas`: {fora_da_lista}")

    def col(c, tipo):
        return f"{c}{sx[tipo]}"

    # Confere se todas as colunas de entrada existem (erro claro se algum nome não bate).
    # Nas coberturas de valor fixo, a taxa é opcional.
    necessarias = [col(c, t) for t in ("t", "ismin", "ismax") for c in coberturas]
    necessarias += [col(c, "taxa") for c in coberturas if c not in cvf]
    necessarias += [cfg["valor"] for cfg in cvf.values()]
    necessarias += list(chave_grupo) + [col_agravo1, col_agravo2]
    faltando = [n for n in necessarias if n not in df.columns]
    if faltando:
        raise KeyError(f"Colunas não encontradas no dataframe: {faltando}. "
                       f"Confira os nomes ou ajuste o parâmetro `sufixos`.")

    base = df.reset_index(drop=True).copy()
    grupo = base.groupby(chave_grupo, dropna=False, sort=False).ngroup().to_numpy()

    # 1) Matrizes (linhas x coberturas)
    n = len(base)

    def coluna(nome, padrao=0.0):
        return base[nome].to_numpy(float) if nome in base.columns else np.full(n, padrao)

    # Valor fixo de cada cobertura: agravos na especial, coluna de valor nas de
    # valor fixo, 0 nas demais
    fixo = np.zeros((n, len(coberturas)))
    fixo[:, k_esp] = np.nan_to_num(coluna(col_agravo1)) + np.nan_to_num(coluna(col_agravo2))
    for c, cfg in cvf.items():
        fixo[:, coberturas.index(c)] = np.nan_to_num(coluna(cfg["valor"]))

    # Ticket recomendado (proporção x ticket total, enviado pelo negócio)
    t0 = np.column_stack([coluna(col(c, "t")) for c in coberturas])
    t0 = np.nan_to_num(t0, nan=0.0).clip(min=0)

    # Taxa efetiva (já dividida pela escala; sem coluna de taxa => 0)
    escala = np.array([cvf[c]["escala_taxa"] if c in cvf else 1.0 for c in coberturas])
    taxa = np.nan_to_num(np.column_stack([coluna(col(c, "taxa")) for c in coberturas])) / escala
    ismin = np.column_stack([coluna(col(c, "ismin")) for c in coberturas])
    ismax = np.column_stack([coluna(col(c, "ismax")) for c in coberturas])

    ativo = t0 > 0                                    # cobertura recomendada no pacote
    tem_taxa = taxa > 0
    eh_valor_fixo = np.array([c in cvf for c in coberturas])[None, :]
    # Cobertura de valor fixo com taxa 0: ticket = valor fixo, fora da otimização
    so_fixo = ativo & ~tem_taxa & eh_valor_fixo & (fixo > 0)
    # Elegível (entra na otimização): no pacote E com taxa > 0. Cobertura comum com
    # ticket recomendado mas taxa 0 (não elegível ao perfil) fica com IS 0 e ticket 0,
    # e o ticket dela é redistribuído para as demais (continua contando em T).
    elegivel = ativo & tem_taxa
    participa = elegivel | so_fixo                    # tem ticket final > 0
    esp_ativa = elegivel[:, k_esp]
    fixo = np.where(participa, fixo, 0.0)

    # 2) Limites de IS já em inteiros e convertidos em limites de ticket
    #    (assim o ótimo contínuo nunca sai do intervalo de IS inteiras válidas)
    lo, hi = limites_inteiros(ismin, ismax)
    L = np.where(participa, np.nan_to_num(lo * taxa) + fixo, 0.0)   # so_fixo: L = U = valor fixo
    U = np.where(participa, np.nan_to_num(hi * taxa) + fixo, 0.0)

    # Faixa sem nenhuma IS inteira possível (ceil(ismin) > floor(ismax)) é descartada
    faixa_valida = (lo[:, k_esp] <= hi[:, k_esp]) | ~esp_ativa
    U[:, k_esp] = np.where(faixa_valida, U[:, k_esp], L[:, k_esp])

    # 3) Ótimo contínuo (bisseção)
    r = resolver_distribuicao(t0, L, U, n_iter=n_iter)
    t_cont = r["t"]

    # 4) Ticket contínuo -> IS contínua -> IS inteira -> ticket final
    with np.errstate(divide="ignore", invalid="ignore"):
        is_cont = np.where(elegivel, (t_cont - fixo) / taxa, 0.0)
    teto = t0.sum(axis=1)
    if col_ticket_total and col_ticket_total in base.columns:
        teto = np.fmin(teto, base[col_ticket_total].to_numpy(float))
    is_int = arredondar_is(is_cont, lo, hi, taxa, fixo, participa,
                           alvo=np.minimum(t_cont.sum(axis=1), teto), teto=teto,
                           casas_ticket=casas_ticket)
    # Valor fixo com taxa 0: a IS não muda o ticket; sai 0 (o ticket é o valor fixo)
    is_int = np.where(so_fixo, 0.0, is_int)
    t_fin = np.where(participa, tickets(is_int, taxa, fixo, casas_ticket), 0.0)

    for j, c in enumerate(coberturas):
        base[col(c, "tf")] = t_fin[:, j]
        base[col(c, "isf")] = is_int[:, j].astype("int64")

    base["sobra"] = r["sobra"]                                   # o que não coube pelos limites
    base["dif_arredondamento"] = t_cont.sum(axis=1) - t_fin.sum(axis=1)   # efeito da IS inteira
    with np.errstate(divide="ignore", invalid="ignore"):
        base["distorcao"] = np.where(ativo, (t_fin - t0) ** 2 / np.where(ativo, t0, 1.0), 0.0).sum(axis=1)
    base["fator_escala"] = r["s"]

    # Checagem de segurança: IS inteira dentro de [ismin, ismax] originais
    fora = elegivel & ((is_int < np.round(ismin, CASAS_LIMPEZA)) | (is_int > np.round(ismax, CASAS_LIMPEZA)))
    fora_limites = fora.any(axis=1) & faixa_valida
    # Checagem de segurança: soma dos tickets finais acima do teto
    # (só é possível se nem as IS mínimas cabem no ticket total)
    acima_teto = passa_do_teto(t_fin.sum(axis=1), teto)

    status = np.where(np.abs(r["sobra"]) <= tol, "ok", "sobra_nao_distribuida")
    status = np.where((r["situacao"] == "piso") | acima_teto, "minimos_excedem_T", status)
    status = np.where(fora_limites, "is_fora_dos_limites", status)
    status = np.where(~faixa_valida, "faixa_invalida", status)
    base["status"] = status

    # 5) Melhor faixa por grupo:
    #    válida > não estoura mínimos > menor |sobra| > menor distorção > desempate de IS
    #    (sobra e distorção do ótimo contínuo, para o arredondamento não decidir a faixa)
    rank = pd.DataFrame({
        "grupo": grupo,
        "invalida": (~faixa_valida).astype(int),
        "piso": ((r["situacao"] == "piso") | acima_teto).astype(int),
        "sobra": np.abs(r["sobra"]).round(6),
        "dist": r["distorcao"].round(6),
        "is_esp": is_int[:, k_esp] * (1 if desempate_is == "menor" else -1),
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
        # pacote, cob_01_t, cob_02_t, cob_03_t, IS máx do Incêndio, IS máx do Roubo
        ("A", 600, 250, 150, 1_000_000, 50_000),
        ("B", 700, np.nan, 300, 1_000_000, 50_000),   # B não tem danos elétricos
        ("C", 600, 250, 150, 300_000, 20_000),        # C limita IS de incêndio e roubo
    ]
    faixas = [  # (início, fim, taxa, agravo_1) — faixas inteiras, sem sobreposição
        (50_000, 200_000, 0.0020, 20),
        (200_001, 500_000, 0.0015, 40),
        (500_001, 1_000_000, 0.0012, 60),
    ]

    linhas = []
    for pac, t1, t2, t3, ismax_inc, ismax_rou in pacotes:
        total = np.nansum([t1, t2, t3])
        for inicio, fim, taxa_f, ag1 in faixas:
            linhas.append({
                "perfil": "P1", "pacote": pac, "ticket_total": total,
                "cob_01_proporcao": t1 / total,
                "cob_02_proporcao": (t2 / total) if t2 == t2 else 0.0,
                "cob_03_proporcao": t3 / total,
                "cob_01_t": t1, "cob_02_t": t2, "cob_03_t": t3,
                # especial: limites da faixa ∩ limites da cobertura
                "cob_01_ismin": max(50_000, inicio), "cob_01_ismax": min(ismax_inc, fim),
                "cob_02_ismin": 5_000, "cob_02_ismax": 30_000,
                "cob_03_ismin": 2_000, "cob_03_ismax": ismax_rou,
                "cob_01_taxa": taxa_f, "cob_02_taxa": 0.005, "cob_03_taxa": 0.01,
                "agravo_1": ag1, "agravo_2": 30,
            })
    df = pd.DataFrame(linhas)

    res = calcular_is(
        df,
        coberturas=["cob_01", "cob_02", "cob_03"],
        especial="cob_01",
        chave_grupo=["perfil", "pacote"],
    )
    cols = ["pacote", "cob_01_ismin", "cob_01_ismax", "status", "sobra", "dif_arredondamento",
            "cob_01_tf", "cob_02_tf", "cob_03_tf", "cob_01_isf", "cob_02_isf", "cob_03_isf"]
    print(res[cols].round(2).to_string(index=False))
