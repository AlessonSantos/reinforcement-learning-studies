"""
Distribuição de ticket entre coberturas e cálculo da Importância Segurada (IS)
VERSÃO 3: versão 2 (ponderação pela IS sugerida pelo negócio) com a IS final
sempre arredondada na centena (múltiplo de 100: 72.000 ou 72.100, nunca 72.101).

Layout de entrada (linhas JÁ explodidas por faixa da cobertura especial):

    colunas_de_perfil..., ticket_total,
    <cob>_proporcao..., <cob>_t...,            # <cob>_t = ticket recomendado
    <cob>_ismin..., <cob>_ismax...,            # na especial, variam por faixa
    <cob>_taxa...,                             # na especial, varia por faixa
    <cob>_isref...                             # IS sugerida pelo negócio (opcional)
    agravo_1, agravo_2                         # agravos da especial

Os sufixos são configuráveis pelo parâmetro `sufixos`.

Saída: uma linha por pacote x perfil (a da faixa escolhida), com
    <cob>_tf          ticket final de cada cobertura (pode ter decimais)
    <cob>_isf         IS final de cada cobertura (múltiplo de 100, dentro de [ismin, ismax])
    <cob>_dif_ticket  ticket recomendado - ticket final
    <cob>_dif_prop    |proporção recomendada - proporção final|
    <cob>_dif_isref   IS final - IS sugerida (0 quando não há IS sugerida)
    + sobra, dif_arredondamento, distorcao, distorcao_is, objetivo,
      fator_escala, status

Modelo (por linha = pacote x perfil x faixa):
    min  sum_i (t_i - t0_i)^2 / t0_i                          (distorção de ticket)
       + beta * sum_i t0_i * ((IS_i - ISref_i) / ISref_i)^2   (distorção de IS)
    s.a. sum_i t_i = T
         L_i <= t_i <= U_i

Como IS_i = (t_i - fixo_i) / taxa_i, os dois termos são quadráticos em t_i e se
juntam em  w_i * (t_i - m_i)^2  (+ constante), com
    a_i = 1 / t0_i
    b_i = beta * t0_i / (taxa_i * ISref_i)^2
    tref_i = fixo_i + taxa_i * ISref_i        (ticket que corresponde à IS sugerida)
    w_i = a_i + b_i
    m_i = (a_i * t0_i + b_i * tref_i) / w_i   (alvo combinado)

Solução (KKT): t_i = clip(m_i + mu / w_i, L_i, U_i), com um único mu por linha,
encontrado por bisseção. Com beta = 0, volta exatamente à versão 1.

IS na centena: os limites viram o primeiro e o último múltiplo de 100 dentro de
[ismin, ismax] antes da otimização; o ótimo contínuo (que é o mesmo da versão 2)
é então arredondado (para baixo + ajuste guloso, de 100 em 100) sem sair desses
limites e sem a soma dos tickets passar do ticket total.
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
    "dif_ticket": "_dif_ticket",  # ticket recomendado - ticket final (saída)
    "dif_prop": "_dif_prop",      # |proporção recomendada - proporção final| (saída)
    "isref": "_isref",            # IS sugerida pelo negócio (entrada, opcional)
    "dif_isref": "_dif_isref",    # IS final - IS sugerida (saída)
}

BETA_PADRAO = 1.0   # peso da distorção de IS em relação à distorção de ticket


# ---------------------------------------------------------------------------
# Núcleo: resolve o problema para N linhas de uma vez (vetorizado)
# ---------------------------------------------------------------------------
def resolver_distribuicao(m: np.ndarray, w: np.ndarray, L: np.ndarray, U: np.ndarray,
                          T: np.ndarray, n_iter: int = 100):
    """
    Resolve, para N linhas de uma vez:
        min  sum_i w_i * (t_i - m_i)^2
        s.a. sum_i t_i = T,   L_i <= t_i <= U_i

    m, w, L, U: matrizes (n_linhas x n_coberturas); T: vetor (n_linhas).
    Coberturas inativas devem ter L = U = 0 (m e w quaisquer, w > 0).

    Solução: t_i(mu) = clip(m_i + mu / w_i, L_i, U_i). A soma de t(mu) cresce com
    mu, então mu é achado por bisseção.

    Retorna dict com t (tickets), mu, sobra e situacao
    ('ok' | 'teto' (soma U < T) | 'piso' (soma L > T)).
    """
    m = np.asarray(m, float)
    w = np.asarray(w, float)
    L = np.asarray(L, float)
    U = np.asarray(U, float)

    soma_L = L.sum(axis=1)
    soma_U = U.sum(axis=1)

    # Intervalo de busca de mu: em mu_lo todas as coberturas estão no piso (L);
    # em mu_hi, todas no teto (U).
    mu_lo = ((L - m) * w).min(axis=1) - 1.0
    mu_hi = ((U - m) * w).max(axis=1) + 1.0

    def soma_t(mu):
        return np.clip(m + mu[:, None] / w, L, U).sum(axis=1)

    # Bisseção vetorizada (soma_t é não decrescente em mu)
    for _ in range(n_iter):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        abaixo = soma_t(mu_mid) < T
        mu_lo = np.where(abaixo, mu_mid, mu_lo)
        mu_hi = np.where(abaixo, mu_hi, mu_mid)
    mu = 0.5 * (mu_lo + mu_hi)

    t = np.clip(m + mu[:, None] / w, L, U)

    situacao = np.full(len(T), "ok", dtype=object)
    teto = soma_U < T
    piso = soma_L > T
    t[teto] = U[teto]
    t[piso] = L[piso]
    situacao[teto] = "teto"
    situacao[piso] = "piso"

    return {"t": t, "mu": mu, "sobra": T - t.sum(axis=1), "situacao": situacao}


# ---------------------------------------------------------------------------
# IS na centena (múltiplos de 100)
# ---------------------------------------------------------------------------
CASAS_LIMPEZA = 6   # casas decimais usadas para limpar ruído de ponto flutuante
PASSO_IS = 100      # a IS final é sempre múltiplo de 100


def limites_centena(ismin: np.ndarray, ismax: np.ndarray):
    """
    Limites de IS válidos em múltiplos de 100:
        lo = menor múltiplo de 100  >= ismin   (ceil)
        hi = maior múltiplo de 100  <= ismax   (floor)
    Ex.: ismin 75.001 -> 75.100; ismax 199.950 -> 199.900.
    Antes de ceil/floor, arredonda para CASAS_LIMPEZA casas, para que ruídos como
    75000.0000000001 não virem 75100 (nem 74999.9999999 vire 74900).
    """
    lo = np.ceil(np.round(ismin / PASSO_IS, CASAS_LIMPEZA)) * PASSO_IS
    hi = np.floor(np.round(ismax / PASSO_IS, CASAS_LIMPEZA)) * PASSO_IS
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
    Converte a IS contínua (ótimo do modelo) em IS múltipla de 100, linha a
    linha, com a garantia de que a soma dos tickets finais NUNCA passa do teto.

    Internamente, tudo é feito em "centenas" (IS / 100): subir 1 unidade
    significa subir a IS em R$ 100 (ex.: 72.000 -> 72.100).
    lo e hi já devem ser múltiplos de 100 (ver limites_centena).

    is_cont, lo, hi, taxa, fixo, ativo: matrizes (linhas x coberturas)
        ticket_i = IS_i * taxa_i + fixo_i   (fixo = agravos na especial, 0 nas demais)
    alvo: soma de tickets que o ótimo contínuo distribuiu (por linha)
    teto: ticket total máximo permitido (por linha)
    casas_ticket: se informado, os tickets são arredondados nessas casas e a
        garantia vale para os tickets já arredondados.

    1) começa com a IS arredondada para baixo na centena, sempre dentro de [lo, hi];
    2) percorre as coberturas da maior para a menor parte fracionária (das centenas)
       e sobe a IS em +100 quando isso aproxima a soma do alvo, sem passar de hi
       e sem a soma passar do teto;
    3) trava final: recalcula a soma exatamente como as colunas <cob>_tf e, se alguma
       linha passar do teto, desce 100 de IS da cobertura de maior taxa que
       ainda pode descer, até caber.
    """
    # Converte para centenas (lo e hi viram inteiros exatos)
    lo = np.round(lo / PASSO_IS)
    hi = np.round(hi / PASSO_IS)
    x = np.round(np.where(ativo, is_cont / PASSO_IS, 0.0), CASAS_LIMPEZA)
    base = np.where(ativo, np.clip(np.floor(x), lo, hi), 0.0)
    frac = np.where(ativo, x - base, -1.0)

    taxa0 = np.where(ativo, np.nan_to_num(taxa), 0.0)
    fixo0 = np.where(ativo, fixo, 0.0)

    def soma(b):
        return tickets(b * PASSO_IS, taxa0, fixo0, casas_ticket).sum(axis=1)

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

    return base * PASSO_IS                             # de volta para reais


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
    beta: float = BETA_PADRAO,
    tol: float = 1e-6,
    n_iter: int = 100,
    retornar_todas_faixas: bool = False,
    linhas_por_bloco: int | None = 300_000,
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
    coberturas_valor_fixo  coberturas cujo ticket é  valor_fixo + IS * taxa.
                  Dicionário {cobertura: {"valor": coluna_do_valor_fixo}}.
                  A taxa já vem dividida por 100, como nas demais coberturas.
                  - A coluna de taxa (<cob>_taxa) é opcional: se não existir, taxa = 0.
                  - Com taxa 0, o ticket final é exatamente o valor fixo e a IS final
                    sai 0. A diferença entre o ticket recomendado (<cob>_t) e o valor
                    fixo é redistribuída entre as demais coberturas (ou retirada
                    delas, se o valor fixo for maior que o recomendado).
                  - Com taxa > 0, a cobertura entra na otimização como qualquer
                    outra, com o valor fixo somado ao ticket.
                  As coberturas daqui também devem estar na lista `coberturas`.
    beta          peso da distorção de IS (afastamento da IS sugerida <cob>_isref)
                  em relação à distorção de ticket. beta = 1: afastar a IS 10% da
                  sugerida custa o mesmo que afastar o ticket 10% do recomendado.
                  beta = 0: ignora a IS sugerida (igual à versão 1).
                  Coberturas sem coluna <cob>_isref (ou com valor vazio/0) não são
                  ponderadas.
    retornar_todas_faixas  se True, devolve também o resultado de cada faixa (auditoria).
    n_iter        iterações da bisseção. 100 já chega à precisão máxima do float.
    linhas_por_bloco  bases grandes são processadas em blocos de ~esse número de
                  linhas, sempre com todas as faixas de um mesmo pacote x perfil no
                  mesmo bloco. Limita o uso de memória sem mudar o resultado.
                  Use None para processar tudo de uma vez.

    Retorna o df com uma linha por grupo (a da faixa escolhida) acrescido de
    <cob>_tf (ticket final, pode ter decimais), <cob>_isf (IS final, múltiplo de 100),
    <cob>_dif_ticket (ticket recomendado - ticket final),
    <cob>_dif_prop (|proporção recomendada - proporção final|),
    <cob>_dif_isref (IS final - IS sugerida),
    sobra, dif_arredondamento, distorcao, distorcao_is, objetivo, fator_escala e status.
    """
    # Bases grandes: processa em blocos de grupos inteiros (resultado idêntico)
    if linhas_por_bloco and len(df) > linhas_por_bloco:
        params = dict(coberturas=coberturas, especial=especial, chave_grupo=chave_grupo,
                      col_agravo1=col_agravo1, col_agravo2=col_agravo2, sufixos=sufixos,
                      desempate_is=desempate_is, col_ticket_total=col_ticket_total,
                      casas_ticket=casas_ticket, coberturas_valor_fixo=coberturas_valor_fixo,
                      beta=beta, tol=tol, n_iter=n_iter, retornar_todas_faixas=retornar_todas_faixas,
                      linhas_por_bloco=None)
        g = df.groupby(chave_grupo, dropna=False, sort=False).ngroup().to_numpy()
        n_blocos = int(np.ceil(len(df) / linhas_por_bloco))
        bloco_do_grupo = (g * n_blocos) // (g.max() + 1)          # grupos inteiros por bloco
        saidas = [calcular_is(df[bloco_do_grupo == b], **params) for b in range(n_blocos)
                  if (bloco_do_grupo == b).any()]
        if retornar_todas_faixas:
            return (pd.concat([x[0] for x in saidas], ignore_index=True),
                    pd.concat([x[1] for x in saidas], ignore_index=True))
        return pd.concat(saidas, ignore_index=True)

    sx = {**SUFIXOS_PADRAO, **(sufixos or {})}
    k_esp = coberturas.index(especial)
    cvf = dict(coberturas_valor_fixo or {})
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

    # Taxa (já dividida por 100 no pré-processamento; sem coluna de taxa => 0)
    taxa = np.nan_to_num(np.column_stack([coluna(col(c, "taxa")) for c in coberturas]))
    ismin = np.column_stack([coluna(col(c, "ismin")) for c in coberturas])
    ismax = np.column_stack([coluna(col(c, "ismax")) for c in coberturas])
    # IS sugerida (opcional: sem coluna, vazia ou 0 => sem ponderação de IS)
    isref = np.nan_to_num(np.column_stack([coluna(col(c, "isref")) for c in coberturas])).clip(min=0)

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

    # 2) Limites de IS já na centena e convertidos em limites de ticket
    #    (assim o ótimo contínuo nunca sai do intervalo de IS válidas)
    lo, hi = limites_centena(ismin, ismax)
    L = np.where(participa, np.nan_to_num(lo * taxa) + fixo, 0.0)   # so_fixo: L = U = valor fixo
    U = np.where(participa, np.nan_to_num(hi * taxa) + fixo, 0.0)

    # Faixa sem nenhum múltiplo de 100 entre ismin e ismax é descartada
    faixa_valida = (lo[:, k_esp] <= hi[:, k_esp]) | ~esp_ativa
    U[:, k_esp] = np.where(faixa_valida, U[:, k_esp], L[:, k_esp])
    # Cobertura comum sem nenhum múltiplo de 100 entre ismin e ismax
    # (ex.: ismin 1.050 e ismax 1.090): não há IS válida; a linha é sinalizada
    # com status "cobertura_sem_is_valida"
    nao_esp = (np.arange(len(coberturas)) != k_esp)[None, :]
    sem_multiplo = elegivel & nao_esp & (lo > hi)
    cob_sem_is_valida = sem_multiplo.any(axis=1)
    U = np.where(sem_multiplo, L, U)

    # 3) Alvo combinado (seção 4.5 do markdown)
    #    a = peso da distorção de ticket; b = peso da distorção de IS
    com_ref = elegivel & (isref > 0)                  # coberturas ponderadas pela IS
    with np.errstate(divide="ignore", invalid="ignore"):
        a = np.where(ativo, 1.0 / np.where(ativo, t0, 1.0), 1.0)
        b = np.where(com_ref, beta * t0 / np.where(com_ref, taxa * isref, 1.0) ** 2, 0.0)
    tref = fixo + taxa * isref                        # ticket que corresponde à IS sugerida
    w = a + b
    m = np.where(ativo, (a * t0 + b * tref) / w, 0.0)

    # 4) Ótimo contínuo (bisseção)
    T = t0.sum(axis=1)
    r = resolver_distribuicao(m, w, L, U, T, n_iter=n_iter)
    t_cont = r["t"]

    # 5) Ticket contínuo -> IS contínua -> IS na centena -> ticket final
    with np.errstate(divide="ignore", invalid="ignore"):
        is_cont = np.where(elegivel, (t_cont - fixo) / taxa, 0.0)
    teto = T.copy()
    if col_ticket_total and col_ticket_total in base.columns:
        teto = np.fmin(teto, base[col_ticket_total].to_numpy(float))
    is_int = arredondar_is(is_cont, lo, np.maximum(lo, hi), taxa, fixo, participa,
                           alvo=np.minimum(t_cont.sum(axis=1), teto), teto=teto,
                           casas_ticket=casas_ticket)
    # Valor fixo com taxa 0: a IS não muda o ticket; sai 0 (o ticket é o valor fixo)
    is_int = np.where(so_fixo, 0.0, is_int)
    t_fin = np.where(participa, tickets(is_int, taxa, fixo, casas_ticket), 0.0)

    # Proporções: recomendada = t0 / soma(t0); final = tf / soma(tf)
    with np.errstate(divide="ignore", invalid="ignore"):
        soma_t0 = t0.sum(axis=1, keepdims=True)
        soma_tf = t_fin.sum(axis=1, keepdims=True)
        prop_rec = np.where(soma_t0 > 0, t0 / soma_t0, 0.0)
        prop_fin = np.where(soma_tf > 0, t_fin / soma_tf, 0.0)

    for j, c in enumerate(coberturas):
        base[col(c, "tf")] = t_fin[:, j]
        base[col(c, "isf")] = is_int[:, j].astype("int64")
        base[col(c, "dif_ticket")] = t0[:, j] - t_fin[:, j]
        base[col(c, "dif_prop")] = np.abs(prop_rec[:, j] - prop_fin[:, j])
        base[col(c, "dif_isref")] = np.where(isref[:, j] > 0, is_int[:, j] - isref[:, j], 0.0)

    base["sobra"] = r["sobra"]                                   # o que não coube pelos limites
    base["dif_arredondamento"] = t_cont.sum(axis=1) - t_fin.sum(axis=1)   # efeito da IS na centena

    # Componentes do objetivo: distorção de ticket e distorção de IS
    def componentes(t, is_):
        with np.errstate(divide="ignore", invalid="ignore"):
            d_t = np.where(ativo, (t - t0) ** 2 / np.where(ativo, t0, 1.0), 0.0).sum(axis=1)
            d_is = np.where(com_ref, t0 * ((is_ - isref) / np.where(com_ref, isref, 1.0)) ** 2,
                            0.0).sum(axis=1)
        return d_t, d_is

    d_t, d_is = componentes(t_fin, is_int)           # medidos na solução final (IS na centena)
    base["distorcao"] = d_t                           # distorção de ticket
    base["distorcao_is"] = d_is                       # distorção de IS (sem o beta)
    base["objetivo"] = d_t + beta * d_is
    base["fator_escala"] = 1.0 + r["mu"]              # com beta = 0, é o s da versão 1

    # Checagem de segurança: IS final dentro de [ismin, ismax] originais e na centena
    fora = elegivel & ((is_int < np.round(ismin, CASAS_LIMPEZA)) | (is_int > np.round(ismax, CASAS_LIMPEZA))
                       | (np.round(is_int) % PASSO_IS != 0))
    fora_limites = fora.any(axis=1) & faixa_valida & ~cob_sem_is_valida
    # Checagem de segurança: soma dos tickets finais acima do teto
    # (só é possível se nem as IS mínimas cabem no ticket total)
    acima_teto = passa_do_teto(t_fin.sum(axis=1), teto)

    status = np.where(np.abs(r["sobra"]) <= tol, "ok", "sobra_nao_distribuida")
    status = np.where((r["situacao"] == "piso") | acima_teto, "minimos_excedem_T", status)
    status = np.where(fora_limites, "is_fora_dos_limites", status)
    status = np.where(cob_sem_is_valida, "cobertura_sem_is_valida", status)
    status = np.where(~faixa_valida, "faixa_invalida", status)
    base["status"] = status

    # 6) Melhor faixa por grupo:
    #    válida > não estoura mínimos > menor |sobra| > menor objetivo > desempate de IS
    #    (sobra e objetivo do ótimo contínuo, para o arredondamento não decidir a faixa)
    d_t_cont, d_is_cont = componentes(t_cont, is_cont)
    rank = pd.DataFrame({
        "grupo": grupo,
        "invalida": (~faixa_valida).astype(int),
        "piso": ((r["situacao"] == "piso") | acima_teto).astype(int),
        "sobra": np.abs(r["sobra"]).round(6),
        "dist": (d_t_cont + beta * d_is_cont).round(6),
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
                # IS sugeridas pelo negócio
                "cob_01_isref": 350_000, "cob_02_isref": 30_000, "cob_03_isref": 15_000,
            })
    df = pd.DataFrame(linhas)

    for beta in (0.0, BETA_PADRAO):
        res = calcular_is(
            df,
            coberturas=["cob_01", "cob_02", "cob_03"],
            especial="cob_01",
            chave_grupo=["perfil", "pacote"],
            beta=beta,
        )
        print(f"beta = {beta}")
        cols = ["pacote", "status", "cob_01_tf", "cob_02_tf", "cob_03_tf",
                "cob_01_isf", "cob_02_isf", "cob_03_isf", "cob_01_dif_isref", "cob_03_dif_isref"]
        print(res[cols].round(2).to_string(index=False))
        print()

