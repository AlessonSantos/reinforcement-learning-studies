"""
Treino em lote do modelo de recomendação de coberturas.

Resumo da arquitetura implementada aqui (ver documentação do projeto para detalhes):
- Um único theta, compartilhado entre as coberturas OPCIONAIS (as obrigatórias nunca
  entram no treino, porque nunca têm variação em y: são sempre aceitas).
- x(i) = concat(c, f_i, c⊗f_i), onde c é o perfil do cliente e f_i são os atributos
  semânticos da cobertura (categóricos viram one-hot; numéricos entram como estão).
- theta é ajustado por máxima verossimilhança a posteriori (MAP): verossimilhança de
  Bernoulli + priori gaussiana N(0, 1/lambda * I). Não tem forma fechada (regressão
  logística não é conjugada com priori gaussiana), por isso usamos um otimizador.
- H_theta é a aproximação de Laplace da incerteza de theta, com um piso de curvatura
  (tau) que evita que itens quase certos parem de reduzir sua própria incerteza.
- Para cada cobertura opcional, calcula-se também um "baseline": a taxa de aceitação
  observada SÓ nas rodadas aleatórias reservadas para isso (nunca misturada com o
  treino de theta), usada depois para o teste de significância na escoragem.

Este script assume que os dados chegam em DataFrames já no formato descrito nas
funções abaixo. Para gerar dados de exemplo nesse formato, ver gerar_dados_exemplo.py.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import pickle
from datetime import datetime


# ---------------------------------------------------------------------------
# Etapa 1: construir os vetores f_i (atributos semânticos) de cada cobertura
# ---------------------------------------------------------------------------

def construir_esquema_codificacao(df_catalogo, colunas_categoricas):
    """
    Constrói o esquema de codificação one-hot das colunas categóricas do catálogo,
    de forma determinística (mesma ordem sempre), para poder ser reaplicado
    depois na escoragem sem risco de gerar uma codificação diferente.

    Parâmetros
    ----------
    df_catalogo : pd.DataFrame
        Uma linha por cobertura, com pelo menos as colunas em `colunas_categoricas`.
    colunas_categoricas : list[str]
        Nomes das colunas categóricas do catálogo (ex.: ["categoria_risco", "faixa_premio"]).

    Retorna
    -------
    dict
        Para cada coluna categórica, a lista ordenada de categorias observadas
        no catálogo de treino. Ex.: {"categoria_risco": ["ambiental", "seguranca"]}.
        A ordem dessa lista define a posição de cada categoria no vetor one-hot.
    """
    esquema = {}
    for coluna in colunas_categoricas:
        categorias_unicas = sorted(df_catalogo[coluna].dropna().unique().tolist())
        esquema[coluna] = categorias_unicas
    return esquema


def codificar_categorica_one_hot(valor, categorias_possiveis):
    """
    Codifica um único valor categórico como vetor one-hot, usando uma lista fixa
    e ordenada de categorias possíveis (para garantir a mesma codificação sempre).

    Parâmetros
    ----------
    valor : str
        O valor categórico observado (ex.: "ambiental").
    categorias_possiveis : list[str]
        Lista ordenada de todas as categorias possíveis para essa coluna
        (gerada por construir_esquema_codificacao).

    Retorna
    -------
    np.ndarray
        Vetor one-hot de tamanho len(categorias_possiveis). Se `valor` não estiver
        na lista (categoria nunca vista no treino), retorna um vetor de zeros —
        isso deveria ser raro e vale logar/alertar se acontecer.
    """
    vetor = np.zeros(len(categorias_possiveis))
    if valor in categorias_possiveis:
        vetor[categorias_possiveis.index(valor)] = 1.0
    return vetor


def construir_f_cobertura(linha_catalogo, esquema_codificacao, colunas_categoricas, colunas_numericas):
    """
    Constrói o vetor f_i de UMA cobertura, concatenando as versões one-hot das
    colunas categóricas com as colunas numéricas (já assumidas normalizadas
    previamente pelo usuário, se necessário).

    Parâmetros
    ----------
    linha_catalogo : pd.Series
        Uma linha do df_catalogo, com os atributos daquela cobertura específica.
    esquema_codificacao : dict
        Saída de construir_esquema_codificacao — mapeia cada coluna categórica
        para a lista ordenada de categorias possíveis.
    colunas_categoricas : list[str]
        Nomes das colunas categóricas a incluir em f_i.
    colunas_numericas : list[str]
        Nomes das colunas numéricas a incluir em f_i (entram sem transformação;
        normalização/escala é responsabilidade de quem monta o catálogo).

    Retorna
    -------
    np.ndarray
        O vetor f_i completo (concatenação das partes one-hot com as numéricas).
    """
    partes = []
    for coluna in colunas_categoricas:
        partes.append(codificar_categorica_one_hot(linha_catalogo[coluna], esquema_codificacao[coluna]))
    if colunas_numericas:
        partes.append(linha_catalogo[colunas_numericas].to_numpy(dtype=float))
    return np.concatenate(partes)


def construir_dicionario_f(df_catalogo, esquema_codificacao, colunas_categoricas, colunas_numericas, coluna_id):
    """
    Constrói f_i para TODAS as coberturas do catálogo de uma vez, retornando um
    dicionário {cobertura_id: vetor f_i}.

    Parâmetros
    ----------
    df_catalogo : pd.DataFrame
        Uma linha por cobertura, com a coluna de id e os atributos semânticos.
    esquema_codificacao : dict
        Saída de construir_esquema_codificacao.
    colunas_categoricas, colunas_numericas : list[str]
        Ver construir_f_cobertura.
    coluna_id : str
        Nome da coluna que identifica unicamente cada cobertura (ex.: "cobertura_id").

    Retorna
    -------
    dict
        {cobertura_id: np.ndarray} — o vetor f_i de cada cobertura do catálogo.
    """
    dicionario_f = {}
    for _, linha in df_catalogo.iterrows():
        dicionario_f[linha[coluna_id]] = construir_f_cobertura(
            linha, esquema_codificacao, colunas_categoricas, colunas_numericas
        )
    return dicionario_f


# ---------------------------------------------------------------------------
# Etapa 2: montar x(i) = concat(c, f_i, c⊗f_i) para cada observação
# ---------------------------------------------------------------------------

def montar_x(vetor_c, vetor_f):
    """
    Monta o vetor de contexto x(i) = concat(c, f_i, c⊗f_i) para uma observação.

    Parâmetros
    ----------
    vetor_c : np.ndarray
        Perfil do cliente daquela observação (ex.: [idade_norm, renda_norm]).
    vetor_f : np.ndarray
        Atributos semânticos da cobertura (f_i), já codificados.

    Retorna
    -------
    np.ndarray
        x(i), com dimensão len(vetor_c) + len(vetor_f) + len(vetor_c)*len(vetor_f).
    """
    interacao = np.outer(vetor_c, vetor_f).flatten()
    return np.concatenate([vetor_c, vetor_f, interacao])


# ---------------------------------------------------------------------------
# Etapa 3: preparar as matrizes de treino (X, y) a partir do histórico
# ---------------------------------------------------------------------------

def preparar_dados_treino_theta(df_interacoes, dicionario_f, coberturas_obrigatorias,
                                 colunas_cliente, coluna_cobertura_id, coluna_y,
                                 coluna_origem, valores_origem_excluidos_do_treino):
    """
    Filtra e monta a matriz X (contextos) e o vetor y (rótulos) usados para
    treinar theta — exclui observações de coberturas obrigatórias (não têm
    variação, não ensinam nada) e exclui observações marcadas como "reservadas
    para o baseline" (essas não podem contaminar o treino de theta).

    Parâmetros
    ----------
    df_interacoes : pd.DataFrame
        Uma linha por observação (cliente + cobertura + resultado). Precisa ter,
        no mínimo: as colunas de perfil do cliente (`colunas_cliente`), a coluna
        de id da cobertura, a coluna de resultado (0/1), e a coluna de origem
        da rodada.
    dicionario_f : dict
        Saída de construir_dicionario_f — {cobertura_id: vetor f_i}.
    coberturas_obrigatorias : set ou list
        Ids das coberturas obrigatórias — observações dessas coberturas são
        excluídas do treino de theta.
    colunas_cliente : list[str]
        Nomes das colunas de df_interacoes que formam o vetor c do cliente
        (ex.: ["idade_norm", "renda_norm"]).
    coluna_cobertura_id : str
        Nome da coluna com o id da cobertura em df_interacoes.
    coluna_y : str
        Nome da coluna com o resultado observado (1 = manteve/aceitou, 0 = não).
    coluna_origem : str
        Nome da coluna que indica a origem da rodada daquela observação.
    valores_origem_excluidos_do_treino : list[str]
        Quais valores de `coluna_origem` identificam observações reservadas
        para o cálculo do baseline (e que, por isso, não devem entrar aqui).

    Retorna
    -------
    (np.ndarray, np.ndarray)
        X (N, d) e y (N,), prontos para o ajuste de theta.
    """
    df_filtrado = df_interacoes[
        (~df_interacoes[coluna_cobertura_id].isin(coberturas_obrigatorias))
        & (~df_interacoes[coluna_origem].isin(valores_origem_excluidos_do_treino))
    ]

    linhas_x = []
    for _, linha in df_filtrado.iterrows():
        vetor_c = linha[colunas_cliente].to_numpy(dtype=float)
        vetor_f = dicionario_f[linha[coluna_cobertura_id]]
        linhas_x.append(montar_x(vetor_c, vetor_f))

    X = np.array(linhas_x)
    y = df_filtrado[coluna_y].to_numpy(dtype=float)
    return X, y


# ---------------------------------------------------------------------------
# Etapa 4: ajustar theta via MAP (máxima verossimilhança a posteriori)
# ---------------------------------------------------------------------------

def sigmoide(z):
    """
    Função logística padrão, aplicada de forma numericamente estável
    (evita overflow para valores muito negativos ou muito positivos de z).

    Parâmetros
    ----------
    z : np.ndarray ou float
        O(s) valor(es) de entrada (log-odds).

    Retorna
    -------
    np.ndarray ou float
        sigma(z) = 1 / (1 + exp(-z)), sempre em (0, 1).
    """
    return np.where(z >= 0, 1 / (1 + np.exp(-z)), np.exp(z) / (1 + np.exp(z)))


def neg_log_posteriori(theta, X, y, lam):
    """
    Log-posteriori NEGATIVA (o que o otimizador minimiza), somando a verossimilhança
    de Bernoulli de todas as observações com a penalização da priori gaussiana.

    Parâmetros
    ----------
    theta : np.ndarray
        O vetor de parâmetros sendo avaliado (dimensão d).
    X : np.ndarray
        Matriz de contextos (N, d), uma linha por observação.
    y : np.ndarray
        Vetor de rótulos observados (N,), valores em {0, 1}.
    lam : float
        Coeficiente de regularização (precisão da priori gaussiana N(0, 1/lam * I)).

    Retorna
    -------
    float
        O valor da log-posteriori negativa em theta (a ser minimizado).
    """
    p = sigmoide(X @ theta)
    eps = 1e-12  # evita log(0) por erro numérico quando p fica extremamente perto de 0 ou 1
    log_verossimilhanca = np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
    penalizacao = 0.5 * lam * np.sum(theta ** 2)
    return -(log_verossimilhanca) + penalizacao


def gradiente_neg_log_posteriori(theta, X, y, lam):
    """
    Gradiente da log-posteriori negativa em relação a theta — usado pelo
    otimizador para convergir mais rápido e de forma mais estável do que
    sem gradiente analítico.

    Parâmetros, e retorno
    ----------------------
    Mesmos parâmetros de neg_log_posteriori.
    Retorna um np.ndarray de dimensão d (o gradiente).
    """
    p = sigmoide(X @ theta)
    gradiente_verossimilhanca = X.T @ (y - p)
    gradiente_penalizacao = lam * theta
    return -(gradiente_verossimilhanca) + gradiente_penalizacao


def ajustar_theta(X, y, lam, theta_inicial=None):
    """
    Encontra o theta que maximiza a posteriori (MAP), via otimizador L-BFGS-B.
    Não existe forma fechada aqui (regressão logística não é conjugada com
    priori gaussiana), por isso a necessidade de otimização iterativa.

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de contextos (N, d).
    y : np.ndarray
        Vetor de rótulos (N,).
    lam : float
        Coeficiente de regularização.
    theta_inicial : np.ndarray ou None
        Ponto de partida do otimizador. Se None, começa em zero (o "chute"
        neutro da priori, sem nenhuma informação prévia).

    Retorna
    -------
    np.ndarray
        theta_hat, o vetor de parâmetros ajustado (dimensão d).
    """
    d = X.shape[1]
    if theta_inicial is None:
        theta_inicial = np.zeros(d)

    resultado = minimize(
        fun=neg_log_posteriori,
        x0=theta_inicial,
        args=(X, y, lam),
        jac=gradiente_neg_log_posteriori,
        method="L-BFGS-B",
    )
    return resultado.x


# ---------------------------------------------------------------------------
# Etapa 5: calcular H_theta (aproximação de Laplace, com piso de curvatura)
# ---------------------------------------------------------------------------

def calcular_H_theta(theta_hat, X, lam, tau):
    """
    Calcula H_theta via aproximação de Laplace, com o piso de curvatura tau
    aplicado por observação (max entre o peso real p(1-p) e tau) — isso evita
    que observações de itens "quase certos" parem de reduzir a incerteza,
    mesmo depois de muitas confirmações repetidas (o "efeito kappa").

    Parâmetros
    ----------
    theta_hat : np.ndarray
        O theta já ajustado (saída de ajustar_theta).
    X : np.ndarray
        A MESMA matriz de contextos usada para ajustar theta_hat (N, d).
    lam : float
        O mesmo coeficiente de regularização usado no ajuste (entra como lam*I).
    tau : float
        O piso de curvatura. Ponto de partida sugerido: algo entre 0,02 e 0,10
        (referência: p(1-p) tem máximo 0,25, em p=0,5).

    Retorna
    -------
    np.ndarray
        H_theta, uma matriz (d, d).
    """
    d = X.shape[1]
    p = sigmoide(X @ theta_hat)
    pesos = np.maximum(p * (1 - p), tau)

    # soma ponderada de x_n x_n^T -- equivalente a X^T diag(pesos) X, calculado
    # de forma vetorizada (mais rápido que um loop somando outer products)
    H = lam * np.eye(d) + X.T @ (X * pesos[:, None])
    return H


# ---------------------------------------------------------------------------
# Etapa 6: calcular o baseline (taxa de referência) de cada cobertura opcional
# ---------------------------------------------------------------------------

def calcular_baseline_por_cobertura(df_interacoes, coberturas_opcionais,
                                     coluna_cobertura_id, coluna_y, coluna_origem,
                                     valor_origem_reservada_baseline,
                                     coluna_ofertada_aleatoriamente=None):
    """
    Calcula, para cada cobertura opcional, a taxa de aceitação observada
    SÓ na fatia de rodadas aleatórias reservada para isso — nunca misturando
    com dados usados no treino de theta, e (se a coluna for informada) nunca
    contando adições espontâneas (só contam observações em que a cobertura foi
    genuinamente sorteada para a oferta aleatória).

    Parâmetros
    ----------
    df_interacoes : pd.DataFrame
        O mesmo histórico completo de observações.
    coberturas_opcionais : list
        Ids das coberturas opcionais (as obrigatórias não têm baseline).
    coluna_cobertura_id, coluna_y, coluna_origem : str
        Ver preparar_dados_treino_theta.
    valor_origem_reservada_baseline : str
        O valor de `coluna_origem` que identifica as rodadas reservadas
        para o cálculo do baseline.
    coluna_ofertada_aleatoriamente : str ou None
        Nome da coluna booleana que indica se aquela cobertura foi de fato
        sorteada para a oferta aleatória naquela rodada (True) ou se foi uma
        adição espontânea do cliente (False). Se None, assume que toda linha
        marcada com a origem reservada já representa uma oferta genuína
        (recomendado sempre informar essa coluna, quando disponível).

    Retorna
    -------
    dict
        {cobertura_id: {"n": int, "b": float, "logit_b": float, "var_logit_b": float}}
        Para coberturas sem nenhuma observação reservada ainda, "n" vem como 0
        e os demais campos vêm como None — a Etapa 5 da escoragem (teste de
        significância) precisa tratar esse caso, deixando a cobertura de fora
        da decisão até haver dado suficiente.
    """
    df_baseline = df_interacoes[
        (df_interacoes[coluna_cobertura_id].isin(coberturas_opcionais))
        & (df_interacoes[coluna_origem] == valor_origem_reservada_baseline)
    ]
    if coluna_ofertada_aleatoriamente is not None:
        df_baseline = df_baseline[df_baseline[coluna_ofertada_aleatoriamente] == True]  # noqa: E712

    resultado = {}
    for cobertura_id in coberturas_opcionais:
        subconjunto = df_baseline[df_baseline[coluna_cobertura_id] == cobertura_id]
        n = len(subconjunto)
        if n == 0:
            resultado[cobertura_id] = {"n": 0, "b": None, "logit_b": None, "var_logit_b": None}
            continue

        b = subconjunto[coluna_y].mean()
        # protege contra b=0 ou b=1 exatos, que quebrariam o logit e a variância
        b_ajustado = min(max(b, 1e-4), 1 - 1e-4)
        logit_b = np.log(b_ajustado / (1 - b_ajustado))
        var_logit_b = 1 / (n * b_ajustado * (1 - b_ajustado))

        resultado[cobertura_id] = {"n": n, "b": b, "logit_b": logit_b, "var_logit_b": var_logit_b}

    return resultado


# ---------------------------------------------------------------------------
# Orquestração: treino completo, do histórico bruto ao pickle final
# ---------------------------------------------------------------------------

def treinar_modelo_batch(
    df_interacoes,
    df_catalogo,
    colunas_cliente,
    colunas_categoricas_cobertura,
    colunas_numericas_cobertura,
    coluna_cobertura_id="cobertura_id",
    coluna_obrigatoria="obrigatoria",
    coluna_y="y",
    coluna_origem="origem",
    valor_origem_reservada_baseline="aleatoria_reservada_baseline",
    coluna_ofertada_aleatoriamente="ofertada_aleatoriamente",
    lam=1.0,
    tau=0.05,
    z_critico=1.645,
    caminho_saida="modelo_coberturas.pkl",
):
    """
    Função principal do treino em lote: recebe o histórico e o catálogo brutos
    (DataFrames), monta tudo, ajusta o modelo, calcula os baselines, e salva
    um pickle com tudo que a escoragem vai precisar.

    Parâmetros
    ----------
    df_interacoes : pd.DataFrame
        Histórico completo de observações (uma linha por cliente+cobertura+rodada).
        Colunas mínimas esperadas: as de `colunas_cliente`, `coluna_cobertura_id`,
        `coluna_y`, `coluna_origem`, e (se disponível) `coluna_ofertada_aleatoriamente`.
    df_catalogo : pd.DataFrame
        Uma linha por cobertura do catálogo (14 linhas, no caso de vocês).
        Colunas mínimas: `coluna_cobertura_id`, `coluna_obrigatoria`, e as colunas
        semânticas listadas em `colunas_categoricas_cobertura`/`colunas_numericas_cobertura`.
    colunas_cliente : list[str]
        Nomes das colunas de df_interacoes que formam o vetor c do cliente.
    colunas_categoricas_cobertura, colunas_numericas_cobertura : list[str]
        Nomes das colunas do catálogo que formam f_i (ver construir_f_cobertura).
    coluna_cobertura_id, coluna_obrigatoria, coluna_y, coluna_origem : str
        Nomes de colunas — ajuste conforme o schema real dos seus dados.
    valor_origem_reservada_baseline : str
        Ver calcular_baseline_por_cobertura.
    coluna_ofertada_aleatoriamente : str ou None
        Ver calcular_baseline_por_cobertura.
    lam : float
        Regularização (mesmo lambda usado em theta e em H_theta).
    tau : float
        Piso de curvatura.
    z_critico : float
        Valor crítico do teste de significância usado na escoragem (salvo aqui
        para ficar tudo junto no mesmo pickle). 1,645 = 95% de confiança, um lado.
    caminho_saida : str
        Caminho do arquivo pickle a ser gerado.

    Retorna
    -------
    dict
        O mesmo dicionário que foi salvo no pickle (retornado também em memória,
        útil se quiser inspecionar ou usar sem reler o arquivo).
    """
    coberturas_obrigatorias = set(
        df_catalogo.loc[df_catalogo[coluna_obrigatoria] == True, coluna_cobertura_id]  # noqa: E712
    )
    coberturas_opcionais = [
        c for c in df_catalogo[coluna_cobertura_id].tolist() if c not in coberturas_obrigatorias
    ]

    esquema_codificacao = construir_esquema_codificacao(df_catalogo, colunas_categoricas_cobertura)
    dicionario_f = construir_dicionario_f(
        df_catalogo, esquema_codificacao, colunas_categoricas_cobertura,
        colunas_numericas_cobertura, coluna_cobertura_id,
    )

    X, y = preparar_dados_treino_theta(
        df_interacoes, dicionario_f, coberturas_obrigatorias, colunas_cliente,
        coluna_cobertura_id, coluna_y, coluna_origem, [valor_origem_reservada_baseline],
    )

    theta_hat = ajustar_theta(X, y, lam)
    H_theta = calcular_H_theta(theta_hat, X, lam, tau)

    baseline_por_cobertura = calcular_baseline_por_cobertura(
        df_interacoes, coberturas_opcionais, coluna_cobertura_id, coluna_y,
        coluna_origem, valor_origem_reservada_baseline, coluna_ofertada_aleatoriamente,
    )

    modelo = {
        "theta_hat": theta_hat,
        "H_theta": H_theta,
        "dicionario_f": dicionario_f,
        "esquema_codificacao": esquema_codificacao,
        "colunas_cliente": colunas_cliente,
        "colunas_categoricas_cobertura": colunas_categoricas_cobertura,
        "colunas_numericas_cobertura": colunas_numericas_cobertura,
        "coberturas_obrigatorias": coberturas_obrigatorias,
        "coberturas_opcionais": coberturas_opcionais,
        "baseline_por_cobertura": baseline_por_cobertura,
        "lam": lam,
        "tau": tau,
        "z_critico": z_critico,
        "n_observacoes_treino": len(y),
        "data_treino": datetime.now().isoformat(),
    }

    with open(caminho_saida, "wb") as f:
        pickle.dump(modelo, f)

    print(f"Modelo treinado com {len(y)} observações e salvo em: {caminho_saida}")
    print(f"Coberturas obrigatórias: {sorted(coberturas_obrigatorias)}")
    print(f"Coberturas opcionais: {sorted(coberturas_opcionais)}")
    for cobertura_id, info in baseline_por_cobertura.items():
        print(f"  baseline[{cobertura_id}] -> n={info['n']}, b={info['b']}")

    return modelo


if __name__ == "__main__":
    # Exemplo de uso -- ajuste os caminhos/nomes de coluna para o seu caso real.
    df_interacoes = pd.read_csv("historico_interacoes.csv")
    df_catalogo = pd.read_csv("catalogo_coberturas.csv")

    treinar_modelo_batch(
        df_interacoes=df_interacoes,
        df_catalogo=df_catalogo,
        colunas_cliente=["idade_norm", "renda_norm"],
        colunas_categoricas_cobertura=["categoria_risco", "faixa_premio"],
        colunas_numericas_cobertura=["freq_sinistro_norm"],
        lam=1.0,
        tau=0.05,
        caminho_saida="modelo_coberturas.pkl",
    )
