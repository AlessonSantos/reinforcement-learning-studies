"""
Treino em lote do modelo de recomendação de coberturas -- versão feature store.

O que muda em relação à versão anterior:
- As variáveis explicativas de cliente e de cobertura já chegam TRATADAS
  (dummies já geradas, numéricas já normalizadas), vindas de uma feature
  store -- este script não faz nenhuma codificação categórica. A única
  construção que continua sendo responsabilidade daqui é o termo de
  interação c⊗f_i, porque ele é específico da arquitetura do modelo, não
  algo que uma feature store genérica saberia montar.
- theta é ajustado via sklearn.linear_model.LogisticRegression, não mais por
  um otimizador escrito à mão.
- H_theta continua calculada manualmente -- não é algo que o sklearn expõe --,
  com o mesmo piso de curvatura tau já documentado no pipeline.

Pressuposto de schema: o histórico de interações chega como UM ÚNICO
DataFrame, uma linha por observação (cliente x cobertura x rodada), já
contendo as colunas de cliente, as colunas de cobertura, o id da cobertura,
o resultado (y), a origem da rodada, e o flag de oferta aleatória genuína.
Não é mais necessário nenhum DataFrame de catálogo separado para o treino
(a lista de obrigatórias/opcionais passa a ser um parâmetro explícito).
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from datetime import datetime


# ---------------------------------------------------------------------------
# Construção de x(i) = concat(c, f_i, c⊗f_i), em lote (vetorizado)
# ---------------------------------------------------------------------------

def montar_x_lote(df, colunas_cliente, colunas_cobertura):
    """
    Monta a matriz X (uma linha por observação) com x(i) = concat(c, f_i, c⊗f_i),
    de forma vetorizada, a partir de um DataFrame cujas colunas de cliente e
    de cobertura já chegam tratadas (dummies e numéricas já prontas).

    Parâmetros
    ----------
    df : pd.DataFrame
        Uma linha por observação, contendo pelo menos as colunas listadas em
        `colunas_cliente` e `colunas_cobertura`.
    colunas_cliente : list[str]
        Nomes das colunas que formam o vetor c do cliente (ex.: idade
        normalizada, dummies de tipo de imóvel, etc. -- já prontas).
    colunas_cobertura : list[str]
        Nomes das colunas que formam o vetor f_i da cobertura (dummies de
        categoria de risco, faixa de prêmio, etc. -- já prontas).

    Retorna
    -------
    np.ndarray
        Matriz X de dimensão (N, d), com
        d = len(colunas_cliente) + len(colunas_cobertura) + len(colunas_cliente)*len(colunas_cobertura).
        A ordem das colunas de X é sempre [c, f_i, c⊗f_i] -- a mesma ordem é
        usada na escoragem, reaproveitando as listas salvas no modelo.
    """
    C = df[colunas_cliente].to_numpy(dtype=float)      # (N, dc)
    F = df[colunas_cobertura].to_numpy(dtype=float)    # (N, dfi)
    # produto externo linha a linha, achatado -- feito em lote via
    # broadcasting, sem nenhum loop Python por observação
    interacao = (C[:, :, None] * F[:, None, :]).reshape(len(df), -1)  # (N, dc*dfi)
    return np.concatenate([C, F, interacao], axis=1)


# ---------------------------------------------------------------------------
# Ajuste de theta via sklearn
# ---------------------------------------------------------------------------

def ajustar_theta_sklearn(X, y, lam):
    """
    Ajusta theta via sklearn.linear_model.LogisticRegression, reproduzindo a
    mesma penalização L2 (lambda/2)*||theta||^2 usada na derivação MAP do
    pipeline (Etapa 9).

    A correspondência de parâmetros: o sklearn minimiza
        (1/2)*||theta||^2 + C * verossimilhanca_negativa
    enquanto o MAP do pipeline minimiza
        (lambda/2)*||theta||^2 + verossimilhanca_negativa
    -- as duas coincidem fazendo C = 1/lambda (a escala global não muda o
    ponto ótimo).

    fit_intercept=False porque x(i) não tem nenhum termo de intercepto
    separado no desenho do pipeline -- se quiserem um intercepto, ele precisa
    ser incluído como uma coluna constante dentro de colunas_cliente antes de
    chamar esta função, não configurado aqui.

    Parâmetros
    ----------
    X : np.ndarray
        Matriz de contextos (N, d), saída de montar_x_lote.
    y : np.ndarray
        Vetor de rótulos (N,), valores em {0, 1}.
    lam : float
        Coeficiente de regularização (o mesmo lambda usado em H_theta).
        Penalização L2 é o padrão do sklearn, por isso não é passada
        explicitamente (evita o aviso de depreciação em versões recentes).

    Retorna
    -------
    (sklearn.linear_model.LogisticRegression, np.ndarray)
        O modelo sklearn já ajustado (salvo no pickle para reaproveitar
        .predict_proba / .decision_function na escoragem), e theta_hat (o
        vetor de coeficientes, equivalente a modelo.coef_.flatten()).
    """
    modelo = LogisticRegression(
        C=1.0 / lam,
        fit_intercept=False,
        solver="lbfgs",
        max_iter=1000,
    )
    modelo.fit(X, y)
    theta_hat = modelo.coef_.flatten()
    return modelo, theta_hat


# ---------------------------------------------------------------------------
# H_theta -- aproximação de Laplace com piso de curvatura
# ---------------------------------------------------------------------------

def sigmoide(z):
    """
    Função logística padrão, aplicada de forma numericamente estável.

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


def calcular_H_theta(theta_hat, X, lam, tau):
    """
    Calcula H_theta via aproximação de Laplace, com o piso de curvatura tau
    aplicado por observação (max entre o peso real p(1-p) e tau) -- evita que
    observações de itens "quase certos" parem de reduzir a incerteza, mesmo
    depois de muitas confirmações repetidas (o "efeito kappa").

    Parâmetros
    ----------
    theta_hat : np.ndarray
        O theta já ajustado (saída de ajustar_theta_sklearn).
    X : np.ndarray
        A MESMA matriz de contextos usada para ajustar theta_hat (N, d).
    lam : float
        O mesmo coeficiente de regularização usado no ajuste (entra como lam*I).
    tau : float
        O piso de curvatura. Ponto de partida sugerido: entre 0,02 e 0,10.

    Retorna
    -------
    np.ndarray
        H_theta, uma matriz (d, d).
    """
    d = X.shape[1]
    p = sigmoide(X @ theta_hat)
    pesos = np.maximum(p * (1 - p), tau)
    H = lam * np.eye(d) + X.T @ (X * pesos[:, None])
    return H


# ---------------------------------------------------------------------------
# Baseline por cobertura opcional
# ---------------------------------------------------------------------------

def calcular_baseline_por_cobertura(df_interacoes, coberturas_opcionais,
                                     coluna_cobertura_id, coluna_y, coluna_origem,
                                     valor_origem_reservada_baseline,
                                     coluna_ofertada_aleatoriamente=None):
    """
    Calcula, para cada cobertura opcional, a taxa de aceitação observada SÓ
    na fatia de rodadas aleatórias reservada para isso -- nunca misturando
    com dados usados no treino de theta, e (se a coluna for informada) nunca
    contando adições espontâneas.

    Parâmetros
    ----------
    df_interacoes : pd.DataFrame
        O histórico completo de observações (mesmo DataFrame usado no treino,
        antes de filtrar as linhas reservadas para o baseline).
    coberturas_opcionais : list
        Ids das coberturas opcionais (as obrigatórias não têm baseline).
    coluna_cobertura_id, coluna_y, coluna_origem : str
        Nomes de coluna correspondentes.
    valor_origem_reservada_baseline : str
        O valor de `coluna_origem` que identifica as rodadas reservadas para
        o cálculo do baseline.
    coluna_ofertada_aleatoriamente : str ou None
        Nome da coluna booleana que indica se a cobertura foi genuinamente
        sorteada para a oferta aleatória (True) ou uma adição espontânea
        (False). Se None, todas as linhas reservadas contam.

    Retorna
    -------
    dict
        {cobertura_id: {"n": int, "b": float, "logit_b": float, "var_logit_b": float}}
        Para coberturas sem observação reservada ainda, "n" vem como 0 e os
        demais campos como None -- a escoragem precisa tratar esse caso,
        deixando a cobertura fora da decisão até haver dado suficiente.
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
        b_ajustado = min(max(b, 1e-4), 1 - 1e-4)  # protege contra b=0 ou b=1 exatos
        logit_b = float(np.log(b_ajustado / (1 - b_ajustado)))
        var_logit_b = float(1 / (n * b_ajustado * (1 - b_ajustado)))

        resultado[cobertura_id] = {"n": n, "b": float(b), "logit_b": logit_b, "var_logit_b": var_logit_b}

    return resultado


# ---------------------------------------------------------------------------
# Orquestração: treino completo, do histórico já tratado ao pickle final
# ---------------------------------------------------------------------------

def treinar_modelo_batch(
    df_interacoes,
    colunas_cliente,
    colunas_cobertura,
    coberturas_obrigatorias,
    coberturas_opcionais,
    coluna_cobertura_id="cobertura_id",
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
    Função principal do treino em lote: recebe o histórico já tratado (um
    único DataFrame, com as colunas de cliente e de cobertura já prontas),
    ajusta o modelo via sklearn, calcula H_theta e os baselines, e salva um
    pickle com tudo que a escoragem em lote vai precisar.

    Parâmetros
    ----------
    df_interacoes : pd.DataFrame
        Histórico completo de observações. Colunas mínimas esperadas: as de
        `colunas_cliente`, as de `colunas_cobertura`, `coluna_cobertura_id`,
        `coluna_y`, `coluna_origem`, e (se disponível) `coluna_ofertada_aleatoriamente`.
    colunas_cliente, colunas_cobertura : list[str]
        Ver montar_x_lote.
    coberturas_obrigatorias : list ou set
        Ids das coberturas obrigatórias -- excluídas do treino de theta e do
        baseline, entram sempre no pacote na escoragem.
    coberturas_opcionais : list
        Ids das coberturas opcionais -- passam pelo treino de theta e pelo
        teste de significância na escoragem.
    coluna_cobertura_id, coluna_y, coluna_origem : str
        Nomes de colunas -- ajuste conforme o schema real da feature store.
    valor_origem_reservada_baseline, coluna_ofertada_aleatoriamente : ver
        calcular_baseline_por_cobertura.
    lam : float
        Regularização (mesmo lambda usado em theta e em H_theta).
    tau : float
        Piso de curvatura.
    z_critico : float
        Valor crítico do teste de significância usado na escoragem (salvo
        aqui para ficar tudo junto no mesmo pickle).
    caminho_saida : str
        Caminho do arquivo pickle a ser gerado.

    Retorna
    -------
    dict
        O mesmo dicionário salvo no pickle, também retornado em memória.
    """
    coberturas_obrigatorias = set(coberturas_obrigatorias)
    coberturas_opcionais = list(coberturas_opcionais)

    df_treino = df_interacoes[
        (~df_interacoes[coluna_cobertura_id].isin(coberturas_obrigatorias))
        & (df_interacoes[coluna_origem] != valor_origem_reservada_baseline)
    ].reset_index(drop=True)

    X = montar_x_lote(df_treino, colunas_cliente, colunas_cobertura)
    y = df_treino[coluna_y].to_numpy(dtype=float)

    modelo_sklearn, theta_hat = ajustar_theta_sklearn(X, y, lam)
    H_theta = calcular_H_theta(theta_hat, X, lam, tau)

    baseline_por_cobertura = calcular_baseline_por_cobertura(
        df_interacoes, coberturas_opcionais, coluna_cobertura_id, coluna_y,
        coluna_origem, valor_origem_reservada_baseline, coluna_ofertada_aleatoriamente,
    )

    modelo = {
        "modelo_sklearn": modelo_sklearn,
        "theta_hat": theta_hat,
        "H_theta": H_theta,
        "colunas_cliente": colunas_cliente,
        "colunas_cobertura": colunas_cobertura,
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
    print(f"Dimensão de x(i): {X.shape[1]}")
    print(f"Coberturas obrigatórias: {sorted(coberturas_obrigatorias)}")
    print(f"Coberturas opcionais: {sorted(coberturas_opcionais)}")
    for cobertura_id, info in baseline_por_cobertura.items():
        print(f"  baseline[{cobertura_id}] -> n={info['n']}, b={info['b']}")

    return modelo


if __name__ == "__main__":
    # Exemplo de uso -- ajuste os caminhos/nomes de coluna para o seu caso real.
    df_interacoes = pd.read_parquet("historico_interacoes_tratado.parquet")

    colunas_cliente = ["idade_norm", "renda_norm"]
    colunas_cobertura = [
        "risco_ambiental", "risco_seguranca", "risco_estrutural",
        "preco_barato", "preco_medio", "preco_caro", "freq_sinistro_norm",
    ]
    coberturas_obrigatorias = ["Incendio", "Responsabilidade_Civil", "Queda_de_Raio", "Desmoronamento"]
    coberturas_opcionais = [
        "Desastres_Naturais", "Danos_Eletricos", "Roubo_Furto", "Conteudo_Residencial",
        "Vidros", "Vendaval", "Curto_Circuito", "Assistencia_Encanador",
        "Assistencia_Chaveiro", "Equipamentos_Eletronicos",
    ]

    treinar_modelo_batch(
        df_interacoes=df_interacoes,
        colunas_cliente=colunas_cliente,
        colunas_cobertura=colunas_cobertura,
        coberturas_obrigatorias=coberturas_obrigatorias,
        coberturas_opcionais=coberturas_opcionais,
        lam=1.0,
        tau=0.05,
        caminho_saida="modelo_coberturas.pkl",
    )
