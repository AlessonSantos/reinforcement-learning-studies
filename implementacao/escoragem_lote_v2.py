"""
Escoragem EM LOTE do modelo de recomendação de coberturas.

Diferente da versão anterior (um cliente por vez), este script escora um
DataFrame inteiro de uma vez -- uma linha por combinação (cliente, cobertura)
a avaliar, com as colunas de cliente e de cobertura já tratadas (o mesmo
formato de feature store usado no treino). Toda a matemática é vetorizada:
nenhum loop Python por observação.

Resumo da lógica (ver pipeline_c2ucb_seguro_didatico_v4.md para detalhes):
- Coberturas obrigatórias sempre entram no pacote, sem nenhum cálculo.
- Cada linha de cobertura opcional é avaliada de forma independente: eta(i) e
  sua incerteza vêm do modelo treinado; comparam-se com a taxa de referência
  daquela cobertura (o baseline, calculado em lote); só entra no pacote se a
  diferença for estatisticamente significativa (o teste da Etapa 5).
"""

import numpy as np
import pandas as pd
import pickle

from treino_batch import montar_x_lote


def carregar_modelo(caminho_pickle):
    """
    Carrega o modelo treinado (o pickle gerado por treino_batch.py).

    Parâmetros
    ----------
    caminho_pickle : str
        Caminho do arquivo .pkl salvo pelo treino em lote.

    Retorna
    -------
    dict
        O dicionário completo do modelo (modelo_sklearn, theta_hat, H_theta,
        baselines, etc.).
    """
    with open(caminho_pickle, "rb") as f:
        modelo = pickle.load(f)
    return modelo


def calcular_eta_e_variancia_lote(df, colunas_cliente, colunas_cobertura, modelo_sklearn, H_theta):
    """
    Calcula eta(i) = theta^T x(i) e Var[eta(i)] = x(i)^T H_theta^-1 x(i) para
    TODAS as linhas do DataFrame de uma vez (vetorizado).

    eta(i) é obtido via modelo_sklearn.decision_function(X) -- reaproveitando
    o modelo sklearn já ajustado, em vez de recalcular manualmente X @ theta_hat
    (as duas dão o mesmo resultado, já que o modelo foi ajustado com
    fit_intercept=False; usar o método do sklearn deixa explícito que a
    inferência reaproveita o objeto de modelo salvo, não só o vetor de
    coeficientes).

    Parâmetros
    ----------
    df : pd.DataFrame
        Uma linha por observação a escorar, com as colunas de
        `colunas_cliente` e `colunas_cobertura` já tratadas.
    colunas_cliente, colunas_cobertura : list[str]
        Ver montar_x_lote -- devem ser exatamente as mesmas listas, na mesma
        ordem, usadas no treino (vêm salvas no pickle do modelo).
    modelo_sklearn : sklearn.linear_model.LogisticRegression
        O modelo já ajustado (modelo["modelo_sklearn"]).
    H_theta : np.ndarray
        A matriz de incerteza (modelo["H_theta"]).

    Retorna
    -------
    (np.ndarray, np.ndarray, np.ndarray)
        eta (N,), variancia_eta (N,) e p_estimado (N,) -- um valor por linha
        de `df`, na mesma ordem.
    """
    X = montar_x_lote(df, colunas_cliente, colunas_cobertura)

    eta = modelo_sklearn.decision_function(X)
    p_estimado = modelo_sklearn.predict_proba(X)[:, 1]

    # H_theta é fixa (não muda por observação) -- inverte-se uma única vez e
    # reaproveita-se para todas as linhas do lote, em vez de resolver um
    # sistema linear por observação
    H_inv = np.linalg.inv(H_theta)
    variancia_eta = np.einsum("ij,jk,ik->i", X, H_inv, X)

    return eta, variancia_eta, p_estimado


def testar_significancia_lote(cobertura_ids, eta, variancia_eta, baseline_por_cobertura,
                               z_critico, razao_de_chances_minima=1.0, piso_absoluto=None):
    """
    Roda o teste de significância da Etapa 5 para um lote inteiro de uma vez,
    combinando dois testes complementares via "ou": um teste relativo (a
    propensão estimada é significativamente maior que a taxa de referência,
    por uma razão de chances mínima) e um teste absoluto (a probabilidade
    estimada já é alta o suficiente por si só, mesmo sem comparação com a
    referência) -- qualquer um dos dois bastando para a cobertura entrar no
    pacote.

    O teste absoluto existe para cobrir um caso de borda do teste relativo:
    em coberturas de baseline já alto, um cliente pode estar abaixo da média
    daquela cobertura específica e, mesmo assim, ter uma probabilidade
    absoluta de aceitação bastante alta -- alta o suficiente para justificar
    a oferta, independente da comparação com a população.

    Parâmetros
    ----------
    cobertura_ids : pd.Series ou array-like
        O id da cobertura de cada linha (mesma ordem de eta/variancia_eta).
    eta, variancia_eta : np.ndarray
        Saída de calcular_eta_e_variancia_lote.
    baseline_por_cobertura : dict
        modelo["baseline_por_cobertura"].
    z_critico : float
        Valor crítico do teste (ex.: 1,645 para 95% de confiança, um lado).
    razao_de_chances_minima : float ou dict
        A razão de chances mínima exigida entre a propensão estimada e a
        taxa de referência (teste relativo). 1.0 (padrão) equivale a exigir
        só significância estatística, sem margem de negócio. Pode ser um
        único valor ou um dict {cobertura_id: valor}.
    piso_absoluto : float, dict ou None
        A probabilidade mínima absoluta (teste absoluto) -- em (0, 1). Se
        None (padrão), o teste absoluto nunca passa (comportamento
        equivalente a não ter esse piso configurado). Pode ser um único
        valor ou um dict {cobertura_id: valor}. Deve ficar acima de 0.5 para
        preservar a leitura de "mais provável aceitar que recusar"; na
        prática, algo entre 0.6 e 0.75 tende a evitar tanto um piso nunca
        acionado quanto pacotes grandes demais.

    Retorna
    -------
    dict de np.ndarray, todos de tamanho N (uma posição por linha):
        {"d", "variancia_d", "limite_inferior_relativo", "passa_teste_relativo",
         "limite_inferior_absoluto", "passa_teste_absoluto",
         "passa_no_teste", "tem_baseline"}
        "passa_no_teste" é o "ou" dos dois -- o resultado final da Etapa 5.
        Onde não há baseline (n=0), o teste relativo vem como não-passou,
        mas o teste absoluto continua podendo passar normalmente (ele não
        depende do baseline).
    """
    n = len(eta)
    logit_b = np.full(n, np.nan)
    var_logit_b = np.full(n, np.nan)
    tem_baseline = np.zeros(n, dtype=bool)

    for idx, cobertura_id in enumerate(cobertura_ids):
        info = baseline_por_cobertura.get(cobertura_id)
        if info is not None and info["n"] > 0:
            logit_b[idx] = info["logit_b"]
            var_logit_b[idx] = info["var_logit_b"]
            tem_baseline[idx] = True

    if isinstance(razao_de_chances_minima, dict):
        margem_log = np.log(np.array([razao_de_chances_minima.get(c, 1.0) for c in cobertura_ids]))
    else:
        margem_log = np.log(razao_de_chances_minima)

    # --- teste relativo ---
    d = eta - logit_b
    variancia_d = variancia_eta + var_logit_b
    limite_inferior_relativo = d - z_critico * np.sqrt(variancia_d) - margem_log
    passa_teste_relativo = tem_baseline & (limite_inferior_relativo > 0)

    # --- teste absoluto ---
    if piso_absoluto is None:
        limite_inferior_absoluto = np.full(n, -np.inf)
        passa_teste_absoluto = np.zeros(n, dtype=bool)
    else:
        if isinstance(piso_absoluto, dict):
            logit_piso = np.array([
                np.log(piso_absoluto.get(c, 0.999999) / (1 - piso_absoluto.get(c, 0.999999)))
                for c in cobertura_ids
            ])
        else:
            logit_piso = np.log(piso_absoluto / (1 - piso_absoluto))
        limite_inferior_absoluto = eta - z_critico * np.sqrt(variancia_eta) - logit_piso
        passa_teste_absoluto = limite_inferior_absoluto > 0

    passa_no_teste = passa_teste_relativo | passa_teste_absoluto

    return {
        "d": d,
        "variancia_d": variancia_d,
        "limite_inferior_relativo": limite_inferior_relativo,
        "passa_teste_relativo": passa_teste_relativo,
        "limite_inferior_absoluto": limite_inferior_absoluto,
        "passa_teste_absoluto": passa_teste_absoluto,
        "passa_no_teste": passa_no_teste,
        "tem_baseline": tem_baseline,
    }


def escorar_lote(df_scoring, modelo, coluna_cobertura_id="cobertura_id",
                  razao_de_chances_minima=1.0, piso_absoluto=None):
    """
    Escora um lote inteiro de observações (cliente x cobertura) de uma vez.

    Parâmetros
    ----------
    df_scoring : pd.DataFrame
        Uma linha por combinação (cliente, cobertura) a avaliar -- precisa
        conter `coluna_cobertura_id` e as colunas de modelo["colunas_cliente"]
        + modelo["colunas_cobertura"], já tratadas. Pode conter linhas de
        coberturas obrigatórias (serão marcadas como sempre incluídas, sem
        nenhum cálculo) ou só linhas de opcionais -- os dois casos são
        tratados corretamente.
    modelo : dict
        Saída de carregar_modelo.
    coluna_cobertura_id : str
        Nome da coluna com o id da cobertura em df_scoring.
    razao_de_chances_minima : float ou dict
        A razão de chances mínima exigida pelo negócio para o teste relativo
        (ver testar_significancia_lote). 1.0 (padrão) equivale a exigir só
        significância estatística, sem margem de negócio.
    piso_absoluto : float, dict ou None
        A probabilidade mínima absoluta para o teste absoluto (ver
        testar_significancia_lote). None (padrão) desativa esse teste --
        só o teste relativo decide.

    Retorna
    -------
    pd.DataFrame
        Uma cópia de df_scoring com as colunas adicionadas: eta, variancia_eta,
        p_estimado, d, variancia_d, limite_inferior_relativo,
        passa_teste_relativo, limite_inferior_absoluto, passa_teste_absoluto,
        entra_no_pacote, motivo.
    """
    df = df_scoring.reset_index(drop=True).copy()

    colunas_numericas = [
        "eta", "variancia_eta", "p_estimado", "d", "variancia_d",
        "limite_inferior_relativo", "limite_inferior_absoluto",
    ]
    for coluna in colunas_numericas:
        df[coluna] = np.nan
    df["passa_teste_relativo"] = False
    df["passa_teste_absoluto"] = False
    df["entra_no_pacote"] = False
    df["motivo"] = ""

    eh_obrigatoria = df[coluna_cobertura_id].isin(modelo["coberturas_obrigatorias"])

    # obrigatórias: sempre entram, sem nenhum cálculo de eta/teste
    df.loc[eh_obrigatoria, "entra_no_pacote"] = True
    df.loc[eh_obrigatoria, "motivo"] = "cobertura obrigatória -- sempre incluída"

    # opcionais: rodam o pipeline completo (Etapas 3 a 5), em lote
    idx_opcionais = df.index[~eh_obrigatoria]
    if len(idx_opcionais) > 0:
        df_opcionais = df.loc[idx_opcionais]

        eta, variancia_eta, p_estimado = calcular_eta_e_variancia_lote(
            df_opcionais, modelo["colunas_cliente"], modelo["colunas_cobertura"],
            modelo["modelo_sklearn"], modelo["H_theta"],
        )

        resultado_teste = testar_significancia_lote(
            df_opcionais[coluna_cobertura_id].to_numpy(), eta, variancia_eta,
            modelo["baseline_por_cobertura"], modelo["z_critico"],
            razao_de_chances_minima=razao_de_chances_minima,
            piso_absoluto=piso_absoluto,
        )

        idx_opcionais_pos = idx_opcionais
        df.loc[idx_opcionais_pos, "eta"] = eta
        df.loc[idx_opcionais_pos, "variancia_eta"] = variancia_eta
        df.loc[idx_opcionais_pos, "p_estimado"] = p_estimado
        df.loc[idx_opcionais_pos, "d"] = resultado_teste["d"]
        df.loc[idx_opcionais_pos, "variancia_d"] = resultado_teste["variancia_d"]
        df.loc[idx_opcionais_pos, "limite_inferior_relativo"] = resultado_teste["limite_inferior_relativo"]
        df.loc[idx_opcionais_pos, "passa_teste_relativo"] = resultado_teste["passa_teste_relativo"]
        df.loc[idx_opcionais_pos, "limite_inferior_absoluto"] = resultado_teste["limite_inferior_absoluto"]
        df.loc[idx_opcionais_pos, "passa_teste_absoluto"] = resultado_teste["passa_teste_absoluto"]
        df.loc[idx_opcionais_pos, "entra_no_pacote"] = resultado_teste["passa_no_teste"]

        motivo = np.where(
            resultado_teste["passa_teste_relativo"] & resultado_teste["passa_teste_absoluto"],
            "aprovada pelos dois testes (relativo e absoluto)",
            np.where(
                resultado_teste["passa_teste_relativo"],
                "aprovada pelo teste relativo (vs. baseline)",
                np.where(
                    resultado_teste["passa_teste_absoluto"],
                    "aprovada pelo teste absoluto (piso de probabilidade)",
                    np.where(
                        ~resultado_teste["tem_baseline"],
                        "reprovada -- cobertura ainda sem baseline suficiente, e não atingiu o piso absoluto",
                        "reprovada nos dois testes",
                    ),
                ),
            ),
        )
        df.loc[idx_opcionais_pos, "motivo"] = motivo

    return df


def montar_pacotes_por_cliente(df_escorado, coluna_cliente_id="cliente_id", coluna_cobertura_id="cobertura_id"):
    """
    Agrega o resultado de escorar_lote por cliente, retornando o pacote final
    recomendado para cada um.

    Parâmetros
    ----------
    df_escorado : pd.DataFrame
        Saída de escorar_lote.
    coluna_cliente_id, coluna_cobertura_id : str
        Nomes das colunas de id de cliente e de cobertura em df_escorado.

    Retorna
    -------
    pd.DataFrame
        Uma linha por cliente, com as colunas: coluna_cliente_id,
        pacote_recomendado (lista de cobertura_id), qtd_coberturas.
    """
    pacotes = (
        df_escorado[df_escorado["entra_no_pacote"]]
        .groupby(coluna_cliente_id)[coluna_cobertura_id]
        .apply(list)
        .reset_index(name="pacote_recomendado")
    )
    pacotes["qtd_coberturas"] = pacotes["pacote_recomendado"].apply(len)
    return pacotes


if __name__ == "__main__":
    # Exemplo de uso -- escora um lote de clientes x coberturas de uma vez.
    modelo = carregar_modelo("modelo_coberturas.pkl")

    df_scoring = pd.read_parquet("lote_para_escorar.parquet")
    # df_scoring esperado: uma linha por (cliente_id, cobertura_id), com as
    # colunas de modelo["colunas_cliente"] + modelo["colunas_cobertura"] já
    # tratadas -- normalmente as 14 coberturas (obrigatórias + opcionais)
    # para cada cliente do lote.

    df_escorado = escorar_lote(df_scoring, modelo, coluna_cobertura_id="cobertura_id")
    pacotes = montar_pacotes_por_cliente(df_escorado, coluna_cliente_id="cliente_id", coluna_cobertura_id="cobertura_id")

    print(df_escorado.head(20))
    print()
    print(pacotes.head(10))
