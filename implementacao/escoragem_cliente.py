"""
Escoragem de UM cliente por execução, para todas as coberturas do catálogo.

Resumo da lógica (ver documentação do projeto para detalhes):
- As coberturas obrigatórias sempre entram no pacote, sem passar por nenhum teste.
- Cada cobertura opcional é avaliada de forma independente: calcula-se eta(i) e sua
  incerteza (via theta e H_theta do modelo treinado), compara-se com a taxa de
  referência daquela cobertura (o baseline, calculado em lote), e só entra no
  pacote se a diferença for estatisticamente significativa (o teste da Etapa 5).
- Não existe processo guloso nem ordem de avaliação -- todas as opcionais são
  avaliadas de uma vez, de forma simétrica.

Este script consome o pickle gerado por treino_batch.py.
"""

import numpy as np
import pickle

from treino_batch import construir_f_cobertura, montar_x, sigmoide


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
        O dicionário completo do modelo (theta_hat, H_theta, baselines, etc.).
    """
    with open(caminho_pickle, "rb") as f:
        modelo = pickle.load(f)
    return modelo


def calcular_eta_e_variancia(vetor_c, cobertura_id, modelo):
    """
    Calcula eta(i) = theta^T x(i) e Var[eta(i)] = x(i)^T H_theta^-1 x(i)
    para UM cliente (vetor_c) e UMA cobertura específica.

    Parâmetros
    ----------
    vetor_c : np.ndarray
        Perfil do cliente, na mesma ordem/escala usada no treino
        (ver modelo["colunas_cliente"] para conferir a ordem esperada).
    cobertura_id : hashable
        Id da cobertura a avaliar -- precisa existir em modelo["dicionario_f"].
    modelo : dict
        O modelo carregado por carregar_modelo.

    Retorna
    -------
    (float, float)
        eta(i) e Var[eta(i)].
    """
    vetor_f = modelo["dicionario_f"][cobertura_id]
    x = montar_x(vetor_c, vetor_f)

    eta = float(modelo["theta_hat"] @ x)

    # resolve H^-1 x via sistema linear, mais estável/eficiente que inverter H inteira
    H_inv_x = np.linalg.solve(modelo["H_theta"], x)
    variancia_eta = float(x @ H_inv_x)

    return eta, variancia_eta


def testar_significancia(eta, variancia_eta, info_baseline, z_critico):
    """
    Roda o teste de significância da Etapa 5: compara eta(i) contra a taxa de
    referência daquela cobertura (em escala log-odds), levando em conta as
    duas incertezas (a de eta e a do baseline) somadas.

    Parâmetros
    ----------
    eta : float
        eta(i) calculado por calcular_eta_e_variancia.
    variancia_eta : float
        Var[eta(i)] calculado por calcular_eta_e_variancia.
    info_baseline : dict
        O dicionário {"n", "b", "logit_b", "var_logit_b"} daquela cobertura,
        vindo de modelo["baseline_por_cobertura"][cobertura_id].
    z_critico : float
        Valor crítico do teste (ex.: 1,645 para 95% de confiança, um lado).

    Retorna
    -------
    dict
        {
          "d": a diferença eta - logit(baseline) (None se não houver baseline ainda),
          "variancia_d": a variância combinada (None se não houver baseline ainda),
          "limite_inferior": D - z*sqrt(Var[D]) (None se não houver baseline ainda),
          "passa_no_teste": bool -- True só se o limite inferior for maior que zero,
          "motivo_se_nao_passa": string explicando por que não passou, quando aplicável,
        }
    """
    if info_baseline["n"] == 0:
        return {
            "d": None,
            "variancia_d": None,
            "limite_inferior": None,
            "passa_no_teste": False,
            "motivo_se_nao_passa": "cobertura ainda sem observações reservadas para o baseline",
        }

    d = eta - info_baseline["logit_b"]
    variancia_d = variancia_eta + info_baseline["var_logit_b"]
    limite_inferior = d - z_critico * np.sqrt(variancia_d)

    return {
        "d": d,
        "variancia_d": variancia_d,
        "limite_inferior": limite_inferior,
        "passa_no_teste": bool(limite_inferior > 0),
        "motivo_se_nao_passa": None if limite_inferior > 0 else "diferença não é estatisticamente significativa",
    }


def escorar_cliente(perfil_cliente, modelo):
    """
    Escora UM cliente para TODAS as coberturas do catálogo (obrigatórias e
    opcionais), retornando o detalhe de cada uma e o pacote final recomendado.

    Parâmetros
    ----------
    perfil_cliente : dict ou pd.Series
        Os valores do perfil do cliente, com as MESMAS chaves/colunas listadas
        em modelo["colunas_cliente"], na mesma ordem em que foram usadas no
        treino (ex.: {"idade_norm": 0.5, "renda_norm": 0.6}).
    modelo : dict
        O modelo carregado por carregar_modelo.

    Retorna
    -------
    dict
        {
          "pacote_recomendado": list de cobertura_id -- obrigatórias + opcionais
                                  que passaram no teste,
          "detalhe_por_cobertura": dict {cobertura_id: {...}} com o resultado
                                     completo de cada cobertura (útil para
                                     auditoria/debug -- mostra eta, variância,
                                     o teste, e se entrou ou não no pacote),
        }
    """
    vetor_c = np.array([perfil_cliente[coluna] for coluna in modelo["colunas_cliente"]], dtype=float)

    detalhe_por_cobertura = {}
    pacote_recomendado = []

    # obrigatórias: sempre entram, sem nenhum cálculo de eta/teste
    for cobertura_id in sorted(modelo["coberturas_obrigatorias"]):
        detalhe_por_cobertura[cobertura_id] = {
            "obrigatoria": True,
            "entra_no_pacote": True,
            "motivo": "cobertura obrigatória -- sempre incluída",
        }
        pacote_recomendado.append(cobertura_id)

    # opcionais: cada uma passa pelo teste de significância, de forma independente
    for cobertura_id in modelo["coberturas_opcionais"]:
        eta, variancia_eta = calcular_eta_e_variancia(vetor_c, cobertura_id, modelo)
        p_estimado = float(sigmoide(np.array([eta]))[0])

        info_baseline = modelo["baseline_por_cobertura"][cobertura_id]
        resultado_teste = testar_significancia(eta, variancia_eta, info_baseline, modelo["z_critico"])

        entra = resultado_teste["passa_no_teste"]
        detalhe_por_cobertura[cobertura_id] = {
            "obrigatoria": False,
            "eta": eta,
            "variancia_eta": variancia_eta,
            "p_estimado": p_estimado,
            "baseline": info_baseline["b"],
            "n_baseline": info_baseline["n"],
            **resultado_teste,
            "entra_no_pacote": entra,
        }
        if entra:
            pacote_recomendado.append(cobertura_id)

    return {
        "pacote_recomendado": pacote_recomendado,
        "detalhe_por_cobertura": detalhe_por_cobertura,
    }


# ---------------------------------------------------------------------------
# Auxiliar opcional: decisão da rota (Etapa 2) -- aleatória vs. modelo
# ---------------------------------------------------------------------------

def sortear_rota(epsilon, rng=None):
    """
    Sorteia se a rodada vai pela rota aleatória (Z_t=1) ou pela rota do modelo
    (Z_t=0). Não é usado dentro de escorar_cliente -- é uma função auxiliar
    para quem for orquestrar o fluxo completo de produção (Etapa 2 do pipeline).

    Parâmetros
    ----------
    epsilon : float
        Probabilidade fixa de ir pela rota aleatória (definida pela equipe,
        nunca decai automaticamente, nunca pode ser zero).
    rng : np.random.Generator ou None
        Gerador de números aleatórios. Se None, cria um novo a cada chamada
        (prefira passar um gerador único por processo, para reprodutibilidade).

    Retorna
    -------
    bool
        True se a rodada deve seguir pela rota aleatória; False se deve
        seguir pela rota do modelo (escorar_cliente).
    """
    if rng is None:
        rng = np.random.default_rng()
    return bool(rng.random() < epsilon)


def sortear_pacote_aleatorio(coberturas_opcionais, coberturas_obrigatorias, rng=None):
    """
    Sorteia um pacote pela rota aleatória: um subconjunto entre as coberturas
    opcionais (sorteado de uma vez, não cobertura por cobertura), mais as
    obrigatórias, que sempre entram.

    Parâmetros
    ----------
    coberturas_opcionais : list
        Ids de todas as coberturas opcionais do catálogo.
    coberturas_obrigatorias : set ou list
        Ids das coberturas obrigatórias.
    rng : np.random.Generator ou None
        Ver sortear_rota.

    Retorna
    -------
    list
        O pacote sorteado (obrigatórias + o subconjunto sorteado de opcionais).
    """
    if rng is None:
        rng = np.random.default_rng()

    tamanho_sorteado = rng.integers(0, len(coberturas_opcionais) + 1)  # inclui a possibilidade de 0 e de todas
    opcionais_sorteadas = list(rng.choice(coberturas_opcionais, size=tamanho_sorteado, replace=False))

    return list(coberturas_obrigatorias) + opcionais_sorteadas


if __name__ == "__main__":
    # Exemplo de uso -- escora um único cliente fictício.
    modelo = carregar_modelo("modelo_coberturas.pkl")

    perfil_cliente_exemplo = {"idade_norm": 0.5, "renda_norm": 0.6}
    resultado = escorar_cliente(perfil_cliente_exemplo, modelo)

    print("Pacote recomendado:", resultado["pacote_recomendado"])
    print()
    for cobertura_id, detalhe in resultado["detalhe_por_cobertura"].items():
        print(cobertura_id, "->", detalhe)
