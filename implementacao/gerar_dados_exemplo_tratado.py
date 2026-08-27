"""
Gera um histórico sintético JÁ TRATADO (dummies e numéricas prontas, como se
tivesse vindo de uma feature store) e um lote de escoragem, só para testar
treino_batch.py e escoragem_lote.py de ponta a ponta.
"""

import numpy as np
import pandas as pd


COBERTURAS_OBRIGATORIAS = ["Incendio", "Responsabilidade_Civil", "Queda_de_Raio", "Desmoronamento"]
COBERTURAS_OPCIONAIS = [
    "Desastres_Naturais", "Danos_Eletricos", "Roubo_Furto", "Conteudo_Residencial",
    "Vidros", "Vendaval", "Curto_Circuito", "Assistencia_Encanador",
    "Assistencia_Chaveiro", "Equipamentos_Eletronicos",
]
TODAS_COBERTURAS = COBERTURAS_OBRIGATORIAS + COBERTURAS_OPCIONAIS

COLUNAS_CLIENTE = ["idade_norm", "renda_norm"]
COLUNAS_COBERTURA = [
    "risco_ambiental", "risco_seguranca", "risco_estrutural",
    "preco_barato", "preco_medio", "preco_caro", "freq_sinistro_norm",
]


def gerar_atributos_coberturas(seed=0):
    """
    Gera, para cada cobertura do catálogo fictício, seu vetor f_i já em
    formato de dummies -- simulando o que a feature store entregaria.

    Retorna
    -------
    dict
        {cobertura_id: dict com os valores de cada coluna de COLUNAS_COBERTURA}
    """
    rng = np.random.default_rng(seed)
    atributos = {}
    for cobertura_id in TODAS_COBERTURAS:
        risco = rng.choice(["ambiental", "seguranca", "estrutural"])
        preco = rng.choice(["barato", "medio", "caro"])
        atributos[cobertura_id] = {
            "risco_ambiental": float(risco == "ambiental"),
            "risco_seguranca": float(risco == "seguranca"),
            "risco_estrutural": float(risco == "estrutural"),
            "preco_barato": float(preco == "barato"),
            "preco_medio": float(preco == "medio"),
            "preco_caro": float(preco == "caro"),
            "freq_sinistro_norm": float(rng.uniform(0, 1)),
        }
    return atributos


def gerar_historico_tratado(atributos_coberturas, n_clientes=800, epsilon=0.1,
                             fracao_reservada_baseline=0.3, seed=1):
    """
    Gera um histórico sintético já no formato tratado (uma linha por
    observação cliente x cobertura opcional, com todas as colunas de cliente
    e de cobertura já preenchidas como dummies/numéricas prontas).

    Retorna
    -------
    pd.DataFrame
        Colunas: cliente_id, idade_norm, renda_norm, cobertura_id,
        + COLUNAS_COBERTURA, y, origem, ofertada_aleatoriamente.
    """
    rng = np.random.default_rng(seed)
    afinidade_risco = {"risco_ambiental": 0.15, "risco_seguranca": -0.05, "risco_estrutural": 0.05}
    afinidade_preco = {"preco_barato": 0.2, "preco_medio": 0.0, "preco_caro": -0.2}

    linhas = []
    for cliente_id in range(n_clientes):
        idade_norm = rng.uniform(0, 1)
        renda_norm = rng.uniform(0, 1)

        vai_aleatorio = rng.random() < epsilon
        if vai_aleatorio:
            e_reservada = rng.random() < fracao_reservada_baseline
            origem = "aleatoria_reservada_baseline" if e_reservada else "aleatoria_treino"
            tamanho_sorteado = rng.integers(0, len(COBERTURAS_OPCIONAIS) + 1)
            ofertadas = set(rng.choice(COBERTURAS_OPCIONAIS, size=tamanho_sorteado, replace=False))
        else:
            origem = "modelo"
            ofertadas = set(COBERTURAS_OPCIONAIS)  # simplificação do exemplo sintético

        for cobertura_id in COBERTURAS_OPCIONAIS:
            attrs = atributos_coberturas[cobertura_id]

            logit_verdadeiro = (
                -0.3
                + 1.2 * (idade_norm - 0.5)
                + 0.8 * (renda_norm - 0.5)
                + sum(afinidade_risco[k] * attrs[k] for k in afinidade_risco)
                + sum(afinidade_preco[k] * attrs[k] for k in afinidade_preco)
            )
            p_verdadeiro = 1 / (1 + np.exp(-logit_verdadeiro))

            foi_ofertada = cobertura_id in ofertadas
            adicionou_espontaneamente = (not foi_ofertada) and (rng.random() < p_verdadeiro * 0.3)

            if not foi_ofertada and not adicionou_espontaneamente:
                continue

            y = int(rng.random() < p_verdadeiro)

            linha = {
                "cliente_id": cliente_id,
                "idade_norm": idade_norm,
                "renda_norm": renda_norm,
                "cobertura_id": cobertura_id,
                "y": y,
                "origem": origem,
                "ofertada_aleatoriamente": foi_ofertada if vai_aleatorio else None,
            }
            linha.update(attrs)
            linhas.append(linha)

    colunas_ordenadas = (
        ["cliente_id"] + COLUNAS_CLIENTE + ["cobertura_id"] + COLUNAS_COBERTURA
        + ["y", "origem", "ofertada_aleatoriamente"]
    )
    return pd.DataFrame(linhas)[colunas_ordenadas]


def gerar_lote_para_escorar(atributos_coberturas, n_clientes=20, seed=42):
    """
    Gera um lote de clientes novos, com uma linha por (cliente, cobertura) --
    as 14 coberturas do catálogo para cada cliente -- pronto para
    escorar_lote.

    Retorna
    -------
    pd.DataFrame
        Colunas: cliente_id, idade_norm, renda_norm, cobertura_id,
        + COLUNAS_COBERTURA.
    """
    rng = np.random.default_rng(seed)
    linhas = []
    for cliente_id in range(1000, 1000 + n_clientes):
        idade_norm = rng.uniform(0, 1)
        renda_norm = rng.uniform(0, 1)
        for cobertura_id in TODAS_COBERTURAS:
            linha = {
                "cliente_id": cliente_id,
                "idade_norm": idade_norm,
                "renda_norm": renda_norm,
                "cobertura_id": cobertura_id,
            }
            linha.update(atributos_coberturas[cobertura_id])
            linhas.append(linha)

    colunas_ordenadas = ["cliente_id"] + COLUNAS_CLIENTE + ["cobertura_id"] + COLUNAS_COBERTURA
    return pd.DataFrame(linhas)[colunas_ordenadas]


if __name__ == "__main__":
    atributos_coberturas = gerar_atributos_coberturas()

    df_historico = gerar_historico_tratado(atributos_coberturas)
    df_historico.to_parquet("historico_interacoes_tratado.parquet", index=False)
    print(f"Histórico tratado gerado: {len(df_historico)} linhas -> historico_interacoes_tratado.parquet")

    df_lote = gerar_lote_para_escorar(atributos_coberturas)
    df_lote.to_parquet("lote_para_escorar.parquet", index=False)
    print(f"Lote para escorar gerado: {len(df_lote)} linhas -> lote_para_escorar.parquet")
