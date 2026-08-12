"""
Gera um catálogo de coberturas e um histórico de interações sintéticos,
só para permitir testar treino_batch.py e escoragem_cliente.py de ponta a ponta.

Os dados aqui não representam nenhum padrão real de negócio -- servem só
para exercitar o código. Ajuste os nomes de coluna e a lógica de geração
livremente para o formato real que vocês tiverem.
"""

import numpy as np
import pandas as pd


def gerar_catalogo(seed=0):
    """
    Gera um catálogo fictício de 14 coberturas: 4 obrigatórias, 10 opcionais,
    cada uma com uma categoria de risco, faixa de prêmio, e frequência de
    sinistro (já normalizada, entre 0 e 1).

    Parâmetros
    ----------
    seed : int
        Semente para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        Uma linha por cobertura, com as colunas:
        cobertura_id, obrigatoria, categoria_risco, faixa_premio, freq_sinistro_norm.
    """
    rng = np.random.default_rng(seed)

    nomes = [
        "Incendio", "Responsabilidade_Civil", "Queda_de_Raio", "Desmoronamento",  # obrigatórias
        "Desastres_Naturais", "Danos_Eletricos", "Roubo_Furto", "Conteudo_Residencial",
        "Vidros", "Vendaval", "Curto_Circuito", "Assistencia_Encanador",
        "Assistencia_Chaveiro", "Equipamentos_Eletronicos",  # opcionais
    ]
    obrigatorias = [True, True, True, True] + [False] * 10

    categorias_risco = ["ambiental", "seguranca", "estrutural"]
    faixas_premio = ["barato", "medio", "caro"]

    df = pd.DataFrame({
        "cobertura_id": nomes,
        "obrigatoria": obrigatorias,
        "categoria_risco": rng.choice(categorias_risco, size=len(nomes)),
        "faixa_premio": rng.choice(faixas_premio, size=len(nomes)),
        "freq_sinistro_norm": rng.uniform(0, 1, size=len(nomes)),
    })

    return df


def gerar_historico(df_catalogo, n_clientes=800, epsilon=0.1, fracao_reservada_baseline=0.3, seed=1):
    """
    Gera um histórico sintético de interações: para cada cliente sorteado,
    decide se a rodada é aleatória ou "do modelo" (aqui simplificado -- não
    simula um modelo de verdade, só usa uma regra sintética de propensão para
    gerar y de forma plausível), e registra uma linha por cobertura opcional
    avaliada. Coberturas obrigatórias não geram linha nenhuma (nunca entram
    no treino).

    Parâmetros
    ----------
    df_catalogo : pd.DataFrame
        Saída de gerar_catalogo.
    n_clientes : int
        Quantos "clientes" (rodadas) simular.
    epsilon : float
        Probabilidade de uma rodada ser aleatória.
    fracao_reservada_baseline : float
        Fração das rodadas aleatórias reservada para o cálculo do baseline.
    seed : int
        Semente para reprodutibilidade.

    Retorna
    -------
    pd.DataFrame
        Uma linha por observação (cliente x cobertura opcional), com as colunas:
        cliente_id, idade_norm, renda_norm, cobertura_id, y, origem,
        ofertada_aleatoriamente. Essa última só tem significado nas linhas de
        origem "aleatoria_*" (True = sorteada para a oferta; False = adição
        espontânea dentro de uma rodada aleatória); nas linhas de origem
        "modelo", vem como None, porque a pergunta não se aplica a elas.
    """
    rng = np.random.default_rng(seed)
    coberturas_opcionais = df_catalogo.loc[~df_catalogo["obrigatoria"], "cobertura_id"].tolist()

    # propensao "verdadeira" sintética, só para gerar y de forma não totalmente aleatória
    afinidade_categoria = {"ambiental": 0.15, "seguranca": -0.05, "estrutural": 0.05}
    afinidade_faixa = {"barato": 0.2, "medio": 0.0, "caro": -0.2}

    linhas = []
    for cliente_id in range(n_clientes):
        idade_norm = rng.uniform(0, 1)
        renda_norm = rng.uniform(0, 1)

        vai_aleatorio = rng.random() < epsilon

        if vai_aleatorio:
            e_reservada = rng.random() < fracao_reservada_baseline
            origem = "aleatoria_reservada_baseline" if e_reservada else "aleatoria_treino"
            tamanho_sorteado = rng.integers(0, len(coberturas_opcionais) + 1)
            ofertadas = set(rng.choice(coberturas_opcionais, size=tamanho_sorteado, replace=False))
        else:
            origem = "modelo"
            ofertadas = set(coberturas_opcionais)  # simplificação: no exemplo sintético, o "modelo" oferece todas

        for cobertura_id in coberturas_opcionais:
            linha_cat = df_catalogo.loc[df_catalogo["cobertura_id"] == cobertura_id].iloc[0]

            logit_verdadeiro = (
                -0.3
                + 1.2 * (idade_norm - 0.5)
                + 0.8 * (renda_norm - 0.5)
                + afinidade_categoria[linha_cat["categoria_risco"]]
                + afinidade_faixa[linha_cat["faixa_premio"]]
            )
            p_verdadeiro = 1 / (1 + np.exp(-logit_verdadeiro))

            foi_ofertada = cobertura_id in ofertadas
            adicionou_espontaneamente = (not foi_ofertada) and (rng.random() < p_verdadeiro * 0.3)

            if not foi_ofertada and not adicionou_espontaneamente:
                continue  # cliente nunca viu essa cobertura nessa rodada -- sem observação

            y = int(rng.random() < p_verdadeiro)

            linhas.append({
                "cliente_id": cliente_id,
                "idade_norm": idade_norm,
                "renda_norm": renda_norm,
                "cobertura_id": cobertura_id,
                "y": y,
                "origem": origem,
                "ofertada_aleatoriamente": foi_ofertada if vai_aleatorio else None,
            })

    return pd.DataFrame(linhas)


if __name__ == "__main__":
    df_catalogo = gerar_catalogo()
    df_historico = gerar_historico(df_catalogo)

    df_catalogo.to_csv("catalogo_coberturas.csv", index=False)
    df_historico.to_csv("historico_interacoes.csv", index=False)

    print(f"Catálogo gerado: {len(df_catalogo)} coberturas -> catalogo_coberturas.csv")
    print(f"Histórico gerado: {len(df_historico)} observações -> historico_interacoes.csv")
    print()
    print(df_catalogo)
    print()
    print(df_historico.head(10))
