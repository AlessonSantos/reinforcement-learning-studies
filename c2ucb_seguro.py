"""
c2ucb_seguro.py
================================================================================
Bandit contextual combinatorio (C2UCB) para recomendacao de pacotes de
coberturas de seguro, com:

  - Termo de item: relevancia individual do cliente por cobertura, incluindo
    o tamanho do pacote parcial no contexto (efeito de fadiga aprendido, nao
    imposto).
  - Termo de par: sinergia (efeito positivo) ou canibalizacao/redundancia
    (efeito negativo) entre pares de coberturas, aprendido separadamente.
  - Modelo logistico (link sigmoid), ajustado em lote via regressao
    logistica regularizada (MAP) com aproximacao de Laplace para a
    incerteza -- os scores ficam sempre em (0,1).
  - Piso de curvatura para evitar que itens com propensao muito proxima de
    0 ou 1 parem de reduzir sua propria incerteza (limitacao conhecida de
    modelos GLM-bandit).
  - Oraculo guloso sem restricao de cardinalidade fixa: o tamanho do pacote
    emerge do ganho marginal esperado, nao de um k pre-definido.
  - Sem nenhum termo de custo/premio monetario na funcao de recompensa.

Dependencias: numpy, pandas, scipy.

Uso rapido (via a interface de alto nivel RecomendadorPacotes, toda ela
baseada em DataFrames):

    from c2ucb_seguro import RecomendadorPacotes
    import pandas as pd

    catalogo_df = pd.read_csv("coverages.csv")
    historico_df = pd.read_csv("historico_interacoes.csv")
    clientes_df = pd.read_csv("novos_clientes.csv")

    recomendador = RecomendadorPacotes.novo(
        catalogo_df, client_feature_cols=["client_risco", "client_alto_valor", "client_urbano"]
    )
    recomendador.treinar_com_log(historico_df)
    recomendador.salvar("serving")

    recomendacoes_df = recomendador.recomendar_lote(clientes_df)
    # recomendacoes_df: uma linha por (client_id, cobertura recomendada)

Quem precisar de controle fino sobre alguma etapa (montar o vetor de
contexto manualmente, rodar o oraculo para um unico cliente ja em memoria,
etc.) pode usar as funcoes/classes de nivel mais baixo (Coverage,
build_item_context, LogisticBanditModel, greedy_oracle, train_from_log)
diretamente -- ver as secoes abaixo. Todas as funcoes que recebem dados
tabulares (load_catalog, train_from_log, recomendar_lote) aceitam tanto um
caminho de arquivo CSV quanto um DataFrame ja carregado.
"""

from __future__ import annotations
import json
import itertools
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ==============================================================================
# Catalogo de coberturas
# ==============================================================================

CATEGORIAS_RISCO = ["climatico", "eletrico_estrutural", "roubo_furto", "outros"]


@dataclass
class Coverage:
    """Representa uma cobertura de seguro disponivel para compor pacotes."""
    id: str
    name: str
    categoria_risco: str               # uma das categorias em CATEGORIAS_RISCO
    premium_band: float      # faixa de premio normalizada (0-1)
    mandatory: bool          # se a cobertura e obrigatoria no produto
    claim_freq: float        # frequencia historica de sinistro, normalizada (0-1)

    def semantic_features(self) -> np.ndarray:
        """Vetor de atributos de dominio (nao inclui o ID da cobertura)."""
        categoria_risco_onehot = np.array([1.0 if self.categoria_risco == p else 0.0 for p in CATEGORIAS_RISCO])
        return np.concatenate([
            categoria_risco_onehot,
            [self.premium_band, float(self.mandatory), self.claim_freq],
        ])

    def __repr__(self) -> str:
        return f"Coverage({self.id!r}, {self.name!r}, categoria_risco={self.categoria_risco!r})"


def load_catalog(catalogo: Union[str, pd.DataFrame]) -> Tuple[Dict[str, Coverage], List[str]]:
    """
    Monta o catalogo a partir de um CSV (caminho do arquivo) ou de um
    DataFrame ja carregado, com colunas:
        id, name, categoria_risco, premium_band, mandatory, claim_freq
    Retorna (dict id -> Coverage, lista ordenada de ids).
    """
    df = pd.read_csv(catalogo) if isinstance(catalogo, str) else catalogo
    catalog: Dict[str, Coverage] = {}
    for _, row in df.iterrows():
        catalog[row["id"]] = Coverage(
            id=row["id"], name=row["name"], categoria_risco=row["categoria_risco"],
            premium_band=float(row["premium_band"]),
            mandatory=bool(row["mandatory"]),
            claim_freq=float(row["claim_freq"]),
        )
    return catalog, list(df["id"])


def full_feature_vector(coverage: Coverage, all_ids: List[str]) -> np.ndarray:
    """Features semanticas + one-hot do ID (efeito idiossincratico por item)."""
    onehot = np.zeros(len(all_ids))
    onehot[all_ids.index(coverage.id)] = 1.0
    return np.concatenate([coverage.semantic_features(), onehot])


# ==============================================================================
# Construcao dos vetores de contexto
# ==============================================================================

def build_item_context(client: np.ndarray, coverage: Coverage, all_ids: List[str],
                        partial_size: int) -> np.ndarray:
    """
    Contexto de item: perfil do cliente, atributos da cobertura, interacao
    cliente x cobertura, e o tamanho do pacote parcial (fadiga) junto com
    sua interacao com o perfil do cliente.
    """
    f_full = full_feature_vector(coverage, all_ids)
    semantic = coverage.semantic_features()
    interaction = np.outer(client, semantic).flatten()
    size_scalar = np.array([float(partial_size)])
    size_interaction = client * partial_size
    return np.concatenate([client, f_full, interaction, size_scalar, size_interaction])


def build_pair_context(client: np.ndarray, cov_i: Coverage, cov_j: Coverage) -> np.ndarray:
    """Contexto de par: atributos de cada cobertura, sua interacao, e o cliente."""
    f_i = cov_i.semantic_features()
    f_j = cov_j.semantic_features()
    interaction = f_i * f_j
    return np.concatenate([f_i, f_j, interaction, client])


# ==============================================================================
# Modelo bandit logistico
# ==============================================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class LogisticBanditModel:
    """
    Bandit contextual com link logistico, ajustado em lote.

    - observe(x, y): registra uma observacao bruta (nao atualiza o modelo).
    - refit(): reajusta theta e a matriz de incerteza H sobre todo o
      historico acumulado (regressao logistica regularizada + Laplace).
    - p_ucb(x): score otimista (upper confidence bound) no espaco de
      probabilidade, sempre em (0,1).
    - save_serving_params / load_serving_params: persistencia dos
      parametros usados para servir recomendacoes, sem precisar do
      historico bruto.
    """

    def __init__(self, feature_dim: int, lam: float = 1.0, alpha: float = 1.0,
                 min_curvature: float = 0.05):
        """
        feature_dim: dimensao do vetor de contexto.
        lam: forca da regularizacao L2 (tambem usada como valor inicial de H).
        alpha: peso do termo de exploracao (bonus de incerteza).
        min_curvature: piso minimo para o peso p*(1-p) usado na matriz de
            incerteza -- evita que observacoes de itens com propensao muito
            proxima de 0 ou 1 deixem de reduzir a incerteza do modelo.
        """
        self.d = feature_dim
        self.lam = lam
        self.alpha = alpha
        self.min_curvature = min_curvature
        self.theta_hat = np.zeros(feature_dim)
        self.H = lam * np.eye(feature_dim)
        self.X_history: List[np.ndarray] = []
        self.y_history: List[float] = []

    def observe(self, x: np.ndarray, y: float):
        """Registra uma observacao (x, y) com y em {0, 1}."""
        self.X_history.append(x)
        self.y_history.append(float(y))

    def refit(self):
        """Reajusta o modelo (MAP logistico + Laplace) sobre todo o historico."""
        if not self.X_history:
            return
        X = np.array(self.X_history)
        y = np.array(self.y_history)

        def neg_log_posterior(theta):
            z = X @ theta
            log_lik = np.sum(y * z - np.logaddexp(0, z))
            reg = 0.5 * self.lam * np.sum(theta ** 2)
            return -(log_lik) + reg

        def grad(theta):
            z = X @ theta
            p = sigmoid(z)
            g = X.T @ (y - p) - self.lam * theta
            return -g

        res = minimize(
            neg_log_posterior, x0=self.theta_hat, jac=grad,
            method="L-BFGS-B", options={"maxiter": 300},
        )
        self.theta_hat = res.x

        p_hat = sigmoid(X @ self.theta_hat)
        weights = np.maximum(p_hat * (1 - p_hat), self.min_curvature)
        self.H = self.lam * np.eye(self.d) + (X * weights[:, None]).T @ X

    def p_ucb(self, x: np.ndarray) -> float:
        """Score otimista (probabilidade estimada + bonus de exploracao)."""
        eta = self.theta_hat @ x
        H_inv_x = np.linalg.solve(self.H, x)
        bonus = self.alpha * np.sqrt(max(x @ H_inv_x, 0.0))
        return float(sigmoid(eta + bonus))

    @property
    def n_observacoes(self) -> int:
        """Quantas observacoes brutas estao acumuladas (uteis para debug/monitoramento)."""
        return len(self.X_history)

    def __repr__(self) -> str:
        return (f"LogisticBanditModel(feature_dim={self.d}, lam={self.lam}, "
                f"alpha={self.alpha}, min_curvature={self.min_curvature}, "
                f"n_observacoes={self.n_observacoes})")

    def save_serving_params(self, path: str):
        """Salva apenas os parametros necessarios para servir recomendacoes."""
        payload = {
            "d": self.d, "lam": self.lam, "alpha": self.alpha,
            "min_curvature": self.min_curvature,
            "theta_hat": self.theta_hat.tolist(),
            "H": self.H.tolist(),
        }
        with open(path, "w") as f:
            json.dump(payload, f)

    @classmethod
    def load_serving_params(cls, path: str) -> "LogisticBanditModel":
        """Carrega um modelo pronto para servir recomendacoes."""
        with open(path) as f:
            payload = json.load(f)
        model = cls(feature_dim=payload["d"], lam=payload["lam"],
                     alpha=payload["alpha"], min_curvature=payload["min_curvature"])
        model.theta_hat = np.array(payload["theta_hat"])
        model.H = np.array(payload["H"])
        return model


# ==============================================================================
# Oraculo guloso (monta o pacote sem restricao de tamanho fixa)
# ==============================================================================

def greedy_oracle(client: np.ndarray, candidate_ids: List[str],
                   catalog: Dict[str, Coverage], all_ids: List[str],
                   item_model: LogisticBanditModel,
                   pair_model: LogisticBanditModel) -> List[str]:
    """
    Constroi o pacote adicionando, a cada passo, a cobertura com maior ganho
    marginal esperado (score individual + interacao com o que ja esta no
    pacote, ambos centralizados em 0.5). Para quando nenhum candidato
    restante tem ganho marginal positivo -- o tamanho do pacote emerge da
    propria decisao, sem k fixo.

    candidate_ids: coberturas elegiveis para entrar no pacote nesta chamada
        (pode ser um subconjunto do catalogo, ex.: se regras de negocio
        excluirem algumas coberturas para este cliente especifico).
    all_ids: lista completa e ORDENADA do catalogo, usada para construir o
        one-hot idiossincratico de cada cobertura (Coverage.semantic_features
        + full_feature_vector) -- precisa ser sempre a mesma lista, na mesma
        ordem, usada no treino, mesmo que candidate_ids seja menor.
    """
    pacote: List[str] = []
    candidatos_restantes = list(candidate_ids)

    while candidatos_restantes:
        melhor_id, melhor_ganho = None, -np.inf

        for id_candidato in candidatos_restantes:
            cobertura = catalog[id_candidato]
            x_item = build_item_context(client, cobertura, all_ids, partial_size=len(pacote))
            ganho_marginal = item_model.p_ucb(x_item) - 0.5

            for id_no_pacote in pacote:
                z_pair = build_pair_context(client, cobertura, catalog[id_no_pacote])
                ganho_marginal += pair_model.p_ucb(z_pair) - 0.5

            if ganho_marginal > melhor_ganho:
                melhor_ganho, melhor_id = ganho_marginal, id_candidato

        if melhor_id is None or melhor_ganho <= 0:
            break

        pacote.append(melhor_id)
        candidatos_restantes.remove(melhor_id)

    return pacote


# ==============================================================================
# Treino em lote a partir de um log de interacoes
# ==============================================================================
# Formato esperado (CSV ou DataFrame), uma linha por proposta ja finalizada:
#   <colunas de client_feature_cols>, offered, final
# onde `offered` e `final` sao ids de cobertura separados por "|".

def train_from_log(log: Union[str, pd.DataFrame], catalog: Dict[str, Coverage],
                    all_ids: List[str], item_model: LogisticBanditModel,
                    pair_model: LogisticBanditModel, client_feature_cols: List[str]):
    """
    Le o log de interacoes (CSV ou DataFrame), traduz cada linha em
    observacoes de item e de par, e reajusta os dois modelos. Reinicia o
    historico bruto antes de processar (o reajuste sempre reprocessa o log
    completo -- ver observacao no docstring de LogisticBanditModel.refit).

    client_feature_cols: nomes das colunas que compoe o vetor de perfil do
        cliente, na ordem em que devem ser concatenadas (ex.:
        ["client_risco", "client_alto_valor", "client_urbano"]).
    """
    df = pd.read_csv(log) if isinstance(log, str) else log

    item_model.X_history, item_model.y_history = [], []
    pair_model.X_history, pair_model.y_history = [], []

    for _, row in df.iterrows():
        client = np.array([row[col] for col in client_feature_cols], dtype=float)
        offered = str(row["offered"]).split("|") if pd.notna(row["offered"]) else []
        final = set(str(row["final"]).split("|")) if pd.notna(row["final"]) and row["final"] else set()

        partial_size = max(len(offered) - 1, 0)
        for cid in offered:
            x_item = build_item_context(client, catalog[cid], all_ids, partial_size=partial_size)
            y_item = 1.0 if cid in final else 0.0
            item_model.observe(x_item, y_item)

        for cid, jid in itertools.combinations(offered, 2):
            z_pair = build_pair_context(client, catalog[cid], catalog[jid])
            y_pair = 1.0 if (cid in final and jid in final) else 0.0
            pair_model.observe(z_pair, y_pair)

    item_model.refit()
    pair_model.refit()


# ==============================================================================
# Recomendacao em lote (batch) -- DataFrame de clientes -> DataFrame de recomendacoes
# ==============================================================================

def recommend_batch(clientes: Union[str, pd.DataFrame], catalog: Dict[str, Coverage],
                     all_ids: List[str], item_model: LogisticBanditModel,
                     pair_model: LogisticBanditModel, client_feature_cols: List[str],
                     client_id_col: str = "client_id",
                     candidatos: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Aplica o oraculo guloso a um DataFrame inteiro de clientes de uma vez
    (CSV ou DataFrame ja carregado) e devolve um DataFrame "tidy" (uma linha
    por cliente x cobertura recomendada), pronto para ser salvo numa tabela
    ou consumido por outro sistema.

    Colunas do DataFrame de saida:
        client_id, coverage_id, coverage_name, ordem
    `ordem` e a posicao em que o oraculo guloso adicionou aquela cobertura
    ao pacote (1 = primeira escolhida, geralmente a de maior ganho marginal
    isolado). Clientes sem nenhuma cobertura recomendada geram uma unica
    linha com coverage_id/coverage_name/ordem em branco (None).
    """
    df = pd.read_csv(clientes) if isinstance(clientes, str) else clientes
    candidatos_default = candidatos if candidatos is not None else all_ids

    linhas = []
    for _, row in df.iterrows():
        client_vec = np.array([row[col] for col in client_feature_cols], dtype=float)
        pacote = greedy_oracle(client_vec, candidatos_default, catalog, all_ids,
                                item_model, pair_model)

        if not pacote:
            linhas.append({
                client_id_col: row[client_id_col],
                "coverage_id": None, "coverage_name": None, "ordem": None,
            })
        else:
            for ordem, cid in enumerate(pacote, start=1):
                linhas.append({
                    client_id_col: row[client_id_col],
                    "coverage_id": cid, "coverage_name": catalog[cid].name, "ordem": ordem,
                })

    return pd.DataFrame(linhas)


# ==============================================================================
# Interface de alto nivel
# ==============================================================================
# Empacota o catalogo e os dois modelos (item e par) numa unica classe, para
# quem for usar o algoritmo no dia a dia sem precisar lidar diretamente com
# os vetores de contexto ou gerenciar duas instancias de LogisticBanditModel
# separadamente. As funcoes acima (build_item_context, greedy_oracle,
# train_from_log, etc.) continuam disponiveis para quem precisar de controle
# fino sobre alguma etapa especifica.

class RecomendadorPacotes:
    """
    Uso tipico (toda a entrada/saida de dados tabulares e via DataFrame ou
    caminho de CSV -- ver `recomendar_lote` para a recomendacao em lote):

        catalogo_df = pd.read_csv("coverages.csv")
        recomendador = RecomendadorPacotes.novo(
            catalogo_df, client_feature_cols=["client_risco", "client_alto_valor", "client_urbano"]
        )
        recomendador.treinar_com_log("historico_interacoes.csv")   # ou um DataFrame
        recomendador.salvar("serving")   # publica os parametros para o servico

        # em outro processo (o servico de recomendacao em tempo real):
        recomendador = RecomendadorPacotes.carregar("coverages.csv", "serving")
        recomendacoes_df = recomendador.recomendar_lote("novos_clientes.csv")
    """

    def __init__(self, catalog: Dict[str, Coverage], all_ids: List[str],
                 item_model: LogisticBanditModel, pair_model: LogisticBanditModel,
                 client_feature_cols: List[str]):
        self.catalog = catalog
        self.all_ids = all_ids
        self.item_model = item_model
        self.pair_model = pair_model
        self.client_feature_cols = client_feature_cols

    @classmethod
    def novo(cls, catalogo: Union[str, pd.DataFrame], client_feature_cols: List[str],
             lam: float = 1.0, alpha: float = 1.0, min_curvature: float = 0.05) -> "RecomendadorPacotes":
        """
        Cria um recomendador do zero (modelos ainda nao treinados), a partir
        do catalogo de coberturas (CSV ou DataFrame).

        client_feature_cols: nomes das colunas que vao compor o perfil do
            cliente nos DataFrames de treino/recomendacao (ex.:
            ["client_risco", "client_alto_valor", "client_urbano"]).
        """
        catalog, all_ids = load_catalog(catalogo)
        cliente_exemplo = np.zeros(len(client_feature_cols))
        item_dim = len(build_item_context(cliente_exemplo, catalog[all_ids[0]], all_ids, 0))
        pair_dim = len(build_pair_context(cliente_exemplo, catalog[all_ids[0]], catalog[all_ids[1]]))
        item_model = LogisticBanditModel(item_dim, lam, alpha, min_curvature)
        pair_model = LogisticBanditModel(pair_dim, lam, alpha, min_curvature)
        return cls(catalog, all_ids, item_model, pair_model, client_feature_cols)

    @classmethod
    def carregar(cls, catalogo: Union[str, pd.DataFrame], prefixo_arquivos: str) -> "RecomendadorPacotes":
        """
        Carrega um recomendador ja treinado, a partir do catalogo e dos
        parametros de servico previamente salvos com `.salvar(prefixo)`
        (arquivos "{prefixo}_item.json", "{prefixo}_pair.json" e
        "{prefixo}_config.json").
        """
        catalog, all_ids = load_catalog(catalogo)
        item_model = LogisticBanditModel.load_serving_params(f"{prefixo_arquivos}_item.json")
        pair_model = LogisticBanditModel.load_serving_params(f"{prefixo_arquivos}_pair.json")
        with open(f"{prefixo_arquivos}_config.json") as f:
            config = json.load(f)
        return cls(catalog, all_ids, item_model, pair_model, config["client_feature_cols"])

    def treinar_com_log(self, log: Union[str, pd.DataFrame]):
        """Retreino em lote a partir de um log de interacoes (CSV ou DataFrame)."""
        train_from_log(log, self.catalog, self.all_ids, self.item_model,
                        self.pair_model, self.client_feature_cols)

    def recomendar_lote(self, clientes: Union[str, pd.DataFrame],
                         client_id_col: str = "client_id",
                         candidatos: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Interface principal para uso em producao: recebe um DataFrame (ou
        CSV) com varios clientes e devolve um DataFrame com as recomendacoes
        de todos eles (uma linha por cliente x cobertura recomendada -- ver
        `recommend_batch` para o detalhe das colunas de saida).
        """
        return recommend_batch(clientes, self.catalog, self.all_ids, self.item_model,
                                self.pair_model, self.client_feature_cols,
                                client_id_col=client_id_col, candidatos=candidatos)

    def recomendar(self, client: np.ndarray, candidatos: Optional[List[str]] = None) -> List[str]:
        """
        Versao para um unico cliente ja em memoria (um vetor numpy, na mesma
        ordem de `client_feature_cols`) -- util em testes rapidos ou dentro
        de outro loop; para uso em producao com varios clientes de uma vez,
        prefira `recomendar_lote`.
        """
        candidatos = candidatos if candidatos is not None else self.all_ids
        return greedy_oracle(client, candidatos, self.catalog, self.all_ids,
                              self.item_model, self.pair_model)

    def salvar(self, prefixo_arquivos: str):
        """
        Publica os parametros de servico (para o serviço de recomendacao
        carregar): "{prefixo}_item.json", "{prefixo}_pair.json" e
        "{prefixo}_config.json" (este ultimo guarda client_feature_cols).
        """
        self.item_model.save_serving_params(f"{prefixo_arquivos}_item.json")
        self.pair_model.save_serving_params(f"{prefixo_arquivos}_pair.json")
        with open(f"{prefixo_arquivos}_config.json", "w") as f:
            json.dump({"client_feature_cols": self.client_feature_cols}, f)

    def __repr__(self) -> str:
        return (f"RecomendadorPacotes({len(self.catalog)} coberturas, "
                f"client_feature_cols={self.client_feature_cols}, "
                f"item_model: {self.item_model.n_observacoes} obs, "
                f"pair_model: {self.pair_model.n_observacoes} obs)")
