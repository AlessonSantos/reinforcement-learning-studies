# Síntese final: do "top-k UCB" ao algoritmo consolidado
### Toda a trajetória de decisões, problema por problema, até a versão em produção

---

## 1. O problema de negócio original

Múltiplas coberturas de seguro residencial, sem histórico de conversão (cold start), necessidade de montar pacotes personalizados por perfil de cliente. Proposta inicial da equipe: UCB contextual por cobertura, pacote = top-$k$ do ranking.

**Pergunta que abriu a discussão:** essa heurística basta, ou vale a pena um framework mais formal (C²UCB)?

---

## 2. UCB/LinUCB vs. C²UCB — por que "top-k" não tem prova própria

O UCB (Auer et al., 2002) e o LinUCB (Li et al., 2010) foram formulados para escolher **um único braço por rodada**. Cortar um ranking em $k$ é uma adaptação sem regret formal.

**C²UCB (Qin, Chen & Zhu, 2014)** generaliza para escolher um **super-braço** via um oráculo que maximiza uma função de recompensa sujeita a monotonicidade e continuidade de Lipschitz. Quando a recompensa é aditiva e a restrição é cardinalidade pura, o oráculo coincide exatamente com "ordenar e pegar os $k$ maiores" — a heurística da equipe era um caso particular correto, só sem formalização. A vantagem real do C²UCB só aparece com sinergia/canibalização ou restrições além de cardinalidade.

---

## 3. Como o paper original constrói features de item — e por que não serve aqui

O paper usa recomendação de filmes com embeddings de **Probabilistic Matrix Factorization (PMF)**, treinada sobre histórico substancial de ratings — inviável em cold start real. Alternativas propostas: features manuais de domínio, embeddings textuais das condições gerais, ou fatoração aprendida futuramente conforme dados se acumulam.

---

## 4. C²UCB vs. DC²B

**DC²B (Liu, Xiao, Wu, Miao & Zhang, 2019)** difere estruturalmente: feedback binário/implícito, diversidade via **Processo Pontual Determinantal (DPP, multiplicativo)**, estimação por Thompson Sampling bayesiano em vez de ridge regression frequentista.

**Achado decisivo:** o DPP é, por construção, um mecanismo de **repulsão pura** — incapaz de representar sinergia positiva. Descartado para este caso de uso, que exige capturar tanto sinergia quanto canibalização.

---

## 5. "Posso considerar só relevância, sem diversidade?"

Sim — nenhum framework exige diversidade. Zerar o termo de diversidade no C²UCB (`lambda_div=0`) recupera exatamente o **k-LinUCB**, a heurística original. No DC²B isso não é possível sem descaracterizar o modelo (a repulsão do DPP não se desliga).

---

## 6. A virada de requisito: sinergia real, sem $k$ fixo

O usuário definiu o requisito central: pacotes de tamanho variável, sinergia/canibalização real capturada, sem restrição de cardinalidade — o modelo decide quais coberturas entram.

**Problema matemático:** recompensa monótona sem restrição de cardinalidade ⇒ solução ótima trivial é "inclua tudo". Era necessário um mecanismo que fizesse o ganho marginal ficar negativo em algum ponto.

---

## 7. Prêmio como freio — rejeitado

Primeira proposta de freio: subtrair custo de prêmio da recompensa. **Rejeitada pelo usuário**: o freio deveria vir inteiramente do comportamento observado do cliente, não do valor monetário do pacote.

---

## 8. Fadiga aprendida via tamanho do pacote parcial

Solução: incluir $|S_{parcial}|$ (e sua interação com o perfil do cliente) como parte do contexto de cada item — permitindo que o próprio modelo aprenda se e o quanto a propensão cai conforme o pacote cresce, sem impor nada manualmente.

---

## 9. Exclusão e não-adição como sinal negativo genuíno

Com apenas 14 coberturas totalmente visíveis, "não-adição" deixa de ser ambígua (diferente de catálogos com milhares de itens) — é rejeição consciente. Solução (na época, ainda com ridge regression): **centralizar** a contribuição de cada item/par em 0,5 na função de recompensa, para que a rejeição vire prejuízo genuíno, não apenas "menos ganho".

---

## 10. Fadiga + rejeição específica — compatíveis, não conflitantes

Confirmado que os dois mecanismos são ortogonais: fadiga responde "quanto já foi ofertado?"; rejeição específica responde "essa cobertura/par é ruim para esse perfil?". Entram como termos aditivos na mesma regressão, sem conflito.

---

## 11. A vantagem de considerar pares (interação de segunda ordem)

**Exemplo central da conversa:** "Desastres Naturais" e "Danos Elétricos" se sobrepõem em risco (raio → dano elétrico). Sem termo de par, o modelo só-item atribuiria a queda de conversão conjunta **inteiramente à fadiga** — um erro de atribuição que contamina o aprendizado em outros contextos. Com o termo de par $\hat w(i,j)$ (parâmetro compartilhado $\phi$, contexto $z(i,j)$), a canibalização fica corretamente isolada no par específico; o mesmo vale para sinergia positiva, que o modelo só-item não representa de jeito nenhum.

**Custo identificado:** mais dimensionalidade (91 pares com 14 coberturas), mais necessidade de dados cobrindo cada par junto/separado.

---

## 12. Esclarecimento: um único sinal observado, não dois canais

Fadiga e termo de par **não são reward separados** — são componentes que competem para explicar a mesma variável observada (manteve/removeu/adicionou). Um explica pela contagem, o outro pela identidade dos coabitantes do pacote.

---

## 13. "Não devo penalizar pacotes grandes já na amostragem aleatória"

Confirmado que **não** — isso contaminaria o "experimento aleatorizado embutido" da fase de exploração, pré-supondo o efeito de fadiga que se quer medir. O freio deve morar inteiramente no termo aprendido, não na coleta de dados.

---

## 14. O problema que motivou a virada para regressão logística

Ao detalhar o pipeline completo, surgiu a pergunta-chave: **$\hat r(i)$ está garantidamente em $[0,1]$, para a centralização em 0,5 ser exata?**

**Resposta: não**, com ridge regression. $\hat\theta^Tx$ pode sair de $[0,1]$ livremente, e o bônus de exploração é desenhado para empurrar a estimativa para cima de propósito. A centralização, até ali, era só uma aproximação prática.

**Solução: função de ligação logística.** Aplicando o bônus no espaço linear e só depois passando pelo funil $\sigma(\cdot)$, a saída fica sempre entre 0 e 1 — resolvendo a imprecisão de forma exata.

---

## 15. Esclarecimento sobre atualização "online" vs. em lote

"Online" (sequencial) e "em lote" não são incompatíveis. Ridge regression tem atualização somável; regressão logística não — mas isso não impede uso em lote, só exige reajuste periódico via solver padrão. Dado que a equipe já trabalha em lote, a regressão logística deixou de ter a desvantagem inicialmente atribuída a ela, e passou a ser a opção recomendada.

**Ressalva sobre reajuste completo:** a cada lote, reprocessa-se **todo** o histórico (inclusive a fase aleatória inicial), o que assume implicitamente estacionariedade do comportamento do cliente — mitigável no futuro via janela deslizante ou ponderação por recência, se necessário.

---

## 16. Reescrita didática do pipeline + exemplo de vetor de features

O pipeline foi formalizado em 10 etapas (amostragem → scores UCB/logísticos → oráculo guloso → oferta → edição → tradução em observações → reajuste em lote → ciclo), com analogias e um exemplo numérico contínuo ("Cliente A") para tornar tangível a cadeia de dependências entre etapas.

**Sobre as features das coberturas:** confirmado que um ID puro (one-hot) é tecnicamente válido, mas custa a capacidade de generalização entre itens/pares parecidos — especialmente prejudicial no termo de par, que viraria uma "tabela de consulta" sem conseguir inferir nada sobre pares pouco observados. Recomendação: combinar features semânticas de domínio (categoria de risco, faixa de prêmio, obrigatoriedade, frequência de sinistro) **com** o ID, no mesmo vetor — à la Factorization Machines. Exemplo concreto construído para "Danos Elétricos" e "Desastres Naturais", incluindo o vetor de interação $f_i\otimes f_j$ e a limitação identificada (a taxonomia one-hot de risco não capta a sobreposição física real entre climático e elétrico — algo que $\phi$ precisa aprender empiricamente ou que pode ser suprido por dados de correlação de sinistro, se existirem).

---

## 17. Implementação em Python da versão consolidada

A implementação final foi organizada como um **módulo reutilizável**, `c2ucb_seguro.py`, pensado para ser importado por qualquer pessoa do time (não é um script de demonstração — não contém exemplo de execução nem geração de dados, só o algoritmo em si). Toda a entrada e saída de dados tabulares (catálogo, log de treino, clientes a recomendar) é baseada em **DataFrames** (ou caminho de CSV, lido internamente) — refletindo como os dados de fato circulam entre times de dados/produto. Contém:
- `Coverage` / `load_catalog()` — a estrutura de uma cobertura (features semânticas + ID) e a função para carregar o catálogo, aceitando um DataFrame já carregado ou um caminho de CSV.
- `build_item_context` / `build_pair_context` — os vetores $x(i,S_{parcial})$ e $z(i,j)$.
- `LogisticBanditModel` — a classe central: `observe()` (registra uma observação bruta), `refit()` (reajuste em lote via MAP + Laplace, já com o piso de curvatura `min_curvature`), `p_ucb()` (score otimista em tempo real), e `save_serving_params()` / `load_serving_params()` (persistência dos parâmetros de serviço, separada do histórico bruto).
- `greedy_oracle()` — cresce o pacote sem $k$ fixo, parando quando o ganho marginal $\Delta \le 0$, para **um** cliente por vez.
- `train_from_log()` — lê um log de interações históricas (DataFrame ou CSV), traduz cada linha em observações de item e de par, e chama `refit()` nos dois modelos. As colunas que compõem o perfil do cliente são configuráveis (`client_feature_cols`), não fixas.
- **`recommend_batch()`** — a peça nova: recebe um **DataFrame inteiro de clientes** (ou CSV) e devolve um **DataFrame de recomendações**, uma linha por combinação cliente × cobertura recomendada (colunas `client_id`, `coverage_id`, `coverage_name`, `ordem`), aplicando `greedy_oracle()` cliente a cliente por trás dos panos.
- **`RecomendadorPacotes`** — a interface de alto nível que empacota catálogo + os dois modelos + a configuração de colunas do cliente, com métodos `.novo()`, `.treinar_com_log()`, `.recomendar_lote()` (a interface principal, DataFrame→DataFrame), `.recomendar()` (para um único cliente já em memória, útil em testes), `.salvar()` e `.carregar()` — esta última persistindo também `client_feature_cols` num arquivo `{prefixo}_config.json`, para que o processo de serviço não precise redefinir manualmente quais colunas usar.

A geração de dados sintéticos para teste e o exemplo de execução ponta a ponta foram deliberadamente mantidos **fora** deste módulo, em arquivos separados (ver Seção 21). O script de exemplo (`c2ucb_final.py`) foi reescrito para **importar** `c2ucb_seguro.py` em vez de duplicar o algoritmo — evitando o risco de as duas versões divergirem ao longo do tempo — e demonstra exatamente o fluxo em lote: carrega o catálogo e o log como DataFrames, treina, publica os parâmetros, recarrega, e chama `.recomendar_lote()` sobre o DataFrame de novos clientes, imprimindo o DataFrame de recomendações resultante.

---

## 18. O problema descoberto ao formalizar a transformação em probabilidade: o efeito kappa

Ao perguntar se havia problema em transformar o score em probabilidade via sigmoid, identificou-se uma limitação real da literatura de **GLM-UCB** (Filippi et al., 2010): o regret depende de um fator $\kappa$ ligado à derivada mínima da função de ligação. Como $p(1-p)$ (o peso de cada observação na matriz de incerteza $H$) tende a zero quando $p$ está perto de 0 ou 1, **itens "quase certos" (ex.: "Incêndio", quase sempre mantida) paravam de reduzir sua própria incerteza**, mesmo depois de centenas de observações consistentes — o bônus de exploração estacionava artificialmente (confirmado empiricamente: bônus parado em ~0,50 após 300 observações idênticas).

---

## 19. A correção escolhida: piso de curvatura, não contagem bruta

Duas correções foram avaliadas:
- **Piso de curvatura** ($\max(p(1-p), \tau)$ na soma de $H$) — corrige o mecanismo exato onde o problema nasce, permanecendo dentro da geometria contextual do modelo.
- **Bônus híbrido por contagem bruta** (estilo UCB1, contando observações por item/par) — descartada como abordagem principal, por reintroduzir uma lógica tabular incompatível com um sistema deliberadamente contextual (o mesmo item aparece em contextos de cliente diferentes; contar só por identidade do item joga fora a generalização que motivou usar C²UCB desde o início).

**Decisão final:** piso de curvatura como correção principal, implementado e validado (o bônus voltou a decair de forma consistente, de 0,50 para 0,25 após 300 observações, em vez de estacionar).

---

## 20. Documentação final consolidada

Dois artefatos entregues, sincronizados com a decisão da Seção 19:
- **`pipeline_c2ucb_seguro_didatico_v2.md`** — o passo a passo didático completo (10 etapas, analogias, exemplo numérico contínuo, mapa de conexões), já incorporando o piso de curvatura $\tau$ e, a pedido, com descrição explícita de cada termo dos vetores de contexto ($f_i$, $c\otimes f_i$, $|S_{parcial}|$, $c\times|S_{parcial}|$, $f_i\otimes f_j$) e do papel de $\lambda$ em cada etapa onde aparece.
- **`c2ucb_seguro.py`** — o módulo de implementação de referência (algoritmo puro, sem exemplo de execução), já com o parâmetro `min_curvature` incorporado.

**Seção de parâmetros de configuração** foi adicionada ao documento didático, listando o que precisa ser definido operacionalmente pela equipe: $\lambda$, $\alpha$, $\tau$, frequência de reajuste em lote, cronograma de $\epsilon_t$, e as dimensões semânticas do catálogo e do perfil de cliente — deixando claro que nenhum desses valores vem "de fábrica" da matemática, são decisões de calibração da equipe.

---

## 21. Como funciona a implantação (deploy) do retreino automático em lote

A implementação final (`c2ucb_seguro.py`) separa deliberadamente **dois processos independentes**, que rodam em cadências diferentes:

### Processo 1: Treino (job agendado, roda em lote)

**O que faz:** lê o log completo de interações acumuladas (a tabela de propostas + edições do cliente, equivalente ao `historico_interacoes.csv`), reconstrói todas as observações de item e de par (`train_from_log()`), e chama `refit()` nos dois modelos — reprocessando o histórico inteiro do zero, como já discutido na Seção 15.

**Onde roda:** um job agendado — um cron, uma DAG no Airflow, um job no Databricks/EMR, ou qualquer orquestrador que a equipe já use. Não precisa estar no caminho de resposta em tempo real; pode rodar de forma assíncrona, em background.

**Qual parâmetro aciona o retreino:** o gatilho recomendado é **baseado em tempo**, não em volume de eventos — por exemplo, uma execução diária (`REFIT_SCHEDULE = "daily"` ou o equivalente na ferramenta de orquestração escolhida). Dado o volume de dados esperado (catálogo pequeno, algumas centenas/milhares de propostas por dia), reprocessar o histórico inteiro diariamente é computacionalmente trivial (segundos a poucos minutos). Um gatilho por volume (ex.: "a cada 500 novas propostas") também é válido, mas menos previsível operacionalmente — eu recomendaria começar com o agendamento diário e só migrar para volume se o tráfego real for muito irregular.

**Saída do processo de treino:** os arquivos `serving_item.json`, `serving_pair.json` e `serving_config.json` (via `RecomendadorPacotes.salvar("serving")`, que por baixo chama `save_serving_params()` nos dois modelos e grava as colunas do perfil do cliente) — contendo **apenas** $\hat\theta$, $H$, os hiperparâmetros ($\lambda$, $\alpha$, $\tau$) e a configuração de colunas, nunca o histórico bruto. Esse é o único conjunto de artefatos que atravessa a fronteira entre treino e serviço.

### Processo 2: Serviço (responde a requisições em tempo real)

**O que faz:** carrega os parâmetros publicados mais recentes (`LogisticBanditModel.load_serving_params()`) e, para cada requisição de recomendação, monta o contexto do cliente e chama `greedy_oracle()` — uma operação rápida (o catálogo tem só 14 itens, o oráculo guloso é $O(k^2)$ no pior caso).

**Quando recarrega os parâmetros:** duas estratégias comuns, ambas válidas:
- **Recarga no start do processo** — reinicia o serviço (ou os workers) logo após cada execução bem-sucedida do treino, de forma orquestrada pelo próprio pipeline de deploy.
- **Hot-reload periódico** — o serviço verifica periodicamente (ex.: a cada 10-15 minutos) se existe uma versão mais nova do arquivo de parâmetros e recarrega em memória sem reiniciar o processo, evitando qualquer downtime.

Eu recomendaria a segunda opção (hot-reload), por ser operacionalmente mais simples de manter sem downtime, dado que o arquivo de parâmetros é pequeno (poucos KBs, um JSON com vetores/matrizes de dimensão modesta).

### Arquivos de dados para testar o pipeline antes de conectar aos dados reais

Antes de ligar o sistema ao data warehouse de verdade, é possível validar toda a lógica (treino, oráculo, persistência, recomendação) com dados sintéticos gerados propositalmente com padrões conhecidos de fadiga, sinergia e canibalização — permitindo checar se o modelo realmente aprende esses padrões antes de confiar nele com dados reais.

- **`gerar_dados_exemplo.py`** — script que cria os três arquivos abaixo. Simula um ambiente onde a decisão do cliente segue regras conhecidas de propósito (fadiga proporcional ao tamanho do pacote ofertado, canibalização entre coberturas de risco climático e elétrico-estrutural, sinergia entre "Conteúdo Residencial" e "Roubo e Furto") — assim, depois de treinar, dá para conferir se $\hat\phi$ realmente aprendeu essas relações, em vez de confiar apenas na execução sem erros.

- **`coverages.csv`** — o catálogo de coberturas usado por `load_catalog()`. Colunas: `id` (identificador único), `name` (nome de exibição), `risco` (uma das quatro categorias: climático, elétrico-estrutural, roubo-furto, outros), `premium_band` (faixa de prêmio normalizada entre 0 e 1), `mandatory` (booleano — se a cobertura é obrigatória no produto), `claim_freq` (frequência histórica de sinistro, normalizada entre 0 e 1). Em produção, esse arquivo seria substituído pela tabela de produto/catálogo real, com os mesmos atributos definidos pela equipe atuarial.

- **`historico_interacoes.csv`** — o log de interações consumido por `train_from_log()`, com uma linha por proposta já finalizada. Colunas: `client_risco`, `client_alto_valor`, `client_urbano` (o vetor de perfil do cliente, $c$), `offered` (as coberturas que foram ofertadas naquela rodada, com os ids separados por `|`), `final` (as coberturas que o cliente efetivamente manteve/adicionou, no mesmo formato). Em produção, esse arquivo seria substituído por uma consulta à tabela de propostas + edições do cliente no data warehouse, no mesmo formato de colunas.

- **`novos_clientes.csv`** — uma lista pequena de clientes fictícios para testar a geração de recomendações depois do treino. Colunas: `client_id` (identificador do cliente, só para exibição), `client_risco`, `client_alto_valor`, `client_urbano` (o mesmo perfil $c$ usado no histórico). Em produção, isso seria substituído pela requisição em tempo real vinda do sistema que solicita a recomendação (ex.: o app ou o site no momento da cotação).

**Como esses arquivos se conectam ao módulo:** rodar `gerar_dados_exemplo.py` uma vez cria os três CSVs; a partir daí, `RecomendadorPacotes.novo(catalogo_df, client_feature_cols=[...])` carrega o catálogo e cria os dois modelos, `.treinar_com_log(historico_df)` popula e treina (aceita DataFrame ou caminho de CSV), `.salvar("serving")` publica os parâmetros (incluindo `client_feature_cols`, gravado em `serving_config.json`), e `.recomendar_lote(clientes_df)` é chamado **uma única vez sobre o DataFrame inteiro** de novos clientes, devolvendo um DataFrame de recomendações (uma linha por cliente × cobertura recomendada) — o mesmo fluxo que rodaria em produção, só trocando a origem dos dados (arquivos locais → tabelas do data warehouse / API em tempo real). É exatamente esse fluxo, já ponta a ponta, que `c2ucb_final.py` demonstra.

### Cuidados operacionais recomendados antes de automatizar totalmente

1. **Versionamento dos parâmetros publicados.** Guardar não só o `serving_*.json` mais recente, mas um histórico versionado (ex.: `serving_item_2026-07-17.json`) — permite reverter rapidamente para uma versão anterior se o retreino mais recente produzir um modelo com comportamento inesperado (ex.: detectado via as métricas de monitoramento abaixo).
2. **Validação automática pós-treino, antes de publicar.** Antes de sobrescrever o arquivo de serviço, vale rodar uma checagem automática comparando o novo modelo com o anterior em um conjunto de clientes de referência (ex.: os tamanhos médios de pacote recomendado mudaram de forma abrupta? algum $\hat\theta$ individual "explodiu" para um valor extremo?) — barrando a publicação automaticamente se algo parecer anômalo, e alertando a equipe.
3. **Monitoramento contínuo**, não só no momento do treino: acompanhar ao longo do tempo o tamanho médio dos pacotes recomendados (Seção 8 do documento didático já recomendava isso), a taxa de aceitação por cobertura, e como o bônus de exploração está evoluindo para os itens mais frequentes (para confirmar que a correção do efeito kappa, Seção 19, está de fato produzindo o decaimento esperado em produção, não só no teste sintético).
4. **Nunca deixar $\epsilon_t$ chegar a exatamente zero.** Mesmo em produção madura, manter uma fração pequena e constante de exploração aleatória (ver a tabela de parâmetros do documento didático) — isso garante que o sistema continue detectando mudanças de padrão ao longo do tempo, em vez de convergir permanentemente para o que aprendeu nos primeiros meses.

---

## 22. Defesa metodológica do sigmoide, arquitetura via API, e formalização da exploração aleatória

### 22.1 A avaliação da biblioteca `contextualbandits` (LogisticUCB) — descartada

Ao pesquisar alternativas prontas, avaliamos a classe `LogisticUCB` da biblioteca `contextualbandits` (baseada em Cortes, 2018). Conferindo o código-fonte, confirmamos que ela: (a) usa um modelo **disjunto por braço** — uma regressão logística independente por cobertura, sem compartilhar $\theta$ entre coberturas parecidas, jogando fora exatamente a generalização que motivou nossa arquitetura desde a Seção 16; (b) resolve o problema de escolher **um único braço por rodada**, sem nenhuma noção de super-braço ou oráculo combinatório — teríamos que construir toda a camada de pacote/pares por conta própria de qualquer forma; (c) o próprio autor da biblioteca desaconselha essa classe específica para uso sério ("implemented for comparison purposes only"). Conclusão: descartada — mecanicamente é a mesma ideia (GLM-UCB) que já implementamos, só que numa arquitetura incompatível com o nosso problema.

### 22.2 Defesa formal de "recompensa + incerteza dentro do sigmoide"

Formalizamos a defesa contra a objeção "isso não perde a interpretação da probabilidade?" em um documento dedicado: **`defesa_metodologica_sigmoide_incerteza.md`**. Os pontos centrais:
- $p_{UCB}(i)$ é o **supremo** de $\sigma(\theta^Tx(i))$ sobre o **conjunto de confiança** de $\theta$ (o equivalente multidimensional de um intervalo de confiança — uma elipse, não um segmento de reta, já que $\theta$ é um vetor).
- Como $\sigma$ é monótona crescente, o supremo comuta com a composição — somar o bônus no espaço linear e só depois aplicar o funil é a **única** forma matematicamente correta de calcular esse supremo, não uma conveniência.
- A crítica só procederia se o bônus fosse emprestado do modelo linear-gaussiano antigo; não é o caso — $H$ é reconstruída da curvatura da própria verossimilhança de Bernoulli (GLM-UCB, Filippi et al., 2010).
- $p_{UCB}(i)$ **não é** uma probabilidade calibrada — converge para a estimativa calibrada $\hat p(i)=\sigma(\hat\theta^Tx(i))$ só quando o bônus é pequeno; quando o bônus é grande, satura perto de 1 e vira um sinal de decisão para o oráculo guloso, não uma crença sobre o mundo.
- **Caso de igualdade**: dois itens podem ter o mesmo $p_{UCB}$ com decomposições opostas (um "candidato de confiança", outro "candidato de exploração") — intercambiáveis para a decisão de hoje, mas com trajetórias diferentes depois, porque a observação de um item pouco conhecido é mais informativa que a de um item já bem conhecido (a mesma lógica do efeito kappa, Seção 18).

### 22.3 Rodadas aleatórias são necessárias, ou o termo de exploração já bastaria?

O bônus de exploração cobre "testar itens/pares pouco conhecidos", mas **não é estocástico** — dado um estado fixo do modelo, ele sempre desloca a decisão da mesma forma, deterministicamente. Ele nunca gera, sozinho, variação de **tamanho de pacote** descorrelacionada do perfil do cliente — e essa variação é justamente o que a identificação da fadiga precisa (a mesma questão de confusão entre fadiga e efeitos específicos discutida nas Seções 11-12). Além disso, ao abrir mão da restrição de cardinalidade fixa e do regime submodular monótono clássico, também abrimos mão da prova formal que garantiria que o bônus, sozinho, exploraria suficientemente. Conclusão: a fase aleatória continua sendo necessária, não substituível pelo termo de incerteza.

### 22.4 Formalização de $Z_t$ e por que não são "dois componentes aleatórios" competindo

$$Z_t \sim \text{Bernoulli}(\epsilon_t)$$

decide a rota da rodada (aleatória se $Z_t=1$, via modelo se $Z_t=0$) — já incorporada explicitamente na Etapa 2 do documento didático. Esclarecido que isso não compete com o bônus de incerteza do modelo: o bônus é determinístico, $Z_t$ é a única fonte de aleatoriedade deliberada do sistema (a resposta do cliente é aleatória também, mas é ruído do fenômeno, não um mecanismo de exploração). Duas regras preservam a validade da identificação: $\epsilon_t$ nunca pode depender do cliente da rodada (só do tempo/configuração vigente), e $Z_t$ precisa ser sorteado a cada rodada, nunca fixado por cliente.

### 22.5 A distribuição do tamanho do pacote sorteado pode ser customizada

Confirmado que a distribuição de $|S|$ na rota aleatória não precisa ser uniforme — pode priorizar tamanhos centrais ou menores por demanda de negócio, desde que **não dependa do perfil do cliente**. Recomendação: evitar truncar uma Normal contínua (empilha massa artificial nos extremos por causa do corte); preferir uma distribuição definida diretamente sobre os inteiros $\{1,\dots,14\}$ — uma binomial deslocada, ou um vetor de probabilidades customizado por tamanho, mais fácil de traduzir diretamente a partir de uma preferência de negócio.

### 22.6 Misturar dados de rodadas aleatórias e do modelo no retreino não é um problema

O `refit()` faz regressão supervisionada direta sobre $(x,y)$ observados — não é avaliação de política (*off-policy*), que exigiria ponderação por propensão. Sob a suposição usual de não-confundimento (tudo que afeta simultaneamente a escolha do pacote e a resposta do cliente está capturado em $x$), misturar as duas origens no treino é válido sem correção adicional — é, aliás, como LinUCB/C²UCB são treinados na prática, já que a política do próprio sistema muda ao longo do tempo. Um efeito sutil, mas não um viés: observações vindas de rodadas guiadas pelo modelo tendem a ter $p_n$ mais extremo (o modelo já estava confiante), contribuindo menos para reduzir $H$ (mesmo efeito kappa, visto de outro ângulo) — reforçando, mais uma vez, a utilidade informacional da fase aleatória.

### 22.7 $\epsilon$ pode ser ajustado manualmente por estágio do projeto

Nenhuma exigência matemática impõe uma fórmula fechada de decaimento — $\epsilon_t$ pode ser lido de um valor de configuração ajustado manualmente pela equipe (versionado, com data e motivo da mudança registrados), com base em monitoramento real (o bônus está de fato encolhendo nos itens/pares mais raros?) em vez de uma curva calculada sem dados reais. As duas regras da Seção 22.4 continuam valendo, independente de quem decide o valor.

### 22.8 Arquitetura real: recomendação via API em tempo real + retreino em lote, dois fluxos que se realimentam

Confirmado o desenho de produção: **Fluxo 1** (tempo real, cliente a cliente) — a API sorteia $Z_t$ a cada chamada (usando o $\epsilon$ vigente lido da configuração, não um contador de "rodada $t$", o que evita problemas de concorrência entre múltiplas instâncias), gera a recomendação, e grava a decisão final do cliente numa tabela de histórico. **Fluxo 2** (batch, periódico) — o job de retreino lê essa tabela, roda `refit()`, e publica os novos parâmetros, que o Fluxo 1 passa a consumir. Os dois fluxos se realimentam exatamente como descrito na Seção 21, só que agora com o Fluxo 1 acionado por requisição HTTP em vez de loop. Ponto de atenção de engenharia: garantir que a gravação da decisão do cliente (que tipicamente é uma chamada/evento separado da geração da recomendação nessa arquitetura distribuída) esteja de fato implementada e confiável — sem ela, o retreino para de receber dados novos silenciosamente.

### 22.9 Correções de consistência nos documentos didáticos

Ao revisar os três documentos de pipeline para acompanhamento sequencial com a equipe, foram corrigidas referências prematuras do tipo "Etapa $X$ da rodada anterior" nas primeiras etapas de cada documento — problemático porque, com $\epsilon_t$ alto no início, a primeira vez que a Etapa 3 (ou equivalente) roda pode não ser a "rodada 1" literal, já que rodadas aleatórias tendem a vir antes. A frase corrigida deixa claro que $\hat\theta$/$H_\theta$ (e $\hat\phi$/$H_\phi$) refletem todas as atualizações já feitas até aquele ponto — de origem aleatória ou via modelo — e só estão genuinamente zerados na primeiríssima execução daquela etapa, não necessariamente na "rodada 1". Também foi corrigida uma frase confusa na Seção 3 do documento do C²UCB original, que dava a entender que $h(S,X)$ já tinha sido calculada para configurações que só o oráculo (Etapa 4) viria a testar depois.

---

## O algoritmo consolidado, em uma frase

Um C²UCB contextual com dois parâmetros compartilhados aprendidos via **regressão logística regularizada em lote** — um para relevância individual **com fadiga** embutida no contexto (tamanho do pacote parcial), outro para **sinergia/canibalização entre pares** de coberturas — cuja incerteza é estimada por aproximação de Laplace **com piso de curvatura** (para não penalizar itens já bem estabelecidos), decidindo o pacote via um **oráculo guloso sem cardinalidade fixa** que para quando o ganho marginal centralizado em 0,5 deixa de ser positivo, **sem nenhum termo de custo/prêmio**, alimentado por uma fase inicial de **amostragem aleatória de tamanhos variados** que funciona como experimento identificador dos efeitos de fadiga, sinergia e canibalização sem viés.

---

## Tabela-mestra: problema → decisão, do início ao fim

| # | Problema levantado | Decisão / solução |
|---|---|---|
| 2 | Top-k UCB sem prova formal para conjuntos | C²UCB generaliza via oráculo + monotonicidade/Lipschitz |
| 3 | Sem histórico para features tipo PMF | Features manuais de domínio |
| 4 | DC²B usa DPP (só repulsão) | Descartado — incompatível com sinergia positiva |
| 6 | Recompensa monótona sem $k$ tende a incluir tudo | Precisa de termo que zere o ganho marginal |
| 7 | Prêmio como freio | Rejeitado — freio deve vir do comportamento |
| 8 | Freio via fadiga aprendida | Contexto inclui $\lvert S_{parcial}\rvert$ |
| 9 | Rejeição/não-adição sem prejuízo genuíno | Centralização da recompensa em 0,5 |
| 11 | Ranking individual não capta sinergia/canibalização | Termo de par $\hat w(i,j)$ com $\phi$ compartilhado |
| 13 | Penalizar tamanho na amostragem aleatória? | Não — contaminaria a identificação do efeito |
| 14 | Score fora de $[0,1]$ com ridge regression | Regressão logística, bônus no espaço linear + sigmoid |
| 15 | "Online" parecia mais custoso | Atualização em lote com reajuste completo do histórico |
| 16 | Só ID é suficiente para caracterizar cobertura? | Não — combinar features semânticas + ID |
| 18 | Efeito kappa: itens "quase certos" não reduzem incerteza | Identificado e medido empiricamente |
| 19 | Qual correção adotar? | Piso de curvatura $\tau$ (preserva a geometria contextual) |
| 21 | Como implantar o retreino em produção? | Dois processos separados: treino agendado (lote diário) publicando parâmetros versionados; serviço em tempo real com hot-reload periódico |
| 22 | O bônus de exploração dispensa a fase aleatória? Isso não perde interpretação de probabilidade? | Não dispensa (bônus é determinístico, não gera variação de tamanho); $p_{UCB}$ é o supremo sobre o conjunto de confiança, matematicamente bem definido (GLM-UCB) |
