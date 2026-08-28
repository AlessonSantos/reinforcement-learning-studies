# Síntese final: da heurística original ao sistema de recomendação de coberturas
### A jornada completa — decisões, descartes, e a arquitetura final (v4)

---

## Como ler este documento

Este não é um registro cronológico de cada mensagem trocada — é uma reconstrução organizada por tema, cobrindo o raciocínio que efetivamente sobreviveu e chegou à versão final em produção. Cada seção termina numa decisão; onde uma decisão foi revertida depois, isso está registrado explicitamente, com o motivo.

**A arquitetura final está documentada em detalhe, passo a passo, em `pipeline_c2ucb_seguro_didatico_v5.md`** — este documento é o mapa de como chegamos até ali, não uma repetição do pipeline em si.

---

## 1. O ponto de partida: por que não usar a heurística original do time

A ideia inicial do time era simples: ordenar as coberturas por uma nota de relevância e ofertar as top-$k$. O problema identificado logo no início: um $k$ fixo não tem como se adaptar ao cliente — um cliente que genuinamente teria propensão para 8 coberturas recebe só as top-$k$ definidas de antemão, e um cliente com propensão real para só 2 recebe as mesmas $k$, incluindo itens de baixa propensão só para preencher a cota.

Isso motivou a busca por uma alternativa com tamanho de pacote **emergente**, não fixo — o que levou à literatura de bandits contextuais.

---

## 2. Por que C²UCB, e o que ele resolvia (na época)

Comparamos UCB não-contextual, LinUCB, C²UCB e DC²B. A escolha inicial foi o **C²UCB** (Qin, Chen & Zhu, 2014), porque ele generaliza o UCB para escolher um **super-braço** (um conjunto de itens) via um oráculo, em vez de um braço só — encaixando naturalmente no problema de "montar um pacote", em vez de "recomendar um item".

A primeira versão do sistema replicava a estrutura do paper: recompensa linear + termo de diversidade fixo (entropy regularizer), ajuste por ridge regression, oráculo guloso com cardinalidade $k$ fixa.

### 2.1 A descoberta tardia sobre o paper original — e por que não muda nada da decisão

Bem mais adiante na conversa, ao ler o PDF completo do paper (Qin, Chen & Zhu, 2014), descobrimos que a estrutura original é **diferente** do que tínhamos assumido: no paper, $\theta_{\ast}$ é a **preferência de um único usuário específico**, refinada ao longo de rodadas repetidas **com essa mesma pessoa** (o experimento é sobre recomendar filmes repetidamente ao mesmo usuário). Os atributos que são fixos e conhecidos são os dos **itens** (filmes, via fatoração de matriz).

Isso é o oposto estrutural do que construímos: um $\theta$ **compartilhado entre clientes**, aprendendo com a população inteira, não uma preferência pessoal refinada ao longo de várias interações com a mesma pessoa. A adaptação foi deliberada e permanece justificada — no nosso caso, um cliente tipicamente não retorna dezenas de vezes para o sistema aprender só sobre ele; o valor está em generalizar entre clientes diferentes.

---

## 3. Por que a recompensa deixou de ser linear (ridge para regressão logística)

A recompensa real do problema é binária (o cliente aceita ou não uma cobertura) — não uma métrica contínua com ruído gaussiano, que é o que a ridge regression assume. Consequência prática: $\hat r(i)$ podia sair fora do intervalo $[0,1]$, sem nenhuma garantia.

**Decisão:** trocar para regressão logística — $p_n\sim\text{Bernoulli}(\sigma(\theta^Tx_n))$ — garantindo que a saída sempre fique entre 0 e 1 por construção do funil logístico.

### 3.1 A defesa formal de "somar o bônus antes do funil"

Essa mudança gerou uma dúvida legítima, revisitada em detalhe: será que somar um termo de incerteza no espaço linear e só depois aplicar o sigmoide não distorce a interpretação da probabilidade resultante?

**A resposta, com rigor:** não — porque o sigmoide é monótono, o supremo de uma probabilidade sobre um conjunto de confiança comuta exatamente com essa ordem de operações. Essa defesa completa está documentada em `defesa_metodologica_sigmoide_incerteza.md`.

---

## 4. A fase com termo de par ($\phi$) — construída, corrigida, e depois retirada por faseamento

Nas fases intermediárias, o sistema incluía um segundo modelo compartilhado, $\phi$, capturando sinergia/canibalização entre pares de coberturas — permitindo que o pacote refletisse não só o valor de cada item isolado, mas também como eles interagem.

### 4.1 Os bugs reais que $\phi$ carregava, encontrados na revisão

Numa revisão com um matemático do time, dois problemas concretos foram identificados:

1. **Assimetria de ordem**: um offset — $\eta(i)+\eta(j)$, usado para eliminar a redundância entre $y_i$ e $y_{ij}$ — tinha vazado da etapa de treino para a etapa de inferência, fazendo $p_{par}(i,j)$ depender de qual item era avaliado primeiro no processo guloso.
2. **$z(i,j)\neq z(j,i)$**: a concatenação de atributos dentro do vetor de contexto do par não era simétrica por construção.

Ambos tinham correção conhecida (offset restrito só ao treino; ordenação canônica do par antes de montar $z$), mas a discussão expôs quanto de complexidade extra o termo de par trazia.

### 4.2 A decisão de faseamento

Diante da complexidade acumulada (offset, ordenação, threshold de par, risco de confundir ruído de amostra pequena com canibalização real), a decisão foi **lançar sem $\phi$** — uma v1 mais simples, focada só em $\theta$, com $\phi$ reservado para uma fase futura, depois de a v1 validar em produção.

A fase aleatória continua coletando dados de pares mesmo sem $\phi$ decidir nada com eles — então, se $\phi$ for reintroduzido depois, o histórico não precisa ser reconstruído.

---

## 5. A remoção da fadiga

Pelo mesmo espírito de simplificação, o termo de fadiga (dependência de $|S_{parcial}|$ no contexto do item) também foi removido nesta fase — sem $\phi$ decidindo em conjunto, cada cobertura passou a ser decidida de forma totalmente independente das demais, eliminando a necessidade de qualquer processo sequencial ou de ponto fixo para lidar com o tamanho do pacote parcial.

**Consequência aceita conscientemente:** não existe mais nenhum mecanismo — nem fraco, nem forte — freando o crescimento do pacote por sobrecarga de escolha. Isso é uma decisão de faseamento, revisável depois, não uma alegação de que fadiga não importa.

---

## 6. O abandono do bônus de exploração (UCB) em favor de um teste de significância

Esse foi o pivô mais estrutural de toda a jornada. A pergunta que motivou: por que somar recompensa e incerteza dentro do sigmoide, se a decisão de ofertar deveria ser comparável a uma taxa de conversão observada (o baseline da rota aleatória)?

### 6.1 O problema com o bônus para essa finalidade

$p_{UCB}(i)=\sigma(\eta(i)+\text{bônus}(i))$ é, por desenho, **deliberadamente inflado** para fins de exploração — não é uma probabilidade calibrada. Comparar um valor inflado contra uma taxa de conversão bruta mistura duas naturezas de número diferentes.

### 6.2 A solução: decisão determinística, exploração só via $Z_t$

**Decisão final:** tirar o bônus da decisão. A oferta passa a comparar $\hat p(i)=\sigma(\eta(i))$ — sem nenhuma inflação — contra a taxa de referência da cobertura. Toda a exploração migra para ser responsabilidade exclusiva de $Z_t$ (a rota aleatória), que já era necessária por outros motivos (identificação de fadiga, cobertura combinatória de pares) e continua sendo a única fonte de aleatoriedade genuína do sistema.

Essa mudança também simplificou a comparação entre UCB e Thompson Sampling: sem bônus, o sistema deixou de ser "UCB" no sentido estrito — virou a família **epsilon-greedy** (decisão determinística + exploração externa via $Z_t$), mais simples e mais auditável que qualquer uma das duas alternativas de bandit clássicas.

---

## 7. Como decidir sem bônus: do threshold fixo ao teste de significância

### 7.1 Por que 0,5 fixo é arbitrário

Um threshold único pressupõe implicitamente que aceitar e rejeitar uma oferta têm o mesmo peso de decisão para toda cobertura — sem nenhuma razão para ser verdade.

### 7.2 A evolução: baseline, depois Wilson, depois teste de significância completo

1. Primeira correção: threshold = a taxa de aceitação observada da cobertura na rota aleatória ($b_i$), em vez de 0,5 fixo.
2. Refinamento: usar o **limite superior do intervalo de Wilson** de $b_i$ como threshold, protegendo contra decidir com base num baseline ainda pouco amostrado.
3. Versão final: abandonar a comparação pontual em favor de um **teste de significância formal**, combinando a incerteza dos dois lados — a de $\eta(i)$ (via $H_\theta$) e a do baseline (via método delta).

### 7.3 A fórmula final, já com a margem de negócio

$D(i)$ é, por construção, o logaritmo de uma razão de chances entre a propensão estimada e a taxa de referência — pela propriedade $\ln(a)-\ln(b)=\ln(a/b)$, já que $\eta(i)$ e $\text{logit}(b_i)$ são ambos log-odds:

$$D(i) = \eta(i) - \text{logit}(b_i) = \ln\left(\frac{\text{odds}(i)}{\text{odds}(b_i)}\right), \qquad \text{Var}[D(i)] = \text{Var}[\eta(i)] + \text{Var}[\text{logit}(b_i)]$$

$$\text{Oferta} \iff D(i) - z\sqrt{\text{Var}[D(i)]} > \ln(R)$$

onde $R$ é a razão de chances mínima exigida pelo negócio, além da significância estatística — com $R=1$ — nenhuma margem extra, já que $\ln(1)=0$ —, a regra volta a exigir só que exista alguma vantagem estatisticamente significativa. Essa é a Etapa 5 do pipeline final.

### 7.4 Por que somar as duas variâncias, e não checar sobreposição de intervalos

Checar se dois intervalos de confiança separados se sobrepõem é **mais conservador** do que um teste de diferença apropriado — um resultado documentado na literatura (Schenker & Gentleman, 2001): usar dois intervalos de 95% e checar sobreposição corresponde, na prática, a um teste com confiança próxima de 99%, não 95%.

### 7.5 A margem de negócio: por que razão de chances, e não lift de probabilidade

Uma vez com o teste de significância implementado, surgiu uma pergunta legítima: "maior que o baseline" garante confiabilidade estatística, mas não necessariamente uma recomendação que maximize a qualidade prática (precisão/revocação) da oferta. A tentação natural é exigir uma margem — um "lift" mínimo sobre a probabilidade.

**Por que lift multiplicativo de probabilidade não funciona.** Ele deixa de ser factível conforme o baseline sobe: um baseline de 60% com lift de 2× exigiria 120%, impossível. Essa restrição fica mais severa quanto maior o baseline — o oposto do que se quer numa régua de negócio genérica.

**A solução: razão de chances, não lift de probabilidade.** Como $D(i)$ já vive em log-odds, ele já é o logaritmo de uma razão de chances — não é preciso nenhuma estrutura nova. Substituindo o ponto de corte de $D(i)$ de $0$ (razão de chances de 1, "nenhuma vantagem") por $\ln(R)$ (razão de chances $R$, "vantagem relevante"), a mesma fórmula generaliza sem quebrar em nenhuma faixa de baseline — chances não têm teto superior, diferente de probabilidade.

**A derivação, em cinco passos:** (1) $D(i)$ é o log de uma razão de chances, por construção; (2) o teste original testa $H_0: \text{OR}(i)\le1$; (3) generaliza-se para $H_0: \text{OR}(i)\le R$; (4) aplicando log nos dois lados, a hipótese vira $D(i)\le\ln(R)$; (5) a estatística de teste desloca o ponto de corte de $0$ para $\ln(R)$, sem alterar $D(i)$ nem $\text{Var}[D(i)]$.

**Uma propriedade favorável, descoberta ao testar numericamente:** para $R$ fixo, a exigência em pontos percentuais de probabilidade cresce, atinge um pico na faixa de 40–50% de baseline, e depois diminui — o funil logístico "achata" nas duas pontas automaticamente. Isso resolve, sem nenhuma regra especial, a preocupação de que uma cobertura com baseline já alto (ex.: 70%) não deveria precisar de uma exigência tão rigorosa em pontos percentuais quanto uma de baseline baixo.

**Decisão final:** o teste de significância ganha um parâmetro de negócio $R$ (a razão de chances mínima), com $R=1$ como comportamento padrão equivalente ao teste original — uma generalização, não uma substituição.

---

## 8. A reserva de dados para o baseline, e a exogeneidade que ela protege

### 8.1 Por que o baseline precisa de uma fatia própria

Se as mesmas observações alimentassem tanto o treino de $\theta$ quanto o cálculo do baseline, os dois deixariam de ser estatisticamente independentes. **Decisão:** uma fração $\rho$ das rodadas aleatórias é reservada exclusivamente para o baseline, decidida por uma regra independente do resultado observado.

### 8.2 Adições espontâneas nunca contam para o baseline

Mesmo dentro da fatia reservada, só contam para $b_i$ as observações em que a cobertura foi **genuinamente sorteada** para a oferta — nunca uma adição espontânea do cliente. O motivo: $b_i$ precisa representar aceitação sob exposição exógena; uma adição espontânea é uma escolha do cliente, o oposto de exógena, e contá-la infla $b_i$ artificialmente.

### 8.3 A divisão pode ser refeita a cada ciclo

Como o retreino sempre reprocessa o histórico completo, a divisão entre "dados para $\theta$" e "dados reservados para o baseline" pode ser redesenhada a cada ciclo — não precisa ser uma partição fixa no tempo.

---

## 9. $\epsilon$: de cronograma de decaimento a valor fixo

**Decisão inicial:** $\epsilon_t$ decrescente ao longo do tempo, nunca chegando a zero.

**Decisão final:** $\epsilon$ **fixo**, definido pela equipe, sem cronograma automático — mais simples de operar, mantendo as duas exigências que continuam valendo: nunca depender do cliente da rodada, e nunca chegar a zero.

A notação foi ajustada de $\epsilon_t$ para $\epsilon$ (sem índice de tempo), e $Z_t$ manteve seu índice — não porque o parâmetro varie, mas porque é uma variável aleatória nova, sorteada a cada rodada.

**A rota aleatória sorteia um subconjunto direto** entre as 10 coberturas opcionais — não cobertura por cobertura, de forma independente (uma proposta intermediária que chegou a ser considerada e depois descartada, para manter a mecânica mais simples).

---

## 10. Obrigatórias: dentro ou fora do treino — a reviravolta

Esse ponto teve uma correção completa de rumo, valendo registrar com precisão.

**Primeira interpretação:** "obrigatória" significava só que a lei exige a **oferta**, não que o cliente seja impedido de recusar — levando à conclusão de que obrigatórias deveriam entrar no treino de $\theta$ (há variação real a aprender).

**Correção posterior, definitiva:** obrigatória significa $y_i=1$ sempre, sem exceção — não há decisão do cliente envolvida. **Consequência final:** obrigatórias ficam inteiramente fora do treino de $\theta$ — não geram nenhuma observação, evitando ruído e risco de quase-separação numérica em uma variável sem nenhuma variação.

---

## 11. O efeito kappa e o piso de curvatura — a discussão mais extensa da jornada

### 11.1 O achado original

A matriz $H_\theta$ (aproximação de Laplace da incerteza de $\theta$) usa um peso $p_n(1-p_n)$ por observação — máximo em casos "difíceis" ($p\approx0{,}5$), quase zero em casos "quase certos". Consequência medida empiricamente: a incerteza de itens quase certos **para de encolher** (ou até sobe temporariamente) mesmo com centenas de novas confirmações — não é um bug de implementação, é uma propriedade genuína da regressão logística.

### 11.2 Por que isso importa especificamente para o teste de significância

Sem correção, $\text{Var}[\eta(i)]$ trava, e o teste da Etapa 5 nunca ganha confiança suficiente para confirmar uma diferença, mesmo com evidência esmagadora.

### 11.3 As três formulações comparadas, com testes numéricos

$$\text{MAX (original)}: H_\theta = \lambda I + \sum_n \max(p_n(1-p_n),\nu)x_nx_n^T$$

$$\text{SOMA direcional}: H_\theta = \lambda I + \sum_n (p_n(1-p_n)+\nu)x_nx_n^T$$

$$\text{Isotrópica (proposta alternativa)}: H_\theta = \lambda I + \sum_n \big(p_n(1-p_n)x_nx_n^T + \nu I\big)$$

**MAX e SOMA são numericamente quase idênticas** em todo o espectro de $n$ testado — a diferença é só suavidade matemática (SOMA não tem a quina do $\max$), sem efeito prático em nenhum cenário simulado.

**A isotrópica resolve o platô, mas ao custo de vazar informação para direções sem nenhum dado real** — testado explicitamente: uma cobertura nunca observada teve sua incerteza reduzida quase ao mesmo nível de uma cobertura testada 300 vezes, só por causa do volume total de observações do sistema, não por nenhuma relação genuína com a cobertura testada.

### 11.4 O argumento matemático central, sem depender de nenhum exemplo numérico

A informação de Fisher de uma observação é zero, por definição, em qualquer direção onde $x_n$ tem valor zero. $\nu\cdot x_nx_n^T$ nunca viola essa propriedade — só amplifica a informação genuína já presente. $\nu\cdot I$ a viola diretamente, afirmando informação em direções que a própria observação, matematicamente, não tocou.

### 11.5 O dilema que fecha a discussão

$\varepsilon$ não pode, ao mesmo tempo, ser pequeno o suficiente para não vazar **e** grande o suficiente para resolver o platô — porque é o mesmo número, aplicado com a mesma magnitude, em toda direção, por construção da matriz identidade. $\tau$ escapa desse dilema porque multiplica $x_nx_n^T$: pode crescer o quanto for necessário só na direção que precisa, sem nenhuma obrigação de crescer nas demais.

**Decisão final:** manter o piso de curvatura multiplicando $x_nx_n^T$ (a forma MAX, já implementada e testada — a versão SOMA é uma alternativa matematicamente equivalente, disponível se a suavidade vier a importar).

---

## 12. A generalização entre coberturas — o que ela realmente garante

Uma sequência de perguntas técnicas aprofundou exatamente como $\theta$ e $H_\theta$ compartilhados generalizam entre coberturas:

- $H_\theta$ é **uma matriz única** para o sistema inteiro — o que muda por cobertura é só o vetor de consulta $x(i)$, nunca a matriz em si.
- A generalização entre coberturas acontece **na proporção da correlação real** entre seus atributos — nunca de forma uniforme ou garantida.
- O termo de interação ($c\otimes f_i$) permite que o efeito de uma variável do cliente varie por cobertura, evitando que $\theta_c$ precise ser um "compromisso" entre relações opostas em coberturas diferentes.
- Esse mesmo termo de interação também permite capturar, de forma indireta, fatores de confusão não observados (ex.: risco climático regional), na medida em que se correlacionam com o perfil de cliente disponível — e essa captura é válida especificamente porque a rota aleatória garante que a correlação não é um artefato da própria política de recomendação.

---

## 13. O risco de colisão de atributos ($f_i$) — identificado e resolvido

Como $f_i$ é construído só de atributos semânticos (sem nenhum identificador único), duas coberturas com atributos categóricos idênticos gerariam o **mesmo** $x(i)$ para o mesmo cliente — o modelo não teria como distingui-las, aprendendo uma média de comportamento entre as duas.

**Duas soluções possíveis, comparadas:**
1. Incluir um ID (one-hot) em $x(i)$ — garantia estrutural, ao custo de uma cobertura nova perder parte da vantagem de "pontuação imediata sem período de aquecimento".
2. Checagem automática de duplicidade no cadastro do catálogo — garantia de processo, não estrutural, mas sem custo nenhum sobre a extensibilidade já documentada.

**Decisão final:** checagem automática, dado o catálogo pequeno (14 coberturas) e sob controle direto da equipe — preserva integralmente a vantagem de extensibilidade que já tínhamos documentado.

---

## 14. LinUCB: disjoint vs. hybrid — de onde vem a comparação de arquitetura

Ao comparar a arquitetura de vocês com o LinUCB clássico, veio à tona que o LinUCB "padrão" (mais citado) é **disjoint** — uma regressão por braço, sem nenhum compartilhamento — e que existe uma versão **hybrid**, que soma um componente compartilhado a um componente específico por braço.

O que construímos é o caso "puramente compartilhado" (nem citado como caso principal no paper original do LinUCB). A discussão sobre incluir um ID em $x(i)$ é, estruturalmente, reconstituir o **LinUCB hybrid** — não uma ideia nova, mas a formulação mais completa do próprio framework que inspirou essa linha de trabalho desde o início.

**Decisão final:** manter só compartilhado (sem ID), resolvendo o risco de colisão por checagem de processo, não por arquitetura — ver Seção 13.

---

## 15. Por que manter $\theta$ compartilhado, e não uma regressão por cobertura

Essa pergunta foi revisitada diretamente ao final da jornada, depois de todos os problemas encontrados no compartilhamento (colisão de atributos, vazamento via $H_\theta$, todo o debate $\tau$ vs. $\varepsilon$).

**A favor do disjunto (uma regressão por cobertura):** elimina de uma vez toda a superfície de bugs relacionados a compartilhamento — sem colisão de $f_i$, sem vazamento entre coberturas, sem necessidade de discutir $\tau$ vs. $\varepsilon$.

**A favor do compartilhado (decisão final):**
1. A complexidade que sobrou já foi entendida, testada e documentada — não é mais um risco desconhecido.
2. O catálogo é pequeno e estável — o benefício mais forte do disjunto (nenhuma cobertura fica "faminta" de dados) importa menos aqui do que importaria num catálogo grande ou volátil.
3. A parte mais original do trabalho — a interação entre compartilhamento, colisão de atributos e vazamento proporcional à correlação — é justamente o que diferencia esse sistema de uma aplicação direta e menos original da literatura, relevante para a intenção de publicar os achados.

---

## 16. A implementação final

**Primeira versão** — `treino_batch.py` (constrói $f_i$ a partir do catálogo via codificação one-hot própria, monta $x(i)$, ajusta $\theta$ via MAP com `scipy.optimize` escrito à mão, calcula $H_\theta$ com o piso de curvatura, calcula o baseline por cobertura opcional respeitando a reserva $\rho$ e a exclusão de adições espontâneas) e `escoragem_cliente.py` (escora um cliente por vez).

**Evolução para a arquitetura real de produção** — dado que as variáveis explicativas passaram a chegar já tratadas (dummies e numéricas prontas) via feature store, e que a escoragem passou a ser em lote, não cliente por cliente: `treino_batch_v2.py` deixa de fazer qualquer codificação categórica (só monta o termo de interação $c\otimes f_i$, de forma vetorizada) e ajusta $\theta$ via `sklearn.linear_model.LogisticRegression` (com $C=1/\lambda$ e `fit_intercept=False`, reproduzindo exatamente a mesma penalização do MAP) — $H_\theta$ continua calculada manualmente, já que o sklearn não a expõe. `escoragem_lote_v2.py` escora um DataFrame inteiro de uma vez (uma linha por combinação cliente×cobertura), reaproveitando `modelo_sklearn.decision_function` e `.predict_proba`, com $H_\theta^{-1}$ invertida uma única vez e reaproveitada via `einsum` para todas as linhas — e já incorpora o parâmetro de razão de chances mínima $R$ (Seção 7.5) na função de teste de significância.

---

## Tabela-mestra: pergunta e resposta final

| # | Pergunta | Resposta final (v4) |
|---|---|---|
| 1 | Por que não a heurística top-$k$ original? | $k$ fixo não se adapta ao cliente; tamanho de pacote precisa emergir |
| 2 | Por que C²UCB, e o que mudou depois? | Generaliza UCB para super-braços; hoje só a inspiração geral permanece — $\theta$ compartilhado é adaptação nossa, não fidelidade ao paper |
| 3 | Por que regressão logística, não ridge? | Recompensa é binária; ridge não garante saída em $[0,1]$ |
| 4 | Somar bônus antes do funil distorce a probabilidade? | Não — o funil é monótono, o supremo comuta com a composição |
| 5 | Onde está o termo de sinergia/canibalização ($\phi$)? | Retirado por faseamento (v1 mais simples); dados continuam sendo coletados para reintrodução futura |
| 6 | Onde está a fadiga? | Retirada junto com $\phi$, pela mesma lógica de faseamento |
| 7 | Por que não usar mais o bônus de exploração (UCB)? | Ele não é uma probabilidade calibrada, incomparável a uma taxa de conversão real |
| 8 | Como decidir a oferta, sem bônus? | Teste de significância com margem de negócio: $D(i)-z\sqrt{\text{Var}[D(i)]}>\ln(R)$ |
| 9 | Como o baseline é protegido de viés? | Fatia reservada $\rho$, decidida antes do resultado; adições espontâneas nunca contam |
| 10 | $\epsilon$ decai com o tempo? | Não — fixo, definido pela equipe, nunca chega a zero |
| 11 | Obrigatórias entram no treino? | Não — $y_i$ sempre 1, sem variação, sem informação |
| 12 | Qual piso de curvatura usar? | $\nu$ multiplicando $x_nx_n^T$ (MAX ou SOMA) — nunca $\nu\cdot I$ isotrópica |
| 13 | Coberturas colidem se tiverem os mesmos atributos? | Sim, tecnicamente — mitigado por checagem automática no cadastro, não por ID |
| 14 | $\theta$ deveria ser um por cobertura? | Não — compartilhado, decisão final, dado o custo já pago em entendimento e o catálogo pequeno |

---

## O algoritmo consolidado, em um parágrafo

Um único $\theta$, compartilhado entre as 10 coberturas opcionais, estimado por máxima verossimilhança a posteriori sobre uma verossimilhança de Bernoulli — nunca treinado com as 4 coberturas obrigatórias, que não têm variação para ensinar. A cada rodada, uma variável $Z_t\sim\text{Bernoulli}(\epsilon)$, com $\epsilon$ fixo definido pela equipe, decide entre uma rota aleatória (sorteando um subconjunto entre as opcionais) e a rota do modelo, que testa cada cobertura opcional de forma independente contra sua taxa de referência (calculada só sobre uma fatia reservada e exógena dos dados aleatórios), via um teste de significância que soma as incertezas dos dois lados e exige uma razão de chances mínima $R$ definida pelo negócio, não só a ausência de diferença. A incerteza de $\theta$ é aproximada via Laplace, com um piso de curvatura que garante a variância continuar encolhendo mesmo para itens quase certos — multiplicando sempre a direção específica de cada observação, nunca de forma isotrópica. O tamanho do pacote final emerge dessas decisões independentes, sem $k$ fixo e sem nenhum processo guloso.
