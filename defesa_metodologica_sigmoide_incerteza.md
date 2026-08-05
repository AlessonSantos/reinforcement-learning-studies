
# Defesa metodológica: por que "recompensa + incerteza dentro do sigmoide" é matematicamente correto
### Linguagem simplificada, rigor matemático mantido

---

## 1. Por que a recompensa não podia continuar sendo "linear"

A primeira versão do algoritmo usava uma fórmula (ridge regression) desenhada para prever **números contínuos com ruído** — como prever uma nota de 1 a 5 estrelas. Mas a nossa pergunta real é binária: **o cliente aceitou ou não aceitou a cobertura?** Sim ou não, 1 ou 0 — não existe "3,7 de aceitação".

**A matemática:** a fórmula original assume $y_n = \theta^Tx_n + \varepsilon_n$, com $\varepsilon_n$ sendo um ruído gaussiano. Sob essa suposição, nada impede que a saída $\hat r(i)$ ultrapasse 1 ou fique negativa. Se esse número é usado para decidir "inclua a cobertura se o valor passar de 0,5", um valor de 1,3 ou de −0,2 já não tem mais sentido probabilístico nenhum.

---

## 2. A correção: um funil que sempre devolve algo entre 0 e 1

A função logística ($\sigma$, "sigmoide") transforma qualquer número real num valor entre 0 e 1, não importa quão extremo seja o número de entrada:

$$p_n = \sigma(\theta^Tx_n) = \frac{1}{1+e^{-\theta^Tx_n}}$$

Essa troca resolve o problema da Seção 1: agora a saída **sempre** representa algo que pode ser lido como uma probabilidade, nunca escapando do intervalo entre 0 e 1.

---

## 3. A pergunta central: "somar incerteza e só depois passar pelo funil não estraga a probabilidade?"

Essa foi a objeção mais séria que recebemos, e merece ser respondida com cuidado.

### 3.1 O que a nossa probabilidade realmente quer dizer

$p_{UCB}(i)$ não é "a probabilidade real de aceitação". É o **maior valor de probabilidade que ainda é plausível**, dado tudo que já foi observado até agora — o teto do que ainda pode ser verdade, não uma previsão calibrada.

**A matemática:** existe uma região de valores plausíveis para $\theta$ — chamada de **conjunto de confiança** (o equivalente, para um vetor com várias dimensões, do que seria um intervalo de confiança para um único número; como $\theta$ tem várias componentes, a região de plausibilidade vira uma forma geométrica multidimensional, uma elipse, em vez de um segmento de reta):

$$E = \{\theta : (\theta-\hat\theta)^TH(\theta-\hat\theta) \le \beta\}$$

$p_{UCB}(i)$ é o maior valor de $\sigma(\theta^Tx(i))$ dentro dessa elipse:

$$p_{UCB}(i) = \sup_{\theta\in E}\ \sigma(\theta^Tx(i))$$

### 3.2 Por que "bônus primeiro, funil depois" é a única ordem certa

A função sigmoide é **monótona crescente** — número maior na entrada sempre gera número maior (ou igual) na saída, sem nunca inverter a ordem. Isso significa que achar "o valor mais otimista antes do funil, depois converter" dá **exatamente o mesmo resultado** que "converter tudo para probabilidade primeiro, e depois achar o valor mais otimista".

**A matemática:** como $\sigma$ é monótona crescente, o supremo comuta com a composição:

$$\sup_{\theta\in E}\ \sigma(\theta^Tx(i)) \;=\; \sigma\Big(\sup_{\theta\in E}\ \theta^Tx(i)\Big)$$

O lado direito é exatamente o que o código faz: soma o bônus no espaço linear e só depois aplica $\sigma$. Isso não é uma conveniência — é a única forma matematicamente correta de calcular "o teto otimista de uma probabilidade". Se a ordem fosse invertida (aplicar o funil primeiro, somar o bônus depois na escala de probabilidade), a conta perderia sentido — poderia facilmente passar de 1.

### 3.3 Onde a crítica teria razão — e por que, no nosso caso, ela não se aplica

A crítica procederia se o bônus de incerteza fosse emprestado do modelo antigo (calibrado assumindo ruído gaussiano) e simplesmente colado no sigmoide, sem recalcular nada.

**Não foi isso que fizemos.** A matriz de incerteza foi reconstruída inteiramente a partir da curvatura da **própria verossimilhança de Bernoulli**:

$$H = \lambda I + \sum_n p_n(1-p_n)\,x_nx_n^T$$

Essa construção tem nome na literatura acadêmica — **GLM-UCB** (Filippi, Cappé, Garivier & Szepesvári, 2010) — e é uma extensão revisada por pares do UCB para esse tipo de modelo, não uma improvisação nossa.

*(Ao avaliar se valeria usar uma biblioteca pronta em vez da nossa implementação, confirmamos que a lógica interna dela é a mesma ideia — intervalo de confiança a partir da variância dos coeficientes ajustados. Isso reforça que a abordagem é reconhecida; descartamos a biblioteca por outros motivos, ligados à falta de compartilhamento de informação entre coberturas.)*

---

## 4. Os dois problemas reais que essa construção trouxe (e como resolvemos)

### 4.1 Problema 1 — itens "quase certos" param de ficar mais confiantes

Para uma cobertura quase sempre aceita (ex.: Incêndio), o quanto cada nova observação reduzia a incerteza ficava cada vez menor — a ponto de praticamente **parar de diminuir**, mesmo depois de centenas de confirmações repetidas.

**A matemática:** o peso $p_n(1-p_n)$ usado para construir $H$ é **máximo** quando $p_n\approx 0{,}5$ e **quase zero** quando $p_n$ está perto de 0 ou 1. Medimos isso na prática: o bônus de exploração estacionava por volta de 0,50 mesmo após 300 observações idênticas, em vez de continuar caindo.

**A solução — um piso mínimo:**

$$H = \lambda I + \sum_n \max\big(p_n(1-p_n),\ \tau\big)\,x_nx_n^T$$

O piso $\tau$ garante que a incerteza sempre continue diminuindo, sem travar num platô artificial. Depois da correção, o mesmo teste mostrou o bônus caindo de 0,50 para 0,25 após as mesmas 300 observações.

### 4.2 Problema 2 — a matriz de incerteza é uma aproximação, não é exata

A matriz $H$ é construída via **aproximação de Laplace** — fica cada vez mais precisa conforme mais dados chegam, mas não tem a mesma garantia exata que o método usado no modelo linear-gaussiano.

**A matemática:** essa aproximação se torna precisa conforme $N\to\infty$ (teorema de Bernstein–von Mises), mas não é uma garantia exata para poucas observações. É uma limitação conhecida e documentada na literatura — trabalhos posteriores (Faury et al., 2020) propuseram construções mais rigorosas, à custa de uma matemática consideravelmente mais complexa.

**Como tratamos isso:** não existe uma correção definitiva aqui — é uma limitação aceita conscientemente em troca de uma implementação mais simples. O piso de curvatura (4.1) resolve o sintoma mais visível, mas a natureza aproximada dessa matriz continua sendo uma característica do método, não um bug a ser eliminado.

---

## 5. O que a probabilidade final realmente significa, com a incerteza embutida

Antes de entrar nos dois regimes abaixo, vale situar onde $p_{UCB}(i)$ é de fato usado: ele alimenta o **oráculo guloso**, o componente que decide quais coberturas entram no pacote. O oráculo guloso constrói o pacote item por item, comparando o $p_{UCB}$ de cada candidato e adicionando o de maior valor a cada passo, até que nenhum candidato restante compense mais ser adicionado. Ou seja: $p_{UCB}(i)$ nunca é reportado ou lido isoladamente — sua única função é servir de critério de comparação dentro dessa montagem passo a passo do pacote. Essa é a peça que dá sentido prático à distinção entre os dois regimes a seguir.

### 5.1 Quando a incerteza é pequena

$$p_{UCB}(i) \to \sigma(\hat\theta^Tx(i)) = \hat p(i)$$

$p_{UCB}(i)$ converge para $\hat p(i)$, a **estimativa calibrada** do modelo — a melhor crença sobre a propensão real, sem nenhuma inflação. Não é "a probabilidade mais otimista" — nesse ponto, não sobra otimismo para descrever; é simplesmente a melhor resposta do modelo.

### 5.2 Quando a incerteza é grande

$$p_{UCB}(i) = \sigma(\eta(i)+\text{bonus}(i)) \to 1$$

Aqui, $p_{UCB}(i)$ **deixa de significar uma probabilidade de verdade**. Não representa "a chance do item ser testado" nem nenhuma outra probabilidade real — é um número empurrado para perto do teto, cuja única função é vencer a comparação contra os outros candidatos dentro do oráculo guloso. A escolha de testar aquele item é uma **consequência** desse empurrão, não algo que o número em si representa.

### 5.3 Tabela-resumo

| Regime | O que $p_{UCB}(i)$ representa | O que **não** representa |
|---|---|---|
| Incerteza pequena (item bem conhecido) | A melhor estimativa calibrada do modelo, $\hat p(i)=\sigma(\hat\theta^Tx(i))$ | "A probabilidade mais otimista" (não sobra otimismo para descrever) |
| Incerteza grande (item pouco conhecido) | Um número empurrado para cima, só para vencer a comparação na decisão | Qualquer probabilidade real — o número perde sentido probabilístico aqui |

### 5.4 Qual número usar se alguém perguntar "qual é a probabilidade real?"

Se alguém pedir para reportar "a probabilidade real de aceitação", a resposta certa é **sempre** $\hat p(i)=\sigma(\hat\theta^Tx(i))$ — a estimativa **sem** o bônus de incerteza. $p_{UCB}(i)$ nunca deveria ser usado para isso; ele existe só para alimentar a decisão do algoritmo, não para descrever uma crença sobre o mundo real.

**Como provar isso na prática, não só na teoria:** o teste mais convincente é pegar dados novos (fora do treino) e conferir: entre os clientes para os quais o modelo previu $\hat p(i)\approx 0{,}7$, será que realmente cerca de 70% deles aceitaram a cobertura? Isso é um **diagrama de calibração**, e é o argumento mais forte para quem questiona se a probabilidade faz sentido — desloca o debate de "isso é elegante na matemática" para "isso prevê bem a realidade".

---

## 6. Um caso concreto: dois itens com a mesma probabilidade final, por motivos opostos

### 6.1 O exemplo numérico

$p_{UCB}(i)=\sigma(\eta(i)+\text{bonus}(i))$. Dois itens podem chegar à mesma soma com decomposições opostas:

| Item | Score "cru" $\eta$ | Bônus de incerteza | Soma | $p_{UCB}$ | Estimativa calibrada $\hat p=\sigma(\eta)$ |
|---|---|---|---|---|---|
| A | baixo (−1,0) | alto (+3,0) | 2,0 | $\approx 0{,}88$ | $\approx 0{,}27$ |
| B | alto (+1,9) | baixo (+0,1) | 2,0 | $\approx 0{,}88$ | $\approx 0{,}87$ |

Empatam em $p_{UCB}$, mas o modelo acredita coisas opostas sobre eles: para A, provavelmente **não** converte bem (0,27); para B, provavelmente **converte** bem (0,87).

### 6.2 Para a decisão de hoje, tanto faz

O ganho marginal ($p_{UCB}-0{,}5$) é idêntico para os dois — na hora de montar o pacote, o algoritmo os trata como equivalentes. Isso é proposital: é o coração do princípio "otimismo diante da incerteza" — dar a mesma chance a um item que já se sabe ser bom e a um item que ainda pode ser bom, mas que precisa ser testado para descobrir.

### 6.3 Mas os dois carregam histórias diferentes

- **Item A** (bônus alto, score cru baixo): não há motivo forte para achar que é bom, mas também não há dados suficientes para descartar — vale a pena testar. Um **candidato de exploração**.
- **Item B** (bônus baixo, score cru alto): há boa confiança de que é bom. Um **candidato de confirmação**.

### 6.4 O empate se desfaz na rodada seguinte

Se o Item A for testado, a observação ensina **muito** — sua incerteza ainda estava "aberta". Já o Item B, por já ser bem conhecido, ganha pouquíssima informação nova com mais uma observação (o mesmo efeito descrito na Seção 4.1). Depois de uma rodada, o empate tende a se desfazer.

### 6.5 A resposta direta

Duas probabilidades finais iguais **não** significam que os itens são igualmente bons nem igualmente arriscados — significam que, para esta decisão específica, o algoritmo está igualmente disposto a apostar nos dois, por motivos opostos: um porque acredita, o outro porque ainda não sabe e quer descobrir.

---

## 7. Tabela-síntese de toda a defesa

| Pergunta | Resposta |
|---|---|
| Por que trocar a recompensa linear por logística? | A recompensa é binária (aceitar/rejeitar), não contínua com ruído — a fórmula antiga não garantia um resultado entre 0 e 1 |
| Somar incerteza e só depois passar pelo funil perde sentido? | Não — o funil preserva a ordem dos números (é monótono), então fazer nessa ordem dá exatamente o mesmo resultado que "converter tudo em probabilidade primeiro" |
| Isso é um remendo emprestado do modelo antigo? | Não — a incerteza foi recalculada do zero, usando as regras certas para dados binários (GLM-UCB, Filippi et al., 2010) |
| Que problema essa construção trouxe? | Itens "quase certos" paravam de ficar mais confiantes (efeito kappa); e a matriz de incerteza é uma aproximação, não é exata |
| Como foi resolvido? | Um piso mínimo ($\tau$) que impede o platô artificial — validado com teste numérico |
| O que a probabilidade final representa, de fato? | O maior valor plausível dentro da região de confiança de $\theta$ — não a probabilidade real |
| Quando ela vira uma probabilidade real de verdade? | Quando a incerteza é pequena — aí converge para a estimativa calibrada $\hat p(i)$ |
| Quando ela deixa de significar uma probabilidade? | Quando a incerteza é grande — vira só um número empurrado para vencer a comparação na decisão |
| Dois itens empatados são igualmente bons? | Não necessariamente — podem ser um "candidato de confiança" e um "candidato de exploração", indistinguíveis hoje, mas com futuros diferentes |
