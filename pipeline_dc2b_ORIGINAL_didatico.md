# Pipeline do algoritmo DC²B ORIGINAL (Liu, Xiao, Wu, Miao & Zhang, 2019)
### Aplicado ao mesmo case do Cliente A — guia matemático e didático

---

## Nota importante antes de começar

Este documento aplica o DC²B **exatamente como descrito no paper original** — feedback implícito binário, diversidade via Processo Pontual Determinantal (DPP), estimação bayesiana com Thompson Sampling. Como já discutimos, o DPP é um mecanismo de **repulsão pura**: ele consegue representar canibalização/redundância, mas **não consegue representar sinergia positiva** — isso não é uma limitação da nossa aplicação ao case, é uma característica estrutural do próprio modelo. Vamos ver exatamente onde essa limitação aparece no fluxo.

| | DC²B original (este documento) | Versão customizada que construímos |
|---|---|---|
| Estimação | bayesiana, Thompson Sampling (amostra $\theta$ da posterior a cada rodada) | frequentista, UCB (usa o valor mais otimista — o melhor caso plausível — de $\theta$ dentro de um intervalo de confiança) |
| Diversidade | DPP — multiplicativa, só repulsão | termo de par aprendido — pode ser sinergia (+) ou canibalização (−) |
| Tamanho do pacote | $N$ fixo (top-$N$ do DPP) | emergente, sem $N$ fixo |
| Fadiga | não existe no paper original | aprendida via contexto de tamanho do pacote parcial |
| Feedback | binário implícito (clique/não clique), semi-bandit | binário, incluindo adições espontâneas |

**O exemplo que vamos seguir:** o mesmo "Cliente A", com as mesmas 5 coberturas ilustrativas, pacote de tamanho fixo $N=3$.

---

## 0. Setup — antes de existir qualquer cliente

**Analogia:** em vez de uma régua fixa que vai sendo calibrada aos poucos, imagine uma **distribuição de possíveis réguas** — um "leque" de versões plausíveis de $\theta$, que vai se estreitando conforme os dados chegam. A cada rodada, o sistema sorteia uma régua desse leque para tomar a decisão.

**A matemática:** prior gaussiano sobre $\theta$:

$$\theta \sim \mathcal{N}(\mu_0, \Sigma_0)$$

| Símbolo | Descrição |
|---|---|
| $\theta$ | parâmetro compartilhado, tratado como uma variável aleatória (não um número fixo a estimar) |
| $\mu_0, \Sigma_0$ | média e covariância iniciais do "leque" de réguas plausíveis — antes de qualquer dado, $\mu_0=\vec 0$, $\Sigma_0$ larga (muita incerteza) |

**O que entra / o que sai:** nada entra. Sai: $\mu_0, \Sigma_0$, prontos para a Rodada 1.

---

## 1. Uma nova rodada começa: o cliente chega

**Analogia:** o Cliente A entra na loja. Antes de qualquer coisa, você olha a ficha dele.

**A matemática:** observa-se $c\in\mathbb{R}^p$.

**O exemplo rodando:** $c=[0{,}8,\ 1{,}0]$.

**O que entra / o que sai:** entra: nada. Sai: $c$.

---

## 2. Thompson Sampling — sortear uma régua do leque

**Analogia:** em vez de calcular um "bônus de otimismo" como no UCB, o DC²B literalmente **sorteia** uma régua específica do leque de possibilidades — às vezes vai sortear uma régua mais otimista, às vezes mais conservadora, e decide o pacote com base nessa régua sorteada daquela rodada.

**A matemática:**

$$\tilde\theta \sim \mathcal{N}(\mu, \Sigma)$$

onde $\mu, \Sigma$ são a média e covariância **atuais** da posterior aproximada — na primeira rodada, iguais a $\mu_0, \Sigma_0$ do Setup (Etapa 0); a partir da segunda rodada em diante, atualizadas na Etapa 7 da rodada anterior.

| Símbolo | Descrição |
|---|---|
| $\tilde\theta$ | a régua sorteada nesta rodada específica — muda a cada rodada, mesmo com $\mu,\Sigma$ fixos |
| $\mu, \Sigma$ | o "leque" atual de réguas plausíveis, cada vez mais estreito conforme mais dados chegam |

**O exemplo rodando:** suponha que $\mu, \Sigma$ já refletem várias rodadas de aprendizado, e o sorteio desta rodada resulta em $\tilde\theta$ ligeiramente mais otimista que a média para coberturas de risco climático.

**O que entra / o que sai:** entra: $\mu, \Sigma$ — na primeira rodada, do Setup (Etapa 0); depois, da Etapa 7 anterior. Sai: $\tilde\theta$, usado na Etapa 3.

---

## 3. Score de qualidade por item (via link logístico)

**Analogia:** com a régua sorteada em mãos, calcula-se a "nota de qualidade" de cada cobertura para este cliente — sempre entre 0 e 1, porque passa pelo mesmo funil logístico que já vimos.

**A matemática:**

$$p_i = \sigma\big(\tilde\theta^{\,T}x(i)\big) = \frac{1}{1+e^{-\tilde\theta^Tx(i)}}$$

$$q_i = p_i^{\,\gamma}$$

| Símbolo | Descrição |
|---|---|
| $x(i)$ | contexto do item: cliente + atributos da cobertura (sem termo de tamanho de pacote) |
| $p_i$ | probabilidade de engajamento estimada, usando a régua **sorteada** (não a média) |
| $\gamma$ | expoente de ajuste — controla o quanto a "qualidade" domina sobre a "diversidade" na etapa seguinte |
| $q_i$ | qualidade ajustada — o ingrediente de relevância que entra na construção do kernel do DPP |

**O exemplo rodando:** suponha $p_{\text{Incêndio}}=0{,}80$, $p_{\text{Desastres}}=0{,}75$, $p_{\text{Conteúdo}}=0{,}60$, $p_{\text{Roubo}}=0{,}55$, $p_{\text{DanosElétricos}}=0{,}50$, com $\gamma=1$ (logo $q_i=p_i$).

**O que entra / o que sai:** entra: $\tilde\theta$ (Etapa 2), $x(i)$ de cada candidato. Sai: $q_i$ para cada item, usado na Etapa 4.

---

## 4. Construção do kernel do DPP — qualidade × similaridade

**Analogia:** imagine uma tabela onde cada célula $(i,j)$ representa "o quanto essas duas coberturas se parecem, ponderado pela qualidade de cada uma". Coberturas de alta qualidade mas muito parecidas entre si vão gerar uma tabela quase "redundante" — o DPP penaliza isso na próxima etapa.

**A matemática:**

$$L_{ij} = q_i \, q_j \, S_{ij}$$

onde $S_{ij}$ é uma medida de similaridade entre as coberturas $i$ e $j$ (ex.: similaridade de cosseno entre $f_i$ e $f_j$), com $S_{ii}=1$.

| Símbolo | Descrição |
|---|---|
| $L$ | matriz-kernel do DPP — combina qualidade individual com similaridade entre pares |
| $S_{ij}$ | similaridade estrutural entre as coberturas — **fixa**, calculada só a partir das features estáticas (igual ao regularizador do C²UCB original, não aprendida do comportamento do cliente) |
| $L_{ij}$ | alto quando as duas coberturas são boas **e** parecidas — é justamente essa combinação que o DPP penaliza na hora de montar o conjunto, para não repetir "o mesmo tipo de cobertura" duas vezes |

**O exemplo rodando:** suponha $S_{\text{Incêndio},\text{Desastres}}=0{,}3$ (riscos diferentes, pouca sobreposição), $S_{\text{Desastres},\text{DanosElétricos}}=0{,}8$ (sobreposição física real, igual ao exemplo de canibalização que já discutimos). Então:

$$L_{\text{Incêndio},\text{Desastres}} = 0{,}80\times0{,}75\times0{,}3 = 0{,}18$$
$$L_{\text{Desastres},\text{DanosElétricos}} = 0{,}75\times0{,}50\times0{,}8 = 0{,}30$$

Note que o par (Desastres, Danos Elétricos) tem $L$ mais alto — mas isso **não** significa "melhor par": no DPP, alto $S_{ij}$ **reduz** a probabilidade de escolher os dois juntos (é repulsão), como veremos na Etapa 5.

**O que entra / o que sai:** entra: $q_i$ (Etapa 3), $S_{ij}$ (do catálogo). Sai: a matriz $L$ completa, usada na Etapa 5.

---

## 5. O oráculo: MAP guloso para DPP — cardinalidade fixa $N$

**O que é um "oráculo" nesse contexto:** aqui também é o componente que traduz os scores estimados (agora via Thompson Sampling, Etapas 2-3, combinados na matriz $L$ da Etapa 4) numa decisão concreta de pacote. A diferença central em relação aos outros dois documentos: o objetivo que o oráculo maximiza não é mais uma soma de scores — é o **determinante** de uma submatriz de $L$, uma quantidade geométrica (o "volume" ocupado pelos itens escolhidos no espaço de features).

**Por que ainda é "guloso":** encontrar o subconjunto de tamanho $N$ que maximiza $\det(L_S)$ exatamente também é caro computacionalmente. A solução prática — chamada de **MAP guloso para DPP** (*maximum a posteriori* do processo pontual determinantal) — constrói o conjunto item por item, adicionando a cada passo o item que mais aumenta o determinante do conjunto parcial, exatamente na mesma lógica greedy dos outros dois documentos, só que aplicada a uma função objetivo diferente.

**O que essa escolha de oráculo permite (e não permite) fazer:** como o determinante de uma matriz de Gram é uma função submodular do conjunto (retornos marginais decrescentes conforme os itens ficam mais parecidos), o greedy tem garantias de aproximação bem estabelecidas na literatura de inferência para DPP. Mas — e essa é a limitação estrutural que já discutimos — como o objetivo é puramente geométrico (baseado em $S_{ij}$, a similaridade fixa entre coberturas), o oráculo só sabe "afastar" itens parecidos entre si; ele não tem como representar que duas coberturas parecidas às vezes convertem melhor juntas (sinergia real) — só existe o mecanismo de empurrar para longe, nunca de aproximar por benefício aprendido.

**Analogia:** o DPP atribui uma "probabilidade" a cada conjunto possível de coberturas, proporcional ao determinante de $L$ restrito àquele conjunto. Conjuntos com itens muito parecidos entre si têm determinante baixo (quase "achatado", sem volume) — o oráculo busca o conjunto de tamanho $N$ com o maior volume possível.

**A matemática:**

$$p(S) \propto \det(L_S)$$

$$S^{\ast} = \arg\max_{|S|=N} \det(L_S)$$

resolvido de forma aproximada por um algoritmo guloso rápido (MAP guloso para DPP): a cada passo, adiciona-se o item que mais aumenta o determinante do conjunto parcial, até atingir $N$ itens.

| Símbolo | Descrição |
|---|---|
| $\det(L_S)$ | "volume" ocupado pelos itens de $S$ no espaço — maximizado quando os itens são simultaneamente bons ($q_i$ alto) e diferentes entre si ($S_{ij}$ baixo) |
| $N$ | tamanho fixo do pacote, definido de antemão (aqui, $N=3$) |

**O exemplo rodando:** o oráculo tende a **evitar** juntar Desastres Naturais e Danos Elétricos (alto $S_{ij}$, reduz o determinante), preferindo combinações como {Incêndio, Desastres Naturais, Conteúdo Residencial} — cobrindo riscos diferentes. Note que essa preferência por evitar o par que sabemos ser cannibalizado **acontece por acidente geométrico** (baixa similaridade estrutural), não porque o modelo aprendeu que a combinação prejudica a conversão — é a limitação estrutural mencionada na nota inicial.

**Limitação que se torna concreta aqui:** se, ao contrário, "Conteúdo Residencial" e "Roubo e Furto" tivessem sinergia real de conversão (o exemplo que motivou a comparação desde o início da nossa conversa), o DPP **não teria como usar isso a favor** — se as duas também forem estruturalmente parecidas (ambas categoria de risco "roubo_furto"), o modelo vai **evitar** colocá-las juntas, mesmo sendo exatamente o par que mais converteria na prática real.

**O que entra / o que sai:** entra: $L$ (Etapa 4). Sai: $S_{ofertado}$, sempre com $N$ itens.

---

## 6. Oferta ao cliente

**Analogia:** o conjunto de $N$ peças, escolhido para maximizar volume no mapa de similaridade, está pronto para ser mostrado.

**O exemplo rodando:** $S_{ofertado} = \{\text{Incêndio}, \text{Desastres Naturais}, \text{Conteúdo Residencial}\}$.

**O que entra / o que sai:** entra: $S_{ofertado}$ (Etapa 5). Sai: o mesmo conjunto, exposto ao cliente.

---

## 7. Feedback binário e atualização bayesiana da posterior

**Analogia:** cada clique (ou manutenção) observado "empurra" o leque de réguas plausíveis um pouco na direção que explica melhor o que aconteceu — o leque nunca colapsa para uma régua única, só vai ficando mais estreito.

**A matemática:** feedback binário $y_i\in\{0,1\}$ para cada $i\in S_{ofertado}$ (só o que foi ofertado — mesma limitação de feedback semi-bandit do C²UCB original). Atualização da posterior via aproximação variacional (limite de Jaakkola-Jordan para a verossimilhança logística, já que ela não é conjugada com o prior gaussiano):

$$\mu, \Sigma \;\leftarrow\; \text{atualização variacional}(\mu, \Sigma, \{(x(i), y_i)\})$$

| Símbolo | Descrição |
|---|---|
| $\mu, \Sigma$ | atualizados a cada rodada — diferente do reajuste em lote da versão logística que construímos, aqui a atualização é tipicamente **sequencial**, rodada a rodada |

**O que entra / o que sai:** entra: $S_{ofertado}$, feedback $y_i$. Sai: $\mu, \Sigma$ atualizados, voltando para a Etapa 2 da próxima rodada.

---

## 8. O ciclo se fecha

**Analogia:** o próximo cliente entra na loja, e o leque de réguas plausíveis está um pouco mais estreito do que estava para o Cliente A — mas o mecanismo de sorteio (Thompson Sampling) e a repulsão do DPP continuam os mesmos.

**A matemática:** volta-se à Etapa 1, com uma nova amostragem de $\tilde\theta$ na Etapa 2 usando o leque já mais estreito.

---

## Mapa de conexões

```
Etapa 0 (setup)
     │  μ₀, Σ₀ (prior largo)
     ▼
Etapa 1 (cliente chega) ──> c
     ▼
Etapa 2 (Thompson Sampling) ←── μ, Σ (Etapa 7 anterior)
     ▼
Etapa 3 (score de qualidade q_i) ←── θ̃ (sorteado) + c
     ▼
Etapa 4 (kernel L = q·q·S) ←── S_ij (similaridade fixa do catálogo)
     ▼
Etapa 5 (MAP guloso DPP, N fixo) ──> maximiza det(L_S)
     ▼
Etapa 6 (oferta) ──> S_ofertado (tamanho N)
     ▼
Etapa 7 (feedback + atualização bayesiana) ──> novo μ, Σ
     │
     └──────────────► volta para a Etapa 2 (Etapa 8)
```

---

## Tabela-síntese

| Peça | O que faz | Limitação em relação à versão customizada |
|---|---|---|
| Thompson Sampling | explora sorteando réguas plausíveis | mais custoso de implementar (inferência variacional) que o bônus determinístico do UCB |
| $S_{ij}$ (similaridade fixa) | mede parecença estrutural entre coberturas | **não aprende sinergia real** — só repulsão, nunca atração |
| DPP / $\det(L_S)$ | favorece conjuntos "espalhados" | pode evitar pares genuinamente sinérgicos só por serem estruturalmente parecidos |
| $N$ fixo | define o tamanho do pacote de antemão | não deixa o tamanho emergir do comportamento do cliente |
| Feedback semi-bandit puro | só aprende do que foi ofertado | ignora adições espontâneas do cliente |
