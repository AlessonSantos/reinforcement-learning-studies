# Pipeline do algoritmo C²UCB ORIGINAL (Qin, Chen & Zhu, 2014)
### Aplicado ao mesmo case do Cliente A — guia matemático e didático

---

## Nota importante antes de começar

Este documento aplica o algoritmo **exatamente como descrito no paper original**, sem as adaptações que construímos ao longo da nossa conversa (sem fadiga, sem termo de par aprendido, com $k$ fixo, com estimação via ridge regression em vez de regressão logística). O objetivo é mostrar como o C²UCB "de fábrica" se comportaria no mesmo case, para efeito de comparação com a versão customizada (`c2ucb_seguro.py` / `pipeline_c2ucb_seguro_didatico_v2.md`). As diferenças-chave, resumidas:

| | C²UCB original (este documento) | Versão customizada que construímos |
|---|---|---|
| Estimação | ridge regression (linear) | regressão logística (scores sempre entre 0 e 1) |
| Diversidade | regularizador fixo, baseado só nas features estáticas dos itens | termo de par **aprendido** ($\phi$), capturando sinergia e canibalização reais |
| Tamanho do pacote | $k$ fixo, definido antecipadamente | emergente, sem $k$ fixo |
| Fadiga | não existe no paper original | aprendida via contexto de tamanho do pacote parcial |
| Feedback | só dos itens efetivamente ofertados (semi-bandit) | inclui itens adicionados espontaneamente |

**O exemplo que vamos seguir:** o mesmo "Cliente A", perfil de risco médio-alto, avaliado entre as mesmas 5 coberturas ilustrativas (Incêndio, Conteúdo, Roubo, Danos Elétricos, Desastres Naturais), com pacote de tamanho fixo $k=3$.

---

## 0. Setup — antes de existir qualquer cliente

**Analogia:** uma única régua, $\theta$, que vai aprender a traduzir "cliente + cobertura" em uma nota de relevância — sem funil nenhum no final, a nota pode ser qualquer número.

**A matemática:**

$$\theta \in \mathbb{R}^d, \qquad V = \lambda I, \qquad b = \vec 0$$

| Símbolo | Descrição |
|---|---|
| $\theta$ | régua compartilhada: cliente+cobertura → nota de relevância (sem limite superior/inferior definido) |
| $V$ | matriz de precisão — "quanto já vimos de variedade de situação" |
| $b$ | vetor acumulador — combinado com $V^{-1}$, gera a estimativa pontual de $\theta$ |
| $\lambda$ | regularização — trava $\theta$ perto de zero antes de ver dados |
| $k$ | tamanho fixo do pacote — decidido de antemão pela equipe (aqui, $k=3$) |

**O que entra / o que sai:** nada entra. Sai: $\theta, V, b$ zerados, $k$ fixado, prontos para a Rodada 1.

---

## 1. Uma nova rodada começa: o cliente chega

**Analogia:** o Cliente A entra na loja. Antes de qualquer coisa, você olha a ficha dele.

**A matemática:** observa-se $c \in \mathbb{R}^p$.

**O exemplo rodando:** $c = [0{,}8,\ 1{,}0]$ (score de risco; indicador de imóvel de alto valor).

**O que entra / o que sai:** entra: nada. Sai: $c$, usado nas etapas seguintes.

---

## 2. Score do item via ridge regression (LinUCB)

**Analogia:** a régua $\theta$ multiplica o perfil combinado de cliente+cobertura e devolve uma nota — essa nota **pode** sair de qualquer faixa numérica, não existe um funil que a limite entre 0 e 1 (diferente da versão que construímos depois).

**A matemática:**

$$x(i) = \text{concat}\big(c,\; f_i,\; c \otimes f_i\big)$$

$$\hat r(i) = \hat\theta^{T}x(i), \qquad \text{bonus}(i) = \alpha\sqrt{x(i)^{T}V^{-1}x(i)}$$

$$r_{UCB}(i) = \hat r(i) + \text{bonus}(i)$$

| Símbolo | Descrição |
|---|---|
| $x(i)$ | contexto do item: cliente, atributos estáticos da cobertura, interação entre os dois — **sem** nenhum termo de tamanho de pacote |
| $\hat r(i)$ | nota estimada — um número real qualquer, não uma probabilidade |
| $\text{bonus}(i)$ | incerteza — igual à lógica que já vimos, mas sem o funil logístico corrigindo a escala |
| $r_{UCB}(i)$ | nota otimista final, usada diretamente na função de recompensa (Etapa 4) |

**O exemplo rodando:** suponha os seguintes valores de $r_{UCB}(i)$ para o Cliente A, já com o bônus somado:

| Cobertura | $r_{UCB}(i)$ |
|---|---|
| Incêndio | 1,05 |
| Desastres Naturais | 0,95 |
| Conteúdo Residencial | 0,65 |
| Roubo e Furto | 0,55 |
| Danos Elétricos | 0,45 |

Note que os valores passam de 1 — isso é normal aqui, porque não existe funil limitando a escala.

**O que entra / o que sai:** entra: $c$; $\hat\theta$ e $V$ — na primeira rodada, ainda zerados, vindos do Setup (Etapa 0); a partir da segunda rodada em diante, vêm já atualizados da Etapa 7 da rodada anterior. Sai: $r_{UCB}(i)$ para cada candidato, usado na Etapa 4.

---

## 3. O regularizador de diversidade — fixo, baseado só nas features estáticas

**Analogia:** pense em cada cobertura como um ponto num mapa, definido só pelos seus próprios atributos (não pelo cliente). O regularizador de diversidade mede o quanto os pontos escolhidos estão "espalhados" nesse mapa — quanto mais próximos/parecidos entre si, menor o valor; quanto mais espalhados, maior.

**A matemática:**

$$h(S, X) = \frac{1}{2}\log\det\big(X(S)^{T}X(S) + \sigma^2 I\big)$$

onde $X(S)$ é a matriz cujas colunas são os vetores $f_i$ (só os atributos **estáticos** das coberturas em $S$ — o perfil do cliente não entra aqui).

| Símbolo | Descrição |
|---|---|
| $X(S)$ | matriz com os vetores de atributos estáticos das coberturas já escolhidas |
| $\sigma^2$ | constante de suavização, evita determinante zero com poucos itens |
| $h(S,X)$ | valor de diversidade — máximo quando os vetores são ortogonais (coberturas "diferentes"), mínimo quando são colineares (coberturas "parecidas") |

**Diferença crucial em relação à versão customizada:** este termo é **fixo** — calculado só a partir dos atributos de catálogo, sempre o mesmo valor para o mesmo conjunto de coberturas, **não aprende nada com o comportamento do cliente**. Ele não distingue sinergia (efeito positivo real de conversão) de mera dissimilaridade geométrica — só mede se as coberturas "parecem" diferentes no papel.

**O exemplo rodando:** suponha que, usando os vetores de atributos de catálogo (risco, faixa de prêmio etc.), o cálculo do log-determinante resulte em:

$$h(\{\text{Incêndio}\}) = 0, \qquad h(\{\text{Incêndio}, \text{Desastres}\}) \approx 0{,}35$$

(positivo, pois "Incêndio" é categoria de risco "outros" e "Desastres" é categoria de risco "climático" — vetores razoavelmente distintos no catálogo).

**O que entra / o que sai:** entra: $f_i$ de cada cobertura candidata e das já escolhidas. Sai: **a fórmula de $h(S,X)$ pronta para uso** — o exemplo acima calculou seu valor só para uma configuração ilustrativa ($\{\text{Incêndio}\}$ e $\{\text{Incêndio},\text{Desastres}\}$); é o oráculo (Etapa 4) que vai recalculá-la repetidamente, para cada subconjunto $S$ diferente que for testando ao longo do processo guloso.

---

## 4. O oráculo guloso: cardinalidade fixa $k$

**O que é um "oráculo" nesse contexto:** é o componente do algoritmo que, recebendo os scores UCB estimados (Etapa 2) e o regularizador de diversidade (Etapa 3), decide qual conjunto de coberturas oferecer. O nome vem do fato de que o oráculo é tratado como uma "caixa-preta" do ponto de vista da parte estatística do algoritmo — o LinUCB não precisa saber como o pacote é montado, só precisa fornecer os scores; é essa separação que permite ao C²UCB trocar a função de recompensa ou as restrições sem alterar a forma como $\theta$ é estimado.

**Por que "guloso" (greedy):** testar todas as combinações possíveis de $k$ coberturas entre um catálogo maior seria caro computacionalmente. O oráculo guloso evita isso construindo o pacote item por item — a cada passo, adiciona o candidato que mais aumenta a função objetivo (relevância + diversidade), sem nunca reconsiderar escolhas já feitas.

**A garantia formal que se aplica aqui:** este é justamente o caso "de fábrica" descrito no paper original — como a função de recompensa (soma dos scores + regularizador de diversidade) é **submodular e monótona**, e a restrição é cardinalidade fixa ($k$), o algoritmo guloso tem uma prova de aproximação $(1-1/e) \approx 63\%$ do valor ótimo — ou seja, mesmo sem testar todas as combinações, há uma garantia matemática de qualidade mínima do pacote encontrado. Essa é a mesma garantia que se perde quando saímos desse regime "clássico" (ver a ressalva equivalente no documento da versão customizada, onde a recompensa pode ter termos negativos e não há $k$ fixo).

**Analogia:** montar a mala com um número **fixo** de compartimentos — você sabe de antemão que vai levar exatamente 3 peças, e escolhe as 3 que, juntas, maximizam relevância + variedade.

**A matemática:** $S$ começa vazio. A cada passo, até $|S|=k$:

$$i^{\ast} = \arg\max_{i \notin S} \Big[r_{UCB}(i) + \lambda_{div}\big(h(S\cup\{i\}, X) - h(S,X)\big)\Big]$$

Adiciona $i^{\ast}$ a $S$. Repete até $|S|=k$ — **não há critério de parada por ganho marginal negativo**: o algoritmo sempre preenche as $k$ posições, mesmo que o ganho da última seja pequeno ou até prejudicial.

**O exemplo rodando** ($k=3$, $\lambda_{div}=0{,}3$):

**Passo 1:** $S=\emptyset$. Maior $r_{UCB}$ sozinho: Incêndio (1,05). $S=\{\text{Incêndio}\}$.

**Passo 2:** avalia cada candidato somando $r_{UCB}(i)$ ao ganho de diversidade trazido:

| Candidato | $r_{UCB}(i)$ | Ganho de diversidade ($\times 0{,}3$) | Total |
|---|---|---|---|
| Desastres Naturais | 0,95 | $0{,}3\times0{,}35=0{,}105$ | **1,055** |
| Conteúdo Residencial | 0,65 | $0{,}3\times0{,}30=0{,}090$ | 0,740 |
| Roubo e Furto | 0,55 | $0{,}3\times0{,}28=0{,}084$ | 0,634 |
| Danos Elétricos | 0,45 | $0{,}3\times0{,}10=0{,}030$ | 0,480 |

Desastres Naturais vence. $S=\{\text{Incêndio}, \text{Desastres}\}$.

**Passo 3:** processo se repete entre os candidatos restantes; suponha que "Conteúdo Residencial" vença por ter boa relevância e trazer diversidade adicional. $S=\{\text{Incêndio}, \text{Desastres}, \text{Conteúdo}\}$ — pacote completo, $|S|=k=3$, processo encerrado **por ter atingido $k$**, não por decisão de ganho marginal.

**O que entra / o que sai:** entra: $r_{UCB}(i)$ de todos os candidatos, $h(S,X)$ recalculado a cada passo. Sai: o pacote final $S_{ofertado}$, sempre com exatamente $k$ itens.

---

## 5. Oferta ao cliente

**Analogia:** a mala de 3 compartimentos está pronta.

**O exemplo rodando:** $S_{ofertado} = \{\text{Incêndio}, \text{Desastres Naturais}, \text{Conteúdo Residencial}\}$.

**O que entra / o que sai:** entra: $S_{ofertado}$ (Etapa 4). Sai: o mesmo conjunto, exposto ao cliente.

---

## 6. Feedback do cliente — só sobre o que foi ofertado

**Analogia:** diferente da versão customizada (onde o cliente podia adicionar qualquer coisa fora da oferta), aqui o algoritmo original só "escuta" o que aconteceu com os itens que ele mesmo colocou na mala — é a suposição de **feedback semi-bandit** padrão do paper.

**A matemática:** para cada $i \in S_{ofertado}$, observa-se $r_t(i)$ — no paper original, tipicamente uma nota contínua (rating); no nosso contexto de seguros, pode ser $y_i\in\{0,1\}$ (manteve/não manteve), mas sem a garantia de escala $[0,1]$ que a versão logística oferece.

**O exemplo rodando:** suponha que o Cliente A mantém Incêndio e Desastres Naturais, mas remove Conteúdo Residencial. $y_{\text{Incêndio}}=1$, $y_{\text{Desastres}}=1$, $y_{\text{Conteúdo}}=0$.

**Diferença importante:** se o cliente tivesse adicionado espontaneamente "Roubo e Furto" (fora da oferta), **o algoritmo original não teria como usar esse dado** — só aprende sobre os itens que ele mesmo ofertou.

**O que entra / o que sai:** entra: $S_{ofertado}$ (Etapa 5). Sai: as observações $(x(i), y_i)$ para cada item ofertado.

---

## 7. Atualização — ridge regression, soma fechada

**Analogia:** diferente da régua logística (que precisa reprocessar tudo), esta régua pode ser atualizada **somando diretamente** o que aconteceu, sem precisar reprocessar o histórico inteiro.

**A matemática:**

$$V \leftarrow V + \sum_{i\in S_{ofertado}} x(i)x(i)^{T}, \qquad b \leftarrow b + \sum_{i \in S_{ofertado}} y_i \cdot x(i)$$

$$\hat\theta = V^{-1}b$$

**De onde vem essa fórmula, passo a passo:**

Assim como na versão logística (ver a derivação equivalente no documento da versão customizada), o ponto de partida é a **regra de Bayes**, aplicada a $\theta$:

$$\underbrace{P(\theta \mid \text{dados})}_{\text{posteriori}} \;\propto\; \underbrace{P(\text{dados} \mid \theta)}_{\text{verossimilhança}} \;\times\; \underbrace{P(\theta)}_{\text{priori}}$$

| Termo | O que significa aqui |
|---|---|
| **Priori** $P(\theta)$ | o que acreditávamos sobre $\theta$ antes de ver dados — $\theta \sim \mathcal{N}(0, \lambda^{-1}I)$, o mesmo "chute gaussiano perto de zero" da versão logística |
| **Verossimilhança** $P(\text{dados}\mid\theta)$ | o quão prováveis são as notas observadas, assumindo um valor particular de $\theta$ — aqui é onde a suposição de **ruído gaussiano** entra, e é a diferença central em relação à versão logística |
| **Posteriori** $P(\theta\mid\text{dados})$ | a crença atualizada sobre $\theta$ depois de combinar priori e verossimilhança |

**Passo 1 — o que é "ruído gaussiano" e por que ele aparece aqui.** O modelo assume que a nota observada $y_n$ é a nota "verdadeira" prevista pela régua, $\theta^Tx_n$, mais um erro aleatório $\varepsilon_n$ que captura tudo que a régua não consegue prever (humor do cliente no dia, fatores não observados etc.):

$$y_n = \theta^Tx_n + \varepsilon_n, \qquad \varepsilon_n \sim \mathcal{N}(0,\sigma^2) \text{ (i.i.d.)}$$

Isso é uma suposição sobre a **verossimilhança**, não sobre a priori: dado $\theta$, a distribuição de $y_n$ é uma gaussiana centrada em $\theta^Tx_n$:

$$P(y_n\mid\theta) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_n-\theta^Tx_n)^2}{2\sigma^2}\right)$$

Esse é o ponto de contraste mais importante com a versão logística: lá, $y_n\in\{0,1\}$ segue uma **Bernoulli**, com $P(y_n\mid\theta)=p_n^{y_n}(1-p_n)^{1-y_n}$ e $p_n=\sigma(\theta^Tx_n)$; aqui, $y_n$ é assumido contínuo e a verossimilhança é **gaussiana**, centrada diretamente em $\theta^Tx_n$ (sem nenhum funil no meio).

**Passo 2 — a verossimilhança conjunta das $N$ observações**, assumindo independência entre elas dado $\theta$:

$$P(\text{dados}\mid\theta) = \prod_{n=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_n-\theta^Tx_n)^2}{2\sigma^2}\right)$$

**Passo 3 — o logaritmo da verossimilhança.** Aplicando $\log$ (transforma produto em soma, e a exponencial desaparece):

$$\log P(\text{dados}\mid\theta) = -\frac{1}{2\sigma^2}\sum_{n=1}^N(y_n-\theta^Tx_n)^2 + \text{constante (não depende de }\theta\text{)}$$

Note o que aconteceu: **maximizar** essa log-verossimilhança em $\theta$ é exatamente o mesmo que **minimizar** $\sum_n(y_n-\theta^Tx_n)^2$ — a soma dos erros ao quadrado. É essa a ponte entre "assumir ruído gaussiano" e "minimizar erro quadrático": elas são a mesma coisa, matematicamente.

**Passo 4 — o logaritmo da priori**, igual à versão logística: a densidade de $\mathcal{N}(0,\lambda^{-1}I)$ contribui com $-\frac{\lambda}{2}\|\theta\|^2$ (a parte que depende de $\theta$).

**Passo 5 — somar as duas partes** (log-verossimilhança + log-priori) dá o log da posteriori, a menos de constantes:

$$\log P(\theta\mid\text{dados}) \;\propto\; -\frac{1}{2\sigma^2}\sum_{n=1}^N(y_n-\theta^Tx_n)^2 - \frac{\lambda}{2}\|\theta\|^2$$

Maximizar isso em $\theta$ (a estimativa **MAP**, *maximum a posteriori*) é equivalente a minimizar $L(\theta) = \sum_n(y_n-\theta^Tx_n)^2 + \lambda\sigma^2\|\theta\|^2$ — absorvendo a constante $\sigma^2$ dentro de $\lambda$ (uma simplificação padrão), essa é exatamente a função objetivo da ridge regression.

**Passo 6 — por que existe solução fechada aqui (e não na versão logística).** Como tanto o termo de erro quadrático quanto a penalização $\|\theta\|^2$ são **quadráticos** em $\theta$, a derivada de $L(\theta)$ é **linear** em $\theta$. Derivando e igualando a zero:

$$\frac{\partial L}{\partial\theta} = -2\sum_n x_n(y_n-x_n^T\theta) + 2\lambda\theta = 0 \quad\Longrightarrow\quad \Big(\lambda I + \sum_n x_nx_n^T\Big)\theta = \sum_n x_ny_n$$

Batizando $V=\lambda I+\sum_n x_nx_n^T$ e $b=\sum_n x_ny_n$, isso é exatamente $\hat\theta=V^{-1}b$ — um sistema linear, resolvido por uma única inversão de matriz. Diferente da versão logística (onde $p_n=\sigma(\theta^Tx_n)$ entra de forma não-linear dentro do logaritmo, impedindo isolar $\theta$ algebricamente), aqui $V$ e $b$ podem ser somados diretamente, evento a evento ou em lote, sem precisar de nenhum otimizador iterativo.

| Símbolo | Descrição |
|---|---|
| $V, b$ | acumuladores — podem ser atualizados evento a evento **ou** em lote, com resultado idêntico (soma é associativa) |
| $\hat\theta$ | nova estimativa, disponível imediatamente após a atualização, sem precisar de nenhum otimizador iterativo |

**Vantagem prática sobre a versão logística:** essa atualização é matematicamente mais simples — não exige guardar o histórico bruto nem reprocessar tudo a cada lote, ao custo de não garantir que $\hat r(i)$ fique dentro de $[0,1]$ (Etapa 2).

**O que entra / o que sai:** entra: as observações da Etapa 6. Sai: $\hat\theta, V$ atualizados, voltando para a Etapa 2 da próxima rodada.

---

## 8. O ciclo se fecha

**Analogia:** o próximo cliente entra na loja, e a régua $\theta$ já está um pouco mais calibrada do que estava para o Cliente A — mas sempre a mesma régua única, sem funil, sem termo de par aprendido, sem fadiga.

**A matemática:** volta-se à Etapa 1 com um novo cliente, usando $\hat\theta, V$ atualizados.

---

## Mapa de conexões

```
Etapa 0 (setup)
     │  θ̂, V (zerados), k fixo
     ▼
Etapa 1 (cliente chega) ──> c
     ▼
Etapa 2 (score item, ridge)  ←── θ̂, V (Etapa 7 anterior) + c
     ▼
Etapa 3 (diversidade fixa)   ←── f_i de cada candidato (catálogo, não muda)
     ▼
Etapa 4 (oráculo guloso, k fixo) ──> preenche ATÉ k, sem parada antecipada
     ▼
Etapa 5 (oferta) ──> S_ofertado (sempre tamanho k)
     ▼
Etapa 6 (feedback só do ofertado) ──> y_i
     ▼
Etapa 7 (atualização ridge, somável) ──> novo θ̂, V
     │
     └──────────────► volta para a Etapa 2 (Etapa 8)
```

---

## Tabela-síntese

| Peça | O que faz | Limitação em relação à versão customizada |
|---|---|---|
| $\hat r(i)$ (ridge) | nota de relevância | pode sair de $[0,1]$; sem funil |
| $h(S,X)$ (diversidade fixa) | mede dissimilaridade geométrica dos itens escolhidos | não aprende sinergia real; não distingue "parecido mas complementar" de "parecido e redundante" |
| $k$ fixo | define o tamanho do pacote de antemão | não deixa o tamanho emergir do comportamento do cliente |
| Feedback semi-bandit puro | só aprende do que foi ofertado | ignora adições espontâneas do cliente |
| Atualização ridge | soma fechada, simples | troca simplicidade por perda da garantia de escala probabilística |
