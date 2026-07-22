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

$$\hat r(i) = \hat\theta^{\,T}x(i), \qquad \text{bonus}(i) = \alpha\sqrt{x(i)^{T}V^{-1}x(i)}$$

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

**O que entra / o que sai:** entra: $c$, $\hat\theta$, $V$ (da Etapa 7 da rodada anterior). Sai: $r_{UCB}(i)$ para cada candidato, usado na Etapa 4.

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

**O que entra / o que sai:** entra: $f_i$ de cada cobertura candidata e das já escolhidas. Sai: o valor de $h(S,X)$ para cada configuração testada pelo oráculo (Etapa 4).

---

## 4. O oráculo guloso: cardinalidade fixa $k$

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

**De onde vem essa fórmula:** ela nasce de minimizar o erro quadrático entre a nota prevista e a observada, com uma penalização que evita valores extremos: $L(\theta) = \sum_n(y_n - \theta^Tx_n)^2 + \lambda\|\theta\|^2$. Derivando $L$ em relação a $\theta$ e igualando a zero, chega-se a $\big(\lambda I + \sum_n x_nx_n^T\big)\theta = \sum_n x_ny_n$ — batizando $V=\lambda I+\sum_n x_nx_n^T$ e $b=\sum_n x_ny_n$, isso é exatamente $\hat\theta=V^{-1}b$. O ponto importante: essa é uma equação **linear** em $\theta$ (basta inverter uma matriz), diferente da versão logística que veremos ser necessária mais adiante — é por isso que $V$ e $b$ podem ser somados diretamente, evento a evento ou em lote, sem precisar de nenhum otimizador iterativo.

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
