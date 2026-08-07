# Recomendação de pacotes de coberturas de seguro: bandit contextual com regressão logística e threshold calibrado
### Guia matemático e didático

---

## Como usar este documento

Cada seção segue o mesmo padrão: primeiro uma **analogia** para visualizar a ideia sem fórmula nenhuma, depois **a matemática exata** por trás dela, depois um **exemplo rodando** (o mesmo cliente fictício, "Cliente A", acompanhado do início ao fim), e uma caixa **"o que entra / o que sai"** conectando cada etapa à anterior e à seguinte.

**O catálogo:** 14 coberturas no total — **4 obrigatórias** (sempre incluídas no pacote, nunca decididas pelo algoritmo) e **10 opcionais** (cada uma decidida individualmente, com base no perfil do cliente).

---

## 0. Setup — antes de existir qualquer cliente

**Analogia:** imagine uma única régua, $\theta$, compartilhada entre as 10 coberturas opcionais — ela vai aprender a traduzir "perfil do cliente + atributos de uma cobertura" numa nota de propensão. No início, a régua não sabe nada; ela vai sendo calibrada conforme os clientes interagem.

**A matemática:**

$$\theta \in \mathbb{R}^d, \qquad \theta \sim \mathcal N(0,\ \lambda^{-1}I)$$

| Símbolo | Descrição |
|---|---|
| $\theta$ | régua compartilhada entre as 10 coberturas opcionais: perfil do cliente + atributos da cobertura → propensão de aceitação |
| $\lambda$ | regularização — o quanto $\theta$ começa "travada" perto de zero antes de haver evidência forte, e o quanto continua puxada para zero mesmo depois, protegendo contra estimativas extremas |
| $\epsilon_t$ | probabilidade de explorar aleatoriamente nesta rodada (ver Etapa 2) |
| $\rho$ | fração das rodadas aleatórias reservada exclusivamente para calcular a taxa de referência de cada cobertura (ver Etapa 4) — nunca usada para treinar $\theta$ |

**O que entra / o que sai:** nada entra. Sai: $\theta$ zerado, e os parâmetros de configuração — $\lambda$, $\epsilon_t$, $\rho$ — definidos, prontos para a Rodada 1.

---

## 1. Uma nova rodada começa: o cliente chega

**Analogia:** o Cliente A entra na loja. Antes de qualquer coisa, você olha a ficha dele.

**A matemática:** observa-se $c \in \mathbb{R}^p$, o vetor de perfil do cliente.

**O exemplo rodando:** $c = [0{,}8,\ 1{,}0]$ (score de risco normalizado; indicador de imóvel de alto valor).

**O que entra / o que sai:** entra: nada. Sai: $c$, usado em todas as etapas seguintes desta rodada.

---

## 2. Explorar aleatoriamente ou usar o modelo?

**Analogia:** nas primeiras semanas de uma loja nova, o vendedor testa combinações variadas de propósito, mesmo sabendo que algumas vão ser ruins, só para aprender rápido. Com o tempo, ele confia mais no que já sabe — mas nunca para de testar por completo.

**A matemática — a variável que decide o caminho:**

$$Z_t \sim \text{Bernoulli}(\epsilon_t)$$

| Se... | Então... |
|---|---|
| $Z_t=1$ (prob. $\epsilon_t$) | rota aleatória: cada uma das 10 coberturas opcionais entra no pacote de forma independente, com probabilidade fixa (ex.: 50%) — pula para a Etapa 6 |
| $Z_t=0$ (prob. $1-\epsilon_t$) | rota do modelo: segue para a Etapa 3 |

| Símbolo | Descrição |
|---|---|
| $Z_t$ | a única variável genuinamente aleatória que decide o caminho — sorteada de novo a cada rodada, nunca fixada por cliente |
| $\epsilon_t$ | decrescente ao longo do tempo, mas **nunca chega a zero** — continua garantindo que toda cobertura, mesmo uma já bem avaliada pelo modelo, continue sendo testada de vez em quando |

**Por que a rota aleatória oferece cada cobertura de forma independente, sem se preocupar com o tamanho do pacote:** nesta versão, cada cobertura opcional é decidida individualmente, sem nenhum termo que dependa de quantas outras já foram incluídas — então a rota aleatória só precisa garantir que cada cobertura, isoladamente, seja testada com e sem a presença do cliente estar "exposto" a ela, em proporção suficiente para estimar sua taxa de aceitação de referência (Etapa 4).

**O exemplo rodando:** suponha que já passamos da fase de exploração intensa — vamos seguir o Cliente A pela via do modelo (Etapa 3 em diante).

**O que entra / o que sai:** entra: $c$, $\epsilon_t$ (valor vigente no momento). Sai: $Z_t$ sorteado, e a partir dele, sinal para calcular scores (Etapa 3) ou pacote aleatório.

---

## 3. Score da cobertura via regressão logística

**Analogia:** $\theta$ multiplica o perfil combinado de cliente e cobertura, e o resultado passa por um funil (a função logística) que o espreme para dentro de 0 a 1 — nunca escapa desse intervalo, não importa quão extremo seja o número antes do funil.

**A matemática:**

$$x(i) = \text{concat}\big(c,\; f_i,\; c \otimes f_i\big)$$

$$\eta(i) = \theta^{T} x(i)$$

$$\hat p(i) = \sigma\big(\eta(i)\big) = \frac{1}{1+e^{-\eta(i)}}$$

| Símbolo | Descrição |
|---|---|
| $c$ | perfil do cliente |
| $f_i$ | atributos estáticos da cobertura $i$ (categoria de risco, faixa de prêmio, frequência histórica de sinistro) |
| $c \otimes f_i$ | interação entre perfil e cobertura — permite que $\theta$ aprenda relações cruzadas ("clientes de risco alto valorizam mais coberturas de determinada categoria"), em vez de só efeitos isolados de $c$ e de $f_i$ |
| $\eta(i)$ | o **log-odds** da propensão — $\theta^Tx(i)$ é, tecnicamente, $\log\big(p(i)/(1-p(i))\big)$, não a probabilidade em si |
| $\sigma(\cdot)$ | o funil logístico — converte o log-odds de volta em probabilidade, garantindo o resultado sempre entre 0 e 1 |
| $\hat p(i)$ | a **recompensa esperada**: a estimativa calibrada do modelo para a propensão real de aceitação — não inflada por nenhum termo de exploração |

**De onde vem $\theta$: a estimativa de máxima verossimilhança a posteriori (MAP).** $\theta$ não é ajustado minimizando erro quadrático — é ajustado maximizando a verossimilhança de observações binárias (Bernoulli), combinada com a priori gaussiana definida na Etapa 0. Pela regra de Bayes:

$$P(\theta\mid\text{dados}) \propto \underbrace{P(\text{dados}\mid\theta)}_{\text{verossimilhança}} \times \underbrace{P(\theta)}_{\text{priori}}$$

A verossimilhança de uma única observação $(x_n, y_n)$, com $y_n\in\{0,1\}$ e $p_n=\sigma(\theta^Tx_n)$, segue uma distribuição de Bernoulli: $P(y_n\mid\theta) = p_n^{y_n}(1-p_n)^{1-y_n}$ — a expressão vale $p_n$ quando $y_n=1$ e $1-p_n$ quando $y_n=0$, cobrindo os dois casos numa única fórmula. Tomando o logaritmo da verossimilhança conjunta de todas as observações e somando o logaritmo da priori, chega-se à log-posteriori, a menos de uma constante aditiva que não depende de $\theta$:

$$\log P(\theta\mid\text{dados}) = \sum_{n=1}^{N}\Big[y_n\log p_n + (1-y_n)\log(1-p_n)\Big] - \frac{\lambda}{2}\|\theta\|^2 + \text{constante}$$

$$\hat\theta = \arg\max_\theta \;\log P(\theta\mid\text{dados})$$

**Por que isso não tem solução fechada.** Combinar uma verossimilhança de Bernoulli com uma priori gaussiana não produz uma posteriori de nenhuma família conhecida — essa não é uma peculiaridade deste problema específico, é uma propriedade da regressão logística bayesiana em geral, bem documentada na literatura estatística clássica. Como $p_n=\sigma(\theta^Tx_n)$ entra no logaritmo de forma não-linear, a derivada da log-posteriori em relação a $\theta$ também é não-linear — não dá para isolar $\theta$ algebricamente. Por isso, encontrar $\hat\theta$ exige um otimizador numérico iterativo (o gradiente usado é $\nabla_\theta = \sum_n x_n(y_n-p_n) - \lambda\theta$), não uma única inversão de matriz.

**O exemplo rodando:** suponha que, para o Cliente A, o modelo já calculou $\eta(\text{Desastres Naturais})=0{,}90$. A recompensa esperada é:

$$\hat p(\text{Desastres Naturais}) = \sigma(0{,}90) \approx 0{,}71$$

**O que entra / o que sai:** entra: $c$, $\theta$ (do reajuste mais recente — Etapa 9). Sai: $\hat p(i)$ para cada uma das 10 coberturas opcionais, usado na Etapa 5.

---

## 4. O threshold por cobertura: uma referência calibrada a partir dos próprios dados

**Analogia:** antes de decidir se vale a pena confiar na régua do modelo para uma cobertura específica, é preciso saber qual seria a taxa de aceitação **se ninguém estivesse decidindo nada** — só para ter certeza de que o modelo está identificando algo além do óbvio.

**Por que não usar 0,5 fixo para todas as coberturas.** Um ponto de corte fixo pressupõe implicitamente que aceitar e rejeitar uma oferta têm o mesmo peso de decisão para toda cobertura — uma suposição que não há motivo para ser verdadeira: cada cobertura pode ter uma taxa de aceitação "de base" bem diferente das demais. Em vez de impor um valor de negócio arbitrário (que exigiria quantificar custo e benefício de cada cobertura em unidades monetárias), a abordagem adotada aqui deriva o threshold diretamente dos dados: **só vale a pena confiar na segmentação do modelo para um cliente específico se a propensão estimada para ele for genuinamente maior do que a taxa de aceitação típica daquela cobertura**, quando ofertada sem nenhuma seleção.

**A matemática — a taxa de referência:**

$$\text{baseline}_i = \frac{\text{aceitações de } i \text{ nas rodadas reservadas}}{n_i}$$

calculada **só** sobre uma fração $\rho$ das rodadas aleatórias ($Z_t=1$), reservada exclusivamente para este cálculo — nunca usada para treinar $\theta$ (Etapa 9), e nunca misturada com rodadas da rota do modelo.

**Por que essa reserva é necessária, e por que a divisão precisa ser independente do resultado.** Se as mesmas observações alimentassem tanto o treino de $\theta$ quanto o cálculo do baseline, os dois deixariam de ser estatisticamente independentes — a comparação entre eles ficaria enviesada. A separação evita isso por construção. A regra que decide se uma rodada aleatória vai para a reserva do baseline ou para o treino de $\theta$ precisa depender só de algo definido **antes** de conhecer a resposta do cliente (por exemplo, um identificador da rodada, nunca do resultado observado) — senão a separação deixaria de proteger a independência que ela existe para garantir.

**O limite superior de confiança do baseline, via intervalo de Wilson:**

$$\tau_i = \frac{\text{baseline}_i + \dfrac{z^2}{2n_i} + z\sqrt{\dfrac{\text{baseline}_i(1-\text{baseline}_i)}{n_i} + \dfrac{z^2}{4n_i^2}}}{1+\dfrac{z^2}{n_i}}, \qquad z=1{,}645 \text{ (95\% de confiança, um lado)}$$

| Símbolo | Descrição |
|---|---|
| $n_i$ | número de observações reservadas para a cobertura $i$ |
| $\text{baseline}_i$ | proporção amostral pura — frequência de aceitação sem nenhuma seleção |
| $\tau_i$ | o limite superior do intervalo de confiança do baseline — o threshold usado na decisão (Etapa 5) |

**Por que o limite superior, e por que Wilson em vez da aproximação normal simples.** Usar o limite superior (em vez do próprio $\text{baseline}_i$) protege contra decidir com base numa taxa de referência que só pareceu baixa por pouca amostra — quanto menor $n_i$, mais largo o intervalo, mais alto (mais conservador) fica $\tau_i$; conforme $n_i$ cresce, $\tau_i$ converge para o próprio $\text{baseline}_i$. A fórmula de Wilson é usada em vez da aproximação normal mais simples porque se comporta melhor precisamente nos casos mais delicados — poucas observações, ou taxas próximas de 0 ou 1 — sem risco de gerar limites fora do intervalo $[0,1]$.

**O exemplo rodando:** suponha que, na fatia reservada, "Desastres Naturais" foi ofertada aleatoriamente $n_i=120$ vezes, com $\text{baseline}_i=0{,}40$ (40% de aceitação). Aplicando Wilson:

$$\tau_{\text{Desastres}} \approx 0{,}477$$

**O que entra / o que sai:** entra: as observações reservadas de cada cobertura (acumuladas desde o início, atualizadas na Etapa 9). Sai: $\tau_i$ para cada uma das 10 coberturas opcionais, usado na Etapa 5.

---

## 5. A decisão: obrigatórias + o que passar no threshold

**Analogia:** a mala já vem com um compartimento fixo, sempre presente (as obrigatórias). Para cada peça opcional, a pergunta é simples e independente das demais: "esse cliente específico tem propensão genuinamente maior que a média para esta peça?" — sem nenhum laço iterativo, sem reavaliar nada em função do que mais foi escolhido.

**A matemática:**

$$S_{ofertado} = \underbrace{\{\text{4 obrigatórias}\}}_{\text{sempre incluídas}} \;\cup\; \big\{ i \in \{\text{10 opcionais}\} : \hat p(i) > \tau_i \big\}$$

Não existe processo guloso, nem ordem de avaliação, nem dependência entre as decisões — cada uma das 10 coberturas opcionais é comparada contra seu próprio $\tau_i$ de forma independente, e todas as que passarem entram no pacote, junto com as 4 obrigatórias.

**O exemplo rodando:**

| Cobertura opcional | $\hat p(i)$ | $\tau_i$ | Decisão |
|---|---|---|---|
| Desastres Naturais | 0,71 | 0,48 | **oferta** |
| Danos Elétricos | 0,55 | 0,60 | não oferta |
| Roubo e Furto | 0,62 | 0,50 | **oferta** |
| Conteúdo Residencial | 0,45 | 0,40 | **oferta** |
| Assistência Chaveiro | 0,30 | 0,35 | não oferta |

$$S_{ofertado} = \{\text{4 obrigatórias}\} \cup \{\text{Desastres Naturais, Roubo e Furto, Conteúdo Residencial}\}$$

O tamanho do pacote **emerge** dessa comparação item a item — não há $k$ fixo em nenhum lugar da equação, e também não há nenhuma penalidade artificial por tamanho: se numa rodada 8 das 10 opcionais passarem no threshold, as 8 são ofertadas.

**O que entra / o que sai:** entra: $\hat p(i)$ (Etapa 3) e $\tau_i$ (Etapa 4) para cada opcional. Sai: $S_{ofertado}$.

---

## 6. Oferta ao cliente

**Analogia:** a mala está pronta — hora de mostrar ao Cliente A.

**O exemplo rodando:** $S_{ofertado} = \{$4 obrigatórias, Desastres Naturais, Roubo e Furto, Conteúdo Residencial$\}$.

**O que entra / o que sai:** entra: $S_{ofertado}$ (Etapa 5). Sai: o mesmo conjunto, exposto ao cliente.

---

## 7. Cliente edita: mantém, remove, adiciona

**Analogia:** o Cliente A olha a mala pronta e mexe nela: tira uma peça, mantém outras, e às vezes pega algo que nem estava na oferta.

**O exemplo rodando:** o Cliente A mantém Desastres Naturais e Roubo e Furto, remove Conteúdo Residencial, e adiciona Danos Elétricos por conta própria (mesmo não tendo sido ofertada).

**O que entra / o que sai:** entra: $S_{ofertado}$ (Etapa 6). Sai: $S_{final}$, o pacote realmente contratado.

---

## 8. Traduzindo a edição em dado de treino

**Analogia:** cada cobertura opcional mantida, tirada ou acrescentada vira uma "ficha" no arquivo — as 4 obrigatórias nunca geram ficha nenhuma, porque não há decisão a aprender sobre elas.

**A matemática:** para cada uma das 10 coberturas opcionais visíveis ao cliente (ofertada ou não — como o catálogo é pequeno e totalmente exposto, não há ambiguidade sobre o que o cliente "viu"):

| $y_i$ | Condição |
|---|---|
| 1 | se $i \in S_{final}$ |
| 0 | caso contrário |

**Por que as 4 obrigatórias nunca entram nesse cálculo.** Se uma cobertura é genuinamente não-editável pelo cliente, $y_i$ para ela seria sempre 1, em toda observação, sem exceção — uma variável que nunca varia não ensina nada a $\theta$, só adiciona ruído à estimativa e um risco desnecessário de quase-separação numérica. As obrigatórias ficam inteiramente fora dos dados de treino.

**O exemplo rodando:**

| Cobertura opcional | Ofertada? | No final? | $y_i$ |
|---|---|---|---|
| Desastres Naturais | sim | sim | **1** |
| Roubo e Furto | sim | sim | **1** |
| Conteúdo Residencial | sim | não | **0** |
| Danos Elétricos | não | sim (espontânea) | **1** |
| (demais opcionais não ofertadas nem adicionadas) | não | não | **0** |

**O que entra / o que sai:** entra: $S_{ofertado}$, $S_{final}$. Sai: as observações $(x_i, y_i)$ de cada cobertura opcional, marcadas também quanto à origem (rota aleatória reservada para baseline, rota aleatória de treino, ou rota do modelo), para a Etapa 9 saber onde alocar cada uma.

---

## 9. Atualização em lote

**Analogia:** no fim do ciclo (diário, ou por lote de N clientes), a régua $\theta$ é recalibrada do zero usando todas as fichas acumuladas desde sempre — e, separadamente, as taxas de referência de cada cobertura também são recalculadas, usando só a fatia que sempre foi reservada para isso.

**A matemática — reajuste de $\theta$:**

$$\hat\theta = \arg\max_\theta \sum_{n=1}^{N} \Big[y_n \log p_n + (1-y_n)\log(1-p_n)\Big] - \frac{\lambda}{2}\|\theta\|^2, \qquad p_n=\sigma(\theta^Tx_n)$$

somando sobre **todas** as observações de coberturas opcionais acumuladas até o momento — vindas tanto da rota do modelo quanto da parte **não-reservada** das rodadas aleatórias (a fatia $\rho$ reservada para o baseline nunca entra aqui).

**Por que é preciso reprocessar tudo, em vez de simplesmente somar o lote novo ao estado antigo.** Diferente de um ajuste por mínimos quadrados (que teria solução fechada, somável incrementalmente), a regressão logística exige um otimizador a cada reajuste — não existe uma forma de atualizar $\hat\theta$ incrementalmente sem reprocessar o histórico completo. Dado o tamanho do catálogo (10 coberturas opcionais), isso continua sendo computacionalmente trivial — segundos, não minutos — mesmo reprocessando meses de dados.

**A matemática — reajuste do baseline:** recalcula-se $\text{baseline}_i$ e $n_i$ (Etapa 4) usando toda a fatia reservada acumulada até o momento, do mesmo jeito.

**A divisão entre "dados para $\theta$" e "dados reservados para o baseline" pode ser redesenhada a cada ciclo.** Como o reajuste de $\theta$ é sempre uma computação nova e autocontida, não existe exigência de que a mesma observação histórica vá sempre para o mesmo grupo em diferentes ciclos, nem de que a fração $\rho$ seja fixa ao longo do tempo — só precisa valer, dentro de um mesmo ciclo, que a divisão seja decidida antes de conhecer o resultado observado.

**O que entra / o que sai:** entra: todas as observações acumuladas, já marcadas por origem (Etapa 8). Sai: $\hat\theta$ atualizado e as taxas $\text{baseline}_i$/$\tau_i$ atualizadas, voltando para as Etapas 3 e 4 da próxima rodada.

---

## 10. O ciclo se fecha

**Analogia:** o próximo cliente que chegar já se beneficia do que foi aprendido — a régua $\theta$ está um pouco mais calibrada, e as taxas de referência de cada cobertura têm um pouco mais de precisão.

**A matemática:** volta-se à Etapa 1 com um novo cliente, usando $\hat\theta$, $\text{baseline}_i$ e $\tau_i$ já atualizados na Etapa 9.

---

## Mapa de conexões entre as etapas

```
Etapa 0 (setup)
     │  θ zerado, λ, ε_t, ρ definidos
     ▼
Etapa 1 (cliente chega) ──> c
     ▼
Etapa 2 (explorar ou usar modelo, via Z_t) ──> pacote aleatório (→ Etapa 6) OU segue
     ▼
Etapa 3 (score da cobertura, regressão logística) ←── θ (Etapa 9 anterior) + c
     ▼
Etapa 4 (threshold por cobertura, Wilson) ←── dados reservados acumulados
     ▼
Etapa 5 (decisão: obrigatórias + quem passa no threshold) ──> S_ofertado
     ▼
Etapa 6 (oferta) ──> S_ofertado exposto ao cliente
     ▼
Etapa 7 (cliente edita) ──> S_final
     ▼
Etapa 8 (traduz em y_i, só para opcionais, marcado por origem)
     ▼
Etapa 9 (reajuste em lote: θ e baseline/τ, separadamente) ──> novo θ, baseline, τ
     │
     └──────────────► volta para a Etapa 3 da próxima rodada (Etapa 10)
```

---

## Tabela-síntese: o que cada peça resolve

| Peça | O que resolve |
|---|---|
| $\sigma(\cdot)$ (funil logístico) | garante que a recompensa esperada nunca sai do intervalo entre 0 e 1 |
| MAP com priori gaussiana | estima $\theta$ de forma regularizada, protegendo contra estimativas extremas com poucos dados |
| $\theta$ compartilhado entre as 10 opcionais | permite generalizar entre coberturas parecidas, em vez de aprender cada uma isoladamente do zero |
| $Z_t\sim\text{Bernoulli}(\epsilon_t)$ | única fonte de aleatoriedade deliberada — garante que toda cobertura continue sendo testada, mesmo as já bem avaliadas |
| $\tau_i$ via Wilson | threshold calibrado a partir dos próprios dados, sem exigir peso de negócio arbitrário; mais conservador quando há poucos dados, convergindo para o baseline puro conforme a amostra cresce |
| Reserva $\rho$ para o baseline | garante que a comparação $\hat p(i)$ vs. $\tau_i$ seja entre duas estimativas estatisticamente independentes |
| Exclusão das obrigatórias do treino | evita ruído e risco de quase-separação numérica em variáveis que nunca têm variação para aprender |
| Reajuste em lote completo | necessário porque a regressão logística não tem solução fechada — não é um detalhe de implementação, é uma propriedade do modelo |

---

## Parâmetros que precisam ser definidos

| Parâmetro | Onde aparece | O que controla | Ponto de partida sugerido |
|---|---|---|---|
| $\lambda$ (regularização) | Etapas 0 e 9 | O quanto $\theta$ fica "presa" perto de zero antes de haver evidência forte | Testar valores entre 0,5 e 2,0 |
| $\epsilon_t$ e seu cronograma | Etapa 2 | Por quanto tempo, e com que intensidade, a oferta aleatória continua acontecendo | Começar próximo de 1,0 e decair até um piso pequeno (ex.: 0,05) — nunca chegar a zero |
| $\rho$ (fração reservada para o baseline) | Etapas 4 e 9 | Quanto do dado aleatório fica reservado só para calibrar o threshold, em vez de treinar $\theta$ | Um ponto de partida razoável é 30%, ajustável conforme o volume de tráfego |
| $z$ (nível de confiança de Wilson) | Etapa 4 | Quão conservador é o threshold em relação ao baseline puro | 1,645 (95% de confiança, um lado) é um ponto de partida padrão |
| Lista de coberturas obrigatórias | Etapas 5 e 8 | Quais coberturas nunca passam pela decisão do modelo | Definida pela equipe de produto, a partir das regras contratuais do produto |
| Dimensões semânticas de $f_i$ | Etapa 3 | Quais atributos descrevem cada cobertura | Definidas com a equipe atuarial (categoria de risco, faixa de prêmio, frequência de sinistro) |
| Dimensões do perfil do cliente $c$ | Etapa 1 | Quais atributos do cliente entram no modelo | Definidas com a equipe de dados, a partir do que já está disponível no cadastro |
