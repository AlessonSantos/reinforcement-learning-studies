# Recomendação de pacotes de coberturas de seguro: bandit contextual com regressão logística e teste de significância
### Guia matemático e didático

---

## Como usar este documento

Cada seção segue o mesmo padrão: primeiro uma **analogia** para visualizar a ideia sem fórmula nenhuma, depois **a matemática exata** por trás dela, depois um **exemplo rodando** (o mesmo cliente fictício, "Cliente A", acompanhado do início ao fim), e uma caixa **"o que entra / o que sai"** conectando cada etapa à anterior e à seguinte.

**O catálogo:** 14 coberturas no total — **4 obrigatórias** (sempre incluídas no pacote, nunca decididas pelo algoritmo) e **10 opcionais** (cada uma decidida individualmente, com base no perfil do cliente).

---

## 0. Setup — antes de existir qualquer cliente

**Analogia:** imagine uma única régua, $\theta$, compartilhada entre as 10 coberturas opcionais — ela vai aprender a traduzir "perfil do cliente + atributos de uma cobertura" numa nota de propensão. No início, a régua não sabe nada; ela vai sendo calibrada conforme os clientes interagem.

**A matemática:**

$$\theta \in \mathbb{R}^d, \qquad \theta \sim \mathcal N(0,\ \lambda^{-1}I), \qquad H_\theta = \lambda I$$

| Símbolo | Descrição |
|---|---|
| $\theta$ | régua compartilhada entre as 10 coberturas opcionais: perfil do cliente + atributos da cobertura → propensão de aceitação |
| $\lambda$ | regularização — o quanto $\theta$ começa "travada" perto de zero antes de haver evidência forte, e o quanto continua puxada para zero mesmo depois, protegendo contra estimativas extremas |
| $H_\theta$ | matriz de incerteza sobre $\theta$ — "quanto já se aprendeu, e em quais direções", usada para medir a confiança de cada estimativa (Etapa 3) |
| $\nu$ | piso mínimo de curvatura usado ao atualizar $H_\theta$ (Etapa 9) — evita que coberturas com propensão muito próxima de 0 ou 1 parem de reduzir sua própria incerteza |
| $\epsilon_t$ | probabilidade de explorar aleatoriamente nesta rodada (ver Etapa 2) |
| $\rho$ | fração das rodadas aleatórias reservada exclusivamente para calcular a taxa de referência de cada cobertura (ver Etapa 4) — nunca usada para treinar $\theta$ |

**O que entra / o que sai:** nada entra. Sai: $\theta$ e $H_\theta$ zerados, e os parâmetros de configuração — $\lambda$, $\nu$, $\epsilon_t$, $\rho$ — definidos, prontos para a Rodada 1.

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

$$x(i) = \text{concat}\big(c, f_i, c \otimes f_i\big)$$

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

**De onde vêm $\theta$ e $H_\theta$, e por que a recompensa esperada já é uma probabilidade calibrada.** $\theta$ e $H_\theta$ não são calculados aqui — chegam prontos do reajuste mais recente (Etapa 9), onde o desenvolvimento completo é apresentado. O que importa para esta etapa é entender **por que** $\hat p(i)=\sigma(\theta^Tx(i))$ pode ser lida como uma probabilidade de verdade, e não um número arbitrário: $\theta$ é estimado maximizando diretamente a verossimilhança de um modelo probabilístico bem especificado para dados binários (não minimizando erro quadrático, como seria numa regressão comum) — é essa forma de ajuste que garante, sob boas condições, que a recompensa esperada já saia calibrada, sem precisar de nenhuma correção posterior. $H_\theta$, por sua vez, mede a incerteza restante sobre $\theta$ — e dela deriva-se $\text{Var}[\eta(i)] = x(i)^TH_\theta^{-1}x(i)$, a incerteza sobre o score daquele cliente para aquela cobertura, usada no teste de significância da Etapa 5.

**O exemplo rodando:** suponha que, para o Cliente A, o modelo já calculou $\eta(\text{Desastres Naturais})=0{,}90$, com $\text{Var}[\eta(\text{Desastres})] = 0{,}05$ (uma incerteza relativamente baixa, refletindo que $\theta$ já viu bastante dado informativo nessa região do espaço de contexto). A recompensa esperada é:

$$\hat p(\text{Desastres Naturais}) = \sigma(0{,}90) \approx 0{,}71$$

**O que entra / o que sai:** entra: $c$, $\theta$ e $H_\theta$ (do reajuste mais recente — Etapa 9). Sai: $\eta(i)$ e $\text{Var}[\eta(i)]$ para cada uma das 10 coberturas opcionais, usados na Etapa 5.

---

## 4. A taxa de referência de cada cobertura

**Analogia:** antes de decidir se vale a pena confiar na régua do modelo para uma cobertura específica, é preciso saber qual seria a taxa de aceitação **se ninguém estivesse decidindo nada** — só para ter certeza de que o modelo está identificando algo além do óbvio.

**Por que não usar 0,5 fixo para todas as coberturas.** Um ponto de corte fixo pressupõe implicitamente que aceitar e rejeitar uma oferta têm o mesmo peso de decisão para toda cobertura — uma suposição que não há motivo para ser verdadeira: cada cobertura pode ter uma taxa de aceitação "de base" bem diferente das demais. Em vez de impor um valor de negócio arbitrário, a abordagem adotada aqui deriva a referência diretamente dos dados: **só vale a pena confiar na segmentação do modelo para um cliente específico se a propensão estimada para ele for genuinamente maior do que a taxa de aceitação típica daquela cobertura**, quando ofertada sem nenhuma seleção — "genuinamente maior" sendo decidido por um teste de significância (Etapa 5), não por comparação direta de dois números.

**A matemática — a taxa de referência e sua incerteza:**

$$\text{baseline}_i = \frac{\text{aceitações de } i \text{ nas rodadas reservadas}}{n_i}$$

calculada **só** sobre uma fração $\rho$ das rodadas aleatórias ($Z_t=1$), reservada exclusivamente para este cálculo — nunca usada para treinar $\theta$ (Etapa 9), e nunca misturada com rodadas da rota do modelo.

$$\text{logit}(\text{baseline}_i) = \log\left(\frac{\text{baseline}_i}{1-\text{baseline}_i}\right), \qquad \text{Var}\big[\text{logit}(\text{baseline}_i)\big] \approx \frac{1}{n_i\cdot\text{baseline}_i\cdot(1-\text{baseline}_i)}$$

| Símbolo | Descrição |
|---|---|
| $n_i$ | número de observações reservadas para a cobertura $i$ |
| $\text{baseline}_i$ | proporção amostral pura — frequência de aceitação sem nenhuma seleção |
| $\text{logit}(\text{baseline}_i)$ | a mesma taxa de referência, convertida para a escala log-odds — a mesma escala em que $\eta(i)$ vive, necessária para comparar os dois de forma consistente na Etapa 5 |
| $\text{Var}[\text{logit}(\text{baseline}_i)]$ | a incerteza dessa taxa de referência, via método delta — maior quando $n_i$ é pequeno, ou quando $\text{baseline}_i$ está perto de 0 ou 1 |

**Por que essa reserva é necessária, e por que a divisão precisa ser independente do resultado.** Se as mesmas observações alimentassem tanto o treino de $\theta$ quanto o cálculo do baseline, os dois deixariam de ser estatisticamente independentes — a comparação entre eles ficaria enviesada. A separação evita isso por construção. A regra que decide se uma rodada aleatória vai para a reserva do baseline ou para o treino de $\theta$ precisa depender só de algo definido **antes** de conhecer a resposta do cliente (por exemplo, um identificador da rodada, nunca do resultado observado).

**O exemplo rodando:** suponha que, na fatia reservada, "Desastres Naturais" foi ofertada aleatoriamente $n_i=120$ vezes, com $\text{baseline}_i=0{,}40$ (40% de aceitação):

$$\text{logit}(0{,}40) \approx -0{,}405, \qquad \text{Var}\big[\text{logit}(0{,}40)\big] = \frac{1}{120\times 0{,}40\times 0{,}60} \approx 0{,}0347$$

**O que entra / o que sai:** entra: as observações reservadas de cada cobertura (acumuladas desde o início, atualizadas na Etapa 9). Sai: $\text{logit}(\text{baseline}_i)$ e sua variância, para cada uma das 10 coberturas opcionais, usados na Etapa 5.

---

## 5. A decisão: obrigatórias + o que passar no teste de significância

**Analogia:** a mala já vem com um compartimento fixo, sempre presente (as obrigatórias). Para cada peça opcional, a pergunta não é só "a estimativa do modelo é maior que a referência?" — é "essa diferença é grande o suficiente para confiar nela, dado o quanto ainda somos incertos dos dois lados?".

**A matemática — a diferença e sua incerteza combinada:**

$$D(i) = \eta(i) - \text{logit}(\text{baseline}_i)$$

$$\text{Var}[D(i)] = \text{Var}[\eta(i)] + \text{Var}\big[\text{logit}(\text{baseline}_i)\big]$$

**A regra de decisão** — oferta se o limite inferior de confiança da diferença for maior que zero:

$$D(i) - z\sqrt{\text{Var}[D(i)]} > 0, \qquad z=1{,}645 \text{ (95 por cento de confiança, um lado)}$$

$$S_{ofertado} = \lbrace \text{4 obrigatórias}\rbrace \cup \Big\lbrace i \in \lbrace \text{10 opcionais}\rbrace : D(i) - z\sqrt{\text{Var}[D(i)]} > 0 \Big\rbrace$$

| Símbolo | Descrição |
|---|---|
| $D(i)$ | a diferença entre o score do modelo e a referência, na mesma escala (log-odds) |
| $\text{Var}[D(i)]$ | soma das duas incertezas — a de $\eta(i)$ (Etapa 3, via $H_\theta$) e a do baseline (Etapa 4, via método delta) |
| A regra | exige não só que $\hat p(i)$ pareça maior que $\text{baseline}_i$, mas que essa diferença resista a subtrair uma margem proporcional à incerteza combinada — quanto mais incerto qualquer um dos dois lados, maior a margem exigida |

**Por que somar as duas variâncias, e não comparar cada lado separadamente.** Tratar a incerteza de $\eta(i)$ e a incerteza do baseline como dois intervalos separados (checando se eles não se sobrepõem) seria estatisticamente mais conservador do que necessário — não equivale a um teste de diferença apropriado. Somar as variâncias e testar a diferença diretamente é a forma correta: usa exatamente a informação necessária, nem mais, nem menos.

**O exemplo rodando:** para Desastres Naturais, com $\eta=0{,}90$, $\text{Var}[\eta]=0{,}05$ (Etapa 3), e $\text{logit}(\text{baseline})=-0{,}405$, $\text{Var}[\text{logit}(\text{baseline})]=0{,}0347$ (Etapa 4):

$$D(\text{Desastres}) = 0{,}90 - (-0{,}405) = 1{,}305$$

$$\text{Var}[D] = 0{,}05 + 0{,}0347 = 0{,}0847, \qquad \sqrt{\text{Var}[D]} \approx 0{,}291$$

$$D - z\sqrt{\text{Var}[D]} = 1{,}305 - 1{,}645\times 0{,}291 \approx 0{,}826 > 0 \quad\Rightarrow\quad \textbf{oferta}$$

Para Danos Elétricos, suponha $\eta=0{,}201$, $\text{Var}[\eta]=0{,}09$ (uma cobertura ainda menos testada), e $\text{baseline}=0{,}50$ com $n=80$ (logo $\text{logit}(\text{baseline})=0$, $\text{Var}[\text{logit}(\text{baseline})]=0{,}05$):

$$D(\text{Danos Elétricos}) = 0{,}201 - 0 = 0{,}201, \qquad \text{Var}[D] = 0{,}09+0{,}05=0{,}14$$

$$D - z\sqrt{\text{Var}[D]} = 0{,}201 - 1{,}645\times 0{,}374 \approx -0{,}414 < 0 \quad\Rightarrow\quad \textbf{não oferta}$$

Mesmo com $\eta(\text{Danos Elétricos})>\text{logit}(\text{baseline})$ (a estimativa pontual do modelo é maior que a referência), a diferença não é grande o suficiente diante da incerteza combinada — o teste corretamente recusa confiar nela ainda.

$$S_{ofertado} = \lbrace \text{4 obrigatórias}\rbrace \cup \lbrace \text{Desastres Naturais}, \ldots\rbrace$$

O tamanho do pacote **emerge** dessa comparação item a item — não há $k$ fixo em nenhum lugar da equação.

**O que entra / o que sai:** entra: $\eta(i)$, $\text{Var}[\eta(i)]$ (Etapa 3) e $\text{logit}(\text{baseline}_i)$, $\text{Var}[\text{logit}(\text{baseline}_i)]$ (Etapa 4), para cada opcional. Sai: $S_{ofertado}$.

---

## 6. Oferta ao cliente

**Analogia:** a mala está pronta — hora de mostrar ao Cliente A.

**O exemplo rodando:** $S_{ofertado} = \lbrace$4 obrigatórias, Desastres Naturais, Roubo e Furto, Conteúdo Residencial$\rbrace$.

**O que entra / o que sai:** entra: $S_{ofertado}$ (Etapa 5). Sai: o mesmo conjunto, exposto ao cliente.

---

## 7. Cliente edita: mantém, remove, adiciona

**Analogia:** o Cliente A olha a mala pronta e mexe nela: tira uma peça, mantém outras, e às vezes pega algo que nem estava na oferta.

**O exemplo rodando:** o Cliente A mantém Desastres Naturais e Roubo e Furto, remove Conteúdo Residencial, e adiciona Danos Elétricos por conta própria (mesmo não tendo sido ofertada).

**O que entra / o que sai:** entra: $S_{ofertado}$ (Etapa 6). Sai: $S_{final}$, o pacote realmente contratado.

---

## 8. Traduzindo a edição em dado de treino

**Analogia:** cada cobertura mantida, tirada ou acrescentada vira uma "ficha" no arquivo — inclusive as obrigatórias, porque "obrigatória" aqui significa que a lei exige que ela seja **ofertada**, não que o cliente seja impedido de recusá-la. Existe decisão real do cliente para aprender, mesmo nessas 4.

**A matemática:** para **todas** as 14 coberturas visíveis ao cliente (ofertada ou não — como o catálogo é pequeno e totalmente exposto, não há ambiguidade sobre o que o cliente "viu"):

| $y_i$ | Condição |
|---|---|
| 1 | se $i \in S_{final}$ |
| 0 | caso contrário |

**Por que as 4 obrigatórias entram no treino de $\theta$, mesmo nunca precisando do teste de significância.** Como $\theta$ é compartilhado entre todas as 14 coberturas, os dados de aceitação/recusa das obrigatórias ainda ensinam algo útil sobre como perfil de cliente e atributos de cobertura se relacionam — informação que se generaliza para as 10 opcionais, mesmo sem $\eta(i)$ nunca ser usado para decidir se uma obrigatória entra no pacote (ela sempre entra, por exigência legal, independente do que o modelo estimar). Excluir esses dados do treino jogaria fora informação de generalização sem necessidade — a única coisa que as obrigatórias **não** fazem é passar pelo teste da Etapa 5.

**O exemplo rodando:**

| Cobertura | Ofertada? | No final? | $y_i$ |
|---|---|---|---|
| Incêndio (obrigatória) | sim | sim | **1** |
| Responsabilidade Civil (obrigatória) | sim | não | **0** — cliente recusou, mesmo sendo ofertada por exigência legal |
| Desastres Naturais (opcional) | sim | sim | **1** |
| Roubo e Furto (opcional) | sim | sim | **1** |
| Conteúdo Residencial (opcional) | sim | não | **0** |
| Danos Elétricos (opcional) | não | sim (espontânea) | **1** |
| (demais opcionais não ofertadas nem adicionadas) | não | não | **0** |

**O que entra / o que sai:** entra: $S_{ofertado}$, $S_{final}$. Sai: as observações $(x_i, y_i)$ de todas as 14 coberturas, marcadas quanto à origem (rota aleatória reservada para baseline, rota aleatória de treino, ou rota do modelo), para a Etapa 9 saber onde alocar cada uma — lembrando que só as 10 opcionais passam pelo teste da Etapa 5, já que as obrigatórias nunca entram nessa comparação.

---

## 9. Atualização em lote

**Analogia:** no fim do ciclo (diário, ou por lote de N clientes), a régua $\theta$ é recalibrada do zero usando todas as fichas acumuladas desde sempre — e, separadamente, as taxas de referência de cada cobertura também são recalculadas, usando só a fatia que sempre foi reservada para isso.

**A matemática — reajuste de $\theta$, por máxima verossimilhança a posteriori (MAP).** $\theta$ não é ajustado minimizando erro quadrático — é ajustado maximizando a verossimilhança de observações binárias (Bernoulli), combinada com a priori gaussiana definida na Etapa 0. Pela regra de Bayes:

$$P(\theta\mid\text{dados}) \propto \underbrace{P(\text{dados}\mid\theta)}_{\text{verossimilhança}} \times \underbrace{P(\theta)}_{\text{priori}}$$

A verossimilhança de uma única observação $(x_n, y_n)$, com $y_n\in\lbrace 0,1\rbrace$ e $p_n=\sigma(\theta^Tx_n)$, segue uma distribuição de Bernoulli: $P(y_n\mid\theta) = p_n^{y_n}(1-p_n)^{1-y_n}$ — a expressão vale $p_n$ quando $y_n=1$ e $1-p_n$ quando $y_n=0$, cobrindo os dois casos numa única fórmula. Tomando o logaritmo da verossimilhança conjunta de todas as observações e somando o logaritmo da priori, chega-se à log-posteriori, a menos de uma constante aditiva que não depende de $\theta$:

$$\log P(\theta\mid\text{dados}) = \sum_{n=1}^{N}\Big[y_n\log p_n + (1-y_n)\log(1-p_n)\Big] - \frac{\lambda}{2}\Vert \theta\Vert ^2 + \text{constante}$$

$$\hat\theta = \arg\max_\theta \sum_{n=1}^{N} \Big[y_n \log p_n + (1-y_n)\log(1-p_n)\Big] - \frac{\lambda}{2}\Vert \theta\Vert ^2, \qquad p_n=\sigma(\theta^Tx_n)$$

somando sobre **todas** as observações de coberturas opcionais acumuladas até o momento — vindas tanto da rota do modelo quanto da parte **não-reservada** das rodadas aleatórias (a fatia $\rho$ reservada para o baseline nunca entra aqui). É essa forma de ajuste — otimizar diretamente um modelo probabilístico bem especificado, em vez de uma métrica substituta como erro quadrático — que garante, sob boas condições, que $\sigma(\hat\theta^Tx(i))$ já saia calibrada como probabilidade (a propriedade usada, sem prova, na Etapa 3).

**Por que é preciso reprocessar tudo, em vez de simplesmente somar o lote novo ao estado antigo.** Combinar uma verossimilhança de Bernoulli com uma priori gaussiana não produz uma posteriori de nenhuma família conhecida — essa não é uma peculiaridade deste problema específico, é uma propriedade da regressão logística bayesiana em geral, bem documentada na literatura estatística clássica. Como $p_n=\sigma(\theta^Tx_n)$ entra no logaritmo de forma não-linear, a derivada da log-posteriori em relação a $\theta$ também é não-linear — não dá para isolar $\theta$ algebricamente, diferente de um ajuste por mínimos quadrados (que teria solução fechada, somável incrementalmente). Por isso, encontrar $\hat\theta$ exige um otimizador numérico iterativo a cada reajuste (o gradiente usado é $\nabla_\theta = \sum_n x_n(y_n-p_n) - \lambda\theta$) — não existe forma de atualizar $\hat\theta$ incrementalmente sem reprocessar o histórico completo. Dado o tamanho do catálogo (10 coberturas opcionais), isso continua sendo computacionalmente trivial — segundos, não minutos — mesmo reprocessando meses de dados.

**A matemática — reajuste de $H_\theta$, via aproximação de Laplace, no ponto ótimo recém-encontrado.** Como a posteriori não tem forma fechada (não é conjugada), aproxima-se a incerteza em torno de $\hat\theta$ por uma gaussiana local, com covariância dada pelo inverso da curvatura da log-posteriori:

$$H_\theta = \lambda I + \sum_{n=1}^{N} \max\big(p_n(1-p_n),\ \nu\big) x_nx_n^{T}$$

| Símbolo | Descrição |
|---|---|
| $p_n(1-p_n)$ | quanto cada observação de treino contribui para reduzir a incerteza — máximo quando $p_n\approx 0{,}5$, mínimo perto de 0 ou 1 |
| $\nu$ | um piso mínimo para esse peso — sem ele, itens com histórico consistentemente extremo (quase sempre aceitos ou quase sempre rejeitados) parariam de reduzir sua própria incerteza, mesmo com muitos dados confirmando o mesmo padrão repetidamente. É o piso que garante a variância continuar encolhendo conforme mais dados chegam, inclusive nessas regiões |

**A matemática — reajuste do baseline:** recalcula-se $\text{baseline}_i$, $n_i$, $\text{logit}(\text{baseline}_i)$ e sua variância (Etapa 4) usando toda a fatia reservada acumulada até o momento, do mesmo jeito.

**A divisão entre "dados para $\theta$" e "dados reservados para o baseline" pode ser redesenhada a cada ciclo.** Como o reajuste de $\theta$ é sempre uma computação nova e autocontida, não existe exigência de que a mesma observação histórica vá sempre para o mesmo grupo em diferentes ciclos, nem de que a fração $\rho$ seja fixa ao longo do tempo — só precisa valer, dentro de um mesmo ciclo, que a divisão seja decidida antes de conhecer o resultado observado.

**O que entra / o que sai:** entra: todas as observações acumuladas, já marcadas por origem (Etapa 8). Sai: $\hat\theta$, $H_\theta$, $\text{baseline}_i$ e $\text{logit}(\text{baseline}_i)$ atualizados, voltando para as Etapas 3 e 4 da próxima rodada.

---

## 10. O ciclo se fecha

**Analogia:** o próximo cliente que chegar já se beneficia do que foi aprendido — a régua $\theta$ está um pouco mais calibrada, e as taxas de referência de cada cobertura têm um pouco mais de precisão, com incertezas um pouco menores dos dois lados.

**A matemática:** volta-se à Etapa 1 com um novo cliente, usando $\hat\theta$, $H_\theta$, $\text{baseline}_i$ e sua variância já atualizados na Etapa 9.

---

## Mapa de conexões entre as etapas

```
Etapa 0 (setup)
     │  θ zerado, λ, ν, ε_t, ρ definidos
     ▼
Etapa 1 (cliente chega) ──> c
     ▼
Etapa 2 (explorar ou usar modelo, via Z_t) ──> pacote aleatório (→ Etapa 6) OU segue
     ▼
Etapa 3 (score da cobertura + H_θ, regressão logística) ←── θ, H_θ (Etapa 9 anterior) + c
     ▼
Etapa 4 (taxa de referência + sua variância) ←── dados reservados acumulados
     ▼
Etapa 5 (decisão: obrigatórias + teste de significância) ──> S_ofertado
     ▼
Etapa 6 (oferta) ──> S_ofertado exposto ao cliente
     ▼
Etapa 7 (cliente edita) ──> S_final
     ▼
Etapa 8 (traduz em y_i, todas as 14, marcado por origem)
     ▼
Etapa 9 (reajuste em lote: θ, H_θ e baseline, separadamente) ──> novo θ, H_θ, baseline
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
| Piso de curvatura $\nu$ em $H_\theta$ | evita que itens com histórico consistentemente extremo parem de reduzir sua própria incerteza — mantém a variância encolhendo com mais dados, mesmo em regiões de alta certeza |
| Teste de significância ($D(i)$ vs. incerteza combinada) | decide se a diferença entre modelo e referência é grande o suficiente para confiar nela, sem exigir peso de negócio arbitrário e sem depender só de uma comparação pontual |
| Reserva $\rho$ para o baseline | garante que a comparação entre $\eta(i)$ e o baseline seja entre duas estimativas estatisticamente independentes |
| Obrigatórias incluídas no treino, fora do teste | aproveita a variação real de aceitação/recusa das obrigatórias para melhorar a generalização de $\theta$, sem nunca usar o teste para decidir se elas entram no pacote |
| Reajuste em lote completo | necessário porque a regressão logística não tem solução fechada — não é um detalhe de implementação, é uma propriedade do modelo |

---

## Parâmetros que precisam ser definidos

| Parâmetro | Onde aparece | O que controla | Ponto de partida sugerido |
|---|---|---|---|
| $\lambda$ (regularização) | Etapas 0, 3 e 9 | O quanto $\theta$ fica "presa" perto de zero antes de haver evidência forte | Testar valores entre 0,5 e 2,0 |
| $\nu$ (piso de curvatura) | Etapas 3 e 9 | Evita que a variância de itens muito certos pare de encolher com mais dados | Algo entre 0,02 e 0,10 (referência: $p(1-p)$ tem máximo 0,25 em $p=0{,}5$) |
| $\epsilon_t$ e seu cronograma | Etapa 2 | Por quanto tempo, e com que intensidade, a oferta aleatória continua acontecendo | Começar próximo de 1,0 e decair até um piso pequeno (ex.: 0,05) — nunca chegar a zero |
| $\rho$ (fração reservada para o baseline) | Etapas 4 e 9 | Quanto do dado aleatório fica reservado só para calibrar a referência, em vez de treinar $\theta$ | Um ponto de partida razoável é 30%, ajustável conforme o volume de tráfego |
| $z$ (nível de confiança do teste) | Etapa 5 | Quão conservadora é a exigência de significância para ofertar | 1,645 (95% de confiança, um lado) é um ponto de partida padrão |
| Lista de coberturas obrigatórias | Etapas 5 e 8 | Quais coberturas nunca passam pelo teste do modelo | Definida pela equipe de produto, a partir das regras contratuais do produto |
| Dimensões semânticas de $f_i$ | Etapa 3 | Quais atributos descrevem cada cobertura | Definidas com a equipe atuarial (categoria de risco, faixa de prêmio, frequência de sinistro) |
| Dimensões do perfil do cliente $c$ | Etapa 1 | Quais atributos do cliente entram no modelo | Definidas com a equipe de dados, a partir do que já está disponível no cadastro |
