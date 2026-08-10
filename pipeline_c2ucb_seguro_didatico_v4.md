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
| $\epsilon$ | probabilidade de a rodada inteira ir pela rota aleatória — um valor **fixo**, definido pela equipe (ver Etapa 2) |
| $\rho$ | fração das rodadas aleatórias reservada exclusivamente para calcular a taxa de referência de cada cobertura (ver Etapa 4) — nunca usada para treinar $\theta$ |

**O que entra / o que sai:** nada entra. Sai: $\theta, H_\theta$ zerados, e os parâmetros de configuração — $\lambda, \nu, \epsilon, \rho$ — definidos, prontos para a Rodada 1.

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

$$Z_t \sim \text{Bernoulli}(\epsilon)$$

O índice $t$ indica o número da rodada.

| Se... | Então... |
|---|---|
| $Z_t=1$ (prob. $\epsilon$) | rota aleatória: sorteia-se um subconjunto entre as 10 coberturas opcionais — não cobertura por cobertura, o subconjunto inteiro é sorteado de uma vez — pula para a Etapa 6 |
| $Z_t=0$ (prob. $1-\epsilon$) | rota do modelo: segue para a Etapa 3 |

| Símbolo | Descrição |
|---|---|
| $Z_t$ | a única variável genuinamente aleatória que decide o caminho — sorteada de novo a cada rodada, nunca fixada por cliente |
| $\epsilon$ | a probabilidade de a rodada inteira ir pela rota aleatória — um valor **fixo**, definido pela equipe, que não decai automaticamente com o tempo. Precisa continuar **maior que zero** para sempre |

**Por que a rota aleatória precisa cobrir bem as 10 coberturas ao longo do tempo, mesmo sorteando o subconjunto inteiro de uma vez:** nesta versão, cada cobertura opcional é decidida individualmente na Etapa 5, sem nenhum termo que dependa de quantas outras já foram incluídas — então, ao longo de muitas rodadas, a rota aleatória precisa garantir que cada cobertura, isoladamente, apareça (e deixe de aparecer) um número suficiente de vezes, para que a Etapa 4 consiga estimar sua taxa de aceitação de referência com confiança.

**O exemplo rodando:** suponha que o sorteio desta rodada resultou em $Z_t=1$ — o pacote é montado direto pela rota aleatória (pule para a Etapa 6, e volte para a Etapa 3 depois para acompanhar a rota do modelo). Se tivesse dado $Z_t=0$, seguiríamos direto para a Etapa 3.

**O que entra / o que sai:** entra: $c, \epsilon$ (o valor fixo definido pela equipe). Sai: $Z_t$ sorteado, e a partir dele, sinal para calcular scores (Etapa 3) ou pacote aleatório (Etapa 6).

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

**O que entra / o que sai:** entra: $c, \theta, H_\theta$ (do reajuste mais recente — Etapa 9). Sai: $\eta(i), \text{Var}[\eta(i)]$ para cada uma das 10 coberturas opcionais, usados na Etapa 5.

---

## 4. A taxa de referência de cada cobertura

**Analogia:** antes de decidir se vale a pena confiar na régua do modelo para uma cobertura específica, é preciso saber qual seria a taxa de aceitação **se ninguém estivesse decidindo nada** — só para ter certeza de que o modelo está identificando algo além do óbvio.

**Por que não usar 0,5 fixo para todas as coberturas.** Um ponto de corte fixo pressupõe implicitamente que aceitar e rejeitar uma oferta têm o mesmo peso de decisão para toda cobertura — uma suposição que não há motivo para ser verdadeira: cada cobertura pode ter uma taxa de aceitação "de base" bem diferente das demais. Em vez de impor um valor de negócio arbitrário, a abordagem adotada aqui deriva a referência diretamente dos dados: **só vale a pena confiar na segmentação do modelo para um cliente específico se a propensão estimada para ele for genuinamente maior do que a taxa de aceitação típica daquela cobertura**, quando ofertada sem nenhuma seleção — "genuinamente maior" sendo decidido por um teste de significância (Etapa 5), não por comparação direta de dois números.

**A matemática — a taxa de referência (o "baseline") e sua incerteza.** Para abreviar a notação nas fórmulas seguintes, chamamos essa taxa de $b_i$:

$$b_i = \frac{\text{aceitações de } i \text{ nas rodadas reservadas}}{n_i}$$

calculada **só** sobre uma fração $\rho$ das rodadas aleatórias ($Z_t=1$), reservada exclusivamente para este cálculo — nunca usada para treinar $\theta$ (Etapa 9), e nunca misturada com rodadas da rota do modelo.

**Uma condição adicional, além da rodada ser reservada: a cobertura precisa ter sido genuinamente sorteada para aquela oferta.** $n_i$ e o numerador de aceitações de $b_i$ só contam observações em que a cobertura $i$ foi de fato incluída no sorteio independente da Etapa 2 — nunca uma adição espontânea do cliente (Etapa 7), mesmo que ela tenha acontecido numa rodada reservada. O motivo é o mesmo que já vale para o treino de $\theta$ com pares, só aplicado ao item isolado: $b_i$ precisa representar "a taxa de aceitação sob exposição exógena, sem nenhuma seleção" — uma adição espontânea é o oposto disso, é uma escolha do cliente, não um sorteio. Contá-la infla $b_i$ artificialmente, contaminando exatamente a referência que deveria ser neutra. Essa restrição vale só para $b_i$; a mesma adição espontânea continua alimentando o treino de $\theta$ normalmente (Etapa 8), porque ali o viés de auto-seleção é tolerável.

$$\text{logit}(b_i) = \log\left(\frac{b_i}{1-b_i}\right), \qquad \text{Var}\big[\text{logit}(b_i)\big] \approx \frac{1}{n_i b_i(1-b_i)}$$

| Símbolo | Descrição |
|---|---|
| $n_i$ | número de observações reservadas para a cobertura $i$ |
| $b_i$ | o baseline — proporção amostral pura, frequência de aceitação sem nenhuma seleção |
| $\text{logit}(b_i)$ | a mesma taxa de referência, convertida para a escala log-odds — a mesma escala em que $\eta(i)$ vive, necessária para comparar os dois de forma consistente na Etapa 5 |
| $\text{Var}[\text{logit}(b_i)]$ | a incerteza dessa taxa de referência, via método delta — maior quando o número de observações reservadas é pequeno, ou quando $b_i$ está perto de 0 ou 1 |

**Por que essa reserva é necessária, e por que a divisão precisa ser independente do resultado.** Se as mesmas observações alimentassem tanto o treino de $\theta$ quanto o cálculo do baseline, os dois deixariam de ser estatisticamente independentes — a comparação entre eles ficaria enviesada. A separação evita isso por construção. A regra que decide se uma rodada aleatória vai para a reserva do baseline ou para o treino de $\theta$ precisa depender só de algo definido **antes** de conhecer a resposta do cliente (por exemplo, um identificador da rodada, nunca do resultado observado).

**O exemplo rodando:** suponha que, na fatia reservada, "Desastres Naturais" foi ofertada aleatoriamente $n_i=120$ vezes, com $b_i=0{,}40$ (40% de aceitação):

$$\text{logit}(0{,}40) \approx -0{,}405, \qquad \text{Var}\big[\text{logit}(0{,}40)\big] = \frac{1}{120\times 0{,}40\times 0{,}60} \approx 0{,}0347$$

**O que entra / o que sai:** entra: as observações reservadas de cada cobertura (acumuladas desde o início, atualizadas na Etapa 9). Sai: $\text{logit}(b_i)$ e sua variância, para cada uma das 10 coberturas opcionais, usados na Etapa 5.

---

## 5. A decisão: obrigatórias + o que passar no teste de significância

**Analogia:** a mala já vem com um compartimento fixo, sempre presente (as obrigatórias). Para cada peça opcional, a pergunta não é só "a estimativa do modelo é maior que a referência?" — é "essa diferença é grande o suficiente para confiar nela, dado o quanto ainda somos incertos dos dois lados?".

**A matemática — a diferença e sua incerteza combinada:**

$$D(i) = \eta(i) - \text{logit}(b_i)$$

$$\text{Var}[D(i)] = \text{Var}[\eta(i)] + \text{Var}\big[\text{logit}(b_i)\big]$$

**A regra de decisão** — oferta se o limite inferior de confiança da diferença for maior que zero:

$$D(i) - z\sqrt{\text{Var}[D(i)]} > 0, \qquad z=1{,}645 \text{ (95 por cento de confiança, um lado)}$$

$$S_{ofertado} = \lbrace \text{4 obrigatórias}\rbrace \cup \Big\lbrace i \in \lbrace \text{10 opcionais}\rbrace : D(i) - z\sqrt{\text{Var}[D(i)]} > 0 \Big\rbrace$$

| Símbolo | Descrição |
|---|---|
| $D(i)$ | a diferença entre o score do modelo e a referência, na mesma escala (log-odds) |
| $\text{Var}[D(i)]$ | soma das duas incertezas — a de $\eta(i)$ (Etapa 3, via $H_\theta$) e a do baseline (Etapa 4, via método delta) |
| A regra | exige não só que $\hat p(i)$ pareça maior que $b_i$, mas que essa diferença resista a subtrair uma margem proporcional à incerteza combinada — quanto mais incerto qualquer um dos dois lados, maior a margem exigida |

**Por que somar as duas variâncias, e não comparar cada lado separadamente.** Tratar a incerteza de $\eta(i)$ e a incerteza do baseline como dois intervalos separados, checando se eles não se sobrepõem — uma prática comum, às vezes chamada de **método de sobreposição de intervalos** ("overlap method") —, seria estatisticamente mais conservador do que necessário, não equivale a um teste de diferença apropriado. Isso é um resultado bem documentado na literatura estatística: **Schenker & Gentleman (2001)** mostraram que, quando os dois erros-padrão são iguais, checar a sobreposição de dois intervalos de 95% cada corresponde, na prática, a um teste de diferença com nível de significância muito mais rigoroso que 95% — próximo de 99%, não 95% —, rejeitando a hipótese nula com muito mais dificuldade do que o nível nominal sugeriria. Somar as variâncias e testar a diferença diretamente é a forma correta: usa exatamente a informação necessária, nem mais, nem menos.

**O exemplo rodando:** para Desastres Naturais, com $\eta=0{,}90, \text{Var}[\eta]=0{,}05$ (Etapa 3), e $\text{logit}(b)=-0{,}405, \text{Var}[\text{logit}(b)]=0{,}0347$ (Etapa 4):

$$D(\text{Desastres}) = 0{,}90 - (-0{,}405) = 1{,}305$$

$$\text{Var}[D] = 0{,}05 + 0{,}0347 = 0{,}0847, \qquad \sqrt{\text{Var}[D]} \approx 0{,}291$$

$$D - z\sqrt{\text{Var}[D]} = 1{,}305 - 1{,}645\times 0{,}291 \approx 0{,}826 > 0 \quad\Rightarrow\quad \textbf{oferta}$$

Para Danos Elétricos, suponha $\eta=0{,}201, \text{Var}[\eta]=0{,}09$ (uma cobertura ainda menos testada), e $b=0{,}50$ com $n=80$ — logo $\text{logit}(b)=0, \text{Var}[\text{logit}(b)]=0{,}05$:

$$D(\text{Danos Elétricos}) = 0{,}201 - 0 = 0{,}201, \qquad \text{Var}[D] = 0{,}09+0{,}05=0{,}14$$

$$D - z\sqrt{\text{Var}[D]} = 0{,}201 - 1{,}645\times 0{,}374 \approx -0{,}414 < 0 \quad\Rightarrow\quad \textbf{não oferta}$$

Mesmo com $\eta(\text{Danos Elétricos})>\text{logit}(b)$ (a estimativa pontual do modelo é maior que a referência), a diferença não é grande o suficiente diante da incerteza combinada — o teste corretamente recusa confiar nela ainda.

$$S_{ofertado} = \lbrace \text{4 obrigatórias}\rbrace \cup \lbrace \text{Desastres Naturais}, \ldots\rbrace$$

O tamanho do pacote **emerge** dessa comparação item a item — não há $k$ fixo em nenhum lugar da equação.

**O que entra / o que sai:** entra: $\eta(i)$ e sua incerteza (Etapa 3), e a taxa de referência com sua incerteza (Etapa 4), para cada opcional. Sai: $S_{ofertado}$.

---

## 6. Oferta ao cliente

**Analogia:** a mala está pronta — hora de mostrar ao Cliente A.

**O exemplo rodando:** $S_{ofertado} = \lbrace \text{4 obrigatórias, Desastres Naturais, Roubo e Furto, Conteúdo Residencial}\rbrace$.

**O que entra / o que sai:** entra: $S_{ofertado}$ (Etapa 5). Sai: o mesmo conjunto, exposto ao cliente.

---

## 7. Cliente edita: mantém, remove, adiciona

**Analogia:** o Cliente A olha a mala pronta e mexe nela: tira uma peça, mantém outras, e às vezes pega algo que nem estava na oferta.

**O exemplo rodando:** o Cliente A mantém Desastres Naturais e Roubo e Furto, remove Conteúdo Residencial, e adiciona Danos Elétricos por conta própria (mesmo não tendo sido ofertada).

**O que entra / o que sai:** entra: $S_{ofertado}$ (Etapa 6). Sai: $S_{final}$, o pacote realmente contratado.

---

## 8. Traduzindo a edição em dado de treino

**Analogia:** cada cobertura opcional mantida, tirada ou acrescentada vira uma "ficha" no arquivo — as 4 obrigatórias nunca geram ficha nenhuma, porque não há decisão do cliente ali: obrigatória é sempre aceita, sem exceção, sem nenhuma variação para o modelo aprender.

**A matemática:** para cada uma das 10 coberturas opcionais visíveis ao cliente (ofertada ou não — como o catálogo é pequeno e totalmente exposto, não há ambiguidade sobre o que o cliente "viu"):

| $y_i$ | Condição |
|---|---|
| 1 | se $i \in S_{final}$ |
| 0 | caso contrário |

**Por que as 4 obrigatórias nunca entram nesse cálculo.** Como uma cobertura obrigatória é sempre mantida — $y_i=1$ em toda observação, sem exceção —, não existe nenhuma variação ali para $\theta$ aprender: uma variável que nunca muda não ensina nada sobre a relação entre perfil de cliente e atributos de cobertura, só adiciona ruído à estimativa e um risco desnecessário de quase-separação numérica (o mesmo cenário extremo que motiva a regularização $\lambda$). As obrigatórias ficam inteiramente fora dos dados de treino — elas entram no pacote final só pela Etapa 5, nunca pelo aprendizado de $\theta$.

**Como a origem de cada observação é marcada, usando $\rho$.** Toda rodada da rota aleatória ($Z_t=1$) é classificada em uma de duas categorias: uma fração $\rho$ delas, definida pela equipe, é marcada como "reservada para o baseline" (Etapa 4); o restante ($1-\rho$) como "rodada aleatória de treino", alimentando $\theta$ normalmente.

**O exemplo rodando:**

| Cobertura opcional | Ofertada? | No final? | $y_i$ |
|---|---|---|---|
| Desastres Naturais | sim | sim | **1** |
| Roubo e Furto | sim | sim | **1** |
| Conteúdo Residencial | sim | não | **0** |
| Danos Elétricos | não | sim (espontânea) | **1** |
| (demais opcionais não ofertadas nem adicionadas) | não | não | **0** |

**O que entra / o que sai:** entra: $S_{ofertado}, S_{final}$. Sai: as observações $(x_i, y_i)$ de cada uma das 10 coberturas opcionais, marcadas quanto à origem (rota aleatória reservada para baseline, rota aleatória de treino, ou rota do modelo), para a Etapa 9 saber onde alocar cada uma — as obrigatórias nunca geram observação nenhuma. Lembrete: mesmo numa rodada reservada, uma cobertura adicionada espontaneamente (fora de $S_{ofertado}$) alimenta $\theta$ normalmente, mas nunca entra no cálculo de $b_i$ (Etapa 4) — só contam ali as coberturas genuinamente sorteadas para a oferta.

---

## 9. Atualização em lote — e o achado importante sobre itens "quase certos"

**Analogia:** no fim do ciclo (diário, ou por lote de N clientes), a régua $\theta$ é recalibrada do zero usando todas as fichas acumuladas desde sempre — e, separadamente, as taxas de referência de cada cobertura também são recalculadas, usando só a fatia que sempre foi reservada para isso.

**A matemática do ajuste (MAP com prior gaussiano):**

$$\hat\theta = \arg\max_\theta \sum_{n=1}^{N} \Big[y_n \log p_n + (1-y_n)\log(1-p_n)\Big] - \frac{\lambda}{2}\Vert \theta\Vert ^2, \qquad p_n=\sigma(\theta^Tx_n)$$

somando sobre **todas** as observações acumuladas até o momento — vindas tanto da rota do modelo quanto da parte **não-reservada** das rodadas aleatórias (a fatia $\rho$ reservada para o baseline, Etapa 4, nunca entra aqui).

**De onde vem essa fórmula, passo a passo:**

O ponto de partida é a **regra de Bayes**, aplicada a $\theta$ em vez de a um evento simples:

$$P(\theta \mid \text{dados}) \propto P(\text{dados} \mid \theta) \times P(\theta)$$

(posteriori $\propto$ verossimilhança $\times$ priori)

| Termo | O que significa aqui |
|---|---|
| **Priori** $P(\theta)$ | o que acreditávamos sobre $\theta$ **antes** de ver qualquer dado — no nosso caso, $\theta \sim \mathcal{N}(0, \lambda^{-1}I)$: um "chute" de que os valores de $\theta$ tendem a ficar perto de zero, com $\lambda$ controlando o quão fortemente essa crença é mantida |
| **Verossimilhança** $P(\text{dados}\mid\theta)$ | o quão prováveis são os dados observados, **assumindo** um valor particular de $\theta$ — é a peça que faz o modelo "aprender" a partir dos dados |
| **Posteriori** $P(\theta\mid\text{dados})$ | a crença atualizada sobre $\theta$, depois de combinar o que já se sabia (priori) com o que os dados mostraram (verossimilhança) — é essa distribuição que queremos maximizar para achar $\hat\theta$ |

**Passo 1 — a verossimilhança de uma única observação.** Como $y_n \in \lbrace 0,1\rbrace$, cada observação segue uma distribuição de **Bernoulli** com parâmetro $p_n$ (a probabilidade de $y_n=1$). A fórmula fechada de uma Bernoulli é:

$$P(y_n \mid \theta) = p_n^{y_n}(1-p_n)^{1-y_n}$$

Repare no truque dessa fórmula: se $y_n=1$, o expoente de $(1-p_n)$ vira zero, então $(1-p_n)^0=1$ e sobra só $p_n$; se $y_n=0$, o expoente de $p_n$ vira zero e sobra só $(1-p_n)$. Ou seja, essa única expressão cobre os dois casos possíveis ao mesmo tempo.

**Passo 2 — a verossimilhança de todas as $N$ observações juntas.** Assumindo que cada observação é independente das demais (dado $\theta$), a verossimilhança conjunta é o **produto** das verossimilhanças individuais:

$$P(\text{dados}\mid\theta) = \prod_{n=1}^{N} p_n^{y_n}(1-p_n)^{1-y_n}$$

**Passo 3 — passar para o logaritmo.** Produtos de números entre 0 e 1 ficam numericamente instáveis (tendem a zero rapidinho) e são difíceis de derivar. Como $\log$ é uma função crescente, maximizar a verossimilhança é o mesmo que maximizar o **logaritmo** da verossimilhança — e o logaritmo transforma o produto em soma:

$$\log P(\text{dados}\mid\theta) = \sum_{n=1}^{N} \log\Big(p_n^{y_n}(1-p_n)^{1-y_n}\Big) = \sum_{n=1}^{N}\Big[y_n\log p_n + (1-y_n)\log(1-p_n)\Big]$$

usando a propriedade $\log(a^k)=k\log a$ e $\log(ab)=\log a + \log b$.

**Passo 4 — o logaritmo da priori.** A densidade de uma gaussiana $\mathcal{N}(0,\lambda^{-1}I)$ é proporcional a $\exp\big(-\frac{\lambda}{2}\Vert \theta\Vert ^2\big)$. Tomando o logaritmo, a parte que depende de $\theta$ é exatamente:

$$\log P(\theta) = -\frac{\lambda}{2}\Vert \theta\Vert ^2 + \text{constante (não depende de }\theta\text{)}$$

**Passo 5 — somar as duas partes.** Como $\log(\text{verossimilhança}\times\text{priori}) = \log(\text{verossimilhança}) + \log(\text{priori})$, a soma dá o log da posteriori **exatamente** — não é mais uma relação de proporcionalidade, é uma **igualdade**, só que com uma constante **somada** (em vez de multiplicada) que não depende de $\theta$:

$$\log P(\theta\mid\text{dados}) = \sum_{n=1}^{N}\Big[y_n\log p_n + (1-y_n)\log(1-p_n)\Big] - \frac{\lambda}{2}\Vert \theta\Vert ^2 + \text{constante}$$

Maximizar essa expressão em $\theta$ é exatamente a fórmula do início desta seção — o $\hat\theta$ resultante é chamado de estimativa **MAP** (*maximum a posteriori*: o valor de $\theta$ mais provável dado o que já foi observado e o que já se acreditava antes). É essa forma de ajuste — otimizar diretamente um modelo probabilístico bem especificado, em vez de uma métrica substituta como erro quadrático — que garante, sob boas condições, que $\sigma(\hat\theta^Tx(i))$ já saia calibrada como probabilidade (a propriedade usada, sem prova, na Etapa 3).

### Por que isso não tem solução fechada: o problema da conjugação

**O que é "conjugação".** Uma priori é dita **conjugada** com uma verossimilhança quando o produto das duas — a posteriori, antes de normalizar — resulta numa distribuição da **mesma família** da priori. Quando isso acontece, dá para escrever a posteriori inteira em forma fechada, exatamente, não só o ponto de pico dela.

**Se a verossimilhança fosse gaussiana (como numa regressão comum, com ruído contínuo), o ajuste teria solução fechada.** Nesse caso — verossimilhança gaussiana, $y_n\sim\mathcal N(\theta^Tx_n,\sigma^2)$, combinada com priori gaussiana — vale **Gaussiana × Gaussiana = Gaussiana**, a conjugação mais simples que existe. O $\hat\theta$ resultante não seria só "o pico" da posteriori — seria a **média exata** dela, com uma **covariância exata** também disponível em forma fechada, sem nenhuma aproximação.

**O nosso caso (Bernoulli × gaussiana) não é conjugado.** O produto de uma verossimilhança de Bernoulli (via sigmoide) por uma priori gaussiana não resulta em nenhuma distribuição de família conhecida — a posteriori existe (é uma distribuição de probabilidade válida), mas não sabemos escrever sua forma exata. Sabemos só que $P(\theta\mid\text{dados})\propto P(\text{dados}\mid\theta)P(\theta)$, até uma constante — e essa constante em si já é intratável de calcular exatamente. Essa não é uma peculiaridade deste problema específico — é uma propriedade da regressão logística bayesiana em geral, bem documentada na literatura estatística clássica.

**As duas consequências práticas disso, que são problemas distintos:**

1. **Achar o pico já exige um otimizador.** Como $p_n=\sigma(\theta^Tx_n)$ entra no logaritmo de forma não-linear, a derivada de $\log P(\theta\mid\text{dados})$ em relação a $\theta$ não é linear — não dá para isolar $\theta$ com álgebra simples, como aconteceria no caso gaussiano (onde a derivada é linear e vira um sistema resolvido por inversão direta de matriz). É por isso que a maximização precisa de um otimizador iterativo — o gradiente usado é $\nabla_\theta = \sum_n x_n(y_n-p_n) - \lambda\theta$ —, em vez de uma única inversão de matriz — e por isso não existe forma de atualizar $\hat\theta$ incrementalmente sem reprocessar o histórico completo a cada reajuste. Dado o tamanho do catálogo (10 coberturas opcionais), isso continua sendo computacionalmente trivial — segundos, não minutos — mesmo reprocessando meses de dados.

2. **Mesmo depois de achar o pico, não sabemos o formato exato da incerteza ao redor dele.** Esta é a parte mais sutil, e o motivo pelo qual precisamos da **aproximação de Laplace**: como a posteriori não tem forma fechada, não existe uma "covariância exata" para extrair. A aproximação de Laplace contorna isso **aproximando a posteriori real (intratável) por uma gaussiana**, centrada no pico $\hat\theta$ que acabamos de achar, com covariância igual ao inverso da curvatura (o Hessiano negativo) da log-posteriori naquele ponto — é isso que $H_\theta$ representa. A justificativa: perto do pico, qualquer função suave se parece com uma parábola (expansão de Taylor de segunda ordem), e o exponencial de uma parábola é, por definição, uma gaussiana. **$H_\theta$ não é a covariância exata da posteriori — é a covariância da aproximação gaussiana local a ela**, o que explica por que essa garantia é assintótica (fica mais precisa com mais dados, via o teorema de Bernstein–von Mises), não exata em amostra finita, como veremos a seguir no efeito kappa.

Aqui $\lambda$ aparece como o termo $-\frac{\lambda}{2}\Vert \theta\Vert ^2$: uma penalização que cresce conforme $\theta$ se afasta de zero. Na prática, isso impede que $\hat\theta$ "dispare" para valores extremos quando os dados são escassos ou quando uma cobertura é aceita/rejeitada de forma quase unânime — sem essa penalização, a otimização poderia tentar empurrar $\theta$ ao infinito para explicar perfeitamente um padrão determinístico nos dados.

### O achado importante: o efeito "kappa"

Ao calcular a incerteza (matriz $H_\theta$) via aproximação de Laplace, a fórmula "ingênua" seria:

$$H_\theta = \lambda I + \sum_{n=1}^{N} p_n(1-p_n) x_n x_n^{T}$$

**Analogia do problema:** imagine um jurado que só atualiza o quanto confia num veredito quando o caso é "difícil" (meio a meio). Se o caso é óbvio — todo mundo já sabe a resposta —, ele para de prestar atenção, mesmo vendo o mesmo padrão se confirmar centenas de vezes. É exatamente isso que $p_n(1-p_n)$ faz: esse peso é **máximo** quando $p_n\approx 0{,}5$ (caso "difícil") e **quase zero** quando $p_n$ está perto de 0 ou 1 (caso "óbvio" — ex.: uma cobertura quase sempre mantida).

**Por que isso não é um defeito da nossa implementação — é uma propriedade real da regressão logística.** À primeira vista, parece um paradoxo: por que a incerteza ficaria alta justamente para as coberturas mais bem confirmadas? A resposta exige separar duas perguntas que parecem a mesma coisa, mas não são: "estou confiante sobre o que vai acontecer?" — a probabilidade $p(i)$ — e "estou confiante sobre o valor exato de $\eta(i)$, o log-odds que gera essa probabilidade?" — essa segunda pergunta é o que $\text{Var}[\eta(i)]$ mede, e a resposta genuína pode ser "não", mesmo quando a primeira resposta é um "sim" categórico.

Isso acontece porque o funil $\sigma(\cdot)$ **satura** nos extremos: duas hipóteses bem diferentes de $\eta(i)$ — digamos, $\eta=4$ (equivale a $p\approx 0{,}982$) e $\eta=7$ (equivale a $p\approx 0{,}999$) — produzem, na prática, o mesmo comportamento observável: quase sempre "aceitou". Centenas de observações confirmam que $\eta(i)$ está "em algum lugar bem alto", mas não conseguem discriminar **exatamente quão alto**, porque uma faixa larga de valores de $\eta$ diferentes comprime para quase a mesma probabilidade. É isso que $p_n(1-p_n)$ mede de verdade: não "quão certo o resultado parece", mas "quanto essa observação ajuda a **localizar precisamente** $\eta$ no espaço log-odds" — e, perto dos extremos, ela ajuda genuinamente pouco. Se você removesse toda a engenharia deste documento e usasse a aproximação de Laplace exatamente como a teoria estatística clássica define, esse comportamento apareceria do mesmo jeito — não é uma falha de implementação, é assim que a matemática funciona.

**O que o piso $\nu$ realmente faz, então — uma decisão de engenharia, não uma correção estatística.** $\nu$ não torna a estimativa "mais certa" nem corrige um erro de cálculo: ele é uma escolha deliberada de tratar observações extremas como se carregassem um pouco mais de informação sobre $\eta$ do que elas genuinamente carregam, porque o custo prático de uma cobertura ficar presa (Etapa 5 nunca ganhando confiança suficiente para ofertá-la, mesmo com evidência esmagadora) supera o custo de abrir mão de um pouco de rigor estatístico. É uma troca consciente, não um "conserto" que devolve a resposta matematicamente correta.

**Por que isso pesa mais no teste de significância (Etapa 5) do que na probabilidade em si.** Para $\hat p(i)=\sigma(\eta(i))$ isoladamente, o próprio funil já comprime qualquer incerteza de $\eta$ para uma faixa estreita de probabilidade perto do extremo — então, na escala de probabilidade, o efeito quase desaparece sozinho. Ele vira um problema prático concreto justamente porque o teste de significância da Etapa 5 compara $D(i)$ na escala log-odds (para as variâncias poderem ser somadas de forma limpa) — é nessa escala, onde a saturação do funil não protege mais nada, que uma variância inflada travaria o teste de rejeitar $H_0$ mesmo diante de evidência esmagadora.

**Consequência prática, medida em teste real:** simulamos uma cobertura "quase certa" (sempre aceita) e observamos como a incerteza (e o bônus de exploração que ela alimentava, numa versão anterior deste algoritmo) evoluía conforme mais observações se acumulavam:

| Observações acumuladas | Sem correção | Com correção (piso $\nu$) |
|---|---|---|
| 1 | 1,02 | 1,02 |
| 50 | 0,59 | 0,56 |
| 100 | 0,55 | 0,42 |
| 300 | **0,50** (estacionou) | **0,25** (continua caindo) |

Sem correção, a incerteza **para de cair por volta de 0,50**, mesmo depois de 300 observações idênticas e consistentes — o modelo nunca "aprende a confiar" totalmente nesse item, mesmo tendo evidência esmagadora. Isso é uma manifestação prática do que a literatura de bandits para modelos logísticos chama de dependência em $\kappa$ (Filippi et al., 2010): as garantias formais desses modelos pioram exatamente quando as probabilidades reais estão perto dos extremos — o oposto do que a intuição sugeriria, e exatamente o problema que o teste de significância da Etapa 5 herdaria sem correção (uma cobertura muito bem avaliada nunca teria sua incerteza pequena o suficiente para o teste ganhar confiança).

### A correção: um piso na curvatura

$$H_\theta = \lambda I + \sum_{n=1}^{N} \max\big(p_n(1-p_n),\ \nu\big) x_n x_n^{T}$$

**Analogia da correção:** é como dizer ao jurado "mesmo em casos óbvios, você ainda precisa prestar um mínimo de atenção a cada novo caso" — garantindo que a confiança sempre cresce com mais dados, nunca estaciona artificialmente.

| Símbolo | Descrição |
|---|---|
| $p_n(1-p_n)$ | quanto aquela observação específica reduz a incerteza — máximo em casos "difíceis" (perto de 50/50), quase zero em casos "óbvios" |
| $\nu$ | piso mínimo — garante que mesmo observações de casos óbvios continuem contribuindo |
| $H_\theta$ (corrigido) | nova "memória", agora sem o platô artificial, alimentando o teste de significância da próxima rodada (Etapa 5) |

**Por que essa correção foi escolhida em vez de uma alternativa por contagem bruta:** o sistema é contextual — o mesmo item aparece em contextos diferentes (perfis de cliente diferentes). Uma correção baseada em "contar quantas vezes vi esse item" jogaria fora a generalização entre contextos parecidos, que é o motivo de $\theta$ ser compartilhado entre as 10 coberturas desde o início. O piso de curvatura, em vez disso, corrige o mecanismo exato onde o problema nasce (o peso da observação dentro de $H_\theta$), sem sair da geometria contextual do modelo.

**A matemática — reajuste do baseline:** recalcula-se a taxa de referência de cada cobertura e sua variância (Etapa 4), usando toda a fatia reservada acumulada até o momento, do mesmo jeito.

**A divisão entre "dados para $\theta$" e "dados reservados para o baseline" pode ser redesenhada a cada ciclo.** Como o reajuste de $\theta$ é sempre uma computação nova e autocontida, não existe exigência de que a mesma observação histórica vá sempre para o mesmo grupo em diferentes ciclos, nem de que a fração $\rho$ seja fixa ao longo do tempo — só precisa valer, dentro de um mesmo ciclo, que a divisão seja decidida antes de conhecer o resultado observado.

**O que entra / o que sai:** entra: todas as observações acumuladas, já marcadas por origem (Etapa 8). Sai: $\hat\theta$ e $H_\theta$ atualizados, além da taxa de referência de cada cobertura e sua incerteza, voltando para as Etapas 3 e 4 da próxima rodada.

---

## 10. O ciclo se fecha

**Analogia:** o próximo cliente que chegar já se beneficia do que foi aprendido — a régua $\theta$ está um pouco mais calibrada, e as taxas de referência de cada cobertura têm um pouco mais de precisão, com incertezas um pouco menores dos dois lados.

**A matemática:** volta-se à Etapa 1 com um novo cliente, usando $\hat\theta, H_\theta, b_i$ e sua variância já atualizados na Etapa 9.

---

## Nota operacional: adicionando ou reclassificando coberturas no catálogo

O catálogo não é fixo para sempre — coberturas novas podem surgir, e uma cobertura pode deixar de ser obrigatória (ou passar a ser). Vale registrar como o pipeline lida com essas mudanças, para não esquecer na hora de operar.

**Por que uma cobertura nova pode ser pontuada imediatamente, sem redimensionar nada.** $\theta$ tem dimensão fixa, definida pelo espaço de atributos semânticos de $f_i$ (Etapa 3) — não pelo número de coberturas do catálogo. Como $f_i$ não inclui nenhum identificador one-hot por item, adicionar uma cobertura não muda a dimensão de $x(i)$ nem exige retreinar $\theta$ do zero: basta a equipe atuarial definir o vetor $f_i$ da cobertura nova, nas mesmas dimensões já usadas para as demais, e $\theta$ já consegue estimar $\eta(i)$ para ela — generalizando a partir de coberturas com atributos parecidos, mesmo sem nenhuma observação histórica específica.

**O que ainda precisa de um período de aquecimento: o baseline.** $\theta$ generaliza de cara, mas $b_i$ (Etapa 4) começa em $n_i=0$ — sem observações reservadas, $\text{Var}[\text{logit}(b_i)]$ não é calculável. Uma cobertura nova (ou uma que acabou de deixar de ser obrigatória — que nunca gerou nenhuma observação de treino, já que obrigatórias ficam inteiramente fora dos dados que alimentam $\theta$, Etapa 8) precisa ficar temporariamente de fora do teste de significância da Etapa 5, sendo testada só pela rota aleatória, até acumular dado suficiente — tanto para $\theta$ aprender algo sobre ela quanto para $n_i$ crescer o bastante para o teste fazer sentido estatístico. Nesse caso específico (obrigatória virando opcional), o "aquecimento" é ainda mais completo que o de uma cobertura genuinamente nova: não existe absolutamente nenhum histórico prévio, nem do lado de $\theta$ nem do lado do baseline. É o mesmo mecanismo que já protege o catálogo como um todo — o motivo de $\epsilon$ nunca poder ser zero — só que aplicado à entrada de um item novo.

**O caminho inverso (opcional → obrigatória) é mais simples:** a cobertura só sai do teste de significância e entra direto no conjunto sempre incluído (Etapa 5) — sem precisar de nenhum período de espera.

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
| $Z_t\sim\text{Bernoulli}(\epsilon)$ | única fonte de aleatoriedade deliberada — garante que toda cobertura continue sendo testada, mesmo as já bem avaliadas |
| Piso de curvatura $\nu$ em $H_\theta$ | evita que itens com histórico consistentemente extremo parem de reduzir sua própria incerteza — mantém a variância encolhendo com mais dados, mesmo em regiões de alta certeza |
| Teste de significância — $D(i)$ vs. incerteza combinada | decide se a diferença entre modelo e referência é grande o suficiente para confiar nela, sem exigir peso de negócio arbitrário e sem depender só de uma comparação pontual |
| Reserva $\rho$ para o baseline | garante que a comparação entre $\eta(i)$ e o baseline seja entre duas estimativas estatisticamente independentes |
| Obrigatórias excluídas do treino | evita ruído e risco de quase-separação numérica em uma variável que nunca tem variação para aprender ($y_i=1$ sempre) — elas entram no pacote só pela Etapa 5, nunca pelo aprendizado de $\theta$ |
| Reajuste em lote completo | necessário porque a regressão logística não tem solução fechada — não é um detalhe de implementação, é uma propriedade do modelo |

---

## Parâmetros que precisam ser definidos

| Parâmetro | Onde aparece | O que controla | Ponto de partida sugerido |
|---|---|---|---|
| $\lambda$ (regularização) | Etapas 0, 3 e 9 | O quanto $\theta$ fica "presa" perto de zero antes de haver evidência forte | Testar valores entre 0,5 e 2,0 |
| $\nu$ (piso de curvatura) | Etapas 3 e 9 | Evita que a variância de itens muito certos pare de encolher com mais dados | Algo entre 0,02 e 0,10 (referência: $p(1-p)$ tem máximo 0,25 em $p=0{,}5$) |
| $\epsilon$ (fixo, definido pela equipe) | Etapa 2 | Qual fração das rodadas continua sendo oferta aleatória, indefinidamente | Um ponto de partida razoável é algo entre 0,05 e 0,15 — precisa ser maior que zero, sem exceção |
| $\rho$ (fração reservada para o baseline) | Etapas 4 e 9 | Quanto do dado aleatório fica reservado só para calibrar a referência, em vez de treinar $\theta$ | Um ponto de partida razoável é 30%, ajustável conforme o volume de tráfego |
| $z$ (nível de confiança do teste) | Etapa 5 | Quão conservadora é a exigência de significância para ofertar | 1,645 (95% de confiança, um lado) é um ponto de partida padrão |
| Lista de coberturas obrigatórias | Etapas 5 e 8 | Quais coberturas nunca passam pelo teste do modelo | Definida pela equipe de produto, a partir das regras contratuais do produto |
| Dimensões semânticas de $f_i$ | Etapa 3 | Quais atributos descrevem cada cobertura | Definidas com a equipe atuarial (categoria de risco, faixa de prêmio, frequência de sinistro) |
| Dimensões do perfil do cliente $c$ | Etapa 1 | Quais atributos do cliente entram no modelo | Definidas com a equipe de dados, a partir do que já está disponível no cadastro |
