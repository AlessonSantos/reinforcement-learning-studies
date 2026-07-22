# Pipeline completo: recomendação de pacotes de coberturas via C²UCB
### Versão com regressão logística em lote + correção do efeito kappa — guia matemático e didático

---

## Como usar este documento

Cada etapa tem três partes: **a analogia** (para visualizar o que está acontecendo), **a matemática** (a equação exata), e **o exemplo rodando** (números fictícios de um cliente específico, carregados de uma etapa para a outra). No fim de cada etapa há uma caixa **"o que entra / o que sai"**, mostrando o que essa etapa recebeu da anterior e o que ela entrega para a próxima.

**O exemplo que vamos seguir do início ao fim:** um cliente fictício, "Cliente A", perfil de risco médio-alto, sendo avaliado para um pacote entre 5 coberturas ilustrativas do catálogo (Incêndio, Conteúdo, Roubo, Danos Elétricos, Desastres Naturais).

---

## 0. Setup — antes de existir qualquer cliente

**Analogia:** imagine duas réguas em branco. Uma régua, $\theta$, vai aprender a traduzir "perfil do cliente + cobertura + tamanho do pacote até agora" em "chance de o cliente manter essa cobertura". A outra régua, $\phi$, vai aprender a traduzir "duas coberturas juntas" em "elas combinam ou brigam entre si". No dia zero, nenhuma das duas réguas sabe nada.

**A matemática:**

$$\theta \in \mathbb{R}^d, \qquad \phi \in \mathbb{R}^{d'}$$

Cada uma tem uma **matriz de precisão** (a "memória de quanta variedade de situação já foi vista"):

$$H_\theta = \lambda I, \qquad H_\phi = \lambda I$$

**A novidade desta versão:** além de $\lambda$, fixamos também um **piso de curvatura** $\tau$ (ex.: $\tau=0{,}05$). Vamos ver exatamente para que ele serve na Etapa 9 — por ora, pense nele como uma "garantia mínima de que toda observação, mesmo as mais óbvias, sempre conta um pouco para aumentar a confiança do modelo".

| Símbolo | Descrição |
|---|---|
| $\theta$ | régua compartilhada: perfil+cobertura+tamanho do pacote → propensão de manter |
| $\phi$ | régua compartilhada: par de coberturas → sinergia (+) ou canibalização (−) |
| $H_\theta, H_\phi$ | "quanto já aprendemos" sobre cada régua |
| $\lambda$ | **regularização.** É o peso que "trava" $\theta$ e $\phi$ perto de zero antes (e durante) o ajuste — funciona como um freio de segurança contra estimativas exageradas quando há poucos dados ou quando alguma cobertura é quase sempre aceita/rejeitada (evita que a régua "dispare" para valores extremos). Também é o valor inicial de $H_\theta$ e $H_\phi$ ($\lambda I$): quanto maior $\lambda$, mais o modelo começa "desconfiado", exigindo mais evidência antes de se mover para longe de zero. Um valor típico de partida é algo entre 0,5 e 2 — vale calibrar empiricamente |
| $\tau$ | **(novo)** piso mínimo de confiança ganha por observação — impede que itens muito previsíveis parem de gerar aprendizado. Não confundir com $\lambda$: $\lambda$ atua uma vez (como base fixa de $H$ e como penalização no ajuste), enquanto $\tau$ atua a cada observação individual, dentro da soma (ver Etapa 9) |

**O que entra / o que sai:** nada entra. Sai: $\theta, \phi, H_\theta, H_\phi$ zerados e $\tau$ definido, prontos para a Rodada 1.

---

## 1. Uma nova rodada começa: o cliente chega

**Analogia:** o Cliente A entra na loja. Você olha a ficha dele antes de qualquer coisa.

**A matemática:** observa-se $c \in \mathbb{R}^p$, o vetor de perfil do cliente.

**O exemplo rodando:** $c = [0{,}8,\ 1{,}0]$ (score de risco normalizado; indicador de imóvel de alto valor).

**O que entra / o que sai:** entra: nada. Sai: $c$, usado em todas as etapas seguintes desta rodada.

---

## 2. Explorar aleatoriamente ou usar o modelo?

**Analogia:** nas primeiras semanas de uma loja nova, o vendedor testa combinações variadas de propósito, mesmo sabendo que algumas vão ser ruins, só para aprender rápido. Com o tempo, ele confia mais no que já sabe.

**A matemática:** com probabilidade $\epsilon_t$ (decrescente em $t$), sorteia-se um tamanho de pacote $|S|$ e depois $|S|$ coberturas aleatórias — pulando para a Etapa 6. Caso contrário, segue para a Etapa 3.

**O exemplo rodando:** suponha que já passamos da fase de exploração intensa — vamos seguir o Cliente A pela via do modelo (Etapa 3 em diante).

**O que entra / o que sai:** entra: $c$. Sai: sinal para calcular scores (Etapa 3) ou pacote aleatório.

---

## 3. Score do item via regressão logística

**Analogia:** $\theta$ é uma régua que, para cada combinação "cliente + cobertura + tamanho da mala até agora", devolve um número. Esse número passa por um funil (a função logística) que o espreme para dentro de 0 a 1 — nunca escapa desse intervalo.

**A matemática:**

$$x(i, S_{parcial}) = \text{concat}\big(c,\; f_i,\; c \otimes f_i,\; |S_{parcial}|,\; c \times |S_{parcial}|\big)$$

$$\eta(i) = \hat\theta^{\,T} x(i, S_{parcial}), \qquad \text{bonus}(i) = \alpha\sqrt{x(i,S_{parcial})^{T} H_\theta^{-1} x(i,S_{parcial})}$$

$$p_{UCB}(i) = \sigma\big(\eta(i) + \text{bonus}(i)\big) = \frac{1}{1+e^{-(\eta(i)+\text{bonus}(i))}}$$

| Símbolo | Descrição |
|---|---|
| $c$ | perfil do cliente (a mesma ficha da Etapa 1: score de risco, indicador de imóvel de alto valor etc.) |
| $f_i$ | o vetor de atributos **estáticos** da cobertura $i$ — o mesmo vetor definido no catálogo (categoria de risco, faixa de prêmio, se é obrigatória, frequência histórica de sinistro). É fixo por cobertura, não muda de cliente para cliente |
| $c \otimes f_i$ | o produto elemento a elemento entre o perfil do cliente e os atributos da cobertura — a **interação** entre os dois. É essa peça que permite à régua $\theta$ aprender coisas do tipo "clientes de risco alto valorizam mais coberturas de risco climático", em vez de aprender um efeito genérico de $c$ e um efeito genérico de $f_i$ separadamente, sem conexão entre os dois |
| $\lvert S_{parcial}\rvert$ | um único número: quantas coberturas já estão no pacote parcial neste exato passo do oráculo (Etapa 5). É o que carrega a informação de fadiga — o "quão cheia a mala já está" |
| $c \times \lvert S_{parcial}\rvert$ | o perfil do cliente multiplicado pelo tamanho do pacote parcial — permite que a **intensidade da fadiga varie por perfil** (ex.: um cliente mais avesso a complexidade pode cansar mais rápido do que outro, mesmo com o mesmo tamanho de pacote) |
| $x(i, S_{parcial})$ | a concatenação de tudo acima — o vetor completo que descreve a situação |
| $\eta(i)$ | score "cru": o produto $\hat\theta^Tx(i,S_{parcial})$, antes de passar pelo funil |
| $\text{bonus}(i)$ | o quanto o modelo ainda está incerto sobre essa combinação — depende de $H_\theta^{-1}$, que carrega a novidade desta versão (Etapa 9) |
| $\sigma(\cdot)$ | o funil logístico — garante o resultado sempre entre 0 e 1 |

**O exemplo rodando:** para "Desastres Naturais", suponha $\eta = 0{,}9$ e $\text{bonus}=0{,}1$:

$$p_{UCB}(\text{Desastres}) = \sigma(0{,}9+0{,}1) = \sigma(1{,}0) \approx 0{,}73$$

**O que entra / o que sai:** entra: $c$, $\hat\theta$, $H_\theta$ (da Etapa 9 da rodada anterior), $|S_{parcial}|$. Sai: $p_{UCB}(i)$ por candidato, usado na Etapa 5.

---

## 4. Score do par — sinergia ou canibalização

**Analogia:** $\phi$ é um termômetro de compatibilidade entre duas peças de roupa. Se combinam bem, o termômetro sobe acima do "morno" (0,5). Se competem, esfria abaixo do morno.

**A matemática:**

$$z(i,j) = \text{concat}\big(f_i,\; f_j,\; f_i \otimes f_j,\; c\big)$$

$$p_{UCB}^{par}(i,j) = \sigma\Big(\hat\phi^{\,T}z(i,j) + \alpha\sqrt{z(i,j)^{T} H_\phi^{-1} z(i,j)}\Big)$$

| Símbolo | Descrição |
|---|---|
| $f_i, f_j$ | os mesmos vetores de atributos estáticos das duas coberturas (idênticos ao $f_i$ da Etapa 3 — categoria de risco, faixa de prêmio, obrigatoriedade, frequência de sinistro) |
| $f_i \otimes f_j$ | produto elemento a elemento entre os atributos das duas coberturas — captura o grau de **sobreposição estrutural** entre elas (ex.: se ambas têm alta frequência de sinistro na mesma categoria de risco, essa posição do vetor fica com valor alto) |
| $c$ | perfil do cliente — permite que a sinergia/canibalização também dependa de quem é o cliente, não só da identidade do par |
| $z(i,j)$ | a concatenação de tudo acima — o vetor completo que descreve a situação do par |

**O exemplo rodando:** para o par (Desastres Naturais, Danos Elétricos), com $\eta_{par}=-0{,}8$ e bônus $0{,}1$:

$$p_{UCB}^{par}(\text{Desastres}, \text{Danos Elétricos}) = \sigma(-0{,}7) \approx 0{,}33$$

Abaixo do morno — sinal de canibalização (sobreposição de sinistro entre os dois riscos).

**O que entra / o que sai:** entra: $\hat\phi$, $H_\phi$, $f_i, f_j$, $c$. Sai: $p_{UCB}^{par}(i,j)$, usado na Etapa 5.

---

## 5. O oráculo guloso: montando a mala item por item

**Analogia:** montar uma mala de viagem, peça por peça — só que a peça entra na mala se ela vale a pena tanto sozinha quanto em conjunto com o que já está ali dentro.

**A matemática:** $S_{parcial}$ começa vazio. A cada passo:

$$\Delta(i \mid S_{parcial}) = \Big(p_{UCB}(i) - 0{,}5\Big) \;+\; \sum_{j \in S_{parcial}} \Big(p_{UCB}^{par}(i,j) - 0{,}5\Big)$$

Escolhe-se $i^{\ast} = \arg\max_i \Delta(i\mid S_{parcial})$. Se $\Delta(i^{\ast})>0$, adiciona e repete; se $\Delta(i^{\ast})\le 0$, para.

**O exemplo rodando:**

$$\Delta(\text{Desastres} \mid \emptyset) = 0{,}73-0{,}5 = 0{,}23 > 0 \Rightarrow \text{adiciona}$$

$$\Delta(\text{Danos Elétricos} \mid \{\text{Desastres}\}) = (0{,}68-0{,}5)+(0{,}33-0{,}5) = 0{,}01$$

Quase zero — a canibalização quase anula o valor individual.

**O que entra / o que sai:** entra: $p_{UCB}(i)$ e $p_{UCB}^{par}(i,j)$ de todos os candidatos. Sai: o pacote final $S_{ofertado}$.

---

## 6. Oferta ao cliente

**Analogia:** a mala está pronta — hora de mostrar ao Cliente A.

**O exemplo rodando:** $S_{ofertado} = \{\text{Desastres Naturais}, \text{Danos Elétricos}, \text{Roubo}\}$.

**O que entra / o que sai:** entra: $S_{ofertado}$ (Etapa 5). Sai: o mesmo conjunto, exposto ao cliente.

---

## 7. Cliente edita: mantém, remove, adiciona

**Analogia:** o Cliente A mexe na mala pronta: tira uma peça, mantém outras, e às vezes pega algo que nem estava lá.

**O exemplo rodando:** remove "Danos Elétricos", mantém "Desastres Naturais" e "Roubo", adiciona "Incêndio" por conta própria. $S_{final} = \{\text{Desastres Naturais}, \text{Roubo}, \text{Incêndio}\}$.

**O que entra / o que sai:** entra: $S_{ofertado}$. Sai: $S_{final}$.

---

## 8. Traduzindo a edição em números que a régua entende

**Analogia:** cada peça mantida, tirada ou acrescentada vira uma "ficha" no arquivo.

**A matemática — sinal por item:**

$$y_i = \begin{cases} 1 & \text{se } i \in S_{final} \\ 0 & \text{caso contrário} \end{cases}$$

**Sinal por par** (para pares presentes no pacote ofertado):

$$y_{ij} = \begin{cases} 1 & \text{se ambos } i,j \in S_{final} \\ 0 & \text{se pelo menos um foi removido} \end{cases}$$

**O exemplo rodando:**

| Cobertura | Ofertada? | No final? | $y_i$ |
|---|---|---|---|
| Desastres Naturais | sim | sim | **1** |
| Danos Elétricos | sim | não | **0** |
| Roubo | sim | sim | **1** |
| Incêndio | não | sim (espontânea) | **1** |

| Par | $y_{ij}$ |
|---|---|
| (Desastres, Danos Elétricos) | **0** — reforça canibalização |
| (Desastres, Roubo) | **1** — candidato a sinergia |
| (Danos Elétricos, Roubo) | **0** |

**O que entra / o que sai:** entra: $S_{ofertado}$, $S_{final}$. Sai: observações $(x_i,y_i)$ e $(z_{ij},y_{ij})$.

---

## 9. Atualização em lote — e o achado importante sobre itens "quase certos"

**Analogia:** no fim do dia, você recalibra as duas réguas usando todas as fichas acumuladas desde sempre — não só as de hoje.

**A matemática do ajuste (MAP com prior gaussiano):**

$$\hat\theta = \arg\max_\theta \sum_{n=1}^{N} \Big[y_n \log p_n + (1-y_n)\log(1-p_n)\Big] - \frac{\lambda}{2}\|\theta\|^2, \qquad p_n=\sigma(\theta^Tx_n)$$

**De onde vem essa fórmula:** ela nasce da regra de Bayes — posterior $\propto$ verossimilhança $\times$ prior. A verossimilhança de uma observação binária segue uma distribuição de Bernoulli, $p_n^{y_n}(1-p_n)^{1-y_n}$; somando o logaritmo dessa verossimilhança para as $N$ observações e somando o logaritmo do prior gaussiano em $\theta$ (que vira exatamente o termo $-\frac{\lambda}{2}\|\theta\|^2$), chega-se na expressão acima. Diferente da ridge regression (onde essa mesma lógica — minimizar erro quadrático + penalização — leva a uma equação **linear** em $\theta$, resolvível por inversão direta de matriz, $\hat\theta=V^{-1}b$), aqui $p_n=\sigma(\theta^Tx_n)$ entra **dentro do logaritmo** de forma não-linear. Por isso a maximização não tem solução fechada e precisa de um otimizador iterativo (o `scipy.optimize.minimize` no código) — é exatamente essa diferença que discutimos como o motivo de a regressão logística não permitir a atualização somável da ridge regression.

Aqui $\lambda$ aparece como o termo $-\frac{\lambda}{2}\|\theta\|^2$: uma penalização que cresce conforme $\theta$ se afasta de zero. Na prática, isso impede que $\hat\theta$ "dispare" para valores extremos quando os dados são escassos ou quando uma cobertura é aceita/rejeitada de forma quase unânime (o cenário de quase-separação que motiva boa parte desta etapa) — sem essa penalização, a otimização poderia tentar empurrar $\theta$ ao infinito para explicar perfeitamente um padrão determinístico nos dados.

### O problema que descobrimos: o efeito "kappa"

Ao calcular a incerteza (matriz $H_\theta$) via aproximação de Laplace, a fórmula original era:

$$H_\theta = \lambda I + \sum_{n=1}^{N} p_n(1-p_n)\, x_n x_n^{T}$$

**Analogia do problema:** imagine um jurado que só atualiza o quanto confia num veredito quando o caso é "difícil" (meio a meio). Se o caso é óbvio — todo mundo já sabe a resposta —, ele para de prestar atenção, mesmo vendo o mesmo padrão se confirmar centenas de vezes. É exatamente isso que $p_n(1-p_n)$ faz: esse peso é **máximo** quando $p_n\approx 0{,}5$ (caso "difícil") e **quase zero** quando $p_n$ está perto de 0 ou 1 (caso "óbvio" — ex.: "Incêndio", quase sempre mantida).

**Consequência prática, medida em teste real:** simulamos uma cobertura "quase certa" (sempre aceita) e observamos o bônus de exploração conforme mais observações se acumulam:

| Observações acumuladas | Bônus **sem correção** | Bônus **com correção** |
|---|---|---|
| 1 | 1,02 | 1,02 |
| 50 | 0,59 | 0,56 |
| 100 | 0,55 | 0,42 |
| 300 | **0,50** (estacionou) | **0,25** (continua caindo) |

Sem correção, o bônus **para de cair por volta de 0,50**, mesmo depois de 300 observações idênticas e consistentes — o modelo nunca "aprende a confiar" totalmente nesse item, mesmo tendo evidência esmagadora. Isso é uma manifestação prática do que a literatura de bandits para modelos logísticos chama de dependência em $\kappa$ (Filippi et al., 2010): o regret formal desses algoritmos piora exatamente quando as probabilidades reais estão perto dos extremos — o oposto do que a intuição sugeriria.

### A correção: um piso na curvatura

$$H_\theta = \lambda I + \sum_{n=1}^{N} \max\big(p_n(1-p_n),\ \tau\big)\, x_n x_n^{T}$$

**Analogia da correção:** é como dizer ao jurado "mesmo em casos óbvios, você ainda precisa prestar um mínimo de atenção a cada novo caso" — garantindo que a confiança sempre cresce com mais dados, nunca estaciona artificialmente.

| Símbolo | Descrição |
|---|---|
| $p_n(1-p_n)$ | quanto aquela observação específica reduz a incerteza — máximo em casos "difíceis" (perto de 50/50), quase zero em casos "óbvios" |
| $\tau$ | piso mínimo — garante que mesmo observações de casos óbvios continuem contribuindo |
| $H_\theta$ (corrigido) | nova "memória", agora sem o platô artificial, alimentando o bônus da próxima rodada (Etapa 3) |

**Por que essa correção foi escolhida em vez de uma alternativa por contagem bruta:** o sistema é contextual — o mesmo item aparece em contextos diferentes (perfis de cliente diferentes, tamanhos de pacote diferentes). Uma correção baseada em "contar quantas vezes vi esse item" jogaria fora a generalização entre contextos parecidos, que é o motivo de termos escolhido C²UCB desde o início. O piso de curvatura, em vez disso, corrige o mecanismo exato onde o problema nasce (o peso da observação dentro de $H$), sem sair da geometria contextual do modelo.

### A mesma atualização acontece em paralelo para $\phi$ (o termo de par)

**Analogia:** enquanto uma régua ($\theta$) está sendo recalibrada com as fichas de "cobertura isolada", a segunda régua ($\phi$) está sendo recalibrada ao mesmo tempo, no mesmo fechamento de caixa, mas usando as fichas de "par de coberturas" (as observações $(z_{ij}, y_{ij})$ da Etapa 8) — é o mesmo procedimento, rodando sobre um conjunto de dados diferente.

**A matemática:** exatamente a mesma estrutura da atualização de $\theta$, trocando o contexto de item $x_n$ pelo contexto de par $z_n$, e o rótulo $y_n$ (manteve/removeu aquele item) pelo rótulo $y_n^{par}$ (ambos da dupla sobreviveram juntos ou não):

$$\hat\phi = \arg\max_\phi \sum_{n=1}^{M} \Big[y_n^{par} \log p_n^{par} + (1-y_n^{par})\log(1-p_n^{par})\Big] - \frac{\lambda}{2}\|\phi\|^2, \qquad p_n^{par}=\sigma(\phi^Tz_n)$$

$$H_\phi = \lambda I + \sum_{n=1}^{M} \max\big(p_n^{par}(1-p_n^{par}),\ \tau\big)\, z_n z_n^{T}$$

| Símbolo | Descrição |
|---|---|
| $M$ | número de observações de **par** acumuladas (lembrando: cada rodada gera vários pares a partir do pacote ofertado — ver Etapa 8) |
| $z_n$ | o contexto do par $(i,j)$ daquela observação — o mesmo $z(i,j)$ definido na Etapa 4 |
| $y_n^{par}$ | 1 se as duas coberturas do par sobreviveram juntas no pacote final, 0 caso contrário |
| $\hat\phi$ | a régua de sinergia/canibalização recalibrada — mesma mecânica de MAP + prior gaussiano da atualização de $\theta$ |
| $H_\phi$ (corrigido) | a incerteza sobre $\phi$, já com o mesmo piso de curvatura $\tau$ aplicado — sem ele, pares de coberturas quase sempre aceitas ou quase sempre rejeitadas juntas sofreriam exatamente o mesmo platô artificial que identificamos para $\theta$ |

**Por que isso não apareceu destacado antes:** as duas atualizações ($\theta$/$H_\theta$ e $\phi$/$H_\phi$) rodam de forma totalmente independente uma da outra — usam dados diferentes (observações de item vs. de par) e nenhuma delas depende do resultado da outra dentro da mesma rodada de treino. Por isso, no código (`item_model.refit()` e `pair_model.refit()`), são duas chamadas separadas, mas com exatamente a mesma lógica interna.

**Uma ressalva honesta sobre reajuste em lote (mantida desta versão):** diferente da ridge regression, a regressão logística não permite simplesmente "somar" o lote novo ao estado antigo — é necessário reprocessar. A opção recomendada, dado o catálogo pequeno (14 itens, 91 pares), é o **reajuste completo**: guardar todas as observações desde o início (inclusive a fase aleatória) e reotimizar do zero a cada lote — isso vale tanto para $\theta$ quanto para $\phi$.

**O que entra / o que sai:** entra: todas as observações acumuladas (Etapa 8, mais histórico), $\tau$ (Etapa 0). Sai: $\hat\theta, H_\theta, \hat\phi, H_\phi$ atualizados — sem o platô artificial — voltando para a Etapa 3 da próxima rodada.

---

## 10. O ciclo se fecha

**Analogia:** o próximo cliente parecido com o Cliente A já se beneficia do que foi aprendido — a régua está mais calibrada, a incerteza um pouco menor, e agora **de forma consistente**, sem estacionar artificialmente nos itens mais óbvios.

**A matemática:** volta-se à Etapa 1 com um novo cliente, usando os $\hat\theta, H_\theta, \hat\phi, H_\phi$ atualizados na Etapa 9.

---

## Mapa de conexões entre as etapas (resumo visual)

```
Etapa 0 (setup)
     │  θ̂, H_θ, φ̂, H_φ (zerados) + τ (piso de curvatura)
     ▼
Etapa 1 (cliente chega) ──> c
     ▼
Etapa 2 (explorar ou não) ──> pacote aleatório (→ Etapa 6) OU segue
     ▼
Etapa 3 (score item)  ←── θ̂, H_θ (da Etapa 9 anterior) + c + |S_parcial| (da Etapa 5)
     ▼
Etapa 4 (score par)   ←── φ̂, H_φ (da Etapa 9 anterior) + c
     ▼
Etapa 5 (oráculo guloso) ──> usa p_UCB(i) e p_UCB_par(i,j), atualiza |S_parcial|
     ▼
Etapa 6 (oferta) ──> S_ofertado
     ▼
Etapa 7 (cliente edita) ──> S_final
     ▼
Etapa 8 (traduz em y_i, y_ij)
     ▼
Etapa 9 (reajuste em lote, com piso de curvatura τ) ──> novo θ̂, H_θ, φ̂, H_φ
     │
     └──────────────► volta para a Etapa 3 da próxima rodada (Etapa 10)
```

---

## Tabela-síntese: o que cada peça resolve, com o exemplo em mente

| Peça | Resolve | No exemplo do Cliente A |
|---|---|---|
| $\sigma(\cdot)$ (funil logístico) | garante que a probabilidade nunca sai de $(0,1)$ | $p_{UCB}(\text{Desastres}) = 0{,}73$, nunca $>1$ |
| $\theta$ + tamanho do pacote | relevância individual **e** fadiga, aprendidas juntas | régua aprende como a chance de manter cai conforme a mala enche |
| $\phi$ (par) | sinergia (+) e canibalização (−) específicas | par (Desastres, Danos Elétricos) fica cada vez mais negativo |
| Centralização em 0,5 (exata) | transforma "menos bom" em "genuinamente ruim" | $\Delta(\text{Danos Elétricos}) \approx 0{,}01$ — quase descartado |
| $H_\theta, H_\phi$ (Laplace) | mede incerteza, alimenta o bônus de exploração | par (Desastres, Roubo) ainda com bônus alto — pouco visto |
| **Piso de curvatura $\tau$ (novo)** | evita que itens "quase certos" (ex.: Incêndio) parem de reduzir incerteza | bônus cai de 0,50 (estacionado) para 0,25 após 300 observações |
| Reajuste em lote | atualiza as réguas periodicamente, sem precisar ser a cada evento | fechamento diário, reprocessando o histórico armazenado |

---

## Parâmetros que vocês precisam definir (não vêm "de fábrica")

Tudo que apareceu nas equações acima com uma letra grega ou um nome fixo é uma peça da matemática — mas boa parte deles não tem um valor certo universal: são escolhas de configuração que a equipe precisa fixar (e depois calibrar empiricamente) antes de colocar o sistema no ar.

| Parâmetro | Onde aparece | O que controla | Ponto de partida sugerido |
|---|---|---|---|
| $\lambda$ (regularização) | Etapas 0 e 9 | O quanto $\theta$ e $\phi$ ficam "presos" perto de zero antes de haver evidência forte — protege contra estimativas explosivas com poucos dados ou padrões quase determinísticos | Testar valores entre 0,5 e 2,0; pode (e provavelmente deve) ser diferente para $\theta$ e para $\phi$ |
| $\alpha$ (peso de exploração) | Etapas 3 e 4 | O quanto o bônus de incerteza pesa na decisão final — maior $\alpha$ = mais disposição a testar combinações pouco vistas, mesmo com estimativa pontual baixa | Começar mais alto (ex.: 1,5–2,0) na fase inicial e reduzir gradualmente conforme o histórico cresce |
| $\tau$ (piso de curvatura) | Etapa 9 | Evita que itens/pares "quase certos" parem de reduzir incerteza (o efeito kappa) | Algo entre 0,02 e 0,10 (referência: $p(1-p)$ tem máximo 0,25 em $p=0{,}5$) |
| Frequência de reajuste (tamanho do lote) | Etapa 9 | De quanto em quanto tempo (ou a cada quantos clientes) o histórico completo é reprocessado | Diário costuma ser suficiente para o volume esperado; pode ser ajustado conforme o tráfego real |
| $\epsilon_t$ (probabilidade de exploração aleatória) e seu cronograma de decaimento | Etapa 2 | Por quanto tempo, e com que intensidade, a oferta aleatória (sem modelo) continua acontecendo | Começar próximo de 1,0 (quase só aleatório) e decair até um piso pequeno (ex.: 0,05) ao longo de semanas/meses — nunca chegar a zero, para continuar cobrindo combinações novas |
| Dimensões semânticas de $f_i$ (o catálogo) | Etapas 3 e 4 | Quais atributos descrevem cada cobertura (categoria de risco, faixa de prêmio, obrigatoriedade, frequência de sinistro, e quaisquer outras que a equipe atuarial considere relevantes) | Definido uma vez, manualmente, com a equipe de produto/atuária — pode evoluir depois para embeddings aprendidos, conforme discutido anteriormente |
| Dimensões do perfil do cliente $c$ | Etapa 1 | Quais atributos do cliente entram no modelo (score de risco, valor do imóvel, região, composição familiar etc.) | Definido com a equipe de dados, com base no que já está disponível no cadastro |

**Por que isso importa destacar separadamente:** nenhum desses valores tem uma "resposta certa" que a matemática determine sozinha — são decisões operacionais que afetam diretamente a velocidade de aprendizado e o comportamento inicial do sistema. Vale tratá-los como parte do escopo de calibração/validação antes do lançamento, não como detalhes de implementação a serem decididos casualmente.
