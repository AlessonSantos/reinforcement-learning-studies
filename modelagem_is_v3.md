# Distribuição de ticket entre coberturas e cálculo da Importância Segurada (IS), versão 3

> Documento de modelagem que acompanha o código `otimizacao_is_v3.py`.
>
> A versão 3 é a versão 2 com uma única mudança: a **IS final é sempre arredondada na centena** (múltiplo de 100: 72.000 ou 72.100, nunca 72.101 ou 72.002). A otimização é a mesma da versão 2; só o arredondamento final muda. A regra está na seção 7, e os exemplos da seção 6 foram recalculados.
>
> A versão 2 é a versão 1 com **um termo a mais no objetivo**: a penalização por afastar a IS de cada cobertura da **IS sugerida pelo negócio** (seção 4.5). Com `beta = 0`, ela volta à versão 1.

---

## 1. O problema em uma frase

Cada pacote de seguro residencial tem um **ticket total** $T$, repartido entre coberturas segundo **proporções recomendadas**. Queremos descobrir a **IS de cada cobertura** de forma que três coisas aconteçam:

1. a IS respeite o mínimo e o máximo de cada cobertura e seja um **múltiplo de 100**;
2. o ticket total seja distribuído integralmente, quando isso for possível;
3. as proporções recomendadas sejam distorcidas **o mínimo possível**;
4. *(versão 2)* as IS fiquem, na medida do possível, perto das **IS sugeridas pelo negócio**.

Uma das coberturas (a **especial**) tem preço não linear: a taxa e o agravo 1 dependem da faixa de IS.

---

## 2. A ideia central: trabalhar em "espaço de ticket"

Para as coberturas comuns, ticket e IS estão ligados por uma relação linear:

$$t_i = IS_i \cdot taxa_i \quad\Longleftrightarrow\quad IS_i = \frac{t_i}{taxa_i}$$

Por isso, os limites de IS viram limites de **ticket**:

$$L_i = IS^{min}_i \cdot taxa_i \;\le\; t_i \;\le\; U_i = IS^{max}_i \cdot taxa_i$$

Assim, distribuímos **tickets** dentro de intervalos $[L_i, U_i]$ e só no final convertemos cada ticket em IS. A cobertura especial pede um cuidado extra, explicado na seção 5.

---

## 3. Notação

Cada linha do dataframe (pacote × perfil) é um problema **independente**.

| Símbolo | Significado |
|---|---|
| $T$ | ticket total do pacote, igual a $\sum_i t^0_i$ |
| $t^0_i$ | ticket recomendado da cobertura $i$ (proporção × $T$) |
| $t_i$ | ticket final, que é o que queremos encontrar |
| $L_i, U_i$ | limites de ticket derivados de IS mín/máx |
| $s = 1+\lambda$ | fator de escala comum (explicado abaixo) |
| $IS^{ref}_i$ | *(versão 2)* IS sugerida pelo negócio para a cobertura $i$ |
| $\beta$ | *(versão 2)* peso da distorção de IS em relação à de ticket |
| $\mu$ | *(versão 2)* ajuste comum que generaliza o $s$ (seção 4.5) |

Uma cobertura **fora do pacote** ($t^0_i$ = 0 ou NaN) não participa do problema e fica com IS = 0.

---

## 4. O problema de otimização (coberturas comuns)

**Objetivo:** minimizar a distorção **relativa**.

$$\min_{t} \; \sum_i \frac{(t_i - t^0_i)^2}{t^0_i}$$

**Restrições:**

$$\sum_i t_i = T \qquad\qquad L_i \le t_i \le U_i$$

### 4.1 Por que dividir por $t^0_i$?

Desviar R$ 50 numa cobertura de R$ 600 é pouco (8%), mas numa de R$ 60 é muito (83%). Dividir pelo ticket recomendado mede o desvio **em proporção**, que é o que importa para "preservar a proporção recomendada".

### 4.2 A solução tem forma fechada

Aplicando as condições de otimalidade (KKT) a esse problema:

$$\boxed{\,t_i = \text{clip}\big(s \cdot t^0_i,\; L_i,\; U_i\big)\,}$$

**Em palavras:**

- todas as coberturas são multiplicadas pelo **mesmo fator** $s$;
- quem ultrapassa um limite fica **travado** nele;
- o excesso, ou a falta, é absorvido pelas coberturas livres **na proporção das recomendações originais**.

Assim, entre as coberturas livres, a proporção relativa recomendada é preservada **exatamente**.

### 4.3 Como achar $s$?

A função $f(s) = \sum_i \text{clip}(s \cdot t^0_i, L_i, U_i)$ é **não decrescente** em $s$, então basta uma **bisseção** para achar o $s$ que dá $f(s) = T$:

- Em $s = 0$, todos vão para $L_i$.
- Em $s$ grande, todos vão para $U_i$.
- Faz-se a busca no meio, 100 iterações (parâmetro `n_iter`). No código isso é vetorizado e todas as linhas são resolvidas de uma vez.

Como ler o resultado:

- $s > 1$: alguma cobertura travou no **máximo** e as outras **cresceram** para absorver o excesso.
- $s < 1$: alguma cobertura foi **forçada a subir** até o mínimo e as outras **cederam** espaço.

### 4.4 Casos de viabilidade

| Situação | Significado | O que o código faz |
|---|---|---|
| $\sum L_i \le T \le \sum U_i$ | cabe exatamente | resolve com $s$; `status = ok` |
| $\sum U_i < T$ | nem tudo no máximo absorve o ticket | todos em $U_i$; `status = sobra_nao_distribuida` |
| $\sum L_i > T$ | nem os mínimos cabem | todos em $L_i$; `status = minimos_excedem_T` |

Pela regra de negócio, o terceiro caso não deveria ocorrer no pacote como um todo. Mas ele **pode** ocorrer em uma **faixa específica** da cobertura especial. Por exemplo, uma faixa alta tem ticket mínimo grande demais. Nesse caso, aquela faixa simplesmente perde na escolha.

### 4.5 Versão 2: ponderação pela IS sugerida

**Motivação.** Na versão 1, quando uma cobertura trava num limite, o excesso de ticket é repartido entre as coberturas livres na proporção do ticket recomendado. A taxa não entra nessa conta. Em pacotes com poucas coberturas, cada cobertura livre recebe uma fatia grande do excesso, e nas de taxa baixa isso vira muita IS ($IS = t / taxa$). O resultado fica dentro de [IS mín, IS máx], mas pode ficar bem acima da IS que o negócio costuma praticar.

**Novo objetivo.** Somamos um segundo termo, que mede o quanto cada IS se afasta da IS sugerida:

$$\min_t \;\underbrace{\sum_i \frac{(t_i - t^0_i)^2}{t^0_i}}_{\text{distorção de ticket}} \;+\; \beta \underbrace{\sum_i t^0_i \left(\frac{IS_i - IS^{ref}_i}{IS^{ref}_i}\right)^2}_{\text{distorção de IS}}$$

com as mesmas restrições de antes ($\sum t_i = T$ e $L_i \le t_i \le U_i$).

**Como ler o β.** Os dois termos usam desvios **relativos**, multiplicados pelo ticket recomendado, então ficam na mesma escala:
- $\beta = 1$: afastar a IS 10% da sugerida "custa" o mesmo que afastar o ticket 10% do recomendado (valor padrão);
- $\beta = 0$: ignora a IS sugerida (igual à versão 1);
- $\beta$ grande: a IS sugerida passa a mandar, e a proporção de ticket vira secundária.

Coberturas sem IS sugerida (coluna ausente, vazia ou 0) simplesmente não têm o segundo termo.

**Quando o β faz diferença.** Para uma cobertura comum, mudar o ticket em 10% muda a IS em 10% também. Por isso, se a IS sugerida for **coerente** com o ticket recomendado ($IS^{ref} \cdot taxa = t^0$), os dois termos apontam para o mesmo lugar e o β **não muda nada**. A penalização só age quando a IS sugerida **diverge** do que a proporção de ticket implica. Esse é o caso que motivou a versão 2: as IS dos pacotes do negócio ficam abaixo das que a proporção de ticket gera. Se as IS sugeridas forem coerentes com os tickets recomendados e mesmo assim o acúmulo de IS incomodar, o tratamento certo é outro: limites de IS mais apertados.

**Por que continua tendo forma fechada.** Como $IS_i = (t_i - fixo_i)/taxa_i$, o segundo termo também é quadrático no ticket. Definindo o **ticket que corresponde à IS sugerida**

$$t^{ref}_i = fixo_i + taxa_i \cdot IS^{ref}_i$$

o termo de IS vira $b_i\,(t_i - t^{ref}_i)^2$, com

$$a_i = \frac{1}{t^0_i} \qquad b_i = \frac{\beta\, t^0_i}{(taxa_i \cdot IS^{ref}_i)^2}$$

e a soma dos dois termos de uma cobertura se reescreve como uma única parábola:

$$a_i (t_i - t^0_i)^2 + b_i (t_i - t^{ref}_i)^2 \;=\; w_i\,(t_i - m_i)^2 + \text{constante}$$

$$w_i = a_i + b_i \qquad m_i = \frac{a_i\, t^0_i + b_i\, t^{ref}_i}{a_i + b_i}$$

- $m_i$ é o **alvo combinado**: uma média ponderada entre o ticket recomendado e o ticket da IS sugerida.
- $w_i$ é a **rigidez** da cobertura: quanto maior, mais caro é afastá-la do alvo, e menos ela absorve dos ajustes.

Aplicando as condições KKT, como na seção 4.2:

$$\boxed{\,t_i = \text{clip}\Big(m_i + \frac{\mu}{w_i},\; L_i,\; U_i\Big)\,}$$

O ajuste $\mu$ é um único número por linha, e a soma dos tickets cresce com ele, então é achado pela mesma bisseção da seção 4.3. O intervalo de busca é escolhido de forma que, no extremo inferior, todas as coberturas estejam no piso $L_i$ e, no superior, todas no teto $U_i$.

**Conferência com a versão 1.** Com $\beta = 0$: $b_i = 0$, $m_i = t^0_i$ e $w_i = 1/t^0_i$. A fórmula vira $t_i = \text{clip}(t^0_i + \mu\, t^0_i) = \text{clip}((1+\mu)\, t^0_i)$, que é exatamente a da versão 1, com $s = 1 + \mu$. Por isso a coluna `fator_escala` na versão 2 é $1 + \mu$.

**Cobertura especial.** O $t^{ref}$ da especial usa a taxa e os agravos **da faixa** de cada linha. Assim, cada faixa é avaliada com a sua própria conversão entre ticket e IS, e a escolha da faixa (seção 5.5) passa a usar o objetivo completo.

**Cobertura de valor fixo sem taxa.** A IS dela não muda o ticket, então o termo de IS não se aplica.

### 4.6 Escolha do β padrão

Fizemos uma varredura numa base simulada de 3.000 pacotes com 2 a 4 coberturas. Nela, uma das coberturas tinha taxa baixa e IS sugerida 40% abaixo da implícita pelo ticket recomendado, e outra cobertura travava no máximo e liberava ticket:

| β | Distorção média de proporção | IS/IS sugerida, Incêndio (mediana) | IS/IS sugerida, cobertura de taxa baixa (mediana) |
|---|---|---|---|
| 0 (versão 1) | 0,069 | 1,19 | 1,93 |
| 0,25 | 0,070 | 1,26 | 1,70 |
| 0,5 | 0,073 | 1,28 | 1,59 |
| **1 (padrão)** | **0,077** | **1,32** | **1,49** |
| 2 | 0,080 | 1,34 | 1,41 |
| 5 | 0,083 | 1,35 | 1,35 |
| 10 | 0,084 | 1,36 | 1,33 |

- Com β = 1, a IS da cobertura de taxa baixa cai de 1,93× para 1,49× a sugerida, com aumento pequeno na distorção de proporção (de 0,069 para 0,077).
- Acima de 2, o ganho fica marginal.
- O Incêndio sobe um pouco, porque o ticket total continua sendo distribuído: o que sai da cobertura de taxa baixa vai para as outras. O modelo equilibra os desvios relativos de IS entre as coberturas.

Por isso o padrão é **β = 1**. O ideal é calibrar com os pacotes reais do negócio: rodar alguns valores de β e ver qual deixa a otimização mais perto do que já é praticado.

---

## 5. A cobertura especial

### 5.1 O que muda

$$t_k = IS \cdot taxa(IS) + A_1(IS) + A_2$$

- $A_2$ depende só do perfil, então é constante.
- A **taxa** e o **agravo 1** dependem da **faixa** em que a IS cai.
- A taxa vale sobre a **IS toda**, e não de forma progressiva.

O ticket recomendado da especial já contém os três termos.

### 5.2 Dentro de uma faixa, tudo volta a ser linear

Fixada a faixa $j$:

$$t_k = IS \cdot taxa_j + A_{1j} + A_2 \quad\Longleftrightarrow\quad IS = \frac{t_k - A_{1j} - A_2}{taxa_j}$$

O intervalo de IS permitido nessa faixa é a interseção da faixa com os limites da cobertura. No seu layout, ele já vem pronto nas colunas `<cob>_ismin` e `<cob>_ismax` da especial, linha a linha:

$$[IS^{lo}_j,\; IS^{hi}_j] = [\max(IS^{min}, inf_j),\; \min(IS^{max}, sup_j)]$$

Ele vira um intervalo de ticket:

$$L_{kj} = IS^{lo}_j \cdot taxa_j + A_{1j} + A_2 \qquad U_{kj} = IS^{hi}_j \cdot taxa_j + A_{1j} + A_2$$

Se $IS^{lo}_j > IS^{hi}_j$, a faixa não tem interseção com os limites da cobertura e é descartada.

### 5.3 Por que não dá para resolver "de uma vez"

Como a taxa costuma **cair** nas faixas mais altas, os intervalos de ticket das faixas podem:

- **se sobrepor**, quando um mesmo ticket corresponde a duas IS diferentes;
- **deixar buracos**, quando há tickets impossíveis de atingir.

O conjunto viável deixa de ser um intervalo, e o problema deixa de ser convexo.

### 5.4 Solução: enumerar as faixas

São poucas faixas (cerca de 5), então:

1. Ter **uma linha por faixa** de cada pacote × perfil. No seu dataframe isso já vem pronto (linhas explodidas).
2. Em cada linha (candidato), tratar a especial **como se fosse comum**, com limites $[L_{kj}, U_{kj}]$.
3. Resolver todas as linhas com a fórmula da seção 4.
4. Escolher, por pacote × perfil, a melhor linha.

O problema não convexo vira cerca de 5 problemas convexos simples, e o melhor deles é o **ótimo exato**.

### 5.5 Critério de escolha, em ordem de prioridade

1. Faixa **válida** (com interseção com IS mín/máx).
2. Faixa que **não** viola os mínimos ($\sum L \le T$).
3. **Menor sobra** não distribuída, em $|T - \sum t|$.
4. **Menor objetivo**: distorção de ticket + β × distorção de IS (na versão 1, só a distorção de ticket).
5. **Desempate pela IS da especial** (parâmetro `desempate_is = "menor"` ou `"maior"`).

O critério 5 existe por causa da sobreposição descrita na seção 5.3: duas faixas podem chegar ao **mesmo ticket**, com a mesma distorção, mas com **IS diferentes**. Quem decide qual preferir é a regra de negócio.

### 5.6 Coberturas de valor fixo

Algumas coberturas têm ticket dado por um **valor fixo** (que só varia por perfil, como o agravo 2) mais uma parte proporcional à IS:

$$t = valor\_fixo + IS \cdot taxa$$

com a taxa já dividida por 100 no pré-processamento, como nas demais coberturas.

É a mesma estrutura da especial (parte fixa + IS × taxa), só que sem faixas. Como as demais, essas coberturas têm um **ticket recomendado** (`<cob>_t`, proporção × ticket total), que não precisa ser igual ao valor fixo.

**Sem taxa (situação atual).** A coluna `<cob>_taxa` nem precisa existir; a taxa é considerada 0.
- O ticket final é **exatamente o valor fixo**, e a IS final sai **0**.
- A cobertura continua fazendo parte da otimização. A diferença entre o ticket recomendado e o valor fixo é absorvida pelas demais coberturas, com os mesmos critérios de sempre (máximo aproveitamento do ticket e mínima distorção):
  - recomendado **maior** que o valor fixo: a sobra vai para as outras;
  - recomendado **menor** que o valor fixo: o que falta é retirado das outras.

Exemplo (pacote A, valor fixo de R$ 50):

| Ticket recomendado | Incêndio | Roubo | Valor fixo | Total |
|---|---|---|---|---|
| 80 (sobram 30) | 704 | 176 | 50 | 1.080 |
| 20 (faltam 30) | 656 | 164 | 50 | 1.020 |

**Com taxa (se for incorporada no futuro).** Basta criar a coluna `<cob>_taxa`. O ticket da cobertura passa a variar entre $valor\_fixo + IS^{min} \cdot taxa$ e $valor\_fixo + IS^{max} \cdot taxa$, e a IS volta a ser calculada por $IS = (t - valor\_fixo) / taxa$, na centena e dentro dos limites.

**Fora do pacote:** ticket recomendado e valor fixo zerados. A cobertura fica com IS 0 e ticket 0.

---

## 6. Exemplo numérico completo

Usamos o perfil P1 com as coberturas Incêndio (especial), Danos Elétricos (DE) e Roubo.

**Coberturas comuns:**

| Cobertura | Taxa | IS mín | IS máx | Ticket possível |
|---|---|---|---|---|
| Danos Elétricos | 0,5% | 5.000 | 30.000 | 25 – 150 |
| Roubo | 1% | 2.000 | 50.000 | 20 – 500 |

**Faixas do Incêndio** ($A_2 = 30$, IS entre 50 mil e 1 milhão):

| Faixa | IS | Taxa | $A_1$ | Ticket possível |
|---|---|---|---|---|
| F1 | 50k – 200k | 0,20% | 20 | 150 – 450 |
| F2 | 200.001 – 500k | 0,15% | 40 | ≈370 – 820 |
| F3 | 500.001 – 1M | 0,12% | 60 | ≈690 – 1.290 |

As faixas não se encostam: F1 vai até 200.000 e F2 começa em 200.001. Com a IS na centena (seção 7), a F2 começa, na prática, em 200.100.

Repare na sobreposição **de ticket**: o ticket 700, por exemplo, cabe em F2 e em F3.

### 6.1 Pacote A: redistribuição clássica

$T = 1000$, com recomendação Incêndio 600 (60%), DE 250 (25%) e Roubo 150 (15%).

DE queria 250, mas o máximo é 150. Fica travado em 150, e sobram 100 para redistribuir.

| Faixa | Incêndio | DE | Roubo | Sobra | Distorção |
|---|---|---|---|---|---|
| F1 | 450 (travado no máx.) | 150 | 400 | 0 | 494,2 |
| **F2** | **680** | **150** | **170** | **0** | **53,3** ✅ |
| F3 | 690 (travado no mín.) | 150 | 160 | 0 | 54,2 |

Os valores dessa tabela são do ótimo contínuo, antes do arredondamento na centena; é com eles que a faixa é escolhida (seção 7.3).

Em F2, Incêndio e Roubo foram escalados pelo mesmo fator: $s = 850/750 \approx 1{,}133$. A proporção entre eles, 600:150 = 680:170 = 4:1, ficou **intacta**.

**IS finais:**
- **Incêndio:** $(680 - 40 - 30)/0{,}0015 = 406.666{,}67$, que vira **R$ 406.600** na centena (cai de fato em F2 ✔). Subir para 406.700 daria ticket de 680,05, e o total passaria de 1.000 (seção 7.3). O ticket final é recalculado a partir da IS na centena: $406.600 \times 0{,}0015 + 70 = 679{,}90$. Os R$ 0,10 que sobram aparecem na coluna `dif_arredondamento`.
- **Danos Elétricos:** $150/0{,}005 =$ **R$ 30.000**
- **Roubo:** $170/0{,}01 =$ **R$ 17.000**

**Diferenças por cobertura** (colunas `<cob>_dif_ticket` e `<cob>_dif_prop`):

| Cobertura | Ticket recomendado | Ticket final | `dif_ticket` | Proporção recomendada | Proporção final | `dif_prop` |
|---|---|---|---|---|---|---|
| Incêndio | 600 | 679,90 | −79,90 (recebeu) | 60% | 68% | 0,08 |
| Danos Elétricos | 250 | 150 | +100 (cedeu) | 25% | 15% | 0,10 |
| Roubo | 150 | 170 | −20 (recebeu) | 15% | 17% | 0,02 |

### 6.2 Pacote B: cobertura fora do pacote e empate entre faixas

Incêndio 700, sem DE, Roubo 300 ($T = 1000$).

- **F1:** o Incêndio trava em 450, e Roubo só vai até 500. Sobram 50.
- **F2:** o ticket 700 cabe direto, com distorção 0 e IS de **420.000**.
- **F3:** o ticket 700 também cabe direto, com distorção 0 e IS de 508.333,33, que vira **508.300** na centena (ticket 699,96).

F2 e F3 empatam em tudo, e o desempate pela IS decide: com `desempate_is="menor"` a escolha é F2 (IS 420.000); com `"maior"`, F3 (IS 508.300).

### 6.3 Pacote C: não cabe tudo

É igual ao pacote A, mas com IS máx de Incêndio = 300 mil e IS máx de Roubo = 20 mil (ticket máx. 200).

- **F3:** sem interseção com a IS máx (500k > 300k), então é descartada.
- **F1:** os máximos somam 450 + 150 + 200 = 800, o que deixa sobra de 200.
- **F2:** com o máximo de 300k, o ticket do Incêndio vai até 520; os máximos somam 870, o que deixa **sobra de 130** ✅.

Resultado: F2, com todas as coberturas no máximo e `status = sobra_nao_distribuida` (130 não distribuídos).

### 6.4 Versão 2: o mesmo exemplo com IS sugeridas

Suponha que o negócio sugira IS de 350.000 para o Incêndio, 30.000 para Danos Elétricos e 15.000 para Roubo. Rodando com β = 0 (versão 1) e β = 1:

| Pacote | β | Incêndio: ticket / IS | Roubo: ticket / IS | Desvio da IS sugerida (Incêndio / Roubo) |
|---|---|---|---|---|
| A | 0 | 679,9 / 406.600 | 170 / 17.000 | +16% / +13% |
| A | 1 | 676,9 / 404.600 | 173 / 17.300 | +16% / +15% |
| B | 0 | 700 / 420.000 | 300 / 30.000 | +20% / **+100%** |
| B | 1 | 784 / 476.000 | 216 / 21.600 | +36% / **+44%** |
| C | 0 e 1 | 520 / 300.000 | 200 / 20.000 | −14% / +33% (tudo no máximo) |

- **Pacote A:** quase nada muda, porque as IS sugeridas são quase coerentes com os tickets recomendados ($IS^{ref} \times taxa \approx t^0$; seção 4.5).
- **Pacote B:** o Roubo recomendado (300) corresponde a IS de 30.000, o dobro da sugerida. Com β = 1, parte do ticket do Roubo vai para o Incêndio, e os dois desvios ficam equilibrados (+36% e +44%), em vez de um deles ficar em +100%. O custo é a proporção de ticket: Roubo passa de 30% para 21,6% do pacote.
- **Pacote C:** todas as coberturas estão no máximo, então não há o que reequilibrar.

---

## 7. IS na centena e precisão numérica

A IS do cliente precisa ser um **múltiplo de 100**: 72.000 ou 72.100, nunca 72.002 ou 72.101. O **ticket** pode ter decimais.

As faixas da cobertura especial são inteiras (por exemplo, F1 de 0 a 75.000 e F2 a partir de 75.001). Na centena, a F2 passa a começar, na prática, em **75.100**: não existe IS na centena entre 75.001 e 75.099.

### 7.1 Por que não arredondar "de qualquer jeito"

Tratar a IS como variável "múltipla de 100" dentro da otimização transformaria o problema num problema inteiro, bem mais pesado. Também não é necessário: a otimização continua **exatamente a mesma da versão 2**, e só a última etapa, o arredondamento, muda.

O efeito no ticket é pequeno. Mudar a IS em R$ 100 muda o ticket em 100 × taxa: R$ 0,15 numa cobertura de taxa 0,0015 e R$ 1,00 numa de taxa 0,01.

O código faz em três etapas:

**1. Limites na centena antes de otimizar.** Os limites de IS de cada cobertura viram o primeiro e o último múltiplo de 100 dentro do intervalo:

$$IS^{min}_{100} = 100 \cdot \left\lceil \frac{IS^{min}}{100} \right\rceil \qquad IS^{max}_{100} = 100 \cdot \left\lfloor \frac{IS^{max}}{100} \right\rfloor$$

Só então eles são convertidos em limites de ticket. Uma IS mínima de 75.001, por exemplo, vira 75.100, e uma IS máxima de 199.950 vira 199.900.

- Se uma **faixa da especial** não tem nenhum múltiplo de 100 (por exemplo, de 200.001 a 200.050), ela é descartada, como na seção 5.4.
- Se uma **cobertura comum** não tem nenhum múltiplo de 100 entre a IS mínima e a máxima (por exemplo, de 5.050 a 5.090), não existe IS válida para ela. A linha sai com `status = cobertura_sem_is_valida`, e a IS dessa cobertura deve ser revista na base.

**2. Otimização contínua, como antes.** A bisseção encontra o ótimo com IS contínuas, que já ficam dentro dos limites na centena.

**3. Arredondamento controlado**, em cada linha:
- todas as IS começam arredondadas **para baixo** na centena, o que nunca fica abaixo do mínimo;
- percorrendo as coberturas da maior para a menor "sobra" de centena, a IS sobe **100** quando isso **aproxima** a soma dos tickets do ótimo contínuo, **não passa** do máximo e **não faz o ticket total ser ultrapassado**;
- o ticket final é **recalculado** a partir da IS na centena: $tf = IS \cdot taxa\ (+\ parte\ fixa)$.

Com isso, ticket e IS sempre batem entre si. A diferença em relação ao ótimo contínuo é de centavos ou poucos reais: num teste com 20 mil pacotes de 5 coberturas, a mediana foi de R$ 0,10 e o máximo, R$ 1,63 por pacote.

**Exemplo** (pacote A, seção 6.1): o Incêndio contínuo é 406.666,67. Para baixo, vira 406.600 (ticket 679,90). Subir para 406.700 levaria o total a 1.000,05, acima do ticket total, então fica em 406.600.

### 7.2 Cuidados com ponto flutuante

O computador representa decimais de forma aproximada: um limite que deveria ser 75.000 pode chegar como 75.000,0000000001. Um arredondamento para cima ingênuo transformaria isso em 75.100 e cortaria uma IS válida. Por isso, antes de qualquer arredondamento para cima ou para baixo, os valores (já divididos por 100) são arredondados para 6 casas decimais, o que remove esse ruído.

Além disso, o código faz uma **checagem final**: se alguma IS ficar fora de [IS mín, IS máx] ou não for múltipla de 100, a linha recebe `status = is_fora_dos_limites`. Isso não deve acontecer; a checagem existe como rede de segurança.

### 7.3 O ticket total nunca é ultrapassado

Regra obrigatória: a soma dos tickets finais **nunca passa** do ticket total do pacote. O teto de cada linha é

$$teto = \min(ticket\_total,\; \textstyle\sum t_{cob})$$

Assim, se a coluna `ticket_total` vier um pouco menor que a soma dos `<cob>_t`, prevalece o menor dos dois.

A garantia tem duas camadas:

1. **No ajuste guloso:** a IS só sobe 1 unidade se a nova soma de tickets continuar dentro do teto.
2. **Trava final:** a soma é recalculada do zero, exatamente como as colunas `<cob>_tf` são geradas. Se alguma linha ainda passar do teto, desce 1 unidade de IS da cobertura de maior taxa que ainda pode descer (sem ficar abaixo do mínimo), até caber.

**Empatar com o teto é permitido.** No pacote B, 700 + 300 = 1.000 exatos. A única folga aceita na comparação é o ruído de representação binária do computador (cerca de 0,00000000001 num ticket de R$ 1.000). Assim, 1000,0000000000002 conta como 1.000, mas qualquer excesso real é bloqueado.

**Tickets em centavos:** com `casas_ticket=2`, os `<cob>_tf` saem arredondados em centavos e a garantia vale já com esse arredondamento. Isso importa porque arredondar os tickets *depois* da função, por conta própria, poderia fazer a soma passar do total em 1 centavo.

**Quanto se perde:** o preço da trava e da centena é deixar um pouco de ticket sem distribuir. No teste com 20 mil pacotes, a mediana foi de R$ 0,10 por pacote, com 99% dos pacotes abaixo de R$ 1,33 e máximo de R$ 1,63.

| Coluna | Significado |
|---|---|
| `sobra` | ticket que não coube por causa dos limites de IS (vem do ótimo contínuo) |
| `dif_arredondamento` | ótimo contínuo − soma dos tickets finais (efeito da IS na centena; centavos ou poucos reais, nunca negativo quando o ótimo contínuo usa o total inteiro) |

O único caso em que o total não pode ser respeitado é quando **nem as IS mínimas cabem nele**. Pela sua regra de negócio isso não ocorre, mas, se ocorrer, a linha recebe `status = minimos_excedem_T`.

A escolha da faixa usa a sobra e o objetivo do ótimo contínuo, para que o arredondamento não decida a faixa.

---

## 8. Algoritmo passo a passo

```
para cada linha (pacote × perfil × faixa):      ← vetorizado no código
    1. T = soma dos <cob>_t recomendados
    2. coberturas comuns: L = ismin·taxa,  U = ismax·taxa
    3. especial: L = ismin·taxa + agravo_1 + agravo_2;  U = ismax·taxa + agravo_1 + agravo_2
       se não houver múltiplo de 100 entre ismin e ismax: faixa inválida
    4. alvo combinado (versão 2, seção 4.5):
         tref = fixo + taxa·ISref;  a = 1/t0;  b = β·t0/(taxa·ISref)²
         w = a + b;  m = (a·t0 + b·tref)/w
       bisseção em μ:  Σ clip(m + μ/w, L, U) = T
       t = clip(m + μ/w, L, U); sobra; objetivo
    (antes de tudo: limites na centena  ismin → 100·ceil(ismin/100),  ismax → 100·floor(ismax/100))
    5. converter: IS_comum = t / taxa;  IS_esp = (t − agravo_1 − agravo_2) / taxa
    6. IS na centena: para baixo + ajuste guloso (+100) sem sair dos limites na centena
       e sem passar do ticket total
       tf = IS_centena · taxa (+ parte fixa)
para cada pacote × perfil:
    7. escolher a linha: válida > não-piso > menor |sobra| > menor objetivo > desempate IS
    8. reportar <cob>_tf, <cob>_isf, <cob>_dif_ticket, <cob>_dif_prop, <cob>_dif_isref,
       sobra, dif_arredondamento, distorcao, distorcao_is, objetivo, status
```

---

## 9. Estrutura dos dados

Um único dataframe, com **uma linha por pacote × perfil × faixa** da cobertura especial:

| Coluna | Descrição |
|---|---|
| colunas de perfil | identificam o perfil do cliente |
| `ticket_total` | ticket total do pacote |
| `<cob>_proporcao` | proporção recomendada de cada cobertura (informativa) |
| `<cob>_t` | ticket recomendado de cada cobertura (NaN/0 = fora do pacote); é o que entra no modelo |
| `<cob>_ismin`, `<cob>_ismax` | limites de IS. Na **especial**, variam por faixa (limites da faixa ∩ limites da cobertura); nas demais, se repetem |
| `<cob>_taxa` | taxa. Na **especial**, a da faixa; nas demais, se repete |
| `agravo_1` | agravo 1 da especial naquela faixa |
| `agravo_2` | agravo 2 da especial (depende só do perfil) |
| coluna de valor fixo | valor fixo das coberturas de valor fixo (seção 5.6). Nelas, `<cob>_taxa` é opcional |
| `<cob>_isref` | *(versão 2, opcional)* IS sugerida pelo negócio. Coluna ausente, vazia ou 0: a cobertura não é ponderada pela IS |

Exemplo do pacote A (3 linhas, uma por faixa; só mudam as colunas da especial `cob_01` e o `agravo_1`):

| perfil | pacote | cob_01_t | cob_02_t | cob_03_t | cob_01_ismin | cob_01_ismax | cob_01_taxa | agravo_1 | agravo_2 | … |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | A | 600 | 250 | 150 | 50.000 | 200.000 | 0,0020 | 20 | 30 | … |
| P1 | A | 600 | 250 | 150 | 200.001 | 500.000 | 0,0015 | 40 | 30 | … |
| P1 | A | 600 | 250 | 150 | 500.001 | 1.000.000 | 0,0012 | 60 | 30 | … |

Os nomes seguem o padrão `<cob>` + sufixo. Os sufixos padrão são `_t`, `_ismin`, `_ismax`, `_taxa`, `_isref` (entrada) e `_tf`, `_isf`, `_dif_ticket`, `_dif_prop`, `_dif_isref` (saída). Se os da sua base forem outros, passe só os que mudam:

```python
sufixos={"t": "_ticket", "ismin": "_is_min_final", "ismax": "_is_max_final"}
```

Se algum nome não bater, a função para com um erro listando as colunas que não encontrou.

---

## 10. Como usar o código

```python
from otimizacao_is_v3 import calcular_is

resultado = calcular_is(
    df,
    coberturas=["cob_01", "cob_02", "cob_03", ...],   # todas, incluindo a especial
    especial="cob_01",
    chave_grupo=["regiao", "tipo_imovel", "pacote"],  # identifica um pacote × perfil
    sufixos={"ismax": "_is_max_final"},               # só os sufixos que diferem do padrão
    desempate_is="menor",                             # ou "maior"
    col_ticket_total="ticket_total",                  # teto: min(ticket_total, soma dos <cob>_t)
    casas_ticket=None,                                # ex.: 2 para tickets em centavos
    coberturas_valor_fixo={                           # seção 5.6
        "cob_04": {"valor": "cob_04_valor"},
    },
    beta=1.0,                                         # peso da IS sugerida (0 = versão 1)
)
```

**`chave_grupo`** são as colunas que **não mudam** entre as linhas de faixa de um mesmo pacote × perfil, ou seja, as colunas de perfil mais o que identifica o pacote. Se não houver um id de pacote, as colunas `<cob>_proporcao` servem.

**`resultado`** tem uma linha por pacote × perfil (a linha da faixa escolhida, com todas as colunas originais), acrescida de:
- `<cob>_tf`: ticket final de cada cobertura (pode ter decimais, recalculado a partir da IS na centena);
- `<cob>_isf`: IS final de cada cobertura (**múltiplo de 100**, dentro de [IS mín, IS máx]);
- `<cob>_dif_ticket`: ticket recomendado − ticket final. Positivo: a cobertura **cedeu** ticket (por exemplo, travou na IS máxima). Negativo: **recebeu** ticket de outras;
- `<cob>_dif_prop`: diferença absoluta entre a proporção recomendada ($t^0_i / \sum t^0$) e a proporção final ($tf_i / \sum tf$);
- `<cob>_dif_isref`: IS final − IS sugerida (positivo: acima da sugerida). Fica 0 quando não há IS sugerida;
- `distorcao`: distorção de ticket, $\sum (tf - t^0)^2/t^0$;
- `distorcao_is`: distorção de IS, $\sum t^0 \,((IS - IS^{ref})/IS^{ref})^2$, sem o β;
- `objetivo`: `distorcao` + β × `distorcao_is`;
- `sobra`, `dif_arredondamento`, `fator_escala` ($1 + \mu$; igual ao $s$ da versão 1 quando β = 0; sem significado quando há sobra, porque todas as coberturas estão travadas no máximo) e `status`.

A soma de `<cob>_dif_ticket` de todas as coberturas é igual a `sobra` + `dif_arredondamento`: o que não foi redistribuído entre as coberturas é justamente o que ficou sem distribuir.

**Bases grandes:** a função processa a base em blocos de cerca de 300 mil linhas (parâmetro `linhas_por_bloco`), sempre com todas as faixas de um mesmo pacote × perfil no mesmo bloco. Isso limita o uso de memória sem mudar o resultado. Num teste com 2.752.512 linhas (393.216 pacotes × perfis, 7 faixas e 10 coberturas), a execução levou cerca de 45 s e o pico de memória foi de 2,3 GB, incluindo o próprio dataframe. O parâmetro `n_iter` (iterações da bisseção) tem padrão 100, que já chega à precisão máxima do computador, e não precisa ser alterado.

Com `retornar_todas_faixas=True`, a função devolve também todas as linhas (todas as faixas) com seus `<cob>_tf`, `<cob>_isf`, sobra e distorção, para auditar **por que** uma faixa ganhou.

**Valores de `status`:**

| status | significado |
|---|---|
| `ok` | ticket totalmente distribuído |
| `sobra_nao_distribuida` | todos os tetos somados < T; veja a coluna `sobra` |
| `minimos_excedem_T` | nem as IS mínimas cabem no ticket total (não deveria ocorrer) |
| `sem_faixa_valida` | em nenhuma faixa existe múltiplo de 100 entre `ismin` e `ismax` da especial |
| `cobertura_sem_is_valida` | alguma cobertura comum não tem múltiplo de 100 entre `ismin` e `ismax` (ex.: 5.050 a 5.090); revise os limites dela |
| `is_fora_dos_limites` | rede de segurança: alguma IS saiu de [IS mín, IS máx] ou não é múltipla de 100 (não deve ocorrer) |

---

## 11. Premissas e pontos de atenção

- **Taxas já divididas por 100:** todas as colunas `<cob>_taxa` devem chegar já divididas por 100 (ex.: 0,0015), e o ticket é sempre $IS \cdot taxa$, mais a parte fixa quando houver.
- **Coberturas fora do pacote ou não elegíveis:**
  - *Fora do pacote:* `<cob>_t` = 0 (com `ismin`/`ismax` zerados). A cobertura sai do problema, com IS 0 e ticket 0.
  - *Não elegível ao perfil:* `<cob>_taxa` = 0. A IS e o ticket ficam em 0 mesmo que o pacote recomende um ticket para ela, e esse ticket é redistribuído para as demais coberturas. Na especial, os agravos também não são cobrados.
- **Limites da especial:** `ismin = max(IS mín, início da faixa)` e `ismax = min(IS máx, fim da faixa)`. Se a especial estiver fora do pacote, tanto faz o resultado dessa conta, porque o ticket 0 já a tira do problema.
- **Taxa sobre a IS toda:** a IS toda usa a taxa da faixa em que cai, e não há cálculo progressivo.
- **Bordas das faixas:** `ismin` e `ismax` são inclusivos. Com a IS na centena, F1 até 75.000 e F2 a partir de 75.001 viram, na prática, F1 até 75.000 e F2 a partir de 75.100. Se as faixas da sua base vierem encostadas (F1 até 75.000 e F2 a partir de 75.000), a IS de exatamente 75.000 é aceita nas duas, e o critério de escolha decide.
- **Arredondamento:** a IS sai sempre múltipla de 100 (seção 7). O 100 é fixo no código (constante `PASSO_IS`).
- **IS sugerida e β (versão 2):** a penalização só age onde a IS sugerida diverge do que o ticket recomendado implica (seção 4.5). Calibre o β com os pacotes reais do negócio; o padrão 1 dá peso igual aos desvios relativos de ticket e de IS.
- **Métrica de distorção:** a forma $\sum (t - t^0)^2 / t^0$ é a que gera a regra "escala proporcional + travas". Outras métricas (por exemplo, desvio absoluto) mudam a solução e perdem a forma fechada.
