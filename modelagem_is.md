# Distribuição de ticket entre coberturas e cálculo da Importância Segurada (IS)

> Documento de modelagem que acompanha o código `otimizacao_is.py`.

---

## 1. O problema em uma frase

Cada pacote de seguro residencial tem um **ticket total** $T$, repartido entre coberturas segundo **proporções recomendadas**. Queremos descobrir a **IS de cada cobertura** de forma que três coisas aconteçam:

1. a IS respeite o mínimo e o máximo de cada cobertura;
2. o ticket total seja distribuído integralmente, quando isso for possível;
3. as proporções recomendadas sejam distorcidas **o mínimo possível**.

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
- Faz-se a busca no meio, cerca de 200 iterações. No código isso é vetorizado e todas as linhas são resolvidas de uma vez.

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

---

## 5. A cobertura especial

### 5.1 O que muda

$$t_k = IS \cdot taxa(IS) + A_1(IS) + A_2$$

- $A_2$ depende só do perfil, então é constante.
- A **taxa** e o **agravo 1** dependem da **faixa** em que a IS cai.
- A taxa vale sobre a **IS inteira**, e não de forma progressiva.

O ticket recomendado da especial já contém os três termos.

### 5.2 Dentro de uma faixa, tudo volta a ser linear

Fixada a faixa $j$:

$$t_k = IS \cdot taxa_j + A_{1j} + A_2 \quad\Longleftrightarrow\quad IS = \frac{t_k - A_{1j} - A_2}{taxa_j}$$

O intervalo de IS permitido nessa faixa é a interseção da faixa com os limites da cobertura. No seu layout, ele já vem pronto nas colunas `ismin_` e `ismax_` da especial, linha a linha:

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
4. **Menor distorção** relativa.
5. **Desempate pela IS da especial** (parâmetro `desempate_is = "menor"` ou `"maior"`).

O critério 5 existe por causa da sobreposição descrita na seção 5.3: duas faixas podem chegar ao **mesmo ticket**, com a mesma distorção, mas com **IS diferentes**. Quem decide qual preferir é a regra de negócio.

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
| F2 | 200k – 500k | 0,15% | 40 | 370 – 820 |
| F3 | 500k – 1M | 0,12% | 60 | 690 – 1.290 |

Repare na sobreposição: o ticket 700, por exemplo, cabe em F2 e em F3.

### 6.1 Pacote A: redistribuição clássica

$T = 1000$, com recomendação Incêndio 600 (60%), DE 250 (25%) e Roubo 150 (15%).

DE queria 250, mas o máximo é 150. Fica travado em 150, e sobram 100 para redistribuir.

| Faixa | Incêndio | DE | Roubo | Sobra | Distorção |
|---|---|---|---|---|---|
| F1 | 450 (travado no máx.) | 150 | 400 | 0 | 494,2 |
| **F2** | **680** | **150** | **170** | **0** | **53,3** ✅ |
| F3 | 690 (travado no mín.) | 150 | 160 | 0 | 54,2 |

Em F2, Incêndio e Roubo foram escalados pelo mesmo fator: $s = 850/750 \approx 1{,}133$. A proporção entre eles, 600:150 = 680:170 = 4:1, ficou **intacta**.

**IS finais:**
- **Incêndio:** $(680 - 40 - 30)/0{,}0015 \approx$ **R$ 406.667** (cai de fato em F2 ✔)
- **Danos Elétricos:** $150/0{,}005 =$ **R$ 30.000**
- **Roubo:** $170/0{,}01 =$ **R$ 17.000**

### 6.2 Pacote B: cobertura fora do pacote e empate entre faixas

Incêndio 700, sem DE, Roubo 300 ($T = 1000$).

- **F1:** o Incêndio trava em 450, e Roubo só vai até 500. Sobram 50.
- **F2:** o ticket 700 cabe direto, com distorção 0 e IS de **420.000**.
- **F3:** o ticket 700 também cabe direto, com distorção 0 e IS de **508.333**.

F2 e F3 empatam em tudo, e o desempate pela IS decide: com `desempate_is="menor"` a escolha é F2 (IS 420 mil); com `"maior"`, F3 (IS 508 mil).

### 6.3 Pacote C: não cabe tudo

É igual ao pacote A, mas com IS máx de Incêndio = 300 mil e IS máx de Roubo = 20 mil (ticket máx. 200).

- **F3:** sem interseção com a IS máx (500k > 300k), então é descartada.
- **F1:** os máximos somam 450 + 150 + 200 = 800, o que deixa sobra de 200.
- **F2:** com o máximo de 300k, o ticket do Incêndio vai até 520; os máximos somam 870, o que deixa **sobra de 130** ✅.

Resultado: F2, com todas as coberturas no máximo e `status = sobra_nao_distribuida` (130 não distribuídos).

---

## 7. Algoritmo passo a passo

```
para cada linha (pacote × perfil × faixa):      ← vetorizado no código
    1. T = soma dos t_<cob> recomendados
    2. coberturas comuns: L = ismin·taxa,  U = ismax·taxa
    3. especial: L = ismin·taxa + agravo_1 + agravo_2;  U = ismax·taxa + agravo_1 + agravo_2
       se ismin > ismax: faixa inválida
    4. bisseção em s:  Σ clip(s·t0, L, U) = T
       t = clip(s·t0, L, U); sobra; distorção
    5. converter: isf_comum = tf / taxa;  isf_esp = (tf − agravo_1 − agravo_2) / taxa
para cada pacote × perfil:
    6. escolher a linha: válida > não-piso > menor |sobra| > menor distorção > desempate IS
    7. reportar tf_<cob>, isf_<cob>, sobra, distorção, status
```

---

## 8. Estrutura dos dados

Um único dataframe, com **uma linha por pacote × perfil × faixa** da cobertura especial:

| Coluna | Descrição |
|---|---|
| colunas de perfil | identificam o perfil do cliente |
| `ticket_total` | ticket total do pacote |
| `proporcao_<cob>` | proporção recomendada de cada cobertura (informativa) |
| `t_<cob>` | ticket recomendado de cada cobertura (NaN/0 = fora do pacote); é o que entra no modelo |
| `ismin_<cob>`, `ismax_<cob>` | limites de IS. Na **especial**, variam por faixa (limites da faixa ∩ limites da cobertura); nas demais, se repetem |
| `taxa_<cob>` | taxa. Na **especial**, a da faixa; nas demais, se repete |
| `agravo_1` | agravo 1 da especial naquela faixa |
| `agravo_2` | agravo 2 da especial (depende só do perfil) |

Exemplo do pacote A (3 linhas, uma por faixa; só mudam as colunas da especial `cob_01` e o `agravo_1`):

| perfil | pacote | t_cob_01 | t_cob_02 | t_cob_03 | ismin_cob_01 | ismax_cob_01 | taxa_cob_01 | agravo_1 | agravo_2 | … |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | A | 600 | 250 | 150 | 50.000 | 200.000 | 0,0020 | 20 | 30 | … |
| P1 | A | 600 | 250 | 150 | 200.000 | 500.000 | 0,0015 | 40 | 30 | … |
| P1 | A | 600 | 250 | 150 | 500.000 | 1.000.000 | 0,0012 | 60 | 30 | … |

Se os prefixos forem outros, passe `prefixos={"t": "...", "ismin": "...", ...}`.

---

## 9. Como usar o código

```python
from otimizacao_is import calcular_is

resultado = calcular_is(
    df,
    coberturas=["cob_01", "cob_02", "cob_03", ...],   # todas, incluindo a especial
    especial="cob_01",
    chave_grupo=["regiao", "tipo_imovel", "pacote"],  # identifica um pacote × perfil
    desempate_is="menor",                             # ou "maior"
)
```

**`chave_grupo`** são as colunas que **não mudam** entre as linhas de faixa de um mesmo pacote × perfil, ou seja, as colunas de perfil mais o que identifica o pacote. Se não houver um id de pacote, as colunas `proporcao_<cob>` servem.

**`resultado`** tem uma linha por pacote × perfil (a linha da faixa escolhida, com todas as colunas originais), acrescida de:
- `tf_<cob>`: ticket final de cada cobertura;
- `isf_<cob>`: IS final de cada cobertura;
- `sobra`, `distorcao`, `fator_escala` (o $s$) e `status`.

Com `retornar_todas_faixas=True`, a função devolve também todas as linhas (todas as faixas) com seus `tf_`, `isf_`, sobra e distorção, para auditar **por que** uma faixa ganhou.

**Valores de `status`:**

| status | significado |
|---|---|
| `ok` | ticket totalmente distribuído |
| `sobra_nao_distribuida` | todos os tetos somados < T; veja a coluna `sobra` |
| `minimos_excedem_T` | nem os mínimos cabem em T (não deveria ocorrer) |
| `sem_faixa_valida` | em nenhuma faixa `ismin <= ismax` para a especial |

---

## 10. Premissas e pontos de atenção

- **Taxa sobre a IS inteira:** a IS inteira usa a taxa da faixa em que cai, e não há cálculo progressivo.
- **Bordas das faixas:** o código trata `ismin` e `ismax` como inclusivos. Se F1 termina em 200k e F2 começa em 200k, a IS de exatamente 200k é aceita nas duas, e o critério de escolha decide.
- **Arredondamento:** a IS sai com casas decimais. Se for preciso arredondar (por exemplo, para múltiplos de R$ 1.000), faça isso depois. O ticket recalculado mudará um pouco, e vale recalcular o prêmio a partir da IS arredondada.
- **Métrica de distorção:** a forma $\sum (t - t^0)^2 / t^0$ é a que gera a regra "escala proporcional + travas". Outras métricas (por exemplo, desvio absoluto) mudam a solução e perdem a forma fechada.
