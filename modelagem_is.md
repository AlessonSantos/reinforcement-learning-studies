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
4. **Menor distorção** relativa.
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

**Com taxa (se for incorporada no futuro).** Basta criar a coluna `<cob>_taxa`. O ticket da cobertura passa a variar entre $valor\_fixo + IS^{min} \cdot taxa$ e $valor\_fixo + IS^{max} \cdot taxa$, e a IS volta a ser calculada por $IS = (t - valor\_fixo) / taxa$, inteira e dentro dos limites.

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

Como a IS é inteira, as faixas não se encostam: F1 vai até 200.000 e F2 começa em 200.001.

Repare na sobreposição **de ticket**: o ticket 700, por exemplo, cabe em F2 e em F3.

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
- **Incêndio:** $(680 - 40 - 30)/0{,}0015 = 406.666{,}67$, que vira **R$ 406.666** (cai de fato em F2 ✔). Arredondar para 406.667 daria ticket de 680,0005, e o total passaria de 1.000 (seção 7.3). O ticket final é recalculado a partir da IS inteira: $406.666 \times 0{,}0015 + 70 = 679{,}999$.
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

## 7. IS inteira e precisão numérica

A IS do cliente precisa ser um **número inteiro** (não existe IS de R$ 500.000,50). O **ticket** pode ter decimais. Como as faixas também são inteiras (por exemplo, F1 de 0 a 75.000 e F2 a partir de 75.001), não existe IS entre uma faixa e outra.

### 7.1 Por que não arredondar "de qualquer jeito"

Tratar a IS como variável inteira dentro da otimização transformaria o problema num problema inteiro, bem mais pesado. Também não é necessário, porque o efeito de mudar a IS em R$ 1 sobre o ticket é minúsculo: é a própria taxa, de alguns décimos de centavo.

Então o código faz em três etapas:

**1. Limites inteiros antes de otimizar.** Os limites de IS de cada cobertura viram

$$IS^{min}_{int} = \lceil IS^{min} \rceil \qquad IS^{max}_{int} = \lfloor IS^{max} \rfloor$$

e só então são convertidos em limites de ticket. Uma IS mínima de 1.000,40, por exemplo, vira 1.001, porque 1.000 ficaria abaixo do mínimo. Se uma faixa não tem nenhum inteiro possível ($\lceil IS^{min} \rceil > \lfloor IS^{max} \rfloor$), ela é descartada.

**2. Otimização contínua, como antes.** A bisseção encontra o ótimo com IS contínuas, que já ficam dentro dos limites inteiros.

**3. Arredondamento controlado**, em cada linha:
- todas as IS começam em $\lfloor IS \rfloor$, o que nunca fica abaixo do mínimo inteiro;
- percorrendo as coberturas da maior para a menor parte fracionária, a IS sobe 1 unidade quando isso **aproxima** a soma dos tickets do ótimo contínuo, **não passa** do máximo inteiro e **não faz o ticket total ser ultrapassado**;
- o ticket final é **recalculado** a partir da IS inteira: $tf = IS \cdot taxa\ (+\ agravo_1 + agravo_2)$.

Com isso, ticket e IS sempre batem entre si, e a diferença em relação ao ótimo contínuo é de centavos (no teste com 60 mil pacotes, no máximo R$ 0,02).

### 7.2 Cuidados com ponto flutuante

O computador representa decimais de forma aproximada: um limite que deveria ser 75.000 pode chegar como 75.000,0000000001. Um $\lceil \cdot \rceil$ ingênuo transformaria isso em 75.001 e cortaria uma IS válida. Por isso, antes de qualquer $\lceil \cdot \rceil$ ou $\lfloor \cdot \rfloor$, os valores são arredondados para 6 casas decimais, o que remove esse ruído.

Além disso, o código faz uma **checagem final**: se alguma IS inteira ficar fora de [IS mín, IS máx], a linha recebe `status = is_fora_dos_limites`. Isso não deve acontecer; a checagem existe como rede de segurança.

### 7.3 O ticket total nunca é ultrapassado

Regra obrigatória: a soma dos tickets finais **nunca passa** do ticket total do pacote. O teto de cada linha é

$$teto = \min(ticket\_total,\; \textstyle\sum t_{cob})$$

Assim, se a coluna `ticket_total` vier um pouco menor que a soma dos `<cob>_t`, prevalece o menor dos dois.

A garantia tem duas camadas:

1. **No ajuste guloso:** a IS só sobe 1 unidade se a nova soma de tickets continuar dentro do teto.
2. **Trava final:** a soma é recalculada do zero, exatamente como as colunas `<cob>_tf` são geradas. Se alguma linha ainda passar do teto, desce 1 unidade de IS da cobertura de maior taxa que ainda pode descer (sem ficar abaixo do mínimo), até caber.

**Empatar com o teto é permitido.** No pacote B, 700 + 300 = 1.000 exatos. A única folga aceita na comparação é o ruído de representação binária do computador (cerca de 0,00000000001 num ticket de R$ 1.000). Assim, 1000,0000000000002 conta como 1.000, mas qualquer excesso real é bloqueado.

**Tickets em centavos:** com `casas_ticket=2`, os `<cob>_tf` saem arredondados em centavos e a garantia vale já com esse arredondamento. Isso importa porque arredondar os tickets *depois* da função, por conta própria, poderia fazer a soma passar do total em 1 centavo.

**Quanto se perde:** o preço da trava é deixar alguns centavos sem distribuir. No teste com 60 mil pacotes, foram no máximo R$ 0,02 por pacote.

| Coluna | Significado |
|---|---|
| `sobra` | ticket que não coube por causa dos limites de IS (vem do ótimo contínuo) |
| `dif_arredondamento` | ótimo contínuo − soma dos tickets finais (efeito da IS inteira; centavos, nunca negativo quando o ótimo contínuo usa o total inteiro) |

O único caso em que o total não pode ser respeitado é quando **nem as IS mínimas cabem nele**. Pela sua regra de negócio isso não ocorre, mas, se ocorrer, a linha recebe `status = minimos_excedem_T`.

A escolha da faixa usa a sobra e a distorção do ótimo contínuo, para que o arredondamento (centavos) não decida a faixa.

---

## 8. Algoritmo passo a passo

```
para cada linha (pacote × perfil × faixa):      ← vetorizado no código
    1. T = soma dos <cob>_t recomendados
    2. coberturas comuns: L = ismin·taxa,  U = ismax·taxa
    3. especial: L = ismin·taxa + agravo_1 + agravo_2;  U = ismax·taxa + agravo_1 + agravo_2
       se ceil(ismin) > floor(ismax): faixa inválida
    4. bisseção em s:  Σ clip(s·t0, L, U) = T
       t = clip(s·t0, L, U); sobra; distorção
    (antes de tudo: limites inteiros  ismin → ceil(ismin),  ismax → floor(ismax))
    5. converter: IS_comum = t / taxa;  IS_esp = (t − agravo_1 − agravo_2) / taxa
    6. IS inteira: floor + ajuste guloso (+1) sem sair de [ceil(ismin), floor(ismax)]
       tf = IS_inteira · taxa (+ agravos na especial)
para cada pacote × perfil:
    7. escolher a linha: válida > não-piso > menor |sobra| > menor distorção > desempate IS
    8. reportar <cob>_tf, <cob>_isf, sobra, dif_arredondamento, distorção, status
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

Exemplo do pacote A (3 linhas, uma por faixa; só mudam as colunas da especial `cob_01` e o `agravo_1`):

| perfil | pacote | cob_01_t | cob_02_t | cob_03_t | cob_01_ismin | cob_01_ismax | cob_01_taxa | agravo_1 | agravo_2 | … |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | A | 600 | 250 | 150 | 50.000 | 200.000 | 0,0020 | 20 | 30 | … |
| P1 | A | 600 | 250 | 150 | 200.001 | 500.000 | 0,0015 | 40 | 30 | … |
| P1 | A | 600 | 250 | 150 | 500.001 | 1.000.000 | 0,0012 | 60 | 30 | … |

Os nomes seguem o padrão `<cob>` + sufixo. Os sufixos padrão são `_t`, `_ismin`, `_ismax`, `_taxa` (entrada) e `_tf`, `_isf` (saída). Se os da sua base forem outros, passe só os que mudam:

```python
sufixos={"t": "_ticket", "ismin": "_is_min_final", "ismax": "_is_max_final"}
```

Se algum nome não bater, a função para com um erro listando as colunas que não encontrou.

---

## 10. Como usar o código

```python
from otimizacao_is import calcular_is

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
)
```

**`chave_grupo`** são as colunas que **não mudam** entre as linhas de faixa de um mesmo pacote × perfil, ou seja, as colunas de perfil mais o que identifica o pacote. Se não houver um id de pacote, as colunas `<cob>_proporcao` servem.

**`resultado`** tem uma linha por pacote × perfil (a linha da faixa escolhida, com todas as colunas originais), acrescida de:
- `<cob>_tf`: ticket final de cada cobertura (pode ter decimais, recalculado a partir da IS inteira);
- `<cob>_isf`: IS final de cada cobertura (**inteira**, dentro de [IS mín, IS máx]);
- `sobra`, `dif_arredondamento`, `distorcao`, `fator_escala` (o $s$) e `status`.

Com `retornar_todas_faixas=True`, a função devolve também todas as linhas (todas as faixas) com seus `<cob>_tf`, `<cob>_isf`, sobra e distorção, para auditar **por que** uma faixa ganhou.

**Valores de `status`:**

| status | significado |
|---|---|
| `ok` | ticket totalmente distribuído |
| `sobra_nao_distribuida` | todos os tetos somados < T; veja a coluna `sobra` |
| `minimos_excedem_T` | nem as IS mínimas cabem no ticket total (não deveria ocorrer) |
| `sem_faixa_valida` | em nenhuma faixa existe IS inteira entre `ismin` e `ismax` da especial |
| `is_fora_dos_limites` | rede de segurança: alguma IS inteira saiu de [IS mín, IS máx] (não deve ocorrer) |

---

## 11. Premissas e pontos de atenção

- **Taxas já divididas por 100:** todas as colunas `<cob>_taxa` devem chegar já divididas por 100 (ex.: 0,0015), e o ticket é sempre $IS \cdot taxa$, mais a parte fixa quando houver.
- **Coberturas fora do pacote ou não elegíveis:**
  - *Fora do pacote:* `<cob>_t` = 0 (com `ismin`/`ismax` zerados). A cobertura sai do problema, com IS 0 e ticket 0.
  - *Não elegível ao perfil:* `<cob>_taxa` = 0. A IS e o ticket ficam em 0 mesmo que o pacote recomende um ticket para ela, e esse ticket é redistribuído para as demais coberturas. Na especial, os agravos também não são cobrados.
- **Limites da especial:** `ismin = max(IS mín, início da faixa)` e `ismax = min(IS máx, fim da faixa)`. Se a especial estiver fora do pacote, tanto faz o resultado dessa conta, porque o ticket 0 já a tira do problema.
- **Taxa sobre a IS inteira:** a IS inteira usa a taxa da faixa em que cai, e não há cálculo progressivo.
- **Bordas das faixas:** `ismin` e `ismax` são inclusivos e inteiros: F1 até 75.000 e F2 a partir de 75.001. Se as faixas da sua base vierem encostadas (F1 até 75.000 e F2 a partir de 75.000), a IS de exatamente 75.000 é aceita nas duas, e o critério de escolha decide.
- **Arredondamento:** a IS sai inteira (seção 7). Se no futuro for preciso arredondar para múltiplos maiores (por exemplo, R$ 1.000), a mesma lógica vale trocando 1 pelo passo desejado no ceil/floor e no ajuste.
- **Métrica de distorção:** a forma $\sum (t - t^0)^2 / t^0$ é a que gera a regra "escala proporcional + travas". Outras métricas (por exemplo, desvio absoluto) mudam a solução e perdem a forma fechada.
