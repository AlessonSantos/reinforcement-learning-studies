# Guia do código: `otimizacao_is_v2.py` (versão 2)

> Este guia explica o código parte por parte, incluindo as construções de Python, NumPy e pandas menos óbvias. Ele acompanha o documento de modelagem `modelagem_is_v2.md`.
>
> **O que muda em relação à versão 1:** o núcleo da otimização (seção 3), o cálculo do alvo combinado com a IS sugerida (seção 7.6), a coluna nova `<cob>_dif_isref` (7.8), os componentes do objetivo (7.9) e o critério de escolha da faixa (7.10). Essas partes estão marcadas com **(novo na versão 2)**. O resto do código é igual ao da versão 1. Quando um trecho implementa uma ideia de lá, a seção correspondente é indicada, como **[modelagem 4.2]**.
>
> Sugestão de leitura: comece pelas seções 0 e 1 (mapa e "kit" de NumPy) e depois siga o código na ordem em que ele executa, que é a ordem das seções 3 a 8.

---

## 0. Mapa do arquivo

| Bloco | Linhas (aprox.) | O que faz | Modelagem |
|---|---|---|---|
| Docstring inicial | 1–44 | Resumo do layout de entrada, saídas e modelo (com a dedução do alvo combinado) | 4.5 |
| `SUFIXOS_PADRAO`, `BETA_PADRAO` | 52–65 | Nomes das colunas e β padrão | 9, 4.6 |
| `resolver_distribuicao` | 71–123 | **Núcleo da otimização**: acha os tickets ótimos por bisseção, agora com alvo `m` e rigidez `w` | 4.5 |
| `limites_inteiros` | 130–138 | Transforma IS mín/máx em limites inteiros | 7.1, 7.2 |
| `tickets`, `passa_do_teto` | 144–157 | Cálculo do ticket a partir da IS e comparação segura com o teto | 7.3 |
| `arredondar_is` | 160–213 | Converte a IS contínua em inteira, sem passar do teto | 7.1, 7.3 |
| `calcular_is` | 219–465 | **Função principal**: prepara os dados, monta o alvo combinado, chama o núcleo, arredonda, escolhe a faixa | tudo |
| `if __name__ == "__main__":` | 472 em diante | Exemplo dos pacotes A, B e C com β = 0 e β = 1 | 6.4 |

**Fluxo de execução** (o que acontece quando você chama `calcular_is`):

```
calcular_is(df, ...)
 ├─ base grande? → divide em blocos e chama calcular_is em cada bloco
 ├─ confere nomes de colunas
 ├─ monta matrizes (linhas × coberturas): t0, taxa, ismin, ismax, isref, fixo
 ├─ define máscaras: ativo, elegivel, so_fixo, participa
 ├─ limites_inteiros → lo, hi  →  limites de ticket L, U
 ├─ alvo combinado: a, b, tref → rigidez w e alvo m           (novo na versão 2)
 ├─ resolver_distribuicao(m, w, L, U, T) → ticket contínuo ótimo
 ├─ ticket contínuo → IS contínua → arredondar_is → IS inteira → ticket final
 ├─ colunas de saída: _tf, _isf, _dif_ticket, _dif_prop, _dif_isref, objetivo, status...
 └─ escolhe a melhor faixa de cada pacote × perfil
```

---

## 1. Kit de NumPy e pandas usado no código

O código quase não tem laços sobre as linhas da base. Em vez de resolver pacote por pacote, ele monta **matrizes** e faz cada conta para todos os pacotes de uma vez. Isso é a **vetorização**, e é o que permite rodar milhões de linhas em segundos. As construções abaixo aparecem o tempo todo.

### 1.1 Matriz "linhas × coberturas"

Quase todas as variáveis são matrizes com **uma linha por linha do dataframe** (pacote × perfil × faixa) e **uma coluna por cobertura**, na ordem da lista `coberturas`. Por exemplo, com 3 linhas e as coberturas `["cob_01", "cob_02", "cob_03"]`:

```
t0 = [[600, 250, 150],     ← linha 0 (pacote A, faixa F1)
      [600, 250, 150],     ← linha 1 (pacote A, faixa F2)
      [600, 250, 150]]     ← linha 2 (pacote A, faixa F3)
```

`t0[1, 2]` é o ticket recomendado da `cob_03` na linha 1. `t0[:, 0]` é a coluna inteira da `cob_01` (o `:` quer dizer "todas as linhas").

### 1.2 `axis=1`: somar "por linha"

`t0.sum(axis=1)` soma ao longo das colunas, ou seja, dá **um total por linha**: `[1000, 1000, 1000]`. Já `axis=0` somaria cada cobertura ao longo das linhas, o que o código nunca usa.

### 1.3 Operações elemento a elemento

`lo * taxa + fixo` multiplica e soma **casa a casa** duas matrizes do mesmo formato. Não é multiplicação de matrizes da álgebra linear. Com matrizes 3×3, são 9 contas independentes.

### 1.4 Máscaras booleanas e `& | ~`

`ativo = t0 > 0` gera uma matriz de `True`/`False` do mesmo formato de `t0`. Máscaras se combinam com:

| Operador | Significado | Exemplo no código |
|---|---|---|
| `a & b` | E | `elegivel = ativo & tem_taxa` |
| `a \| b` | OU | `participa = elegivel \| so_fixo` |
| `~a` | NÃO | `~faixa_valida` |

Em NumPy se usam `&`, `|` e `~`, e **não** `and`, `or`, `not`, que só funcionam com um valor por vez. Os parênteses são obrigatórios quando há comparações, como em `(a > 0) & (b < 1)`.

### 1.5 `np.where(condição, se_verdadeiro, se_falso)`

É um "SE" do Excel aplicado casa a casa:

```python
L = np.where(participa, np.nan_to_num(lo * taxa) + fixo, 0.0)
```

Onde `participa` é `True`, pega `lo * taxa + fixo` (o limite mínimo de ticket); onde é `False`, pega `0`.

Um truque recorrente é usar `np.where` **dentro da divisão** para nunca dividir por zero:

```python
a = np.where(ativo, 1.0 / np.where(ativo, t0, 1.0), 1.0)
```

O `np.where` de dentro troca o `t0` por 1 onde a cobertura não está ativa, então a divisão nunca é por zero. O de fora descarta o resultado dessas casas e coloca 1 no lugar.

### 1.6 `with np.errstate(divide="ignore", invalid="ignore"):`

O NumPy às vezes calcula os dois lados de um `np.where` antes de escolher, e emite avisos de "divisão por zero" mesmo quando o resultado é descartado. Esse bloco `with` apenas **silencia esses avisos** no trecho indentado. Ele não muda nenhum resultado.

### 1.7 Broadcasting: `[:, None]` e `keepdims=True`

Às vezes é preciso combinar um **vetor por linha** (um número por pacote) com uma **matriz** (linhas × coberturas). Exemplo: o ajuste `mu` tem um valor por linha, e queremos `mu / w`.

- `mu` tem formato `(n,)`, um vetor.
- `mu[:, None]` tem formato `(n, 1)`, uma "coluna".
- `mu[:, None] / w` divide o `mu` de cada linha pela rigidez de cada cobertura daquela linha. O NumPy "estica" a coluna para todas as coberturas. Isso é o **broadcasting**.

```
mu[:, None] = [[0.6]],     w = [[0.00397, 1, 0.01667]]   →   [[151.2, 0.6, 36]]
```

`keepdims=True` em `t0.sum(axis=1, keepdims=True)` faz a mesma coisa de outra forma: a soma já sai no formato `(n, 1)`, pronta para dividir a matriz (`t0 / soma_t0`).

O contrário também aparece: `np.array([...])[None, :]` transforma um vetor **por cobertura** em formato `(1, k)`, para ser comparado com todas as linhas.

### 1.8 `np.clip(x, mínimo, máximo)`

Trava cada valor no intervalo: abaixo do mínimo vira o mínimo, acima do máximo vira o máximo. É a operação "clip" das fórmulas **[modelagem 4.2]**. Mínimo e máximo podem ser matrizes, e então cada casa tem o próprio limite.

### 1.9 `np.column_stack`

Junta vetores lado a lado como colunas de uma matriz:

```python
t0 = np.column_stack([coluna(col(c, "t")) for c in coberturas])
```

A list comprehension `[... for c in coberturas]` gera uma lista com um vetor por cobertura (a coluna `<cob>_t` de cada uma), e o `column_stack` monta a matriz linhas × coberturas.

### 1.10 Indexação avançada: `matriz[linhas, i]`

Quando `linhas` e `i` são **vetores do mesmo tamanho**, `matriz[linhas, i]` pega um elemento por par:

```
linhas = [0, 1, 2]
i      = [2, 0, 1]
base[linhas, i]   →   [base[0,2], base[1,0], base[2,1]]
```

O código usa isso para mexer, **em cada linha, numa cobertura diferente**, como "a cobertura com maior parte fracionária daquela linha". Também funciona para atribuir valores: `base[linhas[sobe], i[sobe]] += 1` soma 1 só nas casas escolhidas. Nesse caso, `linhas[sobe]` filtra os vetores pela máscara `sobe`.

### 1.11 `np.argsort` e `np.argmax`

- `np.argsort(-frac, axis=1)` devolve, para cada linha, **os índices das coberturas** em ordem decrescente de `frac`. O sinal de menos inverte a ordem, já que o `argsort` ordena de forma crescente.
- `np.argmax(x, axis=1)` devolve, para cada linha, o índice da coluna com o maior valor.

### 1.12 `nan_to_num`, `round`, `ceil`, `floor`, `fmin`, `spacing`

| Função | O que faz |
|---|---|
| `np.nan_to_num(x)` | troca valores vazios (NaN) por 0 |
| `np.round(x, 6)` | arredonda em 6 casas decimais |
| `np.ceil(x)` / `np.floor(x)` | arredonda para cima / para baixo, para o inteiro |
| `np.fmin(a, b)` | mínimo casa a casa, ignorando NaN (se um lado for NaN, fica o outro) |
| `np.spacing(x)` | a menor diferença representável perto de `x` no computador (≈ 1,1e-13 para x = 1.000) |

### 1.13 pandas: `groupby(...).ngroup()`

```python
grupo = base.groupby(chave_grupo, dropna=False, sort=False).ngroup().to_numpy()
```

Dá um **número de grupo** para cada linha: todas as linhas com os mesmos valores em `chave_grupo` (as 7 faixas de um mesmo pacote × perfil) recebem o mesmo número. `dropna=False` não descarta linhas com chave vazia, e `sort=False` numera na ordem em que os grupos aparecem.

---

## 2. Constantes e nomes de colunas

```python
SUFIXOS_PADRAO = {          # resumido; no arquivo há um item por linha, com comentários
    "t": "_t", "ismin": "_ismin", "ismax": "_ismax", "taxa": "_taxa",
    "tf": "_tf", "isf": "_isf",
    "dif_ticket": "_dif_ticket", "dif_prop": "_dif_prop",
    "isref": "_isref", "dif_isref": "_dif_isref",          # novos na versão 2
}

BETA_PADRAO = 1.0
```

O nome de cada coluna é `cobertura + sufixo` (`"cob_01" + "_ismax"` = `"cob_01_ismax"`). Na função principal:

```python
sx = {**SUFIXOS_PADRAO, **(sufixos or {})}
```

- `{**a, **b}` cria um dicionário novo com as chaves de `a` e, por cima, as de `b`. Assim, os sufixos que você passa substituem só os que mudam.
- `sufixos or {}` usa um dicionário vazio quando `sufixos` é `None`.

A função interna `col(c, tipo)` só monta o nome: `col("cob_01", "ismax")` → `"cob_01_ismax"`.

As outras constantes:
- `BETA_PADRAO = 1.0` *(novo na versão 2)*: peso padrão da distorção de IS **[modelagem 4.6]**. É usado como valor padrão do parâmetro `beta` de `calcular_is`;
- `CASAS_LIMPEZA = 6`: casas decimais usadas para limpar ruído antes de `ceil`/`floor` **[modelagem 7.2]**;
- `ULPS_TOLERANCIA = 64`: tamanho da folga contra ruído binário na trava do teto **[modelagem 7.3]**.

---

## 3. `resolver_distribuicao`: o núcleo da otimização (novo na versão 2)

**[modelagem 4.5]**. Na versão 1, o núcleo recebia `t0` e aplicava o mesmo fator `s` a todas as coberturas. Na versão 2, ele recebe o **alvo** `m` e a **rigidez** `w` de cada cobertura (calculados em `calcular_is`, seção 7.6 deste guia) e resolve, para todas as linhas ao mesmo tempo:

$$\min \sum_i w_i (t_i - m_i)^2 \quad \text{com} \quad \sum_i t_i = T,\;\; L_i \le t_i \le U_i$$

cuja solução é $t_i = \text{clip}(m_i + \mu / w_i,\; L_i,\; U_i)$, com um único $\mu$ por linha.

**Intuição:** cada cobertura parte do seu alvo `m`. Se a soma dos alvos não fecha o ticket total, todas recebem (ou cedem) um ajuste `μ / w`. Quem tem `w` grande (rígida) se move pouco; quem tem `w` pequeno absorve mais. Na versão 1, `w = 1/t0`, então o ajuste era proporcional ao ticket recomendado.

### 3.1 Preparação

```python
m = np.asarray(m, float)
w = np.asarray(w, float)
L = np.asarray(L, float)
U = np.asarray(U, float)

soma_L = L.sum(axis=1)
soma_U = U.sum(axis=1)
```

- `np.asarray(x, float)` garante que é uma matriz de números decimais.
- `soma_L` e `soma_U` são vetores com um valor por linha: o mínimo e o máximo possíveis da soma. O total `T` agora chega pronto como argumento.

### 3.2 Intervalo de busca do `mu`

```python
mu_lo = ((L - m) * w).min(axis=1) - 1.0
mu_hi = ((U - m) * w).max(axis=1) + 1.0
```

A bisseção precisa de um intervalo que com certeza contenha a resposta:
- Para uma cobertura chegar ao piso, é preciso $m + \mu/w \le L$, ou seja, $\mu \le (L - m) \cdot w$. O **menor** desses valores na linha (`.min(axis=1)`) leva **todas** as coberturas ao piso. O `- 1.0` dá folga.
- Pelo mesmo raciocínio, o **maior** $(U - m) \cdot w$ leva todas ao teto.

Assim, em `mu_lo` a soma é a mínima possível e em `mu_hi`, a máxima.

### 3.3 A função `soma_t` e a bisseção

```python
def soma_t(mu):
    return np.clip(m + mu[:, None] / w, L, U).sum(axis=1)

for _ in range(n_iter):
    mu_mid = 0.5 * (mu_lo + mu_hi)
    abaixo = soma_t(mu_mid) < T
    mu_lo = np.where(abaixo, mu_mid, mu_lo)
    mu_hi = np.where(abaixo, mu_hi, mu_mid)
mu = 0.5 * (mu_lo + mu_hi)
```

- `soma_t(mu)` calcula, para cada linha, a soma dos tickets que um ajuste `mu` produziria: aplica o ajuste (`m + mu[:, None] / w`, ver 1.7), trava (`np.clip`) e soma (`.sum(axis=1)`). Como `w > 0`, essa soma só cresce quando `mu` cresce.
- Em cada iteração:
  1. testa o ponto do meio, `mu_mid`;
  2. `abaixo` é `True` nas linhas em que a soma ficou **menor que T**, ou seja, o `mu` precisa ser maior;
  3. nessas linhas, o limite de baixo sobe para `mu_mid`; nas outras, o de cima desce para `mu_mid`.
- `for _ in range(n_iter)`: o `_` é o nome convencional para "variável que não vou usar". Só queremos repetir `n_iter` vezes.
- Cada iteração **corta o intervalo pela metade em todas as linhas de uma vez**. É uma bisseção vetorizada: os `np.where` fazem o "se" de cada linha sem precisar de um `if` por linha. Depois de 100 iterações, o intervalo é menor que a precisão do computador.

### 3.4 Tickets e casos de viabilidade

```python
t = np.clip(m + mu[:, None] / w, L, U)

situacao = np.full(len(T), "ok", dtype=object)
teto = soma_U < T
piso = soma_L > T
t[teto] = U[teto]
t[piso] = L[piso]
```

- `t` aplica a fórmula da solução com o `mu` encontrado.
- `np.full(n, "ok", dtype=object)` cria um vetor de textos com "ok" em todas as posições.
- `teto` e `piso` são máscaras por linha, os casos da tabela **[modelagem 4.4]**. `t[teto] = U[teto]` substitui as **linhas inteiras** marcadas: quando nem todas as coberturas no máximo absorvem o ticket, a resposta é "todas no máximo".

### 3.5 Retorno

```python
return {"t": t, "mu": mu, "sobra": T - t.sum(axis=1), "situacao": situacao}
```

O retorno é um **dicionário**: quem chama acessa `r["t"]`, `r["mu"]` etc. Diferente da versão 1, a distorção não é mais calculada aqui. Ela passou para `calcular_is` (seção 7.9), porque agora tem dois componentes.

---

## 4. `limites_inteiros`: IS mínima e máxima em inteiros

**[modelagem 7.1 e 7.2]**

```python
lo = np.ceil(np.round(ismin, CASAS_LIMPEZA))
hi = np.floor(np.round(ismax, CASAS_LIMPEZA))
```

- `ceil` no mínimo: a menor IS **inteira** que não fica abaixo do mínimo (1.000,40 → 1.001).
- `floor` no máximo: a maior IS inteira que não passa do máximo.
- O `np.round(..., 6)` antes limpa o ruído do computador: 75.000,0000000001 vira 75.000, e o `ceil` não o transforma em 75.001.

---

## 5. `tickets` e `passa_do_teto`

```python
def tickets(is_int, taxa0, fixo0, casas_ticket=None):
    t = is_int * taxa0 + fixo0
    return np.round(t, casas_ticket) if casas_ticket is not None else t
```

É a fórmula do ticket a partir da IS: $IS \cdot taxa + fixo$. A parte fixa são os agravos na especial e o valor fixo nas coberturas de valor fixo. A última linha usa o "if de uma linha" do Python, `A if condição else B`: arredonda em centavos só se `casas_ticket` foi informado. O ponto importante é que **toda** conta de ticket final passa por essa função, então a trava do teto e as colunas de saída usam exatamente o mesmo cálculo.

```python
def passa_do_teto(soma, teto):
    return soma > teto + ULPS_TOLERANCIA * np.spacing(np.abs(teto))
```

Compara a soma com o teto aceitando só uma folga do tamanho do erro de representação do computador **[modelagem 7.3]**. `np.spacing(1000)` ≈ 1,1e-13, então a folga é de cerca de 7e-12 num teto de R$ 1.000. Empatar com o teto é permitido; qualquer excesso real é bloqueado.

---

## 6. `arredondar_is`: da IS contínua para a inteira

**[modelagem 7.1, etapa 3, e 7.3]**. Recebe a IS contínua (o ótimo) e devolve IS inteiras, sem sair de `[lo, hi]` e sem a soma dos tickets passar do teto.

### 6.1 Ponto de partida: `floor`

```python
x = np.round(np.where(ativo, is_cont, 0.0), CASAS_LIMPEZA)
base = np.where(ativo, np.clip(np.floor(x), lo, hi), 0.0)
frac = np.where(ativo, x - base, -1.0)
```

- `base` começa com todas as IS arredondadas para baixo e travadas nos limites inteiros.
- `frac` é a **parte fracionária** que foi descartada (406.666,67 → 0,67). Nas coberturas inativas recebe −1, para ficarem por último na ordenação.

```python
taxa0 = np.where(ativo, np.nan_to_num(taxa), 0.0)
fixo0 = np.where(ativo, fixo, 0.0)

def soma(b):
    return tickets(b, taxa0, fixo0, casas_ticket).sum(axis=1)
```

`soma(b)` dá, por linha, a soma dos tickets correspondente a uma matriz de IS `b`. É uma função definida dentro da outra, que "enxerga" `taxa0`, `fixo0` e `casas_ticket`.

### 6.2 Ajuste guloso: subir +1 onde ajuda

```python
linhas = np.arange(x.shape[0])
ordem = np.argsort(-frac, axis=1)
s_atual = soma(base)
for pos in range(x.shape[1]):
    i = ordem[:, pos]
    pode = ativo[linhas, i] & (base[linhas, i] + 1 <= hi[linhas, i]) & (taxa0[linhas, i] > 0)
    teste = base.copy()
    teste[linhas[pode], i[pode]] += 1
    s_teste = soma(teste)
    melhora = np.abs(alvo - s_teste) < np.abs(alvo - s_atual) - 1e-12
    sobe = pode & melhora & ~passa_do_teto(s_teste, teto)
    base[linhas[sobe], i[sobe]] += 1
    s_atual = np.where(sobe, s_teste, s_atual)
```

Passo a passo:
- `linhas = [0, 1, ..., n-1]` é o índice de cada linha. `x.shape[0]` é o número de linhas e `x.shape[1]`, o de coberturas.
- `ordem` diz, para cada linha, a ordem das coberturas da maior para a menor parte fracionária (ver 1.11).
- O laço `for pos in range(k)` roda **uma vez por cobertura**, não por linha. Na volta `pos`, `i = ordem[:, pos]` é, em cada linha, a cobertura que está nessa posição da fila. Na primeira volta, é a de maior parte fracionária de cada linha.
- `pode`: a cobertura está ativa, subir 1 não passa do máximo inteiro e ela tem taxa (sem taxa, subir a IS não muda nada).
- `teste`: uma cópia em que a IS sobe 1 onde `pode`. Usa a indexação de 1.10.
- `melhora`: a soma do teste fica **mais perto do alvo** (a soma do ótimo contínuo) do que a atual. O `- 1e-12` evita trocas por diferenças desprezíveis.
- `sobe`: pode, melhora e **não passa do teto**. Só aí a subida é aplicada de verdade em `base`.
- `s_atual` é atualizado só nas linhas que subiram.

No pacote A com β = 0 (igual à versão 1), a IS do Incêndio contínua é 406.666,67. O `floor` dá 406.666 (ticket 679,999), e subir para 406.667 daria ticket 680,0005, com total de 1.000,0005 > 1.000. A subida é bloqueada, e por isso a IS final é 406.666 **[modelagem 6.1]**.

### 6.3 Trava final

```python
for _ in range(10 * x.shape[1]):
    pode_descer = ativo & (base - 1 >= lo) & (taxa0 > 0)
    excede = passa_do_teto(soma(base), teto) & pode_descer.any(axis=1)
    if not excede.any():
        break
    i = np.argmax(np.where(pode_descer, taxa0, -np.inf), axis=1)
    base[linhas[excede], i[excede]] -= 1
```

Rede de segurança **[modelagem 7.3]**. Recalcula a soma do zero e, nas linhas que ainda passam do teto (por exemplo, por causa do arredondamento em centavos), desce 1 unidade de IS da cobertura de **maior taxa** que ainda pode descer:
- `pode_descer.any(axis=1)`: a linha tem pelo menos uma cobertura que pode descer.
- `np.where(pode_descer, taxa0, -np.inf)`: troca por menos infinito as coberturas que não podem descer, para o `argmax` nunca escolhê-las.
- `if not excede.any(): break`: assim que nenhuma linha excede, sai do laço. Normalmente isso acontece na primeira volta.

---

## 7. `calcular_is`: a função principal

### 7.1 Processamento em blocos

```python
if linhas_por_bloco and len(df) > linhas_por_bloco:
    params = dict(coberturas=coberturas, ..., linhas_por_bloco=None)
    g = df.groupby(chave_grupo, dropna=False, sort=False).ngroup().to_numpy()
    n_blocos = int(np.ceil(len(df) / linhas_por_bloco))
    bloco_do_grupo = (g * n_blocos) // (g.max() + 1)
    saidas = [calcular_is(df[bloco_do_grupo == b], **params) for b in range(n_blocos)
              if (bloco_do_grupo == b).any()]
    ...
    return pd.concat(saidas, ignore_index=True)
```

- Se a base é grande, a função **chama a si mesma** em pedaços (recursão), com `linhas_por_bloco=None` para que cada chamada processe o seu pedaço de uma vez.
- `g` é o número do grupo de cada linha (ver 1.13).
- `bloco_do_grupo = (g * n_blocos) // (g.max() + 1)` reparte os números de grupo em `n_blocos` faixas de tamanho parecido. `//` é a divisão inteira. Como a conta usa o **número do grupo**, as 7 faixas de um mesmo pacote sempre caem no mesmo bloco.
- `**params` "desempacota" o dicionário como argumentos nomeados: equivale a escrever `coberturas=..., especial=..., ...`.
- `pd.concat(..., ignore_index=True)` empilha os resultados dos blocos e renumera as linhas.

### 7.2 Conferência dos nomes de colunas

```python
necessarias = [col(c, t) for t in ("t", "ismin", "ismax") for c in coberturas]
necessarias += [col(c, "taxa") for c in coberturas if c not in cvf]
...
faltando = [n for n in necessarias if n not in df.columns]
if faltando:
    raise KeyError(...)
```

- List comprehension com **dois `for`**: gera todas as combinações de tipo × cobertura.
- `if c not in cvf` dentro da comprehension filtra: a taxa só é obrigatória para coberturas que não são de valor fixo.
- `raise KeyError(...)` interrompe a função com uma mensagem clara listando o que falta.

### 7.3 Montagem das matrizes

```python
base = df.reset_index(drop=True).copy()
n = len(base)

def coluna(nome, padrao=0.0):
    return base[nome].to_numpy(float) if nome in base.columns else np.full(n, padrao)
```

- `reset_index(drop=True)` renumera as linhas de 0 a n−1, para que as posições nas matrizes batam com as linhas do dataframe. `.copy()` evita alterar o dataframe original.
- `coluna` lê uma coluna como vetor de números. Se a coluna não existe (taxa de cobertura de valor fixo, por exemplo), devolve um vetor de zeros.

```python
fixo = np.zeros((n, len(coberturas)))
fixo[:, k_esp] = np.nan_to_num(coluna(col_agravo1)) + np.nan_to_num(coluna(col_agravo2))
for c, cfg in cvf.items():
    fixo[:, coberturas.index(c)] = np.nan_to_num(coluna(cfg["valor"]))
```

A matriz `fixo` guarda a **parte fixa do ticket** de cada cobertura: os agravos na coluna da especial (`k_esp` é a posição dela na lista) e o valor fixo nas coberturas de valor fixo **[modelagem 5.1 e 5.6]**. Para as demais, fica 0. `fixo[:, k_esp] = ...` preenche uma coluna inteira da matriz.

```python
t0 = np.column_stack([coluna(col(c, "t")) for c in coberturas])
t0 = np.nan_to_num(t0, nan=0.0).clip(min=0)
taxa = np.nan_to_num(np.column_stack([coluna(col(c, "taxa")) for c in coberturas]))
ismin = np.column_stack([coluna(col(c, "ismin")) for c in coberturas])
ismax = np.column_stack([coluna(col(c, "ismax")) for c in coberturas])
```

As demais matrizes, montadas com `column_stack` (ver 1.9). `.clip(min=0)` impede tickets recomendados negativos.

```python
isref = np.nan_to_num(np.column_stack([coluna(col(c, "isref")) for c in coberturas])).clip(min=0)
```

*(novo na versão 2)* A matriz das IS sugeridas. Como `coluna` devolve zeros quando a coluna não existe, e `nan_to_num` troca vazios por 0, uma cobertura sem IS sugerida fica com `isref = 0` e não é ponderada.

### 7.4 Máscaras: quem participa de quê

```python
ativo = t0 > 0
tem_taxa = taxa > 0
eh_valor_fixo = np.array([c in cvf for c in coberturas])[None, :]
so_fixo = ativo & ~tem_taxa & eh_valor_fixo & (fixo > 0)
elegivel = ativo & tem_taxa
participa = elegivel | so_fixo
esp_ativa = elegivel[:, k_esp]
fixo = np.where(participa, fixo, 0.0)
```

| Máscara | Significado | Modelagem |
|---|---|---|
| `ativo` | a cobertura está no pacote (tem ticket recomendado) | 3 |
| `tem_taxa` | a taxa é maior que zero | 11 |
| `eh_valor_fixo` | a cobertura está em `coberturas_valor_fixo`; formato `(1, k)` para valer em todas as linhas (ver 1.7) | 5.6 |
| `so_fixo` | cobertura de valor fixo, no pacote, sem taxa: ticket = valor fixo | 5.6 |
| `elegivel` | entra na otimização com IS calculada | 11 |
| `participa` | tem ticket final (elegível ou só valor fixo) | — |
| `esp_ativa` | a especial está elegível nessa linha (vetor por linha) | 5 |

A última linha zera a parte fixa de quem não participa. Por exemplo, a especial não elegível não cobra os agravos.

### 7.5 Limites de ticket e faixas válidas

```python
lo, hi = limites_inteiros(ismin, ismax)
L = np.where(participa, np.nan_to_num(lo * taxa) + fixo, 0.0)
U = np.where(participa, np.nan_to_num(hi * taxa) + fixo, 0.0)

faixa_valida = (lo[:, k_esp] <= hi[:, k_esp]) | ~esp_ativa
U[:, k_esp] = np.where(faixa_valida, U[:, k_esp], L[:, k_esp])
```

- `L` e `U` são os limites de ticket **[modelagem 2 e 5.2]**: $IS^{min} \cdot taxa + fixo$ e $IS^{max} \cdot taxa + fixo$. Para `so_fixo`, como a taxa é 0, `L = U = valor fixo`, e o ticket fica travado no valor fixo.
- `faixa_valida`: a faixa da especial tem pelo menos uma IS inteira possível, ou a especial nem está no pacote **[modelagem 5.4]**.
- Nas faixas inválidas, `U` da especial é igualado a `L` só para o núcleo não receber um intervalo invertido (`L > U`). Essas linhas são descartadas depois, na escolha da faixa.

### 7.6 Alvo combinado e otimização contínua (novo na versão 2)

**[modelagem 4.5]**. Antes de chamar o núcleo, o código transforma os dois termos do objetivo (distorção de ticket e distorção de IS) num único alvo `m` e numa rigidez `w` por cobertura.

```python
com_ref = elegivel & (isref > 0)
with np.errstate(divide="ignore", invalid="ignore"):
    a = np.where(ativo, 1.0 / np.where(ativo, t0, 1.0), 1.0)
    b = np.where(com_ref, beta * t0 / np.where(com_ref, taxa * isref, 1.0) ** 2, 0.0)
tref = fixo + taxa * isref
w = a + b
m = np.where(ativo, (a * t0 + b * tref) / w, 0.0)
```

| Variável | Fórmula | Significado |
|---|---|---|
| `com_ref` | elegível e com IS sugerida | coberturas que recebem o termo de IS |
| `a` | $1 / t^0$ | peso da distorção de ticket (o mesmo da versão 1) |
| `b` | $\beta\, t^0 / (taxa \cdot IS^{ref})^2$ | peso da distorção de IS, já convertido para "espaço de ticket" |
| `tref` | $fixo + taxa \cdot IS^{ref}$ | ticket que corresponde à IS sugerida |
| `w` | $a + b$ | rigidez: quanto custa afastar a cobertura do alvo |
| `m` | $(a\, t^0 + b\, t^{ref}) / w$ | alvo combinado: média ponderada entre o ticket recomendado e o ticket da IS sugerida |

Detalhes de sintaxe:
- Os `np.where` internos (`np.where(ativo, t0, 1.0)` e `np.where(com_ref, taxa * isref, 1.0)`) evitam divisão por zero, como em 1.5.
- Nas coberturas **inativas**, `a` recebe 1 (e não 0) para `w` nunca ser zero, porque o núcleo divide por `w`. Isso não afeta nada: essas coberturas têm `L = U = 0`, então o `clip` as mantém em 0.
- `beta` é o parâmetro da função (padrão `BETA_PADRAO = 1.0`). Com `beta = 0`, `b = 0`, `m = t0` e `w = 1/t0`, e o núcleo reproduz exatamente a versão 1.
- Na especial, `taxa` e `fixo` são os da faixa daquela linha, então `tref` e `b` mudam de faixa para faixa.

```python
T = t0.sum(axis=1)
r = resolver_distribuicao(m, w, L, U, T, n_iter=n_iter)
t_cont = r["t"]
```

Chama o núcleo (seção 3 deste guia) para todas as linhas e faixas de uma vez. `t_cont` são os tickets ótimos **contínuos**, antes de a IS virar inteira.

### 7.7 Ticket contínuo → IS inteira → ticket final

```python
is_cont = np.where(elegivel, (t_cont - fixo) / taxa, 0.0)
teto = T.copy()
if col_ticket_total and col_ticket_total in base.columns:
    teto = np.fmin(teto, base[col_ticket_total].to_numpy(float))
is_int = arredondar_is(is_cont, lo, hi, taxa, fixo, participa,
                       alvo=np.minimum(t_cont.sum(axis=1), teto), teto=teto,
                       casas_ticket=casas_ticket)
is_int = np.where(so_fixo, 0.0, is_int)
t_fin = np.where(participa, tickets(is_int, taxa, fixo, casas_ticket), 0.0)
```

- `is_cont`: a conversão $IS = (t - fixo)/taxa$ **[modelagem 5.2]**.
- `teto`: o menor valor entre a soma dos tickets recomendados e a coluna `ticket_total` **[modelagem 7.3]**. `np.fmin` ignora vazios. O `.copy()` cria um vetor novo, para que mexer no teto não altere `T`.
- `alvo`: a soma que o ótimo contínuo distribuiu, limitada ao teto.
- `is_int`: as IS inteiras (seção 6 deste guia). Depois, as coberturas `so_fixo` recebem IS 0 **[modelagem 5.6]**.
- `t_fin`: o ticket final **recalculado a partir da IS inteira**, para que ticket e IS sempre batam.

### 7.8 Colunas de saída por cobertura (incluindo as novas)

```python
soma_t0 = t0.sum(axis=1, keepdims=True)
soma_tf = t_fin.sum(axis=1, keepdims=True)
prop_rec = np.where(soma_t0 > 0, t0 / soma_t0, 0.0)
prop_fin = np.where(soma_tf > 0, t_fin / soma_tf, 0.0)

for j, c in enumerate(coberturas):
    base[col(c, "tf")] = t_fin[:, j]
    base[col(c, "isf")] = is_int[:, j].astype("int64")
    base[col(c, "dif_ticket")] = t0[:, j] - t_fin[:, j]
    base[col(c, "dif_prop")] = np.abs(prop_rec[:, j] - prop_fin[:, j])
    base[col(c, "dif_isref")] = np.where(isref[:, j] > 0, is_int[:, j] - isref[:, j], 0.0)
```

- `prop_rec` e `prop_fin`: cada ticket dividido pelo total da sua linha (o `keepdims` de 1.7 permite dividir a matriz pelo vetor de totais). O `np.where` protege linhas com total 0.
- `enumerate(coberturas)` percorre a lista dando a posição `j` e o nome `c` de cada cobertura. `t_fin[:, j]` é a coluna dessa cobertura na matriz.
- `.astype("int64")` grava a IS como número inteiro.
- **`<cob>_dif_ticket` = recomendado − final.** Positivo: a cobertura cedeu ticket; negativo: recebeu. A soma de todas as coberturas é igual a `sobra + dif_arredondamento`.
- **`<cob>_dif_prop` = |proporção recomendada − proporção final|.** `np.abs` é o módulo.
- **`<cob>_dif_isref` = IS final − IS sugerida** *(novo na versão 2)*. Positivo: acima da sugerida. Fica 0 quando não há IS sugerida.

### 7.9 Colunas gerais e `status`

```python
base["sobra"] = r["sobra"]
base["dif_arredondamento"] = t_cont.sum(axis=1) - t_fin.sum(axis=1)

def componentes(t, is_):
    with np.errstate(divide="ignore", invalid="ignore"):
        d_t = np.where(ativo, (t - t0) ** 2 / np.where(ativo, t0, 1.0), 0.0).sum(axis=1)
        d_is = np.where(com_ref, t0 * ((is_ - isref) / np.where(com_ref, isref, 1.0)) ** 2,
                        0.0).sum(axis=1)
    return d_t, d_is

d_t, d_is = componentes(t_fin, is_int)
base["distorcao"] = d_t
base["distorcao_is"] = d_is
base["objetivo"] = d_t + beta * d_is
base["fator_escala"] = 1.0 + r["mu"]

fora = elegivel & ((is_int < np.round(ismin, CASAS_LIMPEZA)) | (is_int > np.round(ismax, CASAS_LIMPEZA)))
fora_limites = fora.any(axis=1) & faixa_valida
acima_teto = passa_do_teto(t_fin.sum(axis=1), teto)

status = np.where(np.abs(r["sobra"]) <= tol, "ok", "sobra_nao_distribuida")
status = np.where((r["situacao"] == "piso") | acima_teto, "minimos_excedem_T", status)
status = np.where(fora_limites, "is_fora_dos_limites", status)
status = np.where(~faixa_valida, "faixa_invalida", status)
```

- *(novo na versão 2)* `componentes(t, is_)` é uma função interna que calcula os dois termos do objetivo **[modelagem 4.5]** para quaisquer tickets e IS: `d_t` é a distorção de ticket, $\sum (t - t^0)^2/t^0$, e `d_is` é a distorção de IS, $\sum t^0 ((IS - IS^{ref})/IS^{ref})^2$. Ela devolve **dois valores de uma vez** (uma tupla), e `d_t, d_is = componentes(...)` recebe cada um numa variável. Aqui é aplicada à solução **final** (IS inteira); na seção 7.10, à solução contínua.
- `objetivo` = `distorcao` + β × `distorcao_is`.
- `fator_escala = 1 + mu`: com β = 0 é exatamente o `s` da versão 1. Quando há sobra, todas as coberturas estão travadas no máximo e o valor não tem significado.
- `fora` e `acima_teto` são as checagens de segurança **[modelagem 7.2 e 7.3]**. `fora.any(axis=1)` vale `True` na linha se **alguma** cobertura saiu dos limites.
- O `status` é montado em camadas: cada `np.where` sobrescreve o anterior onde a condição vale. Por isso a **última** camada tem prioridade: uma faixa inválida é marcada como tal, mesmo que também tenha sobra.

### 7.10 Escolha da melhor faixa

**[modelagem 5.5]**

```python
d_t_cont, d_is_cont = componentes(t_cont, is_cont)
rank = pd.DataFrame({
    "grupo": grupo,
    "invalida": (~faixa_valida).astype(int),
    "piso": ((r["situacao"] == "piso") | acima_teto).astype(int),
    "sobra": np.abs(r["sobra"]).round(6),
    "dist": (d_t_cont + beta * d_is_cont).round(6),
    "is_esp": is_int[:, k_esp] * (1 if desempate_is == "menor" else -1),
})
idx = rank.sort_values(["grupo", "invalida", "piso", "sobra", "dist", "is_esp"]) \
          .drop_duplicates("grupo").index
resultado = base.loc[idx].sort_index().reset_index(drop=True)
```

Um jeito compacto de escolher "a melhor linha de cada grupo":
1. Monta uma tabela auxiliar `rank` com os critérios, um por coluna, na ordem de prioridade. Os critérios lógicos viram 0/1 (`astype(int)`), e 0 é melhor.
2. `sort_values([...])` ordena por grupo e, dentro de cada grupo, pelos critérios em sequência: válida primeiro, depois sem piso, menor sobra, menor **objetivo** e, por fim, o desempate de IS.
3. `drop_duplicates("grupo")` mantém só a **primeira** linha de cada grupo, que é a melhor depois da ordenação.
4. `.index` pega os números dessas linhas, e `base.loc[idx]` as busca no dataframe completo.

Detalhes:
- O `.round(6)` faz empates "quase exatos" (diferenças de 1e-12) contarem como empate de verdade, para o critério seguinte decidir.
- No desempate de IS, multiplicar por −1 transforma "maior IS primeiro" em "menor número primeiro", porque a ordenação é sempre crescente.
- *(novo na versão 2)* A coluna `"dist"` agora é o **objetivo completo** (distorção de ticket + β × distorção de IS), calculado com `componentes` sobre o ótimo **contínuo** (`t_cont` e `is_cont`). Na versão 1, era só a distorção de ticket.
- Sobra e objetivo vêm do ótimo contínuo, para que o arredondamento da IS (centavos) não decida a faixa **[modelagem 7.3]**.
- O `\` no fim da linha só continua a instrução na linha de baixo.

### 7.11 Retorno

```python
if retornar_todas_faixas:
    return resultado, base.assign(grupo=grupo)
return resultado
```

Com `retornar_todas_faixas=True`, devolve também todas as faixas, para auditoria. `base.assign(grupo=grupo)` acrescenta a coluna com o número do grupo. Ao devolver dois valores, a chamada fica `resultado, todas = calcular_is(...)`.

---

## 8. Exemplo acompanhado: pacote B, faixa F2, com β = 1

Os valores abaixo saem do próprio código. É o pacote B **[modelagem 6.4]**: Incêndio 700 e Roubo 300, sem Danos Elétricos, com IS sugeridas de 350.000 (Incêndio) e 15.000 (Roubo). Na versão 1, o Roubo ficaria com IS de 30.000, o dobro da sugerida.

| Variável | Incêndio | Roubo | Comentário |
|---|---|---|---|
| `t0` | 700 | 300 | ticket recomendado; `T = 1000` |
| `taxa` / `fixo` | 0,0015 / 70 | 0,01 / 0 | Incêndio na F2: agravo 1 (40) + agravo 2 (30) |
| `isref` | 350.000 | 15.000 | IS sugerida |
| IS implícita pelo `t0` | 420.000 | 30.000 | `(t0 − fixo) / taxa`: as duas acima da sugerida |
| `tref` | 595 | 150 | `fixo + taxa × isref` |
| `a` | 0,001429 | 0,003333 | `1 / t0` |
| `b` | 0,002540 | 0,013333 | `β × t0 / (taxa × isref)²` |
| `w` | 0,003968 | 0,016667 | `a + b`: o Roubo é ~4× mais "rígido" |
| `m` | 632,8 | 180,0 | alvo combinado; soma = 812,8 < 1.000 |
| `1 / w` | 252 | 60 | quanto cada uma recebe por unidade de `mu` |
| `mu` (bisseção) | 0,6 | | faltavam 187,2 = 0,6 × (252 + 60) |
| `t_cont` | 632,8 + 151,2 = **784** | 180 + 36 = **216** | `m + mu / w` |
| `is_int` | 476.000 | 21.600 | `(t − fixo) / taxa` |
| desvio da IS sugerida | +36% | +44% | na versão 1: +20% e +100% |
| `fator_escala` | 1,6 | | `1 + mu` |

Leitura: o alvo combinado já puxa o Roubo para baixo (180 em vez de 300) e o Incêndio também (632,8 em vez de 700). Como a soma dos alvos não fecha os 1.000, o que falta é repartido pela regra `mu / w`: o Incêndio, menos rígido, recebe a maior parte. No fim, os dois desvios relativos de IS ficam equilibrados, em vez de o Roubo ficar com o dobro da IS sugerida.

Para reproduzir, rode o exemplo do arquivo, que imprime os pacotes A, B e C com β = 0 e β = 1, ou use a auditoria:

```python
resultado, todas = calcular_is(df, coberturas=["cob_01", "cob_02", "cob_03"],
                               especial="cob_01", chave_grupo=["perfil", "pacote"],
                               beta=1.0, retornar_todas_faixas=True)
todas[todas["pacote"] == "B"]
```

---

## 9. Bloco `if __name__ == "__main__":`

O código dentro desse `if` só roda quando o arquivo é executado diretamente (`python otimizacao_is_v2.py`). Ele **não** roda quando você faz `from otimizacao_is_v2 import calcular_is` no seu notebook. Ali está o exemplo dos pacotes A, B e C: monta um dataframe com uma linha por faixa (as listas `pacotes` e `faixas` e o laço duplo), com IS sugeridas, e chama `calcular_is` duas vezes, com β = 0 e β = 1, imprimindo os dois resultados.
