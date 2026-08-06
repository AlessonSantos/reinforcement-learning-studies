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

**A matemática — a variável que decide o caminho:**

$$Z_t \sim \text{Bernoulli}(\epsilon_t)$$

| Se... | Então... |
|---|---|
| $Z_t=1$ (prob. $\epsilon_t$) | rota aleatória: sorteia-se um tamanho de pacote $\lvert S\rvert$ e depois $\lvert S\rvert$ coberturas — pula para a Etapa 6 |
| $Z_t=0$ (prob. $1-\epsilon_t$) | rota do modelo: segue para a Etapa 3 |

| Símbolo | Descrição |
|---|---|
| $Z_t$ | a única variável genuinamente aleatória que decide o caminho desta rodada — sorteada de novo a cada rodada, nunca fixada por cliente |
| $\epsilon_t$ | a probabilidade de ir pela rota aleatória nesta rodada — decrescente ao longo do tempo, mas nunca chegando a zero (ver a tabela de parâmetros ao final deste documento) |

**Duas regras que $Z_t$ e $\epsilon_t$ precisam respeitar, para a identificação de fadiga/sinergia/canibalização continuar válida:**
1. $\epsilon_t$ pode depender do tempo $t$ (ou de um valor de configuração vigente naquele momento), mas **nunca do perfil do cliente daquela rodada específica** — senão reintroduz o viés de seleção que a fase aleatória existe para evitar.
2. $Z_t$ precisa ser sorteado **a cada rodada**, mesmo para o mesmo cliente ao longo do tempo — nunca "sorteie uma vez por cliente e fixe permanentemente", ou o modelo nunca aprenderia como aquele cliente reage aos dois regimes.

**Um esclarecimento importante: $Z_t$ e o bônus de incerteza do modelo não são dois mecanismos aleatórios competindo entre si — são coisas de natureza diferente.**

- **$Z_t$ é aleatoriedade de verdade.** É um sorteio genuíno: a cada rodada, uma moeda é jogada, e o resultado decide se a rota é aleatória ou via modelo.
- **O bônus de exploração (o termo $\sqrt{x^TH^{-1}x}$ calculado nas Etapas 3 e 4) não é aleatório — é uma regra fixa.** Dado um estado fixo do modelo, ele sempre desloca a decisão exatamente da mesma forma, a favor de itens pouco testados. Não há nenhum sorteio ali dentro: para o mesmo cliente e o mesmo estado do modelo, o oráculo guloso sempre chegaria ao mesmo pacote.

Ou seja: $Z_t$ é a **única** fonte de aleatoriedade deliberada do sistema (existe também a resposta do próprio cliente, que é aleatória, mas essa não é um mecanismo de exploração — é só o ruído inerente do fenômeno sendo modelado).

Essa distinção explica por que a rota aleatória continua sendo necessária mesmo com o bônus já atuando: o bônus resolve "testar itens/pares específicos pouco vistos", mas, por ser uma regra fixa, ele nunca gera sozinho variação de **tamanho de pacote** descorrelacionada do perfil do cliente — e é exatamente essa variação que a identificação da fadiga precisa. Só $Z_t$ (via a rota aleatória) fornece isso.

**O exemplo rodando:** suponha que já passamos da fase de exploração intensa — vamos seguir o Cliente A pela via do modelo (Etapa 3 em diante).

**O que entra / o que sai:** entra: $c$, $\epsilon_t$ (valor vigente no momento). Sai: $Z_t$ sorteado, e a partir dele, sinal para calcular scores (Etapa 3) ou pacote aleatório.

---

## 3. Score do item via regressão logística

**Antes de mais nada: o que é $S_{parcial}$?** Essa variável aparece pela primeira vez aqui, mas quem a **constrói** é o oráculo guloso, só explicado na Etapa 5 — vale adiantar o essencial para esta etapa não ficar no vácuo. $S_{parcial}$ é **o pacote que está sendo montado, item por item, no exato momento em que o oráculo está decidindo**: ele começa vazio (nenhuma cobertura ainda escolhida) e cresce a cada passo do processo guloso, conforme uma nova cobertura é adicionada. Quando esta etapa pergunta "qual é o score da cobertura $i$, dado $S_{parcial}$?", a pergunta real é: *"se eu já tenho estas coberturas na mala até agora, quanto vale adicionar mais esta?"* — e $|S_{parcial}|$ (o **tamanho** desse pacote parcial) é o que carrega a informação de fadiga: quanto maior a mala já está, menor tende a ser a propensão de aceitar mais uma peça. A Etapa 5 mostra exatamente como $S_{parcial}$ cresce, passo a passo, até o pacote final.

**Analogia:** $\theta$ é uma régua que, para cada combinação "cliente + cobertura + tamanho da mala até agora", devolve um número. Esse número passa por um funil (a função logística) que o espreme para dentro de 0 a 1 — nunca escapa desse intervalo.

**A matemática:**

$$x(i, S_{parcial}) = \text{concat}\big(c,\; f_i,\; c \otimes f_i,\; |S_{parcial}|,\; c \times |S_{parcial}|\big)$$

$$\eta(i) = \hat\theta^{T} x(i, S_{parcial}), \qquad \text{bonus}(i) = \alpha\sqrt{x(i,S_{parcial})^{T} H_\theta^{-1} x(i,S_{parcial})}$$

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

**O que entra / o que sai:** entra: $c$; $\hat\theta$ e $H_\theta$ — refletem **todas** as atualizações já feitas via Etapa 9 até este ponto, seja a origem delas uma rodada que passou pelo modelo (Etapas 3-5) ou uma rodada de exploração aleatória (que também gera observações e alimenta a mesma atualização). Só estão genuinamente zerados, vindos do Setup (Etapa 0), na primeiríssima vez que esta etapa é executada — o que, na prática, tende a ser bem mais tarde que a "rodada 1" propriamente dita, já que $\epsilon_t$ começa alto e as primeiras rodadas tendem a ir para o caminho aleatório (Etapa 2), pulando esta seção; $|S_{parcial}|$. Sai: $p_{UCB}(i)$ por candidato, usado na Etapa 5.

---

## 4. Score do par — sinergia ou canibalização

**Analogia:** $\phi$ é um termômetro de compatibilidade entre duas peças de roupa. Se combinam bem, o termômetro sobe acima do "morno" (0,5). Se competem, esfria abaixo do morno.

**A matemática:**

$$z(i,j) = \text{concat}\big(f_i,\; f_j,\; f_i \otimes f_j,\; c\big)$$

$$p_{UCB}^{par}(i,j) = \sigma\Big(\underbrace{\eta(i)+\eta(j)}_{\text{offset fixo, vem de }\theta} + \hat\phi^{T}z(i,j) + \alpha\sqrt{z(i,j)^{T} H_\phi^{-1} z(i,j)}\Big)$$

**Por que existe um "offset" aqui, e não só $\hat\phi^Tz(i,j)$:** essa peça é nova em relação a uma primeira formulação mais ingênua, e resolve um problema real que só fica claro quando pensamos em como $\phi$ é treinada (a derivação completa está na Etapa 9) — em resumo, $y_{ij}$ (o sinal que $\phi$ aprende a prever) não é independente de $y_i$: por definição, $y_{ij}=1$ só quando **ambos** $y_i=1$ e $y_j=1$ ao mesmo tempo. Sem o offset, $\phi$ acabaria reaprendendo (de forma distorcida) parte do que $\theta$ já sabe sobre cada item isoladamente, em vez de aprender só o que é **adicional** — a sinergia ou canibalização genuína. Fixando $\eta(i)+\eta(j)$ (os scores que $\theta$ já estimou, tratados como constantes conhecidas nesta conta) como ponto de partida, $\hat\phi^Tz(i,j)$ passa a representar exclusivamente esse "extra" — positivo quando há sinergia real, negativo quando há canibalização real, e zero quando os dois itens simplesmente não interagem.

| Símbolo | Descrição |
|---|---|
| $\eta(i), \eta(j)$ | os scores lineares que $\theta$ **já estimou** para cada cobertura isoladamente (Etapa 3) — entram aqui como uma constante conhecida, não são reajustados junto com $\phi$ |
| $f_i, f_j$ | os mesmos vetores de atributos estáticos das duas coberturas (idênticos ao $f_i$ da Etapa 3 — categoria de risco, faixa de prêmio, obrigatoriedade, frequência de sinistro) |
| $f_i \otimes f_j$ | produto elemento a elemento entre os atributos das duas coberturas — captura o grau de **sobreposição estrutural** entre elas (ex.: se ambas têm alta frequência de sinistro na mesma categoria de risco, essa posição do vetor fica com valor alto) |
| $c$ | perfil do cliente — permite que a sinergia/canibalização também dependa de quem é o cliente, não só da identidade do par |
| $z(i,j)$ | a concatenação de tudo acima — o vetor completo que descreve a situação do par |

**O exemplo rodando:** para o par (Desastres Naturais, Danos Elétricos), suponha $\eta(\text{Desastres})=0{,}90$ (o mesmo valor da Etapa 3) e $\eta(\text{Danos Elétricos})=0{,}75$ — então o offset é $0{,}90+0{,}75=1{,}65$. Suponha que $\hat\phi^Tz(i,j)=-2{,}40$ (a régua de par já aprendeu uma canibalização forte entre esses dois riscos) e o bônus seja $0{,}05$:

$$p_{UCB}^{par}(\text{Desastres}, \text{Danos Elétricos}) = \sigma(1{,}65 - 2{,}40 + 0{,}05) = \sigma(-0{,}70) \approx 0{,}33$$

Repare que o valor de $\hat\phi^Tz(i,j)$ precisou ser bem negativo ($-2{,}40$) para puxar a probabilidade final para baixo — exatamente porque os dois itens são individualmente bem avaliados (offset alto, $1{,}65$). Isso é o comportamento certo: só um efeito de interação genuinamente forte deveria conseguir reverter dois itens individualmente bons para uma probabilidade conjunta baixa. Abaixo do morno — sinal de canibalização real (sobreposição de sinistro entre os dois riscos), não uma coincidência de dois itens bons isoladamente.

**O que entra / o que sai:** entra: $\hat\phi$ e $H_\phi$ — pela mesma lógica da Etapa 3: refletem todas as atualizações já acumuladas via Etapa 9 (de rodadas aleatórias ou via modelo), só genuinamente zerados na primeiríssima execução desta etapa — além de $\eta(i), \eta(j)$ (vindos da Etapa 3), $f_i, f_j$, $c$. Sai: $p_{UCB}^{par}(i,j)$, usado na Etapa 5.

---

## 5. O oráculo guloso: montando a mala item por item

**O que é um "oráculo" nesse contexto (antes de entrar na matemática):** em bandits combinatórios, chama-se de **oráculo** o componente que resolve o problema de decisão — dado os scores estimados de cada cobertura (e de cada par), o oráculo escolhe o pacote que maximiza a função de recompensa. É chamado assim porque o resto do algoritmo (a parte que estima $\theta$ e $\phi$ via UCB, nas Etapas 3 e 4) não precisa saber como o oráculo decide por dentro — só precisa entregar a ele os scores e receber de volta um pacote. Essa separação é o que permite trocar a função de recompensa ou as restrições de negócio sem mexer em nada da parte estatística: é exatamente essa flexibilidade que motivou a escolha do C²UCB desde o início da nossa conversa.

Como testar todas as combinações possíveis de coberturas seria computacionalmente inviável à medida que o catálogo cresce (o número de subconjuntos cresce exponencialmente), o oráculo raramente resolve o problema de forma exata — ele costuma ser uma **heurística aproximada**. O **oráculo guloso** é a forma mais comum: em vez de testar todas as combinações, ele constrói o pacote item por item, adicionando a cada passo o candidato que mais aumenta a recompensa, até algum critério de parada.

**O papel específico dele na nossa versão:** aqui o critério de parada não é um tamanho fixo — é o próprio **ganho marginal** ($\Delta$) ficando menor ou igual a zero. Isso é o que permite ao pacote crescer ou encolher dinamicamente por cliente, sem nenhum $k$ pré-definido. Vale uma ressalva honesta: a prova formal de aproximação $(1-1/e)$ do greedy clássico (que veremos no documento do C²UCB original) vale para funções de recompensa submodulares e monótonas sob cardinalidade fixa — como a nossa função permite termos negativos (canibalização) e não tem cardinalidade fixa, o guloso aqui é uma heurística muito usada e razoável na prática, mas sem essa garantia formal exata (discutimos isso em detalhe mais cedo na conversa).

**Analogia:** montar uma mala de viagem, peça por peça — só que a peça entra na mala se ela vale a pena tanto sozinha quanto em conjunto com o que já está ali dentro.

**A matemática:** $S_{parcial}$ começa vazio. A cada passo:

$$\Delta(i \mid S_{parcial}) = \Big(p_{UCB}(i) - 0{,}5\Big) \;+\; \sum_{j \in S_{parcial}} \Big(p_{UCB}^{par}(i,j) - 0{,}5\Big)$$

Escolhe-se $i^{\ast} = \arg\max_i \Delta(i\mid S_{parcial})$. Se $\Delta(i^{\ast})>0$, adiciona e repete; se $\Delta(i^{\ast})\le 0$, para.

**O exemplo rodando — tabela completa de scores:**

Antes de rodar o guloso, o modelo já calculou (Etapas 3 e 4) o score $p_{UCB}$ individual de cada uma das 5 coberturas candidatas, e o score $p_{UCB}^{par}$ de todos os pares relevantes entre elas. Eis a tabela completa:

**Scores individuais:**

| Cobertura | $p_{UCB}(i)$ |
|---|---|
| Desastres Naturais | 0,73 |
| Danos Elétricos | 0,68 |
| Roubo e Furto | 0,60 |
| Conteúdo Residencial | 0,58 |
| Incêndio | 0,52 |

**Scores de par** (todas as combinações entre as 5 coberturas):

| Par | $p_{UCB}^{par}(i,j)$ |
|---|---|
| Desastres × Danos Elétricos | 0,33 (canibalização — sobreposição de sinistro) |
| Desastres × Roubo | 0,35 |
| Desastres × Conteúdo | 0,30 |
| Desastres × Incêndio | 0,40 |
| Danos Elétricos × Roubo | 0,58 |
| Danos Elétricos × Conteúdo | 0,45 |
| Danos Elétricos × Incêndio | 0,45 |
| Roubo × Conteúdo | 0,65 (sinergia — proteção complementar) |
| Roubo × Incêndio | 0,50 |

Com essa tabela em mãos, o guloso roda assim:

**Passo 1 — $S_{parcial}=\emptyset$.** Sem nenhum par formado ainda, $\Delta(i\mid\emptyset) = p_{UCB}(i)-0{,}5$ para cada candidato:

| Candidato | $\Delta(i\mid\emptyset)$ |
|---|---|
| **Desastres Naturais** | **0,23** |
| Danos Elétricos | 0,18 |
| Roubo e Furto | 0,10 |
| Conteúdo Residencial | 0,08 |
| Incêndio | 0,02 |

Maior $\Delta$: Desastres Naturais. Adiciona. $S_{parcial}=\{\text{Desastres}\}$.

**Passo 2 — $S_{parcial}=\{\text{Desastres}\}$.** Agora cada candidato soma seu score individual à interação com Desastres:

| Candidato | $p_{UCB}(i)-0{,}5$ | $p_{UCB}^{par}(i,\text{Desastres})-0{,}5$ | $\Delta$ total |
|---|---|---|---|
| **Danos Elétricos** | 0,18 | $0{,}33-0{,}5=-0{,}17$ | **0,01** |
| Roubo e Furto | 0,10 | $0{,}35-0{,}5=-0{,}15$ | −0,05 |
| Incêndio | 0,02 | $0{,}40-0{,}5=-0{,}10$ | −0,08 |
| Conteúdo Residencial | 0,08 | $0{,}30-0{,}5=-0{,}20$ | −0,12 |

Mesmo com a canibalização quase anulando seu valor individual, Danos Elétricos ainda tem o maior $\Delta$ (0,01 — os demais já viraram negativos por causa da baixa interação com Desastres). Adiciona. $S_{parcial}=\{\text{Desastres}, \text{Danos Elétricos}\}$.

**Passo 3 — $S_{parcial}=\{\text{Desastres}, \text{Danos Elétricos}\}$.** Cada candidato agora soma duas interações:

| Candidato | $p_{UCB}(i)-0{,}5$ | par c/ Desastres | par c/ Danos Elétricos | $\Delta$ total |
|---|---|---|---|---|
| **Roubo e Furto** | 0,10 | −0,15 | $0{,}58-0{,}5=0{,}08$ | **0,03** |
| Incêndio | 0,02 | −0,10 | $0{,}45-0{,}5=-0{,}05$ | −0,13 |
| Conteúdo Residencial | 0,08 | −0,20 | $0{,}45-0{,}5=-0{,}05$ | −0,17 |

Roubo e Furto é o único com $\Delta>0$ (0,03 — a boa interação com Danos Elétricos compensa a fraca interação com Desastres). Adiciona. $S_{parcial}=\{\text{Desastres}, \text{Danos Elétricos}, \text{Roubo}\}$.

**Passo 4 — $S_{parcial}=\{\text{Desastres}, \text{Danos Elétricos}, \text{Roubo}\}$.** Verifica os 2 candidatos restantes, agora com três interações cada:

| Candidato | $p_{UCB}(i)-0{,}5$ | par c/ Desastres | par c/ Danos Elétricos | par c/ Roubo | $\Delta$ total |
|---|---|---|---|---|---|
| Conteúdo Residencial | 0,08 | −0,20 | −0,05 | $0{,}65-0{,}5=0{,}15$ | **−0,02** |
| Incêndio | 0,02 | −0,10 | −0,05 | $0{,}50-0{,}5=0{,}00$ | **−0,13** |

Mesmo a sinergia de Conteúdo com Roubo (0,65, a maior da tabela) não é suficiente para compensar sua má interação com Desastres e Danos Elétricos — $\Delta$ fica negativo para os dois candidatos restantes. **O guloso para aqui.**

**O que entra / o que sai:** entra: $p_{UCB}(i)$ e $p_{UCB}^{par}(i,j)$ de todos os candidatos. Sai: o pacote final $S_{ofertado} = \{\text{Desastres Naturais}, \text{Danos Elétricos}, \text{Roubo e Furto}\}$ — três coberturas, não porque um $k$ fixo mandou parar em 3, mas porque foi exatamente onde o ganho marginal esgotou.

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

**A matemática — sinal por item** (considera **qualquer** cobertura visível ao cliente, ofertada ou não — inclusive as adicionadas espontaneamente):

| $y_i$ | Condição |
|---|---|
| 1 | se $i \in S_{final}$ |
| 0 | caso contrário |

**Sinal por par** (considera **apenas** pares em que as duas coberturas estavam no pacote **ofertado** — uma cobertura adicionada espontaneamente nunca entra em nenhum par, nem com itens que também estavam na oferta):

| $y_{ij}$ | Condição |
|---|---|
| 1 | se ambos $i,j \in S_{final}$ **e ambos estavam em $S_{ofertado}$** |
| 0 | se ambos estavam em $S_{ofertado}$ e pelo menos um foi removido |

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

Repare: **nenhum** par envolvendo Incêndio entra na tabela — nem (Incêndio, Desastres) nem (Incêndio, Roubo), mesmo os três tendo terminado juntos em $S_{final}$. Isso é proposital, e a próxima subseção explica exatamente por quê.

### Por que a adição espontânea alimenta $y_i$, mas nunca $y_{ij}$

Essa assimetria é a pergunta mais natural de se fazer aqui, e a resposta exige separar duas perguntas que parecem a mesma coisa, mas não são.

**Pergunta 1 (descritiva, de negócio):** "Entre os clientes cujo pacote final contém Incêndio, a taxa de adoção de Desastres Naturais é maior do que entre os que não têm Incêndio?" — uma correlação observada no resultado final. Tanto faz como Incêndio chegou lá (ofertado ou espontâneo); essa pergunta é bem definida com os dados que já temos, incluindo adições espontâneas.

**Pergunta 2 (a que $\phi$ precisa responder, para o oráculo funcionar):** "Se o oráculo **decidir ofertar** Incêndio junto com Desastres, isso **causa** um aumento na probabilidade de os dois serem mantidos juntos?" — uma pergunta sobre o efeito de uma decisão futura, não sobre uma correlação observada no passado. É essa pergunta que decide se vale a pena empacotar os dois para outros clientes no futuro — e é para responder a ela que $\phi$ existe.

**Um exemplo concreto de onde as duas respostas divergem.** Imagine que existe um segmento de clientes — digamos, jovens urbanos com apartamento pequeno — que, por características não capturadas em $c$, tende a gostar tanto de "Roubo e Furto" quanto de "Assistência Chaveiro", independentemente uma da outra, sem nenhuma sinergia real entre as duas. Um desses clientes recebe "Roubo" na oferta, mantém, e adiciona "Chaveiro" por conta própria. Isso responde "sim" à Pergunta 1 — os dois coexistem no pacote final. Mas isso **não é evidência de sinergia**: é só esse segmento gostando de ambas por razões próprias. Se $\hat\phi$ fosse treinada com esse sinal, o modelo aprenderia "Roubo e Chaveiro têm sinergia" e o oráculo passaria a empacotar as duas **para todo mundo** — inclusive para clientes fora desse segmento, para quem empacotar as duas não traz ganho real nenhum (e pode até prejudicar, pela fadiga).

**Por que o viés é tolerável para o item, mas não para o par.** Para o item, mesmo que a adição espontânea seja uma escolha do próprio cliente (endógena), ela ainda aponta na **direção certa** sobre a preferência daquele cliente por aquela cobertura especificamente — o viés está só na magnitude do sinal (fica inflado, porque só clientes com preferência forte adicionam por conta própria), não na direção. Já para o par, o que $\phi$ precisa isolar é um efeito **incremental** — o que sobra **além** da soma das preferências individuais (é exatamente o papel do offset $\eta(i)+\eta(j)$ que introduzimos na Etapa 4). Uma adição espontânea nunca cria a variação necessária para separar "os dois têm efeito conjunto genuíno" de "esse cliente específico gosta dos dois por razões não observadas" — porque o cliente nunca foi exposto a uma situação em que pudesse ter só um dos dois, e a decisão de incluir ambos foi inteiramente dele, não do oráculo.

### Por que tanto a rota aleatória quanto a rota do modelo ajudam a identificar sinergia/canibalização real

A diferença crucial entre um par vindo da **oferta** (aleatória ou via modelo) e um par vindo de **adição espontânea** é: **quem decidiu** que os dois itens ficariam juntos.

- **Rota aleatória:** o pacote é sorteado sem nenhuma relação com o perfil do cliente ou com qualquer preferência dele. Se dois itens aparecem juntos numa oferta aleatória e sobrevivem, isso é a evidência mais limpa possível de interação real — não há absolutamente nenhuma seleção envolvida na composição do par.
- **Rota do modelo:** o oráculo decide o pacote com base só em $x$ (perfil do cliente + atributos das coberturas) e no estado atual do modelo ($\hat\theta,\hat\phi,H$) — nunca em nenhuma informação privada do cliente que não esteja em $x$. Sob a suposição de não-confundimento que já assumimos (tudo que afeta simultaneamente a oferta e a resposta do cliente está em $x$), essa seleção — mesmo sendo determinística e "confiante" — não introduz o mesmo tipo de viés de variável omitida que a adição espontânea introduz.
- **Adição espontânea:** aqui quem decide é o cliente, com base em algo que **não está em $x$** — alguma característica ou motivação que o modelo não observa. É exatamente essa dependência de algo fora de $x$ que quebra a identificação.

Ou seja: tanto as rodadas aleatórias quanto as rodadas guiadas pelo modelo contribuem para o treino de $\phi$ de forma válida — a rota aleatória é a mais "limpa" (nenhuma dependência de $x$ nem do cliente), e a rota do modelo também é válida sob a suposição de não-confundimento, mas com uma ressalva prática: como o oráculo tende a favorecer pares que ele já acredita serem bons, ele naturalmente gera menos variação (menos vezes testa uma dupla que ele já "decidiu" não ser boa) — é mais um motivo, além dos já discutidos, para nunca deixar a fração de exploração aleatória chegar a zero: ela é o que continua garantindo variação de co-ocorrência genuinamente descorrelacionada, inclusive para pares que o modelo ainda não favorece.

### O que fazer com o sinal da adição espontânea, então — não precisa ser descartado, só usado no lugar certo

A Pergunta 1 (a correlação descritiva) continua sendo útil — só não deveria alimentar $\hat\phi$ diretamente. O caminho recomendado é tratá-la como uma **métrica de monitoramento separada**, calculada sobre todos os pares que terminam juntos no pacote final (independente da origem), usada como **radar de hipóteses**: se ela mostrar uma correlação forte entre duas coberturas que $\phi$ ainda não confirma, isso é um sinal para o time de produto investigar — por exemplo, descobrir se existe um segmento de cliente identificável por trás do padrão (como os "jovens urbanos" do exemplo acima). Se esse segmento puder ser adicionado como uma nova dimensão em $c$, o confundimento é resolvido na raiz — o próprio $\theta$ passa a explicar a preferência através desse atributo, em vez de deixá-la vazar como falsa sinergia. Só depois disso, se o par ainda parecer promissor, vale testar deliberadamente ofertando-o junto para uma amostra mais ampla — gerando, aí sim, dados exógenos que alimentam $\phi$ legitimamente.

**O que entra / o que sai:** entra: $S_{ofertado}$, $S_{final}$. Sai: observações $(x_i,y_i)$ para todo item visível, e $(z_{ij},y_{ij})$ apenas para pares onde ambos estavam em $S_{ofertado}$.

---

## 9. Atualização em lote — e o achado importante sobre itens "quase certos"

**Analogia:** no fim do dia, você recalibra as duas réguas usando todas as fichas acumuladas desde sempre — não só as de hoje.

**A matemática do ajuste (MAP com prior gaussiano):**

$$\hat\theta = \arg\max_\theta \sum_{n=1}^{N} \Big[y_n \log p_n + (1-y_n)\log(1-p_n)\Big] - \frac{\lambda}{2}\|\theta\|^2, \qquad p_n=\sigma(\theta^Tx_n)$$

**De onde vem essa fórmula, passo a passo:**

O ponto de partida é a **regra de Bayes**, aplicada a $\theta$ em vez de a um evento simples:

$$\underbrace{P(\theta \mid \text{dados})}_{\text{posteriori}} \;\propto\; \underbrace{P(\text{dados} \mid \theta)}_{\text{verossimilhança}} \;\times\; \underbrace{P(\theta)}_{\text{priori}}$$

| Termo | O que significa aqui |
|---|---|
| **Priori** $P(\theta)$ | o que acreditávamos sobre $\theta$ **antes** de ver qualquer dado — no nosso caso, $\theta \sim \mathcal{N}(0, \lambda^{-1}I)$: um "chute" de que os valores de $\theta$ tendem a ficar perto de zero, com $\lambda$ controlando o quão fortemente essa crença é mantida |
| **Verossimilhança** $P(\text{dados}\mid\theta)$ | o quão prováveis são os dados observados, **assumindo** um valor particular de $\theta$ — é a peça que faz o modelo "aprender" a partir dos dados |
| **Posteriori** $P(\theta\mid\text{dados})$ | a crença atualizada sobre $\theta$, depois de combinar o que já se sabia (priori) com o que os dados mostraram (verossimilhança) — é essa distribuição que queremos maximizar para achar $\hat\theta$ |

**Passo 1 — a verossimilhança de uma única observação.** Como $y_n \in \{0,1\}$, cada observação segue uma distribuição de **Bernoulli** com parâmetro $p_n$ (a probabilidade de $y_n=1$). A fórmula fechada de uma Bernoulli é:

$$P(y_n \mid \theta) = p_n^{y_n}(1-p_n)^{1-y_n}$$

Repare no truque dessa fórmula: se $y_n=1$, o expoente de $(1-p_n)$ vira zero, então $(1-p_n)^0=1$ e sobra só $p_n$; se $y_n=0$, o expoente de $p_n$ vira zero e sobra só $(1-p_n)$. Ou seja, essa única expressão cobre os dois casos possíveis ao mesmo tempo.

**Passo 2 — a verossimilhança de todas as $N$ observações juntas.** Assumindo que cada observação é independente das demais (dado $\theta$), a verossimilhança conjunta é o **produto** das verossimilhanças individuais:

$$P(\text{dados}\mid\theta) = \prod_{n=1}^{N} p_n^{y_n}(1-p_n)^{1-y_n}$$

**Passo 3 — passar para o logaritmo.** Produtos de números entre 0 e 1 ficam numericamente instáveis (tendem a zero rapidinho) e são difíceis de derivar. Como $\log$ é uma função crescente, maximizar a verossimilhança é o mesmo que maximizar o **logaritmo** da verossimilhança — e o logaritmo transforma o produto em soma:

$$\log P(\text{dados}\mid\theta) = \sum_{n=1}^{N} \log\Big(p_n^{y_n}(1-p_n)^{1-y_n}\Big) = \sum_{n=1}^{N}\Big[y_n\log p_n + (1-y_n)\log(1-p_n)\Big]$$

(usando a propriedade $\log(a^k)=k\log a$ e $\log(ab)=\log a + \log b$).

**Passo 4 — o logaritmo da priori.** A densidade de uma gaussiana $\mathcal{N}(0,\lambda^{-1}I)$ é $\propto \exp\big(-\frac{\lambda}{2}\|\theta\|^2\big)$. Tomando o logaritmo, a parte que depende de $\theta$ é exatamente:

$$\log P(\theta) = -\frac{\lambda}{2}\|\theta\|^2 + \text{constante (não depende de }\theta\text{)}$$

**Passo 5 — somar as duas partes.** Como $\log(\text{verossimilhança}\times\text{priori}) = \log(\text{verossimilhança}) + \log(\text{priori})$, a soma dá o log da posteriori **exatamente** — não é mais uma relação de proporcionalidade, é uma **igualdade**, só que com uma constante **somada** (em vez de multiplicada) que não depende de $\theta$:

$$\log P(\theta\mid\text{dados}) = \sum_{n=1}^{N}\Big[y_n\log p_n + (1-y_n)\log(1-p_n)\Big] - \frac{\lambda}{2}\|\theta\|^2 + \text{constante}$$

**Uma correção de notação que vale destacar:** antes do logaritmo, $P(\theta\mid\text{dados})\propto P(\text{dados}\mid\theta)P(\theta)$ usa $\propto$ corretamente — a constante descartada (a "evidência" $P(\text{dados})$, no denominador da regra de Bayes) é **multiplicativa**. Mas depois de aplicar $\log$, essa mesma constante deixa de multiplicar e passa a **somar** — então o símbolo correto deixa de ser $\propto$ (que denota especificamente uma relação multiplicativa) e passa a ser $=$ com um termo aditivo explícito, como escrito acima. Como essa constante não depende de $\theta$, ela desaparece ao maximizar (não muda onde fica o pico) — mas o símbolo usado para chegar lá precisa refletir a operação certa.

Maximizar essa expressão em $\theta$ é exatamente a fórmula do início desta seção — o $\hat\theta$ resultante é chamado de estimativa **MAP** (*maximum a posteriori*: o valor de $\theta$ mais provável dado o que já foi observado e o que já se acreditava antes).

### Por que isso não tem solução fechada: o problema da conjugação

**O que é "conjugação".** Uma priori é dita **conjugada** com uma verossimilhança quando o produto das duas — a posteriori, antes de normalizar — resulta numa distribuição da **mesma família** da priori. Quando isso acontece, dá para escrever a posteriori inteira em forma fechada, exatamente, não só o ponto de pico dela.

**O caso ridge (C²UCB original) é conjugado, e é por isso que ele tem solução fechada.** Lá, a verossimilhança de cada observação é gaussiana ($y_n\sim\mathcal N(\theta^Tx_n,\sigma^2)$), e a priori sobre $\theta$ também é gaussiana. **Gaussiana × Gaussiana = Gaussiana** — a conjugação mais simples que existe. Isso significa que $\hat\theta=V^{-1}b$ não é só "o pico" da posteriori — é a **média exata** da posteriori, e $V^{-1}$ é a **covariância exata** dela, sem nenhuma aproximação: a posteriori *é*, literalmente, $\mathcal N(V^{-1}b,\ V^{-1})$.

**O nosso caso (Bernoulli × gaussiana) não é conjugado.** O produto de uma verossimilhança de Bernoulli (via sigmoide) por uma priori gaussiana não resulta em nenhuma distribuição de família conhecida — a posteriori existe (é uma distribuição de probabilidade válida), mas não sabemos escrever sua forma exata. Sabemos só que $P(\theta\mid\text{dados})\propto P(\text{dados}\mid\theta)P(\theta)$, até uma constante — e essa constante em si já é intratável de calcular exatamente.

**As duas consequências práticas disso, que são problemas distintos:**

1. **Achar o pico já exige um otimizador.** Como $p_n=\sigma(\theta^Tx_n)$ entra no logaritmo de forma não-linear, a derivada de $\log P(\theta\mid\text{dados})$ em relação a $\theta$ não é linear — não dá para isolar $\theta$ com álgebra simples, como no caso ridge (onde a derivada é linear e vira o sistema $V\theta=b$). É por isso que a maximização precisa de um otimizador iterativo (`scipy.optimize.minimize` no código, usando o gradiente $\nabla_\theta = \sum_n x_n(y_n-p_n) - \lambda\theta$) em vez de uma única inversão de matriz.

2. **Mesmo depois de achar o pico, não sabemos o formato exato da incerteza ao redor dele.** Esta é a parte mais sutil, e o motivo pelo qual precisamos da **aproximação de Laplace**, introduzida na próxima subseção: como a posteriori não tem forma fechada, não existe uma "covariância exata" para extrair, do jeito que $V^{-1}$ era exata no caso ridge. A aproximação de Laplace contorna isso **aproximando a posteriori real (intratável) por uma gaussiana**, centrada no pico $\hat\theta$ que acabamos de achar, com covariância igual ao inverso da curvatura (o Hessiano negativo) da log-posteriori naquele ponto — é isso que $H_\theta$ representa. A justificativa: perto do pico, qualquer função suave se parece com uma parábola (expansão de Taylor de segunda ordem), e o exponencial de uma parábola é, por definição, uma gaussiana. **$H_\theta$ não é a covariância exata da posteriori — é a covariância da aproximação gaussiana local a ela**, o que explica por que essa garantia é assintótica (fica mais precisa com mais dados, via o teorema de Bernstein–von Mises), não exata em amostra finita, como discutimos no efeito kappa logo abaixo.

Aqui $\lambda$ aparece como o termo $-\frac{\lambda}{2}\|\theta\|^2$: uma penalização que cresce conforme $\theta$ se afasta de zero. Na prática, isso impede que $\hat\theta$ "dispare" para valores extremos quando os dados são escassos ou quando uma cobertura é aceita/rejeitada de forma quase unânime (o cenário de quase-separação que motiva boa parte desta etapa) — sem essa penalização, a otimização poderia tentar empurrar $\theta$ ao infinito para explicar perfeitamente um padrão determinístico nos dados.

### O problema que descobrimos: o efeito "kappa"

Ao calcular a incerteza (matriz $H_\theta$) via aproximação de Laplace, a fórmula original era:

$$H_\theta = \lambda I + \sum_{n=1}^{N} p_n(1-p_n) x_n x_n^{T}$$

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

$$H_\theta = \lambda I + \sum_{n=1}^{N} \max\big(p_n(1-p_n),\ \tau\big) x_n x_n^{T}$$

**Analogia da correção:** é como dizer ao jurado "mesmo em casos óbvios, você ainda precisa prestar um mínimo de atenção a cada novo caso" — garantindo que a confiança sempre cresce com mais dados, nunca estaciona artificialmente.

| Símbolo | Descrição |
|---|---|
| $p_n(1-p_n)$ | quanto aquela observação específica reduz a incerteza — máximo em casos "difíceis" (perto de 50/50), quase zero em casos "óbvios" |
| $\tau$ | piso mínimo — garante que mesmo observações de casos óbvios continuem contribuindo |
| $H_\theta$ (corrigido) | nova "memória", agora sem o platô artificial, alimentando o bônus da próxima rodada (Etapa 3) |

**Por que essa correção foi escolhida em vez de uma alternativa por contagem bruta:** o sistema é contextual — o mesmo item aparece em contextos diferentes (perfis de cliente diferentes, tamanhos de pacote diferentes). Uma correção baseada em "contar quantas vezes vi esse item" jogaria fora a generalização entre contextos parecidos, que é o motivo de termos escolhido C²UCB desde o início. O piso de curvatura, em vez disso, corrige o mecanismo exato onde o problema nasce (o peso da observação dentro de $H$), sem sair da geometria contextual do modelo.

### A mesma atualização acontece em paralelo para $\phi$ (o termo de par)

**Analogia:** enquanto uma régua ($\theta$) está sendo recalibrada com as fichas de "cobertura isolada", a segunda régua ($\phi$) está sendo recalibrada ao mesmo tempo, no mesmo fechamento de caixa, mas usando as fichas de "par de coberturas" (as observações $(z_{ij}, y_{ij})$ da Etapa 8) — é o mesmo procedimento, rodando sobre um conjunto de dados diferente.

**A matemática:** estrutura quase idêntica à atualização de $\theta$, trocando o contexto de item $x_n$ pelo contexto de par $z_n$, e o rótulo $y_n$ pelo rótulo $y_n^{par}$ — com uma diferença importante: o **offset** $o_n=\eta(i_n)+\eta(j_n)$ (os scores que $\theta$ já estimou para os dois itens daquela observação, calculados na Etapa 3, tratados aqui como **constantes conhecidas**, não reajustadas junto com $\phi$):

$$\hat\phi = \arg\max_\phi \sum_{n=1}^{M} \Big[y_n^{par} \log p_n^{par} + (1-y_n^{par})\log(1-p_n^{par})\Big] - \frac{\lambda}{2}\|\phi\|^2, \qquad p_n^{par}=\sigma(o_n+\phi^Tz_n)$$

$$H_\phi = \lambda I + \sum_{n=1}^{M} \max\big(p_n^{par}(1-p_n^{par}),\ \tau\big) z_n z_n^{T}$$

**Por que o offset precisa entrar aqui — a redundância entre $y_i$ e $y_{ij}$.** Vale destacar algo que só fica visível quando olhamos as duas atualizações lado a lado: $y_{ij}$ **não é uma observação independente** de $y_i$ e $y_j$ — por definição, $y_{ij}=y_i\times y_j$ (o "E" lógico das duas decisões). Se $\phi$ fosse treinada sem o offset (só com $p_n^{par}=\sigma(\phi^Tz_n)$, como numa primeira formulação mais ingênua), ela acabaria reaprendendo, de forma distorcida, parte do que $\theta$ já sabe sobre cada item isoladamente — porque mesmo sob independência perfeita entre aceitar $i$ e aceitar $j$ (ou seja, **sem nenhuma sinergia real**), a probabilidade conjunta seria $p_i^{\ast}\times p_j^{\ast}$, e $\text{logit}(p_i^{\ast}\times p_j^{\ast}) \neq \text{logit}(p_i^{\ast})+\text{logit}(p_j^{\ast})$ — a curvatura de combinar duas probabilidades não é linear. Sem o offset, $\hat\phi$ seria forçada a capturar essa curvatura matemática mesmo na ausência de qualquer interação comportamental real, e o oráculo (Etapa 5) contaria o efeito individual **duas vezes** — uma via $p_{UCB}(i)$, outra (distorcida) via $p_{UCB}^{par}(i,j)$. Fixando o offset, $\phi$ passa a aprender exclusivamente o **resíduo**: o que sobra além do que a soma dos efeitos individuais já explicaria — e, sob independência perfeita, o ajuste ótimo de $\phi$ converge para zero, como deveria.

| Símbolo | Descrição |
|---|---|
| $M$ | número de observações de **par** acumuladas (lembrando: cada rodada gera vários pares a partir do pacote ofertado — ver Etapa 8) |
| $z_n$ | o contexto do par $(i,j)$ daquela observação — o mesmo $z(i,j)$ definido na Etapa 4 |
| $o_n=\eta(i_n)+\eta(j_n)$ | o offset — a soma dos scores lineares que $\theta$ já estimou para os dois itens, fixada durante o ajuste de $\phi$ |
| $y_n^{par}$ | 1 se as duas coberturas do par sobreviveram juntas no pacote final, 0 caso contrário |
| $\hat\phi$ | a régua de sinergia/canibalização recalibrada — agora representando exclusivamente o efeito incremental, livre da redundância com $\theta$ |
| $H_\phi$ (corrigido) | a incerteza sobre $\phi$, já com o mesmo piso de curvatura $\tau$ aplicado — sem ele, pares de coberturas quase sempre aceitas ou quase sempre rejeitadas juntas sofreriam exatamente o mesmo platô artificial que identificamos para $\theta$ |

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
Etapa 4 (score par)   ←── φ̂, H_φ (da Etapa 9 anterior) + c + η(i),η(j) [offset, da Etapa 3]
     ▼
Etapa 5 (oráculo guloso) ──> usa p_UCB(i) e p_UCB_par(i,j), atualiza |S_parcial|
     ▼
Etapa 6 (oferta) ──> S_ofertado
     ▼
Etapa 7 (cliente edita) ──> S_final
     ▼
Etapa 8 (traduz em y_i p/ todo item; y_ij só p/ pares 100% ofertados)
     ▼
Etapa 9 (reajuste em lote, c/ piso τ e offset em φ) ──> novo θ̂, H_θ, φ̂, H_φ
     │
     └──────────────► volta para a Etapa 3 da próxima rodada (Etapa 10)
```

---

## Tabela-síntese: o que cada peça resolve, com o exemplo em mente

| Peça | Resolve | No exemplo do Cliente A |
|---|---|---|
| $\sigma(\cdot)$ (funil logístico) | garante que a probabilidade nunca sai do intervalo entre 0 e 1 | $p_{UCB}(\text{Desastres}) = 0{,}73$, nunca $>1$ |
| $\theta$ + tamanho do pacote | relevância individual **e** fadiga, aprendidas juntas | régua aprende como a chance de manter cai conforme a mala enche |
| $\phi$ (par) | sinergia (+) e canibalização (−) específicas | par (Desastres, Danos Elétricos) fica cada vez mais negativo |
| Centralização em 0,5 (exata) | transforma "menos bom" em "genuinamente ruim" | $\Delta(\text{Danos Elétricos}) \approx 0{,}01$ — quase descartado |
| $H_\theta, H_\phi$ (Laplace) | mede incerteza, alimenta o bônus de exploração | par (Desastres, Roubo) ainda com bônus alto — pouco visto |
| **Piso de curvatura $\tau$ (novo)** | evita que itens "quase certos" (ex.: Incêndio) parem de reduzir incerteza | bônus cai de 0,50 (estacionado) para 0,25 após 300 observações |
| **Offset $\eta(i)+\eta(j)$ em $\phi$ (novo)** | remove a redundância entre $y_i$ e $y_{ij}$ — $\phi$ passa a aprender só o efeito incremental, não o que $\theta$ já explica | par (Desastres, Danos Elétricos): offset alto (1,65) exige $\hat\phi^Tz$ bem negativo (−2,40) para confirmar canibalização real |
| Reajuste em lote | atualiza as réguas periodicamente, sem precisar ser a cada evento | fechamento diário, reprocessando o histórico armazenado |

---

## Parâmetros que vocês precisam definir (não vêm "de fábrica")

Tudo que apareceu nas equações acima com uma letra grega ou um nome fixo é uma peça da matemática — mas boa parte deles não tem um valor certo universal: são escolhas de configuração que a equipe precisa fixar (e depois calibrar empiricamente) antes de colocar o sistema no ar.

| Parâmetro | Onde aparece | O que controla | Ponto de partida sugerido |
|---|---|---|---|
| $\lambda$ (regularização) | Etapas 0 e 9 | O quanto $\theta$ e $\phi$ ficam "presos" perto de zero antes de haver evidência forte — protege contra estimativas explosivas com poucos dados ou padrões quase determinísticos | Testar valores entre 0,5 e 2,0; pode (e provavelmente deve) ser diferente para $\theta$ e para $\phi$ — faz sentido, por exemplo, o modelo de par ter um $\lambda_\phi$ maior, já que ele tem muito mais parâmetros para estimar (91 pares, com 14 coberturas) a partir da mesma quantidade de dados que o modelo de item, precisando de uma regularização mais forte para não overfitar |
| $\alpha$ (peso de exploração) | Etapas 3 e 4 | O quanto o bônus de incerteza pesa na decisão final — maior $\alpha$ = mais disposição a testar combinações pouco vistas, mesmo com estimativa pontual baixa | Começar mais alto (ex.: 1,5–2,0) na fase inicial e reduzir gradualmente conforme o histórico cresce |
| $\tau$ (piso de curvatura) | Etapa 9 | Evita que itens/pares "quase certos" parem de reduzir incerteza (o efeito kappa) | Algo entre 0,02 e 0,10 (referência: $p(1-p)$ tem máximo 0,25 em $p=0{,}5$) |
| Frequência de reajuste (tamanho do lote) | Etapa 9 | De quanto em quanto tempo (ou a cada quantos clientes) o histórico completo é reprocessado | Diário costuma ser suficiente para o volume esperado; pode ser ajustado conforme o tráfego real |
| $\epsilon_t$ (probabilidade de exploração aleatória, via $Z_t\sim\text{Bernoulli}(\epsilon_t)$) e seu cronograma | Etapa 2 | Por quanto tempo, e com que intensidade, a oferta aleatória (sem modelo) continua acontecendo | Começar próximo de 1,0 (quase só aleatório) e decair até um piso pequeno (ex.: 0,05) ao longo de semanas/meses — nunca chegar a zero. Pode seguir uma fórmula de decaimento (linear, exponencial) ou ser ajustado manualmente por estágio do projeto, lido de um valor de configuração vigente (ex.: `epsilon_atual` num arquivo/tabela versionado) em vez de calculado por uma fórmula fixa — especialmente relevante numa arquitetura de serviço via API, onde não existe mais um contador único de "rodada $t$" |
| Dimensões semânticas de $f_i$ (o catálogo) | Etapas 3 e 4 | Quais atributos descrevem cada cobertura (categoria de risco, faixa de prêmio, obrigatoriedade, frequência de sinistro, e quaisquer outras que a equipe atuarial considere relevantes) | Definido uma vez, manualmente, com a equipe de produto/atuária — pode evoluir depois para embeddings aprendidos, conforme discutido anteriormente |
| Dimensões do perfil do cliente $c$ | Etapa 1 | Quais atributos do cliente entram no modelo (score de risco, valor do imóvel, região, composição familiar etc.) | Definido com a equipe de dados, com base no que já está disponível no cadastro |

**Por que isso importa destacar separadamente:** nenhum desses valores tem uma "resposta certa" que a matemática determine sozinha — são decisões operacionais que afetam diretamente a velocidade de aprendizado e o comportamento inicial do sistema. Vale tratá-los como parte do escopo de calibração/validação antes do lançamento, não como detalhes de implementação a serem decididos casualmente.
