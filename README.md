# TradingBot

Bot de trading automático de criptomoedas para a exchange **Kraken**. O bot
monitoriza continuamente vários pares cotados em EUR, decide quando comprar
e vender com base em médias móveis e regras de gestão de risco, e envia
ordens de mercado através da API da Kraken.

> ⚠️ **Aviso:** este projeto opera com dinheiro real quando ligado a uma conta
> Kraken. Use por sua conta e risco, comece com montantes pequenos e teste
> primeiro com as simulações incluídas.

---

## Índice

1. [Visão geral](#visão-geral)
2. [Estrutura do projeto](#estrutura-do-projeto)
3. [Instalação e configuração](#instalação-e-configuração)
4. [Como funciona o bot principal (`bot.py`)](#como-funciona-o-bot-principal-botpy)
5. [Estratégia alternativa: cruzamento de SMA](#estratégia-alternativa-cruzamento-de-sma)
6. [Ficheiros de estado](#ficheiros-de-estado)
7. [Simulação, otimização e testes](#simulação-otimização-e-testes)
8. [Versão C# (`Bot.cs`)](#versão-c-botcs)
9. [Licença](#licença)

---

## Visão geral

| Item               | Valor                                                           |
|--------------------|-----------------------------------------------------------------|
| Exchange           | Kraken (via biblioteca [`krakenex`](https://github.com/veox/python3-krakenex)) |
| Pares negociados   | `XETHZEUR` (ETH), `XXBTZEUR` (BTC), `MANAEUR`, `GRTEUR`, `LSKEUR`, `SCEUR` |
| Moeda base         | EUR (`ZEUR` no saldo da Kraken)                                 |
| Tipo de ordem      | Ordens de mercado (`market`)                                    |
| Ciclo de execução  | A cada 20 segundos (`bot.py`)                                   |
| Persistência       | Ficheiros JSON locais (`data.json`, `trades.json`)              |

O bot gere **uma posição de cada vez por par**. Para cada par, o estado é
determinado pela última operação registada em `trades.json`:

- sem operações, ou última operação foi uma **venda** → modo **COMPRA**
  (procura um ponto de entrada);
- última operação foi uma **compra** → modo **VENDA** (gere a posição aberta).

---

## Estrutura do projeto

```
TradingBot/
├── bot.py                 # Bot principal (estratégia de tendência da MA + pullback)
├── bot2.py                # Bot alternativo: cruzamento SMA20/SMA50 (trades2.json)
├── sma_crossover_bot.py   # Bot SMA20/SMA50 com ficheiros de estado próprios
├── bot_sim.py             # Versão antiga do bot com saldo simulado (balance.json)
├── Bot.cs / TradingBot.csproj  # Port do bot para C# / .NET 8
│
├── test_bot.py            # Testes unitários do bot.py (unittest + mocks)
├── simulate.py            # Simulação do bot.py com dados OHLC sintéticos
├── update_data.py         # Preenche data.json com ~2 meses de histórico da Kraken
│
├── montecarlo*.py         # Simulações Monte Carlo de diferentes variantes
├── optimize.py            # Varrimento de parâmetros (grid search) via Monte Carlo
├── stop_loss_study.py     # Estudo do stop de break-even
├── compare_bots.py        # Comparação bot.py vs sma_crossover_bot.py vs Hold
│
├── data.json              # Estado: histórico de médias móveis por par
├── trades.json            # Estado: histórico de compras/vendas por par
├── balance.json           # Saldo fictício usado pelo bot_sim.py
├── kraken.key             # Credenciais da API Kraken (NÃO partilhar)
└── Resources/             # Documentação de referência (Turtle Trading System)
```

---

## Instalação e configuração

### Requisitos

- Python 3.9+
- `krakenex` (bot em produção)
- `numpy` (apenas para `compare_bots.py`, `montecarlo.py` e `stop_loss_study.py`)

```bash
pip install krakenex numpy
```

### Chave da API Kraken

O bot lê as credenciais do ficheiro `kraken.key`, na pasta onde é executado,
no formato esperado pelo `krakenex`:

```
<API key>
<private key>
```

A chave deve ter permissões para **consultar saldos** e **criar ordens**.
Não dê permissão de levantamento (*withdraw*).

> 🔒 Este ficheiro contém segredos. Não o envie para repositórios públicos —
> considere adicioná-lo ao `.gitignore` e revogar/rodar a chave caso já tenha
> sido publicada.

### Executar

```bash
python bot.py
```

Na primeira execução são criados automaticamente `trades.json` e `data.json`
vazios, caso não existam. O bot corre indefinidamente; termine-o com `Ctrl+C`.
Opcionalmente, pode pré-carregar histórico de médias móveis com:

```bash
python update_data.py
```

---

## Como funciona o bot principal (`bot.py`)

### Ciclo principal

A cada 20 segundos, para cada par:

1. Obtém as velas OHLC de 1 minuto das **últimas 12 horas** (`OHLC` público da Kraken).
2. Calcula a **média móvel das últimas 100 velas** (máximos, mínimos e fechos)
   e acrescenta esse valor ao histórico em `data.json` (limitado a 200 pontos).
3. Se houver uma posição aberta, atualiza o **preço máximo atingido**
   (`peak_price`) desde a compra.
4. Avalia o sinal de compra ou de venda, consoante o modo do par.

Erros num par (rede, API) são registados e não interrompem os restantes.

### Sinal de compra

A compra é feita quando **todas** as condições seguintes se verificam:

1. **Tendência de subida consistente** — nos últimos 10 valores da média móvel
   de fecho existem pelo menos **5 subidas consecutivas**.
2. **Proximidade da média (pullback)** — a média de `MA / preço` nos últimos
   5 pontos é ≥ `0.97`, ou seja, o preço não está mais de ~3% acima da média
   (evita comprar depois de uma subida já esticada).
3. **Vela de confirmação** — a vela atual é altista (`fecho ≥ abertura`),
   indicando que o preço está a ressaltar e não ainda a cair.

**Dimensionamento da posição:** o saldo em EUR disponível é dividido
igualmente pelos pares que ainda não têm posição aberta
(`saldo EUR / nº de slots livres`). É enviada uma ordem de mercado com o
volume `fundos / preço`.

### Sinal de venda (gestão da posição)

Com uma posição aberta, as regras são avaliadas por esta ordem:

| Regra              | Condição                                                        | Efeito |
|--------------------|-----------------------------------------------------------------|--------|
| **Take profit**    | preço ≥ preço de compra × 1.03                                  | Vende com +3% |
| **Break-even stop**| o pico já atingiu +1% e o preço volta abaixo do preço de compra | Vende a ~0% |
| **Stop-loss**      | (pico ainda não chegou a +1%) e preço < compra × 0.99           | Vende com −1% |
| **Trailing stop**  | o pico ultrapassou +1.5% e o preço cai 1% abaixo do pico        | Vende protegendo o ganho |

A quantidade vendida é a registada na última compra em `trades.json`.

### Parâmetros

Todos os limiares estão definidos como constantes no topo de `bot.py`:

| Constante             | Valor   | Significado                                   |
|-----------------------|---------|-----------------------------------------------|
| `_PROFIT_TARGET`      | `1.03`  | Take profit a +3%                             |
| `_STOP_LOSS`          | `0.99`  | Stop-loss inicial a −1%                       |
| `_BREAK_EVEN_TRIGGER` | `1.01`  | A partir de +1% no pico o stop sobe para 0%   |
| `_TRAILING_STOP`      | `0.99`  | Trailing stop a −1% do pico                   |
| `_PROXIMITY_MIN`      | `0.97`  | Proximidade mínima preço/média para entrar    |
| `_WINDOW_SECONDS`     | `43200` | Janela de dados OHLC (12 horas)               |
| `_MAX_MA_HISTORY`     | `200`   | Nº máximo de pontos de média guardados        |
| `_PAIRS`              | 6 pares | Pares negociados                              |

---

## Estratégia alternativa: cruzamento de SMA

Existem duas implementações de uma estratégia clássica de *golden/death cross*
com médias simples de 20 e 50 velas de 1 minuto:

- **Compra:** a SMA20 cruza **acima** da SMA50 (*golden cross*).
- **Venda:** a SMA20 cruza **abaixo** da SMA50 (*death cross*) com lucro —
  `bot2.py` exige pelo menos +3%, `sma_crossover_bot.py` exige apenas preço
  acima do de compra.
- **Stop-loss:** −1% em relação ao preço de compra, sempre ativo.

| Script                  | Ciclo | Ficheiros de estado                         |
|-------------------------|-------|---------------------------------------------|
| `bot2.py`               | 60 s  | `trades2.json`                              |
| `sma_crossover_bot.py`  | —     | `sma_state.json`, `sma_trades.json`         |

Como usam ficheiros próprios, podem correr em paralelo com `bot.py` sem
conflitos de estado (mas partilham o mesmo saldo na conta Kraken).

---

## Ficheiros de estado

### `trades.json`

Histórico de operações por par. Cada entrada:

```json
{
    "time_stamp": "1700000000",
    "price_usd": 1850.42,
    "bought": true,
    "sold": false,
    "amount": 0.0054
}
```

> Apesar do nome `price_usd`, o preço está na moeda de cotação do par (EUR).

### `data.json`

Para cada par guarda as listas de médias móveis (`high`, `low`, `close`),
as velas já vistas (`prices`) e o `peak_price` da posição aberta. Após cada
compra/venda as listas são truncadas aos últimos 10 valores e o pico é reposto.

Para recomeçar do zero, pare o bot e apague (ou esvazie) estes ficheiros.

---

## Simulação, otimização e testes

Nenhum destes scripts envia ordens reais.

| Script                 | Descrição |
|------------------------|-----------|
| `test_bot.py`          | Testes unitários do `bot.py` com a API da Kraken simulada (`python -m unittest test_bot`). |
| `simulate.py`          | Corre a lógica do `bot.py` sobre velas sintéticas numa pasta temporária. |
| `montecarlo.py`        | Monte Carlo de 1 semana com volatilidade calibrada por par; compara variantes de trailing stop. |
| `montecarlo_1m.py`     | 1 000 000 simulações em memória, multiprocessamento. |
| `montecarlo_eur.py`    | 1 000 000 simulações com €100; mede quantos ciclos até ao primeiro lucro. |
| `montecarlo_3to1.py`   | Testa a relação risco/recompensa 3:1 (TP +3% / SL −1%). |
| `montecarlo_value.py`  | Distribuição do valor final da carteira em 1000 simulações. |
| `optimize.py`          | *Grid search* de take profit, stop-loss, trailing, etc., ordenado por valor esperado, taxa de acerto e Sharpe. |
| `stop_loss_study.py`   | Estudo que levou ao stop de break-even a +1%. |
| `compare_bots.py`      | Compara `bot.py`, `sma_crossover_bot.py` e *buy & hold* nos mesmos caminhos de preço. |
| `bot_sim.py`           | Versão antiga do bot que usa `balance.json` como saldo fictício. |

Exemplos:

```bash
python simulate.py
```

```bash
python compare_bots.py
```

> Alguns scripts de simulação mantêm cópias próprias dos parâmetros e podem
> não refletir exatamente os valores atuais de `bot.py`.

---

## Versão C# (`Bot.cs`)

`Bot.cs` é um port da estratégia para .NET 8 que fala diretamente com a API
REST da Kraken (assinatura HMAC-SHA512 própria) e usa os mesmos ficheiros
`kraken.key`, `data.json` e `trades.json`. Os seus parâmetros (por exemplo,
take profit a +5%) correspondem a uma versão anterior do `bot.py`.

```bash
dotnet run
```

---

## Licença

Distribuído sob a [Apache License 2.0](LICENSE).
