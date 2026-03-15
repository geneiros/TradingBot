#!/usr/bin/env python3
"""
Monte Carlo simulation — final portfolio value analysis.
Starts with $10,000 and shows the distribution of final dollar values
across 1000 independent simulations.
"""

import sys, json, time, random, os, tempfile, statistics, contextlib
from unittest.mock import MagicMock

sys.modules['krakenex'] = MagicMock()
import bot

# ── Suppress internal bot prints ─────────────────────────────────────────────
def _silent_save_trade(close, name, bought, sold, amount):
    trade = {'time_stamp': str(int(time.time())), 'price_usd': close,
             'bought': bought, 'sold': sold, 'amount': amount}
    trades = bot.load_trades()
    trades[name].append(trade)
    with open('trades.json', 'w') as f:
        json.dump(trades, f, indent=4)

def _silent_buy(crypto_data, name):
    bot.clear_crypto_data(name)
    price  = float(crypto_data[-1][4])
    funds  = bot.get_available_funds()
    amount = funds * (1 / price)
    bot.update_balance(amount, name, price, False)
    _silent_save_trade(price, name, True, False, amount)

def _silent_sell(crypto_data, name):
    balance = bot.get_balance()
    bot.clear_crypto_data(name)
    price  = float(crypto_data[-1][4])
    amount = float(balance[name[:-4]])
    bot.update_balance(amount, name, price, True)
    _silent_save_trade(price, name, False, True, amount)

bot.save_trade  = _silent_save_trade
bot.buy_crypto  = _silent_buy
bot.sell_crypto = _silent_sell

# ── Config ────────────────────────────────────────────────────────────────────
NUM_SIMULATIONS = 1000
NUM_CYCLES      = 20
INITIAL_USD     = 10_000.0
PAIRS           = bot.get_pairs()

BASE_PRICES = {
    'XETHZUSD': 2000.0, 'XXBTZUSD': 40000.0,
    'MANAUSD':  0.50,   'GRTUSD':   0.15,
    'LSKUSD':   1.50,   'SCUSD':    0.003,
}

tmpdir = tempfile.mkdtemp(prefix='tradingbot_val_')
os.chdir(tmpdir)

# ── Helpers ───────────────────────────────────────────────────────────────────
def generate_ohlc(base_price, n=200, trend='up', vol=0.004):
    candles, price = [], base_price
    t = int(time.time()) - n * 60
    drift = {'up': (0.0003, 0.0015), 'down': (-0.0015, -0.0003), 'flat': (-0.0005, 0.0005)}
    lo, hi = drift[trend]
    for i in range(n):
        price *= 1 + random.uniform(lo, hi) + random.uniform(-vol, vol)
        price  = max(price, 1e-6)
        high   = price * (1 + abs(random.uniform(0, vol)))
        low    = price * (1 - abs(random.uniform(0, vol)))
        close  = price * (1 + random.uniform(-vol / 2, vol / 2))
        candles.append([t + i * 60,
            f"{price:.8f}", f"{high:.8f}", f"{low:.8f}", f"{close:.8f}",
            f"{(high+low+close)/3:.8f}", f"{random.uniform(10,500):.2f}",
            random.randint(50, 500)])
    return candles

def reset_state():
    with open('balance.json', 'w') as f:
        json.dump({'ZUSD': str(INITIAL_USD), 'EUR.HOLD': '0.0'}, f)
    with open('trades.json', 'w') as f:
        json.dump({p: [] for p in PAIRS}, f)
    bot.mva = bot.make_crypto_data({})
    with open('data.json', 'w') as f:
        json.dump(bot.mva, f)

def run_one(seed):
    random.seed(seed)
    reset_state()

    vol    = random.uniform(0.002, 0.010)
    trends = {p: random.choice(['up', 'down', 'flat']) for p in PAIRS}
    prices = {p: BASE_PRICES[p] * random.uniform(0.80, 1.20) for p in PAIRS}
    hist   = {p: generate_ohlc(prices[p], n=200, trend=trends[p], vol=vol) for p in PAIRS}

    for cycle in range(NUM_CYCLES):
        for pair in PAIRS:
            trades     = bot.load_trades()
            end        = min(100 + cycle, len(hist[pair]))
            cdata      = hist[pair][end - 100: end]
            last_trade = trades[pair][-1] if trades[pair] else None
            should_buy = (last_trade is None) or last_trade['sold']
            bot.check_data(pair, cdata, should_buy)
            bot.mva = bot.load_crypto_data_from_file()

    bal   = bot.get_balance()
    usd   = float(bal.get('ZUSD', 0))
    crypto_val = sum(
        float(bal.get(pair[:-4], 0)) * float(hist[pair][-1][4])
        for pair in PAIRS
    )
    total      = usd + crypto_val
    all_trades = bot.load_trades()
    n_trades   = sum(len(v) for v in all_trades.values())
    n_sells    = sum(1 for tl in all_trades.values() for t in tl if t['sold'])

    return {
        'seed':       seed,
        'usd':        usd,
        'crypto_val': crypto_val,
        'total':      total,
        'pnl':        total - INITIAL_USD,
        'pnl_pct':    (total - INITIAL_USD) / INITIAL_USD * 100,
        'n_trades':   n_trades,
        'n_sells':    n_sells,
        'trends':     trends,
    }

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\n{'═'*66}")
print(f"  MONTE CARLO — FINAL VALUE ANALYSIS")
print(f"  {NUM_SIMULATIONS} simulations  ×  {NUM_CYCLES} cycles  ×  {len(PAIRS)} pairs")
print(f"  Starting capital : ${INITIAL_USD:,.2f}")
print(f"  Strategy         : Take-profit +3%  |  Stop-loss -1%  |  Trailing -2%")
print(f"{'═'*66}\n")

start_t = time.time()
results = []
for i in range(NUM_SIMULATIONS):
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results.append(run_one(i))
    if (i + 1) % 100 == 0:
        elapsed = time.time() - start_t
        eta     = elapsed / (i + 1) * (NUM_SIMULATIONS - i - 1)
        sys.stdout.write(f"  [{i+1:5d}/{NUM_SIMULATIONS}]  {elapsed:5.1f}s  ETA {eta:5.1f}s\n")
        sys.stdout.flush()

print(f"\n  Concluído em {time.time()-start_t:.1f}s\n")

# ── Final value statistics ────────────────────────────────────────────────────
totals   = sorted(r['total']   for r in results)
pnls     = sorted(r['pnl']     for r in results)
pnl_pcts = [r['pnl_pct'] for r in results]

mean_val   = statistics.mean(totals)
median_val = statistics.median(totals)
stdev_val  = statistics.stdev(totals)
min_val    = min(totals)
max_val    = max(totals)

above_initial  = sum(1 for t in totals if t > INITIAL_USD)
above_10500    = sum(1 for t in totals if t >= 10_500)
above_11000    = sum(1 for t in totals if t >= 11_000)
above_12000    = sum(1 for t in totals if t >= 12_000)
below_9500     = sum(1 for t in totals if t < 9_500)
below_9000     = sum(1 for t in totals if t < 9_000)

print(f"{'═'*66}")
print(f"  VALOR FINAL DO PORTFÓLIO (USD + crypto ao preço final)")
print(f"{'═'*66}")
print(f"  Capital inicial : ${INITIAL_USD:>10,.2f}")
print(f"  ─────────────────────────────────────────────────────")
print(f"  Média           : ${mean_val:>10,.2f}   ({mean_val-INITIAL_USD:+,.2f})")
print(f"  Mediana         : ${median_val:>10,.2f}   ({median_val-INITIAL_USD:+,.2f})")
print(f"  Desvio padrão   : ${stdev_val:>10,.2f}")
print(f"  ─────────────────────────────────────────────────────")
print(f"  Mínimo absoluto : ${min_val:>10,.2f}   ({min_val-INITIAL_USD:+,.2f})")
print(f"  Máximo absoluto : ${max_val:>10,.2f}   ({max_val-INITIAL_USD:+,.2f})")
print(f"  ─────────────────────────────────────────────────────")

for label, idx in [("P5  (pior  5%)", 50), ("P10", 100), ("P25", 250),
                   ("P50 (mediana)", 500), ("P75", 750), ("P90", 900), ("P95 (melhor 5%)", 950)]:
    v = totals[idx]
    print(f"  {label:<18}: ${v:>10,.2f}   ({v-INITIAL_USD:+,.2f})")

# ── Probability table ─────────────────────────────────────────────────────────
print(f"\n{'═'*66}")
print(f"  PROBABILIDADES DE RESULTADO")
print(f"{'═'*66}")
thresholds = [
    (f"> $12,000  (+20%)",  lambda t: t > 12_000),
    (f"> $11,500  (+15%)",  lambda t: t > 11_500),
    (f"> $11,000  (+10%)",  lambda t: t > 11_000),
    (f"> $10,500   (+5%)",  lambda t: t > 10_500),
    (f"> $10,000    (0%)",  lambda t: t > 10_000),
    (f"= $10,000    (0%)",  lambda t: abs(t - 10_000) < 10),
    (f"< $10,000    (0%)",  lambda t: t < 10_000),
    (f"< $9,500    (-5%)",  lambda t: t <  9_500),
    (f"< $9,000   (-10%)",  lambda t: t <  9_000),
]
for label, fn in thresholds:
    count = sum(1 for r in results if fn(r['total']))
    pct   = count / NUM_SIMULATIONS * 100
    bar   = '█' * int(pct / 2)
    print(f"  {label}  {bar:<50} {pct:5.1f}%  ({count:4d})")

# ── Dollar histogram ──────────────────────────────────────────────────────────
print(f"\n{'═'*66}")
print(f"  DISTRIBUIÇÃO DO VALOR FINAL ($)")
print(f"{'═'*66}")
BUCKETS = 22
bmin = min_val - 1
bmax = max_val + 1
bsize  = (bmax - bmin) / BUCKETS
counts = [0] * BUCKETS
for t in totals:
    counts[min(int((t - bmin) / bsize), BUCKETS - 1)] += 1
max_c = max(counts)
for i, c in enumerate(counts):
    lo   = bmin + i * bsize
    hi   = lo + bsize
    mid  = (lo + hi) / 2
    bar  = '█' * int(c / max_c * 42)
    marker = ' ◄ inicial' if lo <= INITIAL_USD <= hi else ''
    print(f"  ${mid:>8,.0f} |{bar:<42}| {c:4d}{marker}")

# ── Trade activity ────────────────────────────────────────────────────────────
trades_l = [r['n_trades'] for r in results]
sells_l  = [r['n_sells']  for r in results]

print(f"\n{'═'*66}")
print(f"  ACTIVIDADE DE TRADING")
print(f"{'═'*66}")
print(f"  Média de operações/simulação : {statistics.mean(trades_l):.1f}")
print(f"  Média de vendas/simulação    : {statistics.mean(sells_l):.1f}")
print(f"  Máximo operações numa sim    : {max(trades_l)}")
print(f"  Simulações sem operações     : {sum(1 for t in trades_l if t == 0)}")

# ── Best / worst detail ───────────────────────────────────────────────────────
print(f"\n{'═'*66}")
print(f"  TOP 5 MELHORES RESULTADOS")
print(f"{'═'*66}")
for r in sorted(results, key=lambda x: x['total'], reverse=True)[:5]:
    ts = ' '.join(f"{p.replace('ZUSD','').replace('USD','')[:4]}:{r['trends'][p][0].upper()}" for p in PAIRS)
    print(f"  Seed {r['seed']:4d}  Final=${r['total']:>10,.2f}  P&L={r['pnl_pct']:+.2f}%"
          f"  trades={r['n_trades']:2d}  [{ts}]")

print(f"\n{'═'*66}")
print(f"  TOP 5 PIORES RESULTADOS")
print(f"{'═'*66}")
for r in sorted(results, key=lambda x: x['total'])[:5]:
    ts = ' '.join(f"{p.replace('ZUSD','').replace('USD','')[:4]}:{r['trends'][p][0].upper()}" for p in PAIRS)
    print(f"  Seed {r['seed']:4d}  Final=${r['total']:>10,.2f}  P&L={r['pnl_pct']:+.2f}%"
          f"  trades={r['n_trades']:2d}  [{ts}]")

print(f"\n{'═'*66}\n")
