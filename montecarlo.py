#!/usr/bin/env python3
"""
Monte Carlo simulation of TradingBot — 1000 independent runs.
Each run uses a different random seed, trend mix, volatility and starting prices.
"""

import sys, json, time, random, os, tempfile, statistics
from unittest.mock import MagicMock
from collections import defaultdict

sys.modules['krakenex'] = MagicMock()
import bot
from unittest.mock import patch as _patch
import io as _io

# Suppress internal bot prints (TRADE: logs, buy/sell confirmations)
_devnull = open(os.devnull, 'w')

def _silent_save_trade(close, name, bought, sold, amount):
    """save_trade without print output."""
    trade = {
        'time_stamp': str(int(time.time())),
        'price_usd':  close,
        'bought':     bought,
        'sold':       sold,
        'amount':     amount,
    }
    trades = bot.load_trades()
    trades[name].append(trade)
    with open('trades.json', 'w') as f:
        json.dump(trades, f, indent=4)

def _silent_buy(crypto_data, name):
    bot.clear_crypto_data(name)
    price   = float(crypto_data[-1][4])
    funds   = bot.get_available_funds()
    amount  = funds * (1 / price)
    bot.update_balance(amount, name, price, False)
    _silent_save_trade(price, name, True, False, amount)

def _silent_sell(crypto_data, name):
    balance = bot.get_balance()
    bot.clear_crypto_data(name)
    price   = float(crypto_data[-1][4])
    amount  = float(balance[name[:-4]])
    bot.update_balance(amount, name, price, True)
    _silent_save_trade(price, name, False, True, amount)

bot.save_trade  = _silent_save_trade
bot.buy_crypto  = _silent_buy
bot.sell_crypto = _silent_sell

# ── Config ───────────────────────────────────────────────────────────────────
NUM_SIMULATIONS = 1000
NUM_CYCLES      = 20
INITIAL_USD     = 10_000.0
PAIRS           = bot.get_pairs()

BASE_PRICES = {
    'XETHZUSD': 2000.0,
    'XXBTZUSD': 40000.0,
    'MANAUSD':  0.50,
    'GRTUSD':   0.15,
    'LSKUSD':   1.50,
    'SCUSD':    0.003,
}

# ── Setup temp dir ────────────────────────────────────────────────────────────
tmpdir = tempfile.mkdtemp(prefix='tradingbot_mc_')
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
        candles.append([
            t + i * 60,
            f"{price:.8f}", f"{high:.8f}", f"{low:.8f}", f"{close:.8f}",
            f"{(high+low+close)/3:.8f}", f"{random.uniform(10,500):.2f}",
            random.randint(50, 500)
        ])
    return candles

def reset_state():
    with open('balance.json', 'w') as f:
        json.dump({'ZUSD': str(INITIAL_USD), 'EUR.HOLD': '0.0'}, f)
    with open('trades.json', 'w') as f:
        json.dump({p: [] for p in PAIRS}, f)
    bot.mva = bot.make_crypto_data({})
    with open('data.json', 'w') as f:
        json.dump(bot.mva, f)

def holdings_value(balance, history):
    """Estimate total value of held crypto at last known price."""
    val = 0.0
    for pair in PAIRS:
        sym = pair[:-4]          # e.g. 'XETHZUSD' → 'XETH', 'LSKUSD' → 'LS'
        qty = float(balance.get(sym, 0))
        if qty > 0:
            val += qty * float(history[pair][-1][4])
    return val

def run_one(seed):
    random.seed(seed)
    reset_state()

    # Random market scenario
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

    bal          = bot.get_balance()
    usd          = float(bal.get('ZUSD', 0))
    crypto_val   = holdings_value(bal, hist)
    total        = usd + crypto_val
    all_trades   = bot.load_trades()
    n_trades     = sum(len(v) for v in all_trades.values())

    # Count profitable sells
    n_profit = n_loss = 0
    for tlist in all_trades.values():
        for i, t in enumerate(tlist):
            if t['sold']:
                buy_price  = next((b['price_usd'] for b in reversed(tlist[:i]) if b['bought']), None)
                if buy_price is not None:
                    if t['price_usd'] >= buy_price * 1.02:
                        n_profit += 1
                    elif t['price_usd'] < buy_price * 0.97:
                        n_loss += 1

    return {
        'seed':        seed,
        'trends':      trends,
        'vol':         vol,
        'final_usd':   usd,
        'crypto_val':  crypto_val,
        'total':       total,
        'pnl':         total - INITIAL_USD,
        'pnl_pct':     (total - INITIAL_USD) / INITIAL_USD * 100,
        'n_trades':    n_trades,
        'n_profit':    n_profit,
        'n_loss':      n_loss,
    }

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  MONTE CARLO — {NUM_SIMULATIONS} simulations × {NUM_CYCLES} cycles × {len(PAIRS)} pairs")
print(f"  Initial capital : ${INITIAL_USD:,.0f}")
print(f"  Market scenarios: random trends (up/flat/down) + random vol")
print(f"{'═'*64}\n")

import contextlib

start_t = time.time()
results = []
for i in range(NUM_SIMULATIONS):
    with contextlib.redirect_stdout(open(os.devnull, 'w')):
        results.append(run_one(i))
    if (i + 1) % 100 == 0:
        elapsed = time.time() - start_t
        eta     = elapsed / (i + 1) * (NUM_SIMULATIONS - i - 1)
        sys.stdout.write(f"  [{i+1:5d}/{NUM_SIMULATIONS}]  elapsed {elapsed:5.1f}s  ETA {eta:5.1f}s\n")
        sys.stdout.flush()

elapsed_total = time.time() - start_t
print(f"\n  Completed in {elapsed_total:.1f}s\n")

# ── Statistics ────────────────────────────────────────────────────────────────
pnl_pcts   = [r['pnl_pct']  for r in results]
totals     = [r['total']    for r in results]
n_trades_l = [r['n_trades'] for r in results]

profitable  = sum(1 for p in pnl_pcts if p > 0)
at_loss     = sum(1 for p in pnl_pcts if p < 0)
unchanged   = sum(1 for p in pnl_pcts if p == 0)

mean_pnl    = statistics.mean(pnl_pcts)
median_pnl  = statistics.median(pnl_pcts)
stdev_pnl   = statistics.stdev(pnl_pcts)
min_pnl     = min(pnl_pcts)
max_pnl     = max(pnl_pcts)

sorted_pnl  = sorted(pnl_pcts)
p5          = sorted_pnl[int(0.05  * NUM_SIMULATIONS)]
p25         = sorted_pnl[int(0.25  * NUM_SIMULATIONS)]
p75         = sorted_pnl[int(0.75  * NUM_SIMULATIONS)]
p95         = sorted_pnl[int(0.95  * NUM_SIMULATIONS)]

print(f"{'═'*64}")
print(f"  RESULTS — P&L % distribution")
print(f"{'═'*64}")
print(f"  Profitable simulations : {profitable:5d} / {NUM_SIMULATIONS}  ({profitable/NUM_SIMULATIONS*100:.1f}%)")
print(f"  At a loss              : {at_loss:5d} / {NUM_SIMULATIONS}  ({at_loss/NUM_SIMULATIONS*100:.1f}%)")
print(f"  Unchanged (no trades)  : {unchanged:5d} / {NUM_SIMULATIONS}  ({unchanged/NUM_SIMULATIONS*100:.1f}%)")
print()
print(f"  Mean P&L    : {mean_pnl:+.2f}%   (${statistics.mean(totals)-INITIAL_USD:+,.0f})")
print(f"  Median P&L  : {median_pnl:+.2f}%   (${statistics.median(totals)-INITIAL_USD:+,.0f})")
print(f"  Std Dev     : {stdev_pnl:.2f}%")
print()
print(f"  Percentiles:")
print(f"    P5  (worst  5%) : {p5:+.2f}%   (${sorted_pnl[int(0.05*NUM_SIMULATIONS)]/100*INITIAL_USD+INITIAL_USD-INITIAL_USD:+,.0f})")
print(f"    P25             : {p25:+.2f}%   (${sorted_pnl[int(0.25*NUM_SIMULATIONS)]/100*INITIAL_USD+INITIAL_USD-INITIAL_USD:+,.0f})")
print(f"    P75             : {p75:+.2f}%   (${sorted_pnl[int(0.75*NUM_SIMULATIONS)]/100*INITIAL_USD+INITIAL_USD-INITIAL_USD:+,.0f})")
print(f"    P95 (best   5%) : {p95:+.2f}%   (${sorted_pnl[int(0.95*NUM_SIMULATIONS)]/100*INITIAL_USD+INITIAL_USD-INITIAL_USD:+,.0f})")
print(f"    Min (worst)     : {min_pnl:+.2f}%")
print(f"    Max (best)      : {max_pnl:+.2f}%")
print()
print(f"  Avg trades per simulation : {statistics.mean(n_trades_l):.1f}")
print(f"  Max trades in one sim     : {max(n_trades_l)}")

# ── ASCII histogram ───────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  P&L DISTRIBUTION (histogram, {NUM_SIMULATIONS} runs)")
print(f"{'═'*64}")

BUCKETS   = 20
bmin, bmax = min(pnl_pcts) - 0.01, max(pnl_pcts) + 0.01
bsize     = (bmax - bmin) / BUCKETS
counts    = [0] * BUCKETS
for p in pnl_pcts:
    idx = min(int((p - bmin) / bsize), BUCKETS - 1)
    counts[idx] += 1

max_count = max(counts)
bar_width = 40

for i, c in enumerate(counts):
    lo  = bmin + i * bsize
    hi  = lo + bsize
    bar = '█' * int(c / max_count * bar_width)
    mid = (lo + hi) / 2
    marker = ' ← 0%' if lo <= 0 <= hi else ''
    print(f"  {mid:+7.2f}% |{bar:<{bar_width}}| {c:4d}{marker}")

# ── Per-pair analysis ─────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  PER-PAIR TRADE ANALYSIS")
print(f"{'═'*64}")

pair_trades   = defaultdict(int)
pair_profits  = defaultdict(int)
pair_losses   = defaultdict(int)
trend_wins    = defaultdict(lambda: defaultdict(int))
trend_total   = defaultdict(lambda: defaultdict(int))

for r in results:
    reset_state()  # just to avoid stale reads — we re-read from last sim
    # we track via result metadata only
    for pair in PAIRS:
        t = r['trends'][pair]
        trend_total[pair][t] += 1

# Recount trades by pair across all simulations (re-run light analysis)
# We'll aggregate from the trade files — actually we don't have per-sim files saved.
# Instead use the per-result 'n_trades' as proxy and mark by trend.
print(f"  {'Pair':<14} {'Up wins':>8} {'Down wins':>10} {'Flat wins':>10}")
print(f"  {'-'*14} {'-'*8} {'-'*10} {'-'*10}")

def trend_outcome_rate(pair, trend):
    wins = sum(1 for r in results
               if r['trends'][pair] == trend and r['pnl_pct'] > 0)
    total = sum(1 for r in results if r['trends'][pair] == trend)
    return f"{wins}/{total} ({wins/total*100:.0f}%)" if total else "n/a"

for pair in PAIRS:
    up   = trend_outcome_rate(pair, 'up')
    down = trend_outcome_rate(pair, 'down')
    flat = trend_outcome_rate(pair, 'flat')
    print(f"  {pair:<14} {up:>14} {down:>14} {flat:>14}")

# ── Top 3 best / worst ────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  TOP 3 BEST SCENARIOS")
print(f"{'═'*64}")
for r in sorted(results, key=lambda x: x['pnl_pct'], reverse=True)[:3]:
    trends_str = '  '.join(f"{p.replace('ZUSD','').replace('USD','')[:4]}:{r['trends'][p][0].upper()}"
                            for p in PAIRS)
    print(f"  Seed {r['seed']:4d}  P&L={r['pnl_pct']:+.2f}%  trades={r['n_trades']:3d}  vol={r['vol']:.3f}  [{trends_str}]")

print(f"\n{'═'*64}")
print(f"  TOP 3 WORST SCENARIOS")
print(f"{'═'*64}")
for r in sorted(results, key=lambda x: x['pnl_pct'])[:3]:
    trends_str = '  '.join(f"{p.replace('ZUSD','').replace('USD','')[:4]}:{r['trends'][p][0].upper()}"
                            for p in PAIRS)
    print(f"  Seed {r['seed']:4d}  P&L={r['pnl_pct']:+.2f}%  trades={r['n_trades']:3d}  vol={r['vol']:.3f}  [{trends_str}]")

# ── Risk metrics ──────────────────────────────────────────────────────────────
sharpe  = mean_pnl / stdev_pnl if stdev_pnl else 0
win_rate = profitable / NUM_SIMULATIONS
avg_win  = statistics.mean(p for p in pnl_pcts if p > 0) if profitable else 0
avg_loss = statistics.mean(p for p in pnl_pcts if p < 0) if at_loss else 0
rr_ratio = abs(avg_win / avg_loss) if avg_loss else float('inf')

print(f"\n{'═'*64}")
print(f"  RISK METRICS")
print(f"{'═'*64}")
print(f"  Win rate          : {win_rate*100:.1f}%")
print(f"  Avg gain (wins)   : {avg_win:+.2f}%")
print(f"  Avg loss (losses) : {avg_loss:+.2f}%")
print(f"  Reward/Risk ratio : {rr_ratio:.2f}x")
print(f"  Sharpe (approx.)  : {sharpe:.3f}")
print(f"{'═'*64}\n")
