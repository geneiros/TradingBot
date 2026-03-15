#!/usr/bin/env python3
"""
Monte Carlo simulation — 3:1 Reward/Risk variant
Tests take-profit 3% + stop-loss 1% (guaranteed 3:1 ratio per trade by design).
Does NOT modify bot.py — patches check_opportunity at runtime only.
"""

import sys, json, time, random, os, tempfile, statistics, contextlib
from unittest.mock import MagicMock

sys.modules['krakenex'] = MagicMock()
import bot

# ── Patch check_opportunity with 3:1 thresholds ──────────────────────────────
TAKE_PROFIT   = 1.03   # sell when +3%  (was 1.02)
STOP_LOSS     = 0.99   # sell when -1%  (was 0.97)
TRAILING_STOP = 0.98   # keep trailing at -2% from peak MA

def check_opportunity_3to1(data, name, sell, buy):
    count = 0
    previous_value = 0
    trends = []
    for mva in data['close'][-10:]:
        if previous_value == 0:
            previous_value = mva
        else:
            if mva / previous_value > 1:
                count = 1 if count < 1 else count + 1
                trends.append('UPTREND')
            elif mva / previous_value < 1:
                count = -1 if count > 0 else count - 1
                trends.append('DOWNTREND')
            else:
                trends.append('NOTREND')
            previous_value = mva

    areas = []
    price = float(data['prices'][-1][4])
    for mva in reversed(data['close'][-5:]):
        if sell:
            purchase_price = float(bot.get_purchasing_price(name))
            peak_ma = max(data['close']) if data['close'] else purchase_price
            if price >= purchase_price * TAKE_PROFIT:     # +3%
                return True
            if price < purchase_price * STOP_LOSS:        # -1%
                return True
            if price < peak_ma * TRAILING_STOP:           # trailing -2%
                return True
        areas.append(mva / price)

    if buy:
        if count >= 5:
            counter = sum(areas)
            if len(areas) > 0 and counter / len(areas) >= 0.95:
                return True
        return False

# Patch silently
bot.check_opportunity = check_opportunity_3to1

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

tmpdir = tempfile.mkdtemp(prefix='tradingbot_mc31_')
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

def holdings_value(balance, history):
    val = 0.0
    for pair in PAIRS:
        qty = float(balance.get(pair[:-4], 0))
        if qty > 0:
            val += qty * float(history[pair][-1][4])
    return val

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

    bal        = bot.get_balance()
    usd        = float(bal.get('ZUSD', 0))
    crypto_val = holdings_value(bal, hist)
    total      = usd + crypto_val
    all_trades = bot.load_trades()
    n_trades   = sum(len(v) for v in all_trades.values())

    n_profit = n_loss = 0
    for tlist in all_trades.values():
        for i, t in enumerate(tlist):
            if t['sold']:
                buy_price = next((b['price_usd'] for b in reversed(tlist[:i]) if b['bought']), None)
                if buy_price:
                    if t['price_usd'] >= buy_price * TAKE_PROFIT:
                        n_profit += 1
                    elif t['price_usd'] < buy_price * STOP_LOSS:
                        n_loss += 1

    return {
        'seed': seed, 'trends': trends, 'vol': vol,
        'total': total, 'pnl_pct': (total - INITIAL_USD) / INITIAL_USD * 100,
        'n_trades': n_trades, 'n_profit': n_profit, 'n_loss': n_loss,
    }

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  MONTE CARLO 3:1 — {NUM_SIMULATIONS} simulations × {NUM_CYCLES} cycles × {len(PAIRS)} pairs")
print(f"  Strategy : Take-profit {(TAKE_PROFIT-1)*100:.0f}%  |  Stop-loss {(1-STOP_LOSS)*100:.0f}%  |  Trailing {(1-TRAILING_STOP)*100:.0f}%")
print(f"  Guaranteed RR per trade : {(TAKE_PROFIT-1)/(1-STOP_LOSS):.1f}:1")
print(f"  Initial capital : ${INITIAL_USD:,.0f}")
print(f"{'═'*64}\n")

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
pnl_pcts    = [r['pnl_pct']  for r in results]
n_trades_l  = [r['n_trades'] for r in results]
profitable  = sum(1 for p in pnl_pcts if p > 0)
at_loss     = sum(1 for p in pnl_pcts if p < 0)
unchanged   = sum(1 for p in pnl_pcts if p == 0)
sorted_pnl  = sorted(pnl_pcts)

mean_pnl  = statistics.mean(pnl_pcts)
median_pnl= statistics.median(pnl_pcts)
stdev_pnl = statistics.stdev(pnl_pcts)
avg_win   = statistics.mean(p for p in pnl_pcts if p > 0) if profitable else 0
avg_loss  = statistics.mean(p for p in pnl_pcts if p < 0) if at_loss   else 0
rr_ratio  = abs(avg_win / avg_loss) if avg_loss else float('inf')
sharpe    = mean_pnl / stdev_pnl if stdev_pnl else 0

print(f"{'═'*64}")
print(f"  RESULTS — P&L % distribution")
print(f"{'═'*64}")
print(f"  Profitable simulations :  {profitable:4d} / {NUM_SIMULATIONS}  ({profitable/NUM_SIMULATIONS*100:.1f}%)")
print(f"  At a loss              :  {at_loss:4d} / {NUM_SIMULATIONS}  ({at_loss/NUM_SIMULATIONS*100:.1f}%)")
print(f"  Unchanged (no trades)  :  {unchanged:4d} / {NUM_SIMULATIONS}  ({unchanged/NUM_SIMULATIONS*100:.1f}%)")
print()
print(f"  Mean P&L    : {mean_pnl:+.2f}%   (${mean_pnl/100*INITIAL_USD:+,.0f})")
print(f"  Median P&L  : {median_pnl:+.2f}%   (${median_pnl/100*INITIAL_USD:+,.0f})")
print(f"  Std Dev     :  {stdev_pnl:.2f}%")
print()
print(f"  Percentiles:")
for label, idx in [("P5  (worst  5%)", 50), ("P25", 250), ("P75", 750), ("P95 (best   5%)", 950)]:
    v = sorted_pnl[idx]
    print(f"    {label} : {v:+.2f}%   (${v/100*INITIAL_USD:+,.0f})")
print(f"    Min (worst)     : {min(pnl_pcts):+.2f}%")
print(f"    Max (best)      : {max(pnl_pcts):+.2f}%")
print()
print(f"  Avg trades per simulation : {statistics.mean(n_trades_l):.1f}")
print(f"  Max trades in one sim     : {max(n_trades_l)}")

# ── Histogram ─────────────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  P&L DISTRIBUTION (histogram)")
print(f"{'═'*64}")
BUCKETS = 20
bmin, bmax = min(pnl_pcts) - 0.01, max(pnl_pcts) + 0.01
bsize  = (bmax - bmin) / BUCKETS
counts = [0] * BUCKETS
for p in pnl_pcts:
    counts[min(int((p - bmin) / bsize), BUCKETS - 1)] += 1
max_c = max(counts)
for i, c in enumerate(counts):
    lo  = bmin + i * bsize
    bar = '█' * int(c / max_c * 40)
    mid = lo + bsize / 2
    marker = ' ← 0%' if lo <= 0 <= lo + bsize else ''
    print(f"  {mid:+7.2f}% |{bar:<40}| {c:4d}{marker}")

# ── Per-pair win rates ────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  PER-PAIR WIN RATE BY TREND")
print(f"{'═'*64}")
print(f"  {'Pair':<14} {'Uptrend':>14} {'Downtrend':>14} {'Flat':>14}")
print(f"  {'-'*14} {'-'*14} {'-'*14} {'-'*14}")
for pair in PAIRS:
    def rate(t):
        w = sum(1 for r in results if r['trends'][pair] == t and r['pnl_pct'] > 0)
        n = sum(1 for r in results if r['trends'][pair] == t)
        return f"{w}/{n} ({w/n*100:.0f}%)" if n else "n/a"
    print(f"  {pair:<14} {rate('up'):>14} {rate('down'):>14} {rate('flat'):>14}")

# ── Best / Worst ──────────────────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  TOP 3 BEST  /  TOP 3 WORST")
print(f"{'═'*64}")
for r in sorted(results, key=lambda x: x['pnl_pct'], reverse=True)[:3]:
    ts = '  '.join(f"{p.replace('ZUSD','').replace('USD','')[:4]}:{r['trends'][p][0].upper()}" for p in PAIRS)
    print(f"  [BEST ] Seed {r['seed']:4d}  P&L={r['pnl_pct']:+.2f}%  trades={r['n_trades']:3d}  [{ts}]")
for r in sorted(results, key=lambda x: x['pnl_pct'])[:3]:
    ts = '  '.join(f"{p.replace('ZUSD','').replace('USD','')[:4]}:{r['trends'][p][0].upper()}" for p in PAIRS)
    print(f"  [WORST] Seed {r['seed']:4d}  P&L={r['pnl_pct']:+.2f}%  trades={r['n_trades']:3d}  [{ts}]")

# ── Risk metrics + comparison ────────────────────────────────────────────────
print(f"\n{'═'*64}")
print(f"  RISK METRICS")
print(f"{'═'*64}")
print(f"  Win rate          : {profitable/NUM_SIMULATIONS*100:.1f}%")
print(f"  Avg gain (wins)   : {avg_win:+.2f}%")
print(f"  Avg loss (losses) : {avg_loss:+.2f}%")
print(f"  Reward/Risk ratio : {rr_ratio:.2f}x")
print(f"  Sharpe (approx.)  : {sharpe:.3f}")

print(f"\n{'═'*64}")
print(f"  COMPARISON vs CURRENT STRATEGY (2% TP / 3% SL)")
print(f"{'═'*64}")
print(f"  {'Metric':<28} {'Current (2/3%)':>15} {'3:1 (3/1%)':>15}")
print(f"  {'-'*28} {'-'*15} {'-'*15}")
current = {'win_rate': 86.6, 'mean': 1.95, 'p5': -0.41, 'worst': -2.88, 'rr': 3.85, 'sharpe': 1.138}
new     = {'win_rate': profitable/NUM_SIMULATIONS*100, 'mean': mean_pnl,
           'p5': sorted_pnl[50], 'worst': min(pnl_pcts), 'rr': rr_ratio, 'sharpe': sharpe}
rows = [
    ("Win rate (%)",        f"{current['win_rate']:.1f}%",  f"{new['win_rate']:.1f}%"),
    ("Mean P&L",            f"{current['mean']:+.2f}%",     f"{new['mean']:+.2f}%"),
    ("P5 (worst 5%)",       f"{current['p5']:+.2f}%",       f"{new['p5']:+.2f}%"),
    ("Worst case",          f"{current['worst']:+.2f}%",    f"{new['worst']:+.2f}%"),
    ("Reward/Risk ratio",   f"{current['rr']:.2f}x",        f"{new['rr']:.2f}x"),
    ("Sharpe",              f"{current['sharpe']:.3f}",     f"{new['sharpe']:.3f}"),
]
for label, cur, nw in rows:
    print(f"  {label:<28} {cur:>15} {nw:>15}")
print(f"{'═'*64}\n")
