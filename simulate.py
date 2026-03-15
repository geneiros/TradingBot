#!/usr/bin/env python3
"""
Simulation of TradingBot with synthetic market data.
Runs without real Kraken API keys using generated OHLC candles.
"""

import sys, json, time, random, os, tempfile
from unittest.mock import MagicMock

# ── Mock krakenex before importing bot ───────────────────────────────────────
sys.modules['krakenex'] = MagicMock()
import bot

# ── Isolated temp directory for all JSON files ───────────────────────────────
tmpdir = tempfile.mkdtemp(prefix='tradingbot_sim_')
os.chdir(tmpdir)

PAIRS      = bot.get_pairs()
NUM_CYCLES = 20
random.seed(7)

# ── Helpers ──────────────────────────────────────────────────────────────────
def generate_ohlc(base_price, n=200, trend='up', vol=0.004):
    """Return n OHLC candles as Kraken-style lists."""
    candles, price = [], base_price
    t = int(time.time()) - n * 60
    drift_map = {'up': (0.0003, 0.0015), 'down': (-0.0015, -0.0003), 'flat': (-0.0005, 0.0005)}
    lo, hi = drift_map[trend]
    for i in range(n):
        price *= 1 + random.uniform(lo, hi) + random.uniform(-vol, vol)
        price  = max(price, 0.0001)
        high   = price * (1 + abs(random.uniform(0, vol)))
        low    = price * (1 - abs(random.uniform(0, vol)))
        close  = price * (1 + random.uniform(-vol / 2, vol / 2))
        vwap   = (high + low + close) / 3
        candles.append([
            t + i * 60,
            f"{price:.6f}", f"{high:.6f}", f"{low:.6f}", f"{close:.6f}",
            f"{vwap:.6f}", f"{random.uniform(10, 500):.2f}", random.randint(50, 500)
        ])
    return candles

def section(title):
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")

def divider(title):
    print(f"\n  ── {title} {'─'*(55 - len(title))}")

def fmt_balance(bal):
    usd  = float(bal.get('ZUSD', 0))
    held = {k: float(v) for k, v in bal.items() if k not in ('ZUSD', 'EUR.HOLD')}
    line = f"ZUSD=${usd:>12,.2f}"
    for k, v in held.items():
        line += f"  |  {k}={v:.6f}"
    return line

# ── Initial state ─────────────────────────────────────────────────────────────
initial_balance = {'ZUSD': '10000.00', 'EUR.HOLD': '0.0'}
with open('balance.json', 'w') as f:
    json.dump(initial_balance, f)
with open('trades.json', 'w') as f:
    json.dump({p: [] for p in PAIRS}, f)

bot.mva = bot.make_crypto_data({})
with open('data.json', 'w') as f:
    json.dump(bot.mva, f)

# ── Synthetic market ──────────────────────────────────────────────────────────
history = {
    'XETHZUSD': generate_ohlc(2000.0,  trend='up'),
    'XXBTZUSD': generate_ohlc(40000.0, trend='up'),
    'MANAUSD':  generate_ohlc(0.50,    trend='flat'),
    'GRTUSD':   generate_ohlc(0.15,    trend='down'),
    'LSKUSD':   generate_ohlc(1.50,    trend='up'),
    'SCUSD':    generate_ohlc(0.003,   trend='flat'),
}

# ── Run simulation ────────────────────────────────────────────────────────────
section("TRADINGBOT — SIMULATION")
print(f"  Pairs      : {', '.join(PAIRS)}")
print(f"  Cycles     : {NUM_CYCLES}  (each cycle ≈ 1 Kraken poll interval)")
print(f"  Scenarios  : ETH/BTC uptrend | MANA/SC flat | GRT downtrend | LSK uptrend")
print(f"  Start bal  : {fmt_balance(initial_balance)}")

trade_log = []

for cycle in range(1, NUM_CYCLES + 1):
    divider(f"Cycle {cycle:02d}")
    trades_before = sum(len(v) for v in bot.load_trades().values())

    for pair in PAIRS:
        trades     = bot.load_trades()
        # sliding window: each cycle shifts 1 candle forward
        window_end = min(100 + cycle, len(history[pair]))
        crypto_data = history[pair][window_end - 100 : window_end]
        cur_price   = float(crypto_data[-1][4])

        if len(trades[pair]) > 0:
            last = trades[pair][-1]
            if last['sold']:
                state      = "sold → BUY check"
                should_buy = True
            else:
                state      = f"holding @ ${float(last['price_usd']):.4f} → SELL check"
                should_buy = False
        else:
            state      = "no trades → BUY check"
            should_buy = True

        print(f"    {pair:<14} ${cur_price:<12.4f}  {state}")
        bot.check_data(pair, crypto_data, should_buy)

        # Reload mva from file after potential buy/sell (syncs global with trimmed file)
        bot.mva = bot.load_crypto_data_from_file()

    trades_after = sum(len(v) for v in bot.load_trades().values())
    new_trades   = trades_after - trades_before
    bal          = bot.get_balance()

    print(f"\n    Balance : {fmt_balance(bal)}")
    if new_trades:
        print(f"    *** {new_trades} trade(s) executed this cycle ***")
        for pair in PAIRS:
            ts = bot.load_trades()[pair]
            if ts:
                t = ts[-1]
                if t['time_stamp'] == str(int(time.time())) or True:
                    trade_log.append((cycle, pair, t))

# ── Summary ───────────────────────────────────────────────────────────────────
section("SIMULATION SUMMARY")
final_bal = bot.get_balance()
start_usd = float(initial_balance['ZUSD'])
end_usd   = float(final_bal['ZUSD'])
holdings  = {k: float(v) for k, v in final_bal.items() if k not in ('ZUSD', 'EUR.HOLD')}

print(f"  Starting USD : ${start_usd:>12,.2f}")
print(f"  Ending USD   : ${end_usd:>12,.2f}")
print(f"  USD change   : ${end_usd - start_usd:>+12,.2f}")

if holdings:
    print(f"\n  Current holdings:")
    for sym, qty in holdings.items():
        pair  = sym + 'ZUSD'
        price = float(history.get(pair, [[0]*5])[-1][4]) if pair in history else 0
        val   = qty * price
        print(f"    {sym:<10} qty={qty:.6f}  last_price=${price:.4f}  est_value=${val:.2f}")

all_trades = bot.load_trades()
total      = sum(len(v) for v in all_trades.values())
print(f"\n  Total trades executed: {total}")

if total:
    print(f"\n  Trade history:")
    for pair, ts in all_trades.items():
        if ts:
            for t in ts:
                action = "BUY " if t['bought'] else "SELL"
                print(f"    [{action}]  {pair:<14}  price=${float(t['price_usd']):<12.4f}  qty={float(t['amount']):.6f}")

print(f"\n  MA entries built per pair:")
for pair in PAIRS:
    n = len(bot.mva.get(pair, {}).get('close', []))
    print(f"    {pair:<14} {n} MA values")

print(f"\n{'═'*62}\n")
