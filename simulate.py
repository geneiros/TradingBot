#!/usr/bin/env python3
"""
Simulation of TradingBot (bot.py) with synthetic market data.

Runs the real bot.py decision code (check_data → try_buy / try_sell →
_place_order) against an in-memory fake Kraken (sim_kraken.FakeKraken),
so no API keys or network access are needed and no real orders are sent.
All JSON state files are written to a temporary directory.

Usage:  python simulate.py [cycles] [seed]
"""

import io
import os
import random
import sys
import tempfile
import contextlib
from unittest.mock import MagicMock

# ── Mock krakenex before importing bot (not needed for the simulation) ───────
sys.modules.setdefault('krakenex', MagicMock())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bot                                   # noqa: E402
from sim_kraken import FakeKraken, generate_ohlc, BASE_PRICES  # noqa: E402

NUM_CYCLES = int(sys.argv[1]) if len(sys.argv) > 1 else 300
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
INITIAL_EUR = 1000.0
WINDOW = 100                                 # candles returned per OHLC call

SCENARIOS = {
    'XETHZEUR': 'up', 'XXBTZEUR': 'up', 'MANAEUR': 'flat',
    'GRTEUR': 'down', 'LSKEUR': 'up', 'SCEUR': 'flat',
}


def main():
    rng = random.Random(SEED)
    os.chdir(tempfile.mkdtemp(prefix='tradingbot_sim_'))
    bot._init_files()

    history = {p: generate_ohlc(BASE_PRICES[p], n=WINDOW + NUM_CYCLES,
                                trend=SCENARIOS[p], rng=rng)
               for p in bot.get_pairs()}
    fake = FakeKraken(history, eur=INITIAL_EUR, window=WINDOW)
    bot.k = fake

    print('═' * 62)
    print('  TRADINGBOT — SIMULATION (bot.py on synthetic data)')
    print('═' * 62)
    print(f'  Cycles    : {NUM_CYCLES}  (1 cycle = 1 new 1-minute candle)')
    print(f'  Scenarios : ' + ', '.join(f'{p}={t}' for p, t in SCENARIOS.items()))
    print(f'  Start     : €{INITIAL_EUR:,.2f}')
    print(f'  State dir : {os.getcwd()}')

    mva = bot.load_crypto_data_from_file()
    for _ in range(NUM_CYCLES):
        trades = bot.load_trades()
        for pair in bot.get_pairs():
            last = trades[pair][-1] if trades[pair] else None
            should_buy = last is None or last['sold']
            with contextlib.redirect_stdout(io.StringIO()):   # silence bot logs
                bot.check_data(pair, fake.candles(pair), should_buy, mva)
            mva = bot.load_crypto_data_from_file()
        fake.advance()
    fake.cursor -= 1  # last processed candle

    print('\n' + '─' * 62)
    print('  ORDERS')
    print('─' * 62)
    if not fake.orders:
        print('  (no orders placed)')
    for cursor, pair, side, vol, price in fake.orders:
        print(f'  t={cursor:>4}  {side.upper():<4} {pair:<9} qty={vol:<16.8f} @ €{price:,.6f}')

    trades = bot.load_trades()
    wins = losses = 0
    for pair, ts in trades.items():
        buy = None
        for t in ts:
            if t['bought']:
                buy = t
            elif t['sold'] and buy:
                if t['price_usd'] > buy['price_usd']:
                    wins += 1
                else:
                    losses += 1
                buy = None

    final = fake.total_value()
    print('\n' + '─' * 62)
    print('  SUMMARY')
    print('─' * 62)
    print(f'  Closed trades  : {wins + losses}  (wins={wins}, losses={losses})')
    open_pos = [p for p, ts in trades.items() if ts and ts[-1]['bought']]
    print(f'  Open positions : {", ".join(open_pos) or "none"}')
    print(f'  Cash (EUR)     : €{fake.balance["ZEUR"]:,.2f}')
    print(f'  Total value    : €{final:,.2f}  ({(final / INITIAL_EUR - 1) * 100:+.2f}%)')
    print('═' * 62)


if __name__ == '__main__':
    main()
