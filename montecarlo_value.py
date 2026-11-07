#!/usr/bin/env python3
"""
Monte Carlo simulation — final portfolio value analysis.

Runs the real bot.py decision code against an in-memory fake Kraken
(sim_kraken.FakeKraken) on randomly generated markets and shows the
distribution of the final portfolio value (EUR + crypto marked to market).

Each simulation draws a random volatility, a random trend per pair
(up / down / flat) and a random starting price around each pair's base.

Usage:  python montecarlo_value.py [simulations] [cycles]
"""

import contextlib
import io
import os
import random
import statistics
import sys
import tempfile
import time
from multiprocessing import Pool, cpu_count
from unittest.mock import MagicMock

sys.modules.setdefault('krakenex', MagicMock())
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bot                                   # noqa: E402
from sim_kraken import FakeKraken, generate_ohlc, BASE_PRICES  # noqa: E402

NUM_SIMULATIONS = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
NUM_CYCLES = int(sys.argv[2]) if len(sys.argv) > 2 else 200
INITIAL_EUR = 1000.0
WINDOW = 100
FEE = 0.0026                                  # Kraken taker fee (0.26%)
PAIRS = bot.get_pairs()


def run_one(seed):
    rng = random.Random(seed)
    os.chdir(tempfile.mkdtemp(prefix=f'tb_mc_{seed}_'))
    bot._init_files()

    vol = rng.uniform(0.002, 0.010)
    trends = {p: rng.choice(['up', 'down', 'flat']) for p in PAIRS}
    history = {p: generate_ohlc(BASE_PRICES[p] * rng.uniform(0.8, 1.2),
                                n=WINDOW + NUM_CYCLES, trend=trends[p], vol=vol, rng=rng)
               for p in PAIRS}
    fake = FakeKraken(history, eur=INITIAL_EUR, window=WINDOW, fee=FEE)
    bot.k = fake

    # Keep data.json state in memory (writing it on every pair/cycle dominates runtime)
    state = {'data': bot.make_crypto_data({})}
    bot.load_crypto_data_from_file = lambda: state['data']
    bot.save_crypto_data = lambda data: state.update(data=data)

    mva = bot.load_crypto_data_from_file()
    with contextlib.redirect_stdout(io.StringIO()):
        for _ in range(NUM_CYCLES):
            trades = bot.load_trades()
            for pair in PAIRS:
                last = trades[pair][-1] if trades[pair] else None
                bot.check_data(pair, fake.candles(pair), last is None or last['sold'], mva)
                mva = bot.load_crypto_data_from_file()
            fake.advance()
    fake.cursor -= 1

    total = fake.total_value()
    return {
        'seed': seed,
        'total': total,
        'pnl_pct': (total / INITIAL_EUR - 1) * 100,
        'n_orders': len(fake.orders),
        'n_sells': sum(1 for o in fake.orders if o[2] == 'sell'),
        'trends': trends,
    }


def pct(sorted_vals, q):
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def main():
    W = 66
    print('═' * W)
    print('  MONTE CARLO — FINAL VALUE ANALYSIS (bot.py)')
    print(f'  {NUM_SIMULATIONS} simulations × {NUM_CYCLES} cycles × {len(PAIRS)} pairs')
    print(f'  Starting capital : €{INITIAL_EUR:,.2f}   |   fee {FEE * 100:.2f}% per order')
    print('═' * W)

    t0 = time.time()
    with Pool(cpu_count()) as pool:
        results = pool.map(run_one, range(NUM_SIMULATIONS), chunksize=8)
    print(f'  Concluído em {time.time() - t0:.1f}s\n')

    totals = sorted(r['total'] for r in results)
    mean_val = statistics.mean(totals)
    print('═' * W)
    print('  VALOR FINAL DO PORTFÓLIO (EUR + crypto ao preço final)')
    print('═' * W)
    print(f'  Média           : €{mean_val:>10,.2f}   ({mean_val - INITIAL_EUR:+,.2f})')
    print(f'  Mediana         : €{statistics.median(totals):>10,.2f}')
    if len(totals) > 1:
        print(f'  Desvio padrão   : €{statistics.stdev(totals):>10,.2f}')
    print(f'  Mínimo / Máximo : €{totals[0]:,.2f}  /  €{totals[-1]:,.2f}')
    for label, q in [('P5', .05), ('P25', .25), ('P50', .5), ('P75', .75), ('P95', .95)]:
        v = pct(totals, q)
        print(f'  {label:<15} : €{v:>10,.2f}   ({(v / INITIAL_EUR - 1) * 100:+.2f}%)')

    print('\n' + '═' * W)
    print('  PROBABILIDADES DE RESULTADO')
    print('═' * W)
    for thr in (10, 5, 2, 0, -2, -5, -10):
        if thr >= 0:
            n = sum(1 for r in results if r['pnl_pct'] > thr)
            label = f'> {thr:+d}%'
        else:
            n = sum(1 for r in results if r['pnl_pct'] < thr)
            label = f'< {thr:+d}%'
        p = n / len(results) * 100
        print(f'  {label:<7} {"█" * int(p / 2):<50} {p:5.1f}%')

    print('\n' + '═' * W)
    print('  ACTIVIDADE DE TRADING')
    print('═' * W)
    orders = [r['n_orders'] for r in results]
    print(f'  Média de ordens/simulação : {statistics.mean(orders):.1f}')
    print(f'  Média de vendas/simulação : {statistics.mean(r["n_sells"] for r in results):.1f}')
    print(f'  Simulações sem ordens     : {sum(1 for o in orders if o == 0)}')

    for title, rows in (('TOP 5 MELHORES', sorted(results, key=lambda r: -r['total'])[:5]),
                        ('TOP 5 PIORES', sorted(results, key=lambda r: r['total'])[:5])):
        print('\n' + '═' * W)
        print(f'  {title}')
        print('═' * W)
        for r in rows:
            ts = ' '.join(f'{p[:4]}:{r["trends"][p][0].upper()}' for p in PAIRS)
            print(f'  seed {r["seed"]:5d}  €{r["total"]:>9,.2f}  {r["pnl_pct"]:+6.2f}%'
                  f'  ordens={r["n_orders"]:3d}  [{ts}]')
    print()


if __name__ == '__main__':
    main()
