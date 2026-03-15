#!/usr/bin/env python3
"""
Parameter optimization via Monte Carlo sweep.
Tests all combinations of key strategy parameters and ranks by
expected value, win rate, worst case, and Sharpe ratio.
"""

import random
import time
import sys
from multiprocessing import Pool, cpu_count
from itertools import product

# ── Simulation constants ──────────────────────────────────────────────────────
NUM_SIMS    = 5_000        # per parameter combination
NUM_CYCLES  = 20
INITIAL_USD = 1_000.0
NUM_PAIRS   = 6
BATCH_SIZE  = 500

BASE_PRICES = (2000.0, 40000.0, 0.50, 0.15, 1.50, 0.003)
_UP   = (0.0003,  0.0015)
_DOWN = (-0.0015, -0.0003)
_FLAT = (-0.0005,  0.0005)
DRIFTS = (_UP, _DOWN, _FLAT)

# ── Parameter grid ────────────────────────────────────────────────────────────
GRID = {
    'take_profit':    [1.02, 1.03, 1.04, 1.05],
    'stop_loss':      [0.98, 0.99, 0.995],
    'trailing_stop':  [0.97, 0.98, 0.99],
    'ma_uptrend_min': [4, 5, 7],
    'ma_threshold':   [0.93, 0.95, 0.97],
    'breakeven_pct':  [0.0, 0.5, 1.0],   # % gain before moving stop to breakeven (0=disabled)
}

# ── Core simulation ───────────────────────────────────────────────────────────
def _simulate_one(seed, tp, sl, trail, ma_min, ma_thr, be_pct):
    rng = random.Random(seed)
    vol         = rng.uniform(0.002, 0.010)
    pair_trends = [rng.randint(0, 2) for _ in range(NUM_PAIRS)]
    pair_mults  = [rng.uniform(0.80, 1.20) for _ in range(NUM_PAIRS)]

    hist = []
    for i in range(NUM_PAIRS):
        lo, hi = DRIFTS[pair_trends[i]]
        price  = BASE_PRICES[i] * pair_mults[i]
        closes = []
        for _ in range(200):
            price *= 1.0 + rng.uniform(lo, hi) + rng.uniform(-vol, vol)
            if price < 1e-8: price = 1e-8
            closes.append(price * (1.0 + rng.uniform(-vol * 0.5, vol * 0.5)))
        hist.append(closes)

    usd      = INITIAL_USD
    holdings = [None] * NUM_PAIRS   # None or (qty, buy_price, be_triggered)
    ma_hist  = [[] for _ in range(NUM_PAIRS)]

    be_trigger = 1.0 + be_pct / 100.0  # price level that activates breakeven stop

    for cycle in range(NUM_CYCLES):
        for i in range(NUM_PAIRS):
            end    = 100 + cycle if 100 + cycle < 200 else 200
            window = hist[i][end - 100: end]
            ma     = sum(window) / len(window)
            ma_hist[i].append(ma)
            cur    = window[-1]

            if holdings[i] is None:
                mh = ma_hist[i]
                if len(mh) < 6:
                    continue
                recent = mh[-10:]
                count, prev = 0, 0.0
                for mv in recent:
                    if prev == 0.0: prev = mv; continue
                    if mv > prev:   count = 1 if count < 1 else count + 1
                    elif mv < prev: count = -1 if count > 0 else count - 1
                    prev = mv
                if count < ma_min:
                    continue
                last5     = mh[-5:]
                avg_ratio = sum(mv / cur for mv in last5) / 5
                if avg_ratio < ma_thr:
                    continue
                n_free = NUM_PAIRS - sum(1 for h in holdings if h is not None)
                if n_free > 0 and usd > 0.0:
                    funds       = usd / n_free
                    usd        -= funds
                    holdings[i] = (funds / cur, cur, False)
                    ma_hist[i]  = mh[-10:]
            else:
                qty, buy_price, be_active = holdings[i]
                mh   = ma_hist[i]
                peak = max(mh) if mh else buy_price

                # Activate breakeven stop once price reaches trigger level
                if be_pct > 0 and not be_active and cur >= buy_price * be_trigger:
                    be_active = True
                    holdings[i] = (qty, buy_price, be_active)

                # Effective stop-loss: breakeven (buy_price) or fixed SL
                effective_sl = buy_price if be_active else buy_price * sl

                sell = False
                if   cur >= buy_price * tp:    sell = True
                elif cur <  effective_sl:      sell = True
                elif cur <  peak * trail:      sell = True

                if sell:
                    usd        += qty * cur
                    holdings[i] = None
                    ma_hist[i]  = mh[-10:]

    total = usd
    for i, h in enumerate(holdings):
        if h is not None:
            total += h[0] * hist[i][-1]
    return total


def simulate_batch(args):
    start_seed, tp, sl, trail, ma_min, ma_thr, be_pct = args
    return [_simulate_one(s, tp, sl, trail, ma_min, ma_thr, be_pct)
            for s in range(start_seed, start_seed + BATCH_SIZE)]


def run_combo(params):
    tp, sl, trail, ma_min, ma_thr, be_pct = params
    n_batches = NUM_SIMS // BATCH_SIZE
    args_list = [(i * BATCH_SIZE, tp, sl, trail, ma_min, ma_thr, be_pct)
                 for i in range(n_batches)]
    with Pool(cpu_count()) as pool:
        results = []
        for batch in pool.map(simulate_batch, args_list):
            results.extend(batch)

    n      = len(results)
    mean   = sum(results) / n
    above  = sum(1 for v in results if v > INITIAL_USD) / n * 100
    worst  = min(results)
    sq_sum = sum(v * v for v in results)
    std    = ((sq_sum - n * mean ** 2) / (n - 1)) ** 0.5
    pnl    = (mean - INITIAL_USD) / INITIAL_USD * 100
    sharpe = pnl / (std / INITIAL_USD * 100) if std > 0 else 0
    losses = [v for v in results if v < INITIAL_USD]
    avg_loss = (sum(losses) / len(losses) - INITIAL_USD) if losses else 0
    gains  = [v for v in results if v > INITIAL_USD]
    avg_gain = (sum(gains) / len(gains) - INITIAL_USD) if gains else 0
    rr = abs(avg_gain / avg_loss) if avg_loss != 0 else float('inf')

    return {
        'params': params,
        'mean':   mean,
        'pnl':    pnl,
        'win':    above,
        'worst':  worst,
        'std':    std,
        'sharpe': sharpe,
        'rr':     rr,
        'avg_gain': avg_gain,
        'avg_loss': avg_loss,
    }


if __name__ == '__main__':
    combos = list(product(
        GRID['take_profit'], GRID['stop_loss'], GRID['trailing_stop'],
        GRID['ma_uptrend_min'], GRID['ma_threshold'], GRID['breakeven_pct'],
    ))

    # Benchmark
    t0  = time.time()
    _   = simulate_batch((0, 1.03, 0.99, 0.98, 5, 0.95, 0.0))
    bch = time.time() - t0
    sps = BATCH_SIZE / bch * cpu_count()
    total_sims = len(combos) * NUM_SIMS
    eta = total_sims / sps

    print(f"\n{'═'*70}")
    print(f"  OPTIMIZAÇÃO DE PARÂMETROS — SWEEP COMPLETO")
    print(f"  {len(combos)} combinações × {NUM_SIMS:,} simulações = {total_sims:,} simulações")
    print(f"  CPU cores : {cpu_count()}  |  Velocidade: {sps:,.0f} sims/s  |  ETA: {eta:.0f}s ({eta/60:.1f} min)")
    print(f"{'═'*70}")
    print(f"\n  {'#':>4}  {'TP':>5} {'SL':>6} {'Trail':>6} {'MA≥':>4} {'Thr':>5} {'BE':>4}"
          f"  {'Win%':>6}  {'P&L%':>6}  {'Worst':>8}  {'RR':>5}  {'Sharpe':>7}")
    print(f"  {'─'*4}  {'─'*5} {'─'*6} {'─'*6} {'─'*4} {'─'*5} {'─'*4}"
          f"  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*5}  {'─'*7}")

    start_t = time.time()
    all_results = []

    for idx, combo in enumerate(combos, 1):
        r = run_combo(combo)
        all_results.append(r)
        tp, sl, trail, ma_min, ma_thr, be_pct = combo
        elapsed = time.time() - start_t
        rate    = idx * NUM_SIMS / elapsed if elapsed > 0 else 1
        eta_rem = (len(combos) - idx) * NUM_SIMS / rate
        print(f"  {idx:>4}  {(tp-1)*100:>4.0f}%"
              f" {(1-sl)*100:>5.1f}%"
              f" {(1-trail)*100:>5.1f}%"
              f"  {ma_min:>3}"
              f"  {ma_thr:>4.2f}"
              f"  {be_pct:>3.0f}%"
              f"  {r['win']:>5.1f}%"
              f"  {r['pnl']:>+5.2f}%"
              f"  ${r['worst']:>7.2f}"
              f"  {r['rr']:>4.2f}x"
              f"  {r['sharpe']:>6.3f}"
              f"  ETA:{eta_rem:5.0f}s")
        sys.stdout.flush()

    total_time = time.time() - start_t
    print(f"\n  Concluído em {total_time:.0f}s\n")

    # ── Rankings ──────────────────────────────────────────────────────────────
    def show_ranking(title, key, top_n=10, reverse=True):
        print(f"\n{'═'*70}")
        print(f"  TOP {top_n} — {title}")
        print(f"{'═'*70}")
        print(f"  {'TP':>5} {'SL':>6} {'Trail':>6} {'MA≥':>4} {'Thr':>5} {'BE':>4}"
              f"  {'Win%':>6}  {'P&L%':>6}  {'Worst':>8}  {'RR':>5}  {'Sharpe':>7}")
        ranked = sorted(all_results, key=lambda x: x[key], reverse=reverse)[:top_n]
        for r in ranked:
            tp, sl, trail, ma_min, ma_thr, be_pct = r['params']
            print(f"  {(tp-1)*100:>4.0f}%"
                  f" {(1-sl)*100:>5.1f}%"
                  f" {(1-trail)*100:>5.1f}%"
                  f"  {ma_min:>3}"
                  f"  {ma_thr:>4.2f}"
                  f"  {be_pct:>3.0f}%"
                  f"  {r['win']:>5.1f}%"
                  f"  {r['pnl']:>+5.2f}%"
                  f"  ${r['worst']:>7.2f}"
                  f"  {r['rr']:>4.2f}x"
                  f"  {r['sharpe']:>6.3f}")

    show_ranking("MAIOR P&L MÉDIO",         'pnl',    reverse=True)
    show_ranking("MAIOR WIN RATE",           'win',    reverse=True)
    show_ranking("MENOR PERDA MÁXIMA",       'worst',  reverse=True)
    show_ranking("MELHOR REWARD/RISK",       'rr',     reverse=True)
    show_ranking("MELHOR SHARPE",            'sharpe', reverse=True)

    # ── Combined score: balance all metrics ───────────────────────────────────
    # Normalise each metric to [0,1] then weighted sum
    def norm(vals, reverse=False):
        lo, hi = min(vals), max(vals)
        if hi == lo: return [0.5] * len(vals)
        n = [(v - lo) / (hi - lo) for v in vals]
        return [1 - v for v in n] if reverse else n

    pnls   = [r['pnl']    for r in all_results]
    wins   = [r['win']    for r in all_results]
    worsts = [r['worst']  for r in all_results]
    rrs    = [r['rr']     for r in all_results]
    shps   = [r['sharpe'] for r in all_results]

    n_pnl   = norm(pnls)
    n_win   = norm(wins)
    n_worst = norm(worsts)          # higher worst = better (less loss)
    n_rr    = norm([min(r, 20) for r in rrs])
    n_shp   = norm(shps)

    for i, r in enumerate(all_results):
        r['score'] = (0.30 * n_pnl[i]   +
                      0.20 * n_win[i]    +
                      0.25 * n_worst[i]  +
                      0.15 * n_rr[i]     +
                      0.10 * n_shp[i])

    print(f"\n{'═'*70}")
    print(f"  TOP 10 — SCORE COMBINADO (P&L 30% + Win 20% + Worst 25% + RR 15% + Sharpe 10%)")
    print(f"{'═'*70}")
    print(f"  {'TP':>5} {'SL':>6} {'Trail':>6} {'MA≥':>4} {'Thr':>5} {'BE':>4}"
          f"  {'Win%':>6}  {'P&L%':>6}  {'Worst':>8}  {'RR':>5}  {'Sharpe':>7}  {'Score':>6}")
    for r in sorted(all_results, key=lambda x: x['score'], reverse=True)[:10]:
        tp, sl, trail, ma_min, ma_thr, be_pct = r['params']
        print(f"  {(tp-1)*100:>4.0f}%"
              f" {(1-sl)*100:>5.1f}%"
              f" {(1-trail)*100:>5.1f}%"
              f"  {ma_min:>3}"
              f"  {ma_thr:>4.2f}"
              f"  {be_pct:>3.0f}%"
              f"  {r['win']:>5.1f}%"
              f"  {r['pnl']:>+5.2f}%"
              f"  ${r['worst']:>7.2f}"
              f"  {r['rr']:>4.2f}x"
              f"  {r['sharpe']:>6.3f}"
              f"  {r['score']:>5.3f}")

    best = sorted(all_results, key=lambda x: x['score'], reverse=True)[0]
    tp, sl, trail, ma_min, ma_thr, be_pct = best['params']
    print(f"\n{'═'*70}")
    print(f"  PARÂMETROS ÓPTIMOS RECOMENDADOS")
    print(f"{'═'*70}")
    print(f"  take_profit    = {tp}   (+{(tp-1)*100:.0f}%)")
    print(f"  stop_loss      = {sl}   (-{(1-sl)*100:.1f}%)")
    print(f"  trailing_stop  = {trail}   (-{(1-trail)*100:.0f}%)")
    print(f"  ma_uptrend_min = {ma_min}")
    print(f"  ma_threshold   = {ma_thr}")
    print(f"  breakeven_pct  = {be_pct}%  ({'activo' if be_pct > 0 else 'desactivado'})")
    print(f"  ─────────────────────────────────────────")
    print(f"  Win rate       = {best['win']:.1f}%")
    print(f"  P&L médio      = {best['pnl']:+.2f}%  (+${best['avg_gain']:.2f} por sim)")
    print(f"  Pior caso      = ${best['worst']:.2f}  ({best['worst']-INITIAL_USD:+.2f})")
    print(f"  Reward/Risk    = {best['rr']:.2f}x")
    print(f"  Sharpe         = {best['sharpe']:.3f}")
    print(f"{'═'*70}\n")
