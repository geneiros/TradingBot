#!/usr/bin/env python3
"""
Monte Carlo simulation — 1,000,000 runs, $1,000 initial capital.

Fully in-memory (zero file I/O), multiprocessing across all CPU cores.
Replicates bot.py strategy exactly: TP +3% | SL -1% | Trailing -2%
"""

import random
import time
import sys
from multiprocessing import Pool, cpu_count

# ── Strategy (mirrors bot.py) ─────────────────────────────────────────────────
TAKE_PROFIT    = 1.03
STOP_LOSS      = 0.99
TRAILING_STOP  = 0.98
MA_THRESHOLD   = 0.95
MA_UPTREND_MIN = 5

# ── Simulation config ─────────────────────────────────────────────────────────
NUM_SIMULATIONS = 1_000_000
NUM_CYCLES      = 20
INITIAL_USD     = 1_000.0
NUM_PAIRS       = 6
BATCH_SIZE      = 2_000          # simulations per worker batch

BASE_PRICES = (2000.0, 40000.0, 0.50, 0.15, 1.50, 0.003)
PAIR_NAMES  = ('XETHZUSD', 'XXBTZUSD', 'MANAUSD', 'GRTUSD', 'LSKUSD', 'SCUSD')

_UP   = (0.0003,  0.0015)
_DOWN = (-0.0015, -0.0003)
_FLAT = (-0.0005,  0.0005)
DRIFTS = (_UP, _DOWN, _FLAT)

# ── Core simulation (no imports from bot.py, no file I/O) ────────────────────
def _simulate_one(seed):
    rng = random.Random(seed)

    vol         = rng.uniform(0.002, 0.010)
    pair_trends = [rng.randint(0, 2) for _ in range(NUM_PAIRS)]
    pair_mults  = [rng.uniform(0.80, 1.20) for _ in range(NUM_PAIRS)]

    # Generate close-price history for each pair (200 candles)
    hist = []
    for i in range(NUM_PAIRS):
        lo, hi = DRIFTS[pair_trends[i]]
        price  = BASE_PRICES[i] * pair_mults[i]
        closes = []
        for _ in range(200):
            price *= 1.0 + rng.uniform(lo, hi) + rng.uniform(-vol, vol)
            if price < 1e-8:
                price = 1e-8
            closes.append(price * (1.0 + rng.uniform(-vol * 0.5, vol * 0.5)))
        hist.append(closes)

    # Portfolio state
    usd      = INITIAL_USD
    holdings = [None] * NUM_PAIRS   # None  or  (qty, buy_price)
    ma_hist  = [[]   for _ in range(NUM_PAIRS)]

    for cycle in range(NUM_CYCLES):
        for i in range(NUM_PAIRS):
            end    = 100 + cycle if 100 + cycle < 200 else 200
            window = hist[i][end - 100: end]
            ma     = sum(window) / len(window)
            ma_hist[i].append(ma)
            cur    = window[-1]

            if holdings[i] is None:
                # ── BUY check ────────────────────────────────────────────────
                mh = ma_hist[i]
                if len(mh) < 6:
                    continue

                # Count consecutive uptrends in last ≤10 MA values
                recent = mh[-10:]
                count, prev = 0, 0.0
                for mv in recent:
                    if prev == 0.0:
                        prev = mv; continue
                    if mv > prev:
                        count = 1 if count < 1 else count + 1
                    elif mv < prev:
                        count = -1 if count > 0 else count - 1
                    prev = mv

                if count < MA_UPTREND_MIN:
                    continue

                last5     = mh[-5:]
                avg_ratio = sum(mv / cur for mv in last5) / 5
                if avg_ratio < MA_THRESHOLD:
                    continue

                # Execute buy
                n_free = NUM_PAIRS - sum(1 for h in holdings if h is not None)
                if n_free > 0 and usd > 0.0:
                    funds       = usd / n_free
                    usd        -= funds
                    holdings[i] = (funds / cur, cur)
                    ma_hist[i]  = mh[-10:]          # trim (clear_crypto_data)

            else:
                # ── SELL check ───────────────────────────────────────────────
                qty, buy_price = holdings[i]
                mh   = ma_hist[i]
                peak = max(mh) if mh else buy_price

                if   cur >= buy_price * TAKE_PROFIT:   sell = True
                elif cur <  buy_price * STOP_LOSS:     sell = True
                elif cur <  peak      * TRAILING_STOP: sell = True
                else:                                   sell = False

                if sell:
                    usd        += qty * cur
                    holdings[i] = None
                    ma_hist[i]  = mh[-10:]

    # Final portfolio value (USD + mark-to-market crypto)
    total = usd
    for i, h in enumerate(holdings):
        if h is not None:
            total += h[0] * hist[i][-1]
    return total


def simulate_batch(start_seed):
    return [_simulate_one(s) for s in range(start_seed, start_seed + BATCH_SIZE)]


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ncpus    = cpu_count()
    n_batches = NUM_SIMULATIONS // BATCH_SIZE
    seeds    = [i * BATCH_SIZE for i in range(n_batches)]

    # Quick benchmark to show ETA
    t0    = time.time()
    _     = simulate_batch(0)
    bench = time.time() - t0
    rate_est  = BATCH_SIZE / bench
    eta_est   = NUM_SIMULATIONS / (rate_est * ncpus)

    print(f"\n{'═'*66}")
    print(f"  MONTE CARLO — 1,000,000 SIMULAÇÕES")
    print(f"  Capital inicial : ${INITIAL_USD:,.2f}")
    print(f"  Ciclos/sim      : {NUM_CYCLES}  |  Pares : {NUM_PAIRS}")
    print(f"  Estratégia      : TP +{(TAKE_PROFIT-1)*100:.0f}%  |"
          f"  SL -{(1-STOP_LOSS)*100:.0f}%  |  Trailing -{(1-TRAILING_STOP)*100:.0f}%")
    print(f"  CPU cores       : {ncpus}")
    print(f"  Velocidade est. : {rate_est*ncpus:>10,.0f} sims/s")
    print(f"  ETA estimado    : {eta_est:>6.1f}s ({eta_est/60:.1f} min)")
    print(f"{'═'*66}\n")

    start_t    = time.time()
    all_values = []

    with Pool(ncpus) as pool:
        for done_batches, batch_res in enumerate(
                pool.imap(simulate_batch, seeds, chunksize=5), start=1):
            all_values.extend(batch_res)
            done    = done_batches * BATCH_SIZE
            elapsed = time.time() - start_t
            rate    = done / elapsed if elapsed > 0 else 1
            eta     = (NUM_SIMULATIONS - done) / rate
            if done % 50_000 == 0 or done == NUM_SIMULATIONS:
                sys.stdout.write(
                    f"\r  [{done:>10,}/{NUM_SIMULATIONS:,}]"
                    f"  {elapsed:6.1f}s  ETA {eta:6.1f}s"
                    f"  ({rate:>10,.0f} sims/s)"
                )
                sys.stdout.flush()

    total_time = time.time() - start_t
    print(f"\n\n  Concluído em {total_time:.1f}s"
          f"  ({NUM_SIMULATIONS/total_time:,.0f} sims/s)\n")

    # ── Statistics ────────────────────────────────────────────────────────────
    all_values.sort()
    n        = len(all_values)
    mean_val = sum(all_values) / n
    sum_sq   = sum(v * v for v in all_values)
    std_val  = ((sum_sq - n * mean_val ** 2) / (n - 1)) ** 0.5
    min_val  = all_values[0]
    max_val  = all_values[-1]

    above = sum(1 for v in all_values if v > INITIAL_USD)
    below = sum(1 for v in all_values if v < INITIAL_USD)
    equal = n - above - below

    def pct(i):
        return all_values[int(i / 100 * n)]

    print(f"{'═'*66}")
    print(f"  VALOR FINAL DO PORTFÓLIO (1,000,000 simulações)")
    print(f"{'═'*66}")
    print(f"  Capital inicial  : ${INITIAL_USD:>10,.2f}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Média            : ${mean_val:>10,.2f}   ({mean_val-INITIAL_USD:+,.2f})")
    print(f"  Mediana          : ${pct(50):>10,.2f}   ({pct(50)-INITIAL_USD:+,.2f})")
    print(f"  Desvio padrão    : ${std_val:>10,.2f}")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Mínimo absoluto  : ${min_val:>10,.2f}   ({min_val-INITIAL_USD:+,.2f})")
    print(f"  Máximo absoluto  : ${max_val:>10,.2f}   ({max_val-INITIAL_USD:+,.2f})")
    print(f"  ─────────────────────────────────────────────────────")
    for label, p in [("P1  (pior   1%)", 1), ("P5  (pior   5%)", 5),
                     ("P10", 10), ("P25", 25), ("P50 (mediana)", 50),
                     ("P75", 75), ("P90", 90), ("P95 (melhor 5%)", 95),
                     ("P99 (melhor 1%)", 99)]:
        v = pct(p)
        print(f"  {label:<18}: ${v:>10,.2f}   ({v-INITIAL_USD:+,.2f})")

    # ── Probability table ─────────────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print(f"  PROBABILIDADES DE RESULTADO")
    print(f"{'═'*66}")
    thresholds = [
        (f"> $1,200  (+20%)", lambda v: v > 1_200),
        (f"> $1,150  (+15%)", lambda v: v > 1_150),
        (f"> $1,100  (+10%)", lambda v: v > 1_100),
        (f"> $1,050   (+5%)", lambda v: v > 1_050),
        (f"> $1,010   (+1%)", lambda v: v > 1_010),
        (f"> $1,000    (0%)", lambda v: v > 1_000),
        (f"< $1,000    (0%)", lambda v: v < 1_000),
        (f"< $990     (-1%)", lambda v: v <   990),
        (f"< $950     (-5%)", lambda v: v <   950),
        (f"< $900    (-10%)", lambda v: v <   900),
    ]
    for label, fn in thresholds:
        c   = sum(1 for v in all_values if fn(v))
        p   = c / n * 100
        bar = '█' * int(p / 2)
        print(f"  {label}  {bar:<50} {p:5.2f}%  ({c:>7,})")

    # ── Dollar histogram ──────────────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print(f"  DISTRIBUIÇÃO DO VALOR FINAL ($)")
    print(f"{'═'*66}")
    BUCKETS = 24
    bmin   = min_val - 0.01
    bmax   = max_val + 0.01
    bsize  = (bmax - bmin) / BUCKETS
    counts = [0] * BUCKETS
    for v in all_values:
        counts[min(int((v - bmin) / bsize), BUCKETS - 1)] += 1
    max_c = max(counts)
    for i, c in enumerate(counts):
        lo     = bmin + i * bsize
        mid    = lo + bsize / 2
        bar    = '█' * int(c / max_c * 44)
        marker = ' ◄ $1,000' if lo <= INITIAL_USD <= lo + bsize else ''
        print(f"  ${mid:>7,.1f} |{bar:<44}| {c:>7,}{marker}")

    # ── Gain/loss summary ─────────────────────────────────────────────────────
    print(f"\n{'═'*66}")
    print(f"  RESUMO")
    print(f"{'═'*66}")
    print(f"  Terminam com LUCRO    : {above:>7,}  ({above/n*100:.2f}%)")
    print(f"  Terminam em BREAKEVEN : {equal:>7,}  ({equal/n*100:.2f}%)")
    print(f"  Terminam com PREJUÍZO : {below:>7,}  ({below/n*100:.2f}%)")
    print(f"\n  Ganho esperado por simulação : ${mean_val - INITIAL_USD:+.4f}")
    print(f"  Ganho esperado em 100 sims   : ${(mean_val - INITIAL_USD) * 100:+,.2f}")
    print(f"  Ganho esperado em 1000 sims  : ${(mean_val - INITIAL_USD) * 1000:+,.2f}")
    print(f"{'═'*66}\n")
