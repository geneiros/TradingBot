#!/usr/bin/env python3
"""
Monte Carlo — 1,000,000 simulations, €100 initial capital, EUR pairs.
Tracks how many bot cycles it takes to first reach a profit.

One bot cycle = 20 seconds (bot.py sleeps 20s between loops).
"""

import random, time, sys
from multiprocessing import Pool, cpu_count

# ── Strategy constants (mirror bot.py) ────────────────────────────────────────
TAKE_PROFIT    = 1.05   # sell at +5%
STOP_LOSS      = 0.99   # sell at -1%
TRAILING_STOP  = 0.99   # sell if price < peak_MA * 0.99
MA_THRESHOLD   = 0.93
MA_UPTREND_MIN = 5

# ── Simulation config ──────────────────────────────────────────────────────────
NUM_SIMULATIONS = 1_000_000
NUM_CYCLES      = 500           # 500 × 20s ≈ 2.8 hours of bot run-time
INITIAL_EUR     = 100.0
NUM_PAIRS       = 6
CYCLE_SECONDS   = 20            # bot.py sleep interval
BATCH_SIZE      = 2_000

# EUR base prices (approximate)
BASE_PRICES = (1850.0, 37000.0, 0.46, 0.14, 1.40, 0.0028)
PAIR_NAMES  = ('XETHZEUR', 'XXBTZEUR', 'MANAEUR', 'GRTEUR', 'LSKEUR', 'SCEUR')

_UP   = ( 0.0003,  0.0015)
_DOWN = (-0.0015, -0.0003)
_FLAT = (-0.0005,  0.0005)
DRIFTS = (_UP, _DOWN, _FLAT)


# ── Core simulation ────────────────────────────────────────────────────────────
def _simulate_one(seed):
    """
    Returns (final_value, first_profit_cycle).
    first_profit_cycle = -1 if profit was never reached.
    """
    rng = random.Random(seed)

    vol         = rng.uniform(0.002, 0.010)
    pair_trends = [rng.randint(0, 2) for _ in range(NUM_PAIRS)]
    pair_mults  = [rng.uniform(0.80, 1.20) for _ in range(NUM_PAIRS)]

    # Pre-generate close prices: 200 warm-up + NUM_CYCLES live candles
    total_candles = 200 + NUM_CYCLES
    hist = []
    for i in range(NUM_PAIRS):
        lo, hi = DRIFTS[pair_trends[i]]
        price  = BASE_PRICES[i] * pair_mults[i]
        closes = []
        for _ in range(total_candles):
            price *= 1.0 + rng.uniform(lo, hi) + rng.uniform(-vol, vol)
            price  = max(price, 1e-8)
            closes.append(price * (1.0 + rng.uniform(-vol * 0.5, vol * 0.5)))
        hist.append(closes)

    eur      = INITIAL_EUR
    holdings = [None] * NUM_PAIRS   # None  or  (qty, buy_price)
    ma_hist  = [[]   for _ in range(NUM_PAIRS)]

    first_profit_cycle = -1

    for cycle in range(NUM_CYCLES):
        for i in range(NUM_PAIRS):
            # Slide a 100-candle window forward with each cycle
            end    = 200 + cycle
            window = hist[i][end - 100: end]
            ma     = sum(window) / 100
            ma_hist[i].append(ma)
            cur    = window[-1]

            if holdings[i] is None:
                # ── BUY check ────────────────────────────────────────────────
                mh = ma_hist[i]
                if len(mh) < 6:
                    continue
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
                n_free = NUM_PAIRS - sum(1 for h in holdings if h is not None)
                if n_free > 0 and eur > 0.0:
                    funds        = eur / n_free
                    eur         -= funds
                    holdings[i]  = (funds / cur, cur)
                    ma_hist[i]   = mh[-10:]
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
                    eur         += qty * cur
                    holdings[i]  = None
                    ma_hist[i]   = mh[-10:]

        # Mark-to-market total after each cycle
        total = eur
        for i, h in enumerate(holdings):
            if h is not None:
                total += h[0] * hist[i][199 + cycle]

        if first_profit_cycle == -1 and total > INITIAL_EUR:
            first_profit_cycle = cycle

    # Final value
    final = eur
    for i, h in enumerate(holdings):
        if h is not None:
            final += h[0] * hist[i][-1]

    return final, first_profit_cycle


def simulate_batch(start_seed):
    return [_simulate_one(s) for s in range(start_seed, start_seed + BATCH_SIZE)]


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    ncpus     = cpu_count()
    n_batches = NUM_SIMULATIONS // BATCH_SIZE
    seeds     = [i * BATCH_SIZE for i in range(n_batches)]

    # Benchmark one batch for ETA
    t0    = time.time()
    _     = simulate_batch(0)
    bench = time.time() - t0
    rate_est = BATCH_SIZE / bench
    eta_est  = NUM_SIMULATIONS / (rate_est * ncpus)

    print(f"\n{'═'*68}")
    print(f"  MONTE CARLO — 1,000,000 SIMULATIONS")
    print(f"  Initial capital : €{INITIAL_EUR:,.2f}")
    print(f"  Cycles / sim    : {NUM_CYCLES}  ({NUM_CYCLES * CYCLE_SECONDS / 3600:.1f}h bot run-time each)")
    print(f"  Pairs           : {', '.join(PAIR_NAMES)}")
    print(f"  Strategy        : TP +{(TAKE_PROFIT-1)*100:.0f}%  |  SL -{(1-STOP_LOSS)*100:.0f}%  |  Trailing -{(1-TRAILING_STOP)*100:.0f}%")
    print(f"  CPU cores       : {ncpus}")
    print(f"  Est. speed      : {rate_est * ncpus:>10,.0f} sims/s")
    print(f"  Est. ETA        : {eta_est:>6.1f}s  ({eta_est / 60:.1f} min)")
    print(f"{'═'*68}\n")

    start_t    = time.time()
    all_finals = []
    all_cycles = []  # first-profit cycle, -1 = never

    with Pool(ncpus) as pool:
        for done_b, batch in enumerate(
                pool.imap(simulate_batch, seeds, chunksize=5), start=1):
            for final, fpc in batch:
                all_finals.append(final)
                all_cycles.append(fpc)
            done    = done_b * BATCH_SIZE
            elapsed = time.time() - start_t
            rate    = done / elapsed if elapsed > 0 else 1
            eta     = (NUM_SIMULATIONS - done) / rate
            if done % 50_000 == 0 or done == NUM_SIMULATIONS:
                sys.stdout.write(
                    f"\r  [{done:>10,}/{NUM_SIMULATIONS:,}]"
                    f"  {elapsed:6.1f}s  ETA {eta:5.1f}s"
                    f"  ({rate:>10,.0f} sims/s)"
                )
                sys.stdout.flush()

    total_time = time.time() - start_t
    print(f"\n\n  Completed in {total_time:.1f}s  ({NUM_SIMULATIONS / total_time:,.0f} sims/s)\n")

    # ── Final portfolio value stats ────────────────────────────────────────────
    n = len(all_finals)
    all_finals.sort()
    mean_val = sum(all_finals) / n
    variance = sum((v - mean_val) ** 2 for v in all_finals) / (n - 1)
    std_val  = variance ** 0.5

    above = sum(1 for v in all_finals if v > INITIAL_EUR)
    below = sum(1 for v in all_finals if v < INITIAL_EUR)
    equal = n - above - below

    def pct(p):
        return all_finals[int(p / 100 * n)]

    print(f"{'═'*68}")
    print(f"  FINAL PORTFOLIO VALUE  (€{INITIAL_EUR:.0f} initial)")
    print(f"{'═'*68}")
    print(f"  Mean              : €{mean_val:>8,.2f}   ({mean_val - INITIAL_EUR:+.2f})")
    print(f"  Median            : €{pct(50):>8,.2f}   ({pct(50) - INITIAL_EUR:+.2f})")
    print(f"  Std dev           : €{std_val:>8,.2f}")
    print(f"  Min               : €{all_finals[0]:>8,.2f}   ({all_finals[0] - INITIAL_EUR:+.2f})")
    print(f"  Max               : €{all_finals[-1]:>8,.2f}   ({all_finals[-1] - INITIAL_EUR:+.2f})")
    print()
    for label, p in [("P5  (worst  5%)", 5), ("P25", 25), ("P50", 50),
                     ("P75", 75), ("P95 (best   5%)", 95)]:
        v = pct(p)
        print(f"  {label:<18}: €{v:>8,.2f}   ({v - INITIAL_EUR:+.2f})")
    print()
    print(f"  Profitable sims   : {above:>7,}  ({above / n * 100:.2f}%)")
    print(f"  Break-even        : {equal:>7,}  ({equal / n * 100:.2f}%)")
    print(f"  At a loss         : {below:>7,}  ({below / n * 100:.2f}%)")

    # ── Time-to-first-profit analysis ─────────────────────────────────────────
    reached      = [c for c in all_cycles if c >= 0]
    never        = sum(1 for c in all_cycles if c < 0)
    n_reached    = len(reached)

    print(f"\n{'═'*68}")
    print(f"  TIME TO FIRST PROFIT  (cycle = {CYCLE_SECONDS}s)")
    print(f"{'═'*68}")
    print(f"  Sims that reached profit  : {n_reached:>7,}  ({n_reached / n * 100:.2f}%)")
    print(f"  Sims that never profited  : {never:>7,}  ({never / n * 100:.2f}%)")

    if reached:
        reached.sort()
        nr = len(reached)

        def cpct(p):
            return reached[int(p / 100 * nr)]

        mean_c   = sum(reached) / nr
        mean_t   = mean_c * CYCLE_SECONDS
        median_c = cpct(50)
        median_t = median_c * CYCLE_SECONDS

        def fmt(seconds):
            if seconds < 60:    return f"{seconds:.0f}s"
            if seconds < 3600:  return f"{seconds/60:.1f}min"
            return f"{seconds/3600:.2f}h"

        print()
        print(f"  Among simulations that DID reach profit:")
        print(f"  {'Metric':<25}  {'Cycles':>7}  {'Real time':>10}")
        print(f"  {'-'*25}  {'-'*7}  {'-'*10}")
        print(f"  {'Mean':<25}  {mean_c:>7.1f}  {fmt(mean_t):>10}")
        print(f"  {'Median (P50)':<25}  {median_c:>7}  {fmt(median_t):>10}")

        for label, p in [("Fastest  1% (P1)",  1), ("Fastest  5% (P5)",  5),
                          ("Fastest 10% (P10)", 10), ("Fastest 25% (P25)", 25),
                          ("Slowest 25% (P75)", 75), ("Slowest 10% (P90)", 90),
                          ("Slowest  5% (P95)", 95), ("Slowest  1% (P99)", 99)]:
            c = cpct(p)
            print(f"  {label:<25}  {c:>7}  {fmt(c * CYCLE_SECONDS):>10}")

        # Cumulative distribution: % that reached profit by cycle X
        print(f"\n{'═'*68}")
        print(f"  CUMULATIVE % REACHING PROFIT BY CYCLE")
        print(f"{'═'*68}")
        milestones = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500]
        for m in milestones:
            c_by_m = sum(1 for c in reached if c <= m)
            pct_of_all     = c_by_m / n * 100
            pct_of_reached = c_by_m / nr * 100
            bar = '█' * int(pct_of_reached / 2)
            print(f"  By cycle {m:>3} ({fmt(m * CYCLE_SECONDS):>6})"
                  f"  {bar:<50} {pct_of_reached:5.1f}% of profitable"
                  f"  ({pct_of_all:5.2f}% of all)")

        # ASCII histogram of first-profit cycle
        print(f"\n{'═'*68}")
        print(f"  DISTRIBUTION OF FIRST-PROFIT CYCLE")
        print(f"{'═'*68}")
        BUCKETS = 20
        bmin  = 0
        bmax  = NUM_CYCLES
        bsize = bmax / BUCKETS
        counts = [0] * BUCKETS
        for c in reached:
            counts[min(int(c / bsize), BUCKETS - 1)] += 1
        max_c = max(counts) or 1
        for i, c in enumerate(counts):
            lo  = int(i * bsize)
            hi  = int((i + 1) * bsize)
            bar = '█' * int(c / max_c * 44)
            mid = (lo + hi) / 2
            print(f"  cyc {lo:>3}-{hi:<3} ({fmt(lo * CYCLE_SECONDS):>6})"
                  f"  |{bar:<44}| {c:>7,}")

    # ── Probability table ──────────────────────────────────────────────────────
    print(f"\n{'═'*68}")
    print(f"  PROFIT PROBABILITY TABLE")
    print(f"{'═'*68}")
    thresholds = [
        ("> €110  (+10%)", lambda v: v > 110),
        ("> €105   (+5%)", lambda v: v > 105),
        ("> €102   (+2%)", lambda v: v > 102),
        ("> €101   (+1%)", lambda v: v > 101),
        ("> €100    (0%)", lambda v: v > 100),
        ("< €100    (0%)", lambda v: v < 100),
        ("< €99    (-1%)", lambda v: v <  99),
        ("< €95    (-5%)", lambda v: v <  95),
        ("< €90   (-10%)", lambda v: v <  90),
    ]
    for label, fn in thresholds:
        cnt = sum(1 for v in all_finals if fn(v))
        p   = cnt / n * 100
        bar = '█' * int(p / 2)
        print(f"  {label}  {bar:<50} {p:5.2f}%  ({cnt:>7,})")

    print(f"\n{'═'*68}")
    print(f"  SUMMARY")
    print(f"{'═'*68}")
    print(f"  Expected gain per sim        : €{mean_val - INITIAL_EUR:+.4f}")
    if n_reached > 0:
        mean_c = sum(reached) / len(reached)
        print(f"  Median time to first profit  : {fmt(cpct(50) * CYCLE_SECONDS)}")
        print(f"  Mean   time to first profit  : {fmt(mean_c * CYCLE_SECONDS)}")
    print(f"{'═'*68}\n")
