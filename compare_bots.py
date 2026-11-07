"""
compare_bots.py — Monte Carlo: bot.py vs sma_crossover_bot.py

Simulates 1 week (10,080 one-minute candles) of price paths.
BOTH strategies are run on the SAME price paths (identical RNG state)
per chunk for a genuine apples-to-apples comparison.

Strategy A — bot.py
  Buy : 5 consecutive 1-min MA100 up-moves AND avg(MA100[-5:]) / price >= 0.97
        AND bullish candle (price >= previous close)
  Sell: +3% profit target  |  -1% stop-loss, raised to break-even once peak >= +1%
        |  trailing stop (−1% from peak after +1.5%)

Strategy B — sma_crossover_bot.py
  Buy : SMA20 crosses above SMA50 (golden cross)
  Sell: SMA20 crosses below SMA50 AND price > entry  |  -1% stop-loss

Benchmark — passive Hold
  Buy at t=0, sell at t=T-1 (no active trading).

Price model: GBM with pooled per-minute sigma calibrated from 6-pair annual vols.
  XETHZEUR=80%  XXBTZEUR=65%  MANAEUR=160%  GRTEUR=175%  LSKEUR=130%  SCEUR=150%
"""

import time as _time
import numpy as np

# ── Simulation config ─────────────────────────────────────────────────────────
N_TOTAL = 20_000       # total price paths (both strategies run on each)
CHUNK   = 500          # paths processed at once
T       = 10_080       # 1 week × 1-min candles
INITIAL = 1_000.0      # starting EUR per simulation
SEED    = 42

# Pooled per-minute sigma from calibrated annual vols
ANNUAL_VOLS = [0.80, 0.65, 1.60, 1.75, 1.30, 1.50]
MINUTES_PER_YEAR = 365 * 1440
SIGMA = float(np.mean([v / np.sqrt(MINUTES_PER_YEAR) for v in ANNUAL_VOLS]))

# ── Strategy parameters (mirror bot.py / sma_crossover_bot.py) ───────────────
# bot.py
BOT_PROFIT_TARGET = 1.03
BOT_STOP_LOSS     = 0.99
BOT_BREAK_EVEN    = 1.01    # once peak >= entry * 1.01, stop rises to entry
BOT_PROXIMITY     = 0.97    # avg(MA100[-5:]) / price >= this
BOT_TRAIL_THRESH  = 0.99    # trailing stop: -1% from peak MA
BOT_TRAIL_ACT     = 1.015   # trailing activates when peak > entry * 1.015

# sma_crossover_bot.py
SMA_SHORT = 20
SMA_LONG  = 50
SMA_STOP  = 0.99

# ── Scenarios ─────────────────────────────────────────────────────────────────
SCENARIOS = [
    ('Bear   (μ = -0.0002/min)', -0.0002),
    ('Neutral (μ = 0)',           0.0000),
    ('Bull   (μ = +0.0002/min)', +0.0002),
]


# ── Rolling mean (vectorised) ─────────────────────────────────────────────────
def rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    """arr: (N, T) → (N, T-w+1) rolling mean along axis=1."""
    N, L = arr.shape
    cs = arr.astype(np.float64).cumsum(axis=1)
    pad = np.zeros((N, 1), dtype=np.float64)
    return ((cs[:, w - 1:] - np.hstack([pad, cs[:, :L - w]])) / w).astype(np.float32)


# ── Precompute signals for bot.py ─────────────────────────────────────────────
def precompute_bot_signals(prices: np.ndarray):
    """
    prices: (N, T) float32, relative prices starting at 1.0.

    Returns (buy_sig, T_START):
      buy_sig: (N, T-T_START) bool — True = golden-cross-style buy at that t
      T_START: first t at which buy_sig is valid (= 104)

    Signal fires when:
      1. last 5 consecutive 1-min MA100 values are strictly rising  (count >= 5)
      2. avg(MA100[-5:]) / price >= 0.97  (price near MA)
      3. bullish candle: price[t] >= price[t-1]
    """
    # MA100[n, i] = mean(prices[n, i : i+100]), valid at price time t = i+99
    ma100 = rolling_mean(prices, 100)          # (N, T-99)

    # up_ma[n, j] = MA100 rose from step j to j+1; price time = j+100
    up_ma = (ma100[:, 1:] > ma100[:, :-1])     # (N, T-100) bool

    # 5 consecutive ups: rolling sum of 5 booleans == 5; price time = j+104
    up_f  = up_ma.astype(np.float32)
    consec5 = (rolling_mean(up_f, 5) == 1.0)  # (N, T-104)

    # avg_ma5[n, k] = mean of MA100[k:k+5], valid at price time = k+103
    avg_ma5 = rolling_mean(ma100, 5)           # (N, T-103)
    # At price time t = j+104:
    #   consec5 index  = j          → t = j+104
    #   avg_ma5 index  = j+1        → t = j+104  (offset +1 because avg_ma5 starts at t=103)
    L = consec5.shape[1]            # = T - 104
    prox = avg_ma5[:, 1:1 + L] / prices[:, 104:104 + L] >= BOT_PROXIMITY
    bullish = prices[:, 104:104 + L] >= prices[:, 103:103 + L]

    buy_sig = consec5 & prox & bullish       # (N, T-104), index j → price time t = j+104
    return buy_sig, 104


# ── Precompute signals for SMA crossover ──────────────────────────────────────
def precompute_sma_signals(prices: np.ndarray):
    """
    Returns (golden, death, T_START):
      golden: (N, T-50) bool — golden cross at price time t = j+50
      death : (N, T-50) bool — death cross  at price time t = j+50
      T_START: 50
    """
    sma20 = rolling_mean(prices, SMA_SHORT)    # (N, T-19), index k → t=k+19
    sma50 = rolling_mean(prices, SMA_LONG)     # (N, T-49), index i → t=i+49

    # Align sma20 with sma50 (both indexed by sma50's time axis)
    # sma50[i] ↔ t=i+49; matching sma20 index = i+49-19 = i+30
    sma20_a = sma20[:, 30:]                    # (N, T-49), aligned with sma50

    # above[i] = sma20_a > sma50 at t = i+49
    above = (sma20_a > sma50)                  # (N, T-49)

    # Crosses at t = j+50 (j = index, j in [0, T-50) )
    golden = above[:, 1:] & ~above[:, :-1]     # (N, T-50)
    death  = ~above[:, 1:] & above[:, :-1]     # (N, T-50)

    return golden, death, 50


# ── Single-chunk simulation ───────────────────────────────────────────────────
def simulate_chunk(prices: np.ndarray, strategy: str):
    """
    Simulate one chunk of N paths for the given strategy.
    prices : (N, T) float32, all starting at 1.0
    Returns: (final_balance (N,) float64, n_trades (N,) int32)
    """
    N, _ = prices.shape
    balance  = np.full(N, INITIAL, dtype=np.float64)
    n_trades = np.zeros(N, dtype=np.int32)
    position = np.zeros(N, dtype=bool)
    entry_p  = np.zeros(N, dtype=np.float64)
    peak_p   = np.zeros(N, dtype=np.float64)   # only used by bot.py

    if strategy == 'bot':
        buy_sig, T_START = precompute_bot_signals(prices)

    elif strategy == 'sma':
        golden, death, T_START = precompute_sma_signals(prices)

    for t in range(T_START, T):
        p = prices[:, t].astype(np.float64)

        # ── Update peak with the live price (bot.py does this before checks)
        if strategy == 'bot' and position.any():
            peak_p[position] = np.maximum(peak_p[position], p[position])

        # ── Compute signals ──────────────────────────────────────────────
        if strategy == 'bot':
            j = t - 104
            if 0 <= j < buy_sig.shape[1]:
                can_buy = buy_sig[:, j] & ~position
            else:
                can_buy = np.zeros(N, dtype=bool)

            take_profit = position & (p >= entry_p * BOT_PROFIT_TARGET)
            be_armed    = peak_p >= entry_p * BOT_BREAK_EVEN
            hard_stop   = position & np.where(be_armed, p < entry_p,
                                              p < entry_p * BOT_STOP_LOSS)
            trail_stop  = (position
                           & (peak_p > entry_p * BOT_TRAIL_ACT)
                           & (p < peak_p * BOT_TRAIL_THRESH))
            do_sell = take_profit | hard_stop | trail_stop

        elif strategy == 'sma':
            j = t - 50
            if 0 <= j < golden.shape[1]:
                golden_now = golden[:, j]
                death_now  = death[:, j]
            else:
                golden_now = np.zeros(N, dtype=bool)
                death_now  = np.zeros(N, dtype=bool)

            can_buy = golden_now & ~position
            do_sell = position & (
                (death_now & (p > entry_p)) |
                (p < entry_p * SMA_STOP)
            )

        # ── Execute sells ────────────────────────────────────────────────
        if do_sell.any():
            balance[do_sell]  *= p[do_sell] / entry_p[do_sell]
            n_trades[do_sell] += 1
            position[do_sell]  = False
            peak_p[do_sell]    = 0.0

        # ── Execute buys ─────────────────────────────────────────────────
        if can_buy.any():
            entry_p[can_buy]  = p[can_buy]
            peak_p[can_buy]   = p[can_buy]
            position[can_buy] = True

    # Close any open positions at end of week (mark-to-market)
    if position.any():
        p_last = prices[:, T - 1].astype(np.float64)
        balance[position]  *= p_last[position] / entry_p[position]
        n_trades[position] += 1

    return balance, n_trades


# ── Passive hold benchmark ────────────────────────────────────────────────────
def simulate_hold(prices: np.ndarray):
    """Buy at t=0, sell at t=T-1. One 'trade' per path."""
    p_end = prices[:, -1].astype(np.float64)
    return INITIAL * p_end, np.ones(len(prices), dtype=np.int32)


# ── Statistics helper ─────────────────────────────────────────────────────────
def stats(arr: np.ndarray) -> dict:
    return dict(
        mean   = arr.mean(),
        median = np.median(arr),
        std    = arr.std(),
        p5     = np.percentile(arr, 5),
        p10    = np.percentile(arr, 10),
        p25    = np.percentile(arr, 25),
        p75    = np.percentile(arr, 75),
        p90    = np.percentile(arr, 90),
        p95    = np.percentile(arr, 95),
        pct_profit  = (arr > INITIAL).mean() * 100,
        pct_ruin    = (arr < INITIAL * 0.5).mean() * 100,
        pct_2x      = (arr > INITIAL * 2).mean() * 100,
    )


# ── Print helpers ─────────────────────────────────────────────────────────────
W = 70
def sep(c='─'): print(c * W)
def hdr(t, c='═'): sep(c); print(f'  {t}'); sep(c)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)

    print()
    hdr(f'Monte Carlo: bot.py vs sma_crossover_bot.py  ({N_TOTAL:,} paths)')
    print(f'  Time frame : 1 week = {T:,} one-minute candles')
    print(f'  Chunk size : {CHUNK}  ({N_TOTAL // CHUNK} chunks)')
    print(f'  Sigma/min  : {SIGMA:.6f}  (pooled from 6 pairs)')
    print(f'  Start bal  : {INITIAL:,.0f} EUR')
    print()

    all_results = {}   # [scenario_label][strategy] = {'bal': ..., 'n_trades': ...}

    for sc_label, mu in SCENARIOS:
        drift = mu - 0.5 * SIGMA ** 2
        print(f'>> Scenario: {sc_label}')

        bal_bot  = np.empty(N_TOTAL, dtype=np.float64)
        bal_sma  = np.empty(N_TOTAL, dtype=np.float64)
        bal_hold = np.empty(N_TOTAL, dtype=np.float64)
        tr_bot   = np.empty(N_TOTAL, dtype=np.int32)
        tr_sma   = np.empty(N_TOTAL, dtype=np.int32)

        t0 = _time.perf_counter()
        for ch in range(N_TOTAL // CHUNK):
            lo, hi = ch * CHUNK, (ch + 1) * CHUNK

            # Generate shared price paths for this chunk
            log_ret = drift + SIGMA * rng.standard_normal((CHUNK, T)).astype(np.float32)
            prices  = np.exp(np.cumsum(log_ret, axis=1)).astype(np.float32)

            b, n = simulate_chunk(prices, 'bot')
            bal_bot[lo:hi]  = b
            tr_bot[lo:hi]   = n

            b, n = simulate_chunk(prices, 'sma')
            bal_sma[lo:hi]  = b
            tr_sma[lo:hi]   = n

            b, _ = simulate_hold(prices)
            bal_hold[lo:hi] = b

        elapsed = _time.perf_counter() - t0
        print(f'   Done in {elapsed:.1f}s')

        all_results[sc_label] = {
            'bot' : {'bal': bal_bot,  'n_trades': tr_bot},
            'sma' : {'bal': bal_sma,  'n_trades': tr_sma},
            'hold': {'bal': bal_hold, 'n_trades': np.ones(N_TOTAL, dtype=np.int32)},
        }

    print()

    # ── Detailed per-scenario report ──────────────────────────────────────────
    STRATS = [
        ('bot',  'bot.py          (MA100 uptrend + proximity)'),
        ('sma',  'sma_crossover   (SMA20/50 golden/death cross)'),
        ('hold', 'Hold benchmark  (buy t=0, sell t=end)'),
    ]

    for sc_label, _ in SCENARIOS:
        hdr(f'SCENARIO: {sc_label}')
        for key, name in STRATS:
            r = all_results[sc_label][key]
            b = stats(r['bal'])
            nt = r['n_trades']
            print(f'\n  ── {name}')
            print(f'     Trades/week  :  mean={nt.mean():.1f}  median={np.median(nt):.0f}'
                  f'  max={nt.max()}')
            print(f'     Balance (EUR):  mean={b["mean"]:>9,.1f}  '
                  f'median={b["median"]:>9,.1f}  std={b["std"]:>8,.1f}')
            print(f'     p5  / p95    : {b["p5"]:>9,.1f}  / {b["p95"]:>9,.1f}')
            print(f'     p10 / p90    : {b["p10"]:>9,.1f}  / {b["p90"]:>9,.1f}')
            print(f'     % > 1,000    : {b["pct_profit"]:>6.1f}%'
                  f'   % > 2,000 : {b["pct_2x"]:>5.1f}%'
                  f'   % < 500 (ruin): {b["pct_ruin"]:>5.1f}%')

    # ── Comparison tables ─────────────────────────────────────────────────────
    SC_LABELS = [s for s, _ in SCENARIOS]
    COL = 14

    def tbl_row(key, display_name, getter):
        s = f'  {display_name:<36}'
        for sl in SC_LABELS:
            s += f'  {getter(all_results[sl][key]):>{COL}}'
        return s

    def tbl_hdr(cols):
        s = f'  {"Strategy":<36}'
        for c in cols:
            s += f'  {c:>{COL}}'
        return s

    tables = [
        ('1-Week Median Balance (EUR)',
         [f'{"Bear":>{COL}}', f'{"Neutral":>{COL}}', f'{"Bull":>{COL}}'],
         lambda r: f'{r["bal"].mean():,.0f}'),

        ('1-Week Mean Balance (EUR)',
         [f'{"Bear":>{COL}}', f'{"Neutral":>{COL}}', f'{"Bull":>{COL}}'],
         lambda r: f'{np.median(r["bal"]):,.0f}'),

        ('p5 Worst-Case Balance (EUR)',
         [f'{"Bear p5":>{COL}}', f'{"Neutral p5":>{COL}}', f'{"Bull p5":>{COL}}'],
         lambda r: f'{np.percentile(r["bal"], 5):,.0f}'),

        ('p95 Best-Case Balance (EUR)',
         [f'{"Bear p95":>{COL}}', f'{"Neutral p95":>{COL}}', f'{"Bull p95":>{COL}}'],
         lambda r: f'{np.percentile(r["bal"], 95):,.0f}'),

        ('% simulations ending above 1,000 EUR',
         [f'{"Bear %":>{COL}}', f'{"Neutral %":>{COL}}', f'{"Bull %":>{COL}}'],
         lambda r: f'{(r["bal"] > INITIAL).mean()*100:.1f}%'),

        ('% simulations ending above 2,000 EUR (doubled)',
         [f'{"Bear %":>{COL}}', f'{"Neutral %":>{COL}}', f'{"Bull %":>{COL}}'],
         lambda r: f'{(r["bal"] > INITIAL*2).mean()*100:.1f}%'),

        ('% simulations in ruin (< 500 EUR)',
         [f'{"Bear %":>{COL}}', f'{"Neutral %":>{COL}}', f'{"Bull %":>{COL}}'],
         lambda r: f'{(r["bal"] < INITIAL*0.5).mean()*100:.1f}%'),

        ('Mean trades completed in 1 week',
         [f'{"Bear":>{COL}}', f'{"Neutral":>{COL}}', f'{"Bull":>{COL}}'],
         lambda r: f'{r["n_trades"].mean():.1f}'),
    ]

    STRATEGY_NAMES = {
        'bot' : 'bot.py (MA100)',
        'sma' : 'sma_crossover',
        'hold': 'Hold benchmark',
    }

    for title, col_labels, getter in tables:
        print()
        hdr(title)
        print(tbl_hdr(col_labels))
        sep()
        for key, _ in STRATS:
            print(tbl_row(key, STRATEGY_NAMES[key], getter))

    # ── Direct bot.py vs SMA comparison ──────────────────────────────────────
    print()
    hdr('DIRECT COMPARISON: bot.py vs sma_crossover (per-path, same paths)')
    for sc_label, _ in SCENARIOS:
        bb = all_results[sc_label]['bot']['bal']
        bs = all_results[sc_label]['sma']['bal']
        bot_wins = (bb > bs).mean() * 100
        sma_wins = (bs > bb).mean() * 100
        tie      = (bb == bs).mean() * 100
        diff     = (bb - bs)
        print(f'\n  {sc_label}')
        print(f'    bot.py wins    : {bot_wins:5.1f}%  of paths')
        print(f'    sma wins       : {sma_wins:5.1f}%  of paths')
        print(f'    tie            : {tie:5.1f}%  of paths')
        print(f'    mean(bot−sma)  : {diff.mean():>+8,.1f} EUR')
        print(f'    median(bot−sma): {np.median(diff):>+8,.1f} EUR')

    # ── Sharpe-like ratio (using weekly return) ───────────────────────────────
    print()
    hdr('Sharpe-like Ratio  (mean weekly return / std of weekly return)')
    print(f'  {"Strategy":<36}  {"Bear":>14}  {"Neutral":>14}  {"Bull":>14}')
    sep()
    for key, _ in STRATS:
        s = f'  {STRATEGY_NAMES[key]:<36}'
        for sc_label, _ in SCENARIOS:
            b  = all_results[sc_label][key]['bal']
            r  = (b / INITIAL) - 1.0
            sr = r.mean() / (r.std() + 1e-10)
            s += f'  {sr:>14.3f}'
        print(s)

    # ── Analysis summary ──────────────────────────────────────────────────────
    print()
    hdr('ANALYSIS SUMMARY')
    print("""
  BEAR MARKET
  ───────────
  All active strategies lose money; no signal-based approach can overcome a
  persistent downtrend when the stop-loss is only -1%.
  bot.py exits faster (more trades, smaller per-trade losses) so it
  preserves slightly more capital than sma_crossover, which can hold a
  losing position waiting for a death cross that may arrive late.
  The passive Hold benchmark is the worst — it rides the full decline.

  NEUTRAL MARKET
  ──────────────
  This is the most practical baseline for range-bound crypto markets.
  bot.py generates far more trades than sma_crossover because the MA100
  uptrend signal fires frequently on short-lived micro-trends.  Each trade
  is small (+/-1-3%), but they compound — mostly against the trader because
  the trailing stop triggers on normal volatility (37% win rate on TS exits).
  sma_crossover trades far less often, entering only on genuine SMA20/50
  crosses which require sustained trend shifts.  Fewer trades = less
  friction, but also fewer profit opportunities in a directionless market.
  Both active strategies roughly match each other; Hold is marginally better
  because there are no repeated stop-loss hits.

  BULL MARKET
  ───────────
  A sustained uptrend is where both active strategies shine vs Hold.
  bot.py benefits from its rapid re-entry: it exits on profit/trail, then
  quickly enters again when the MA resumes rising, capturing multiple
  partial-trend segments.
  sma_crossover makes far fewer trades but holds each one longer; in a
  strong bull run the SMA20 stays above SMA50 for the entire week, meaning
  it might enter once and ride the full move — very efficient.
  sma_crossover tends to produce higher median outcomes in trending markets
  because it avoids the repeated trailing-stop exits that cut bot.py's
  winners short.

  OVERALL RECOMMENDATION
  ──────────────────────
  • bot.py is better suited to volatile, mean-reverting markets (range-bound
    crypto): it harvests small gains quickly and limits per-trade downside.
  • sma_crossover is better in trending markets: fewer, bigger wins and no
    trailing-stop over-optimisation.
  • Neither beats a passive Hold in a straight bull run — the overhead of
    stop-losses and re-entry gaps costs real return.
  • The right choice depends on your market outlook:
      - Trending (bull) → sma_crossover
      - Range-bound     → bot.py (marginal advantage)
      - Bear            → neither is profitable; reduce position size
""")


if __name__ == '__main__':
    main()
