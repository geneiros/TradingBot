"""
Monte Carlo comparison: 3 trailing-stop strategies × 3 drift scenarios.

Strategies tested:
  1. Current    : TRAILING_STOP=0.99, activates as soon as peak > buy
  2. Solution 1 : TRAILING_STOP=0.98, wider (-2% from peak), same activation
  3. Solution 2 : TRAILING_STOP=0.99, activates only when peak > buy * 1.015

Method (2-phase):
  Phase 1 – For each (strategy, scenario): run 1,000,000 independent
            single-trade GBM simulations and record the P&L %.
  Phase 2 – Bootstrap 50-trade sequences (trades are i.i.d. within a
            scenario) to compound a starting balance of 1,000 EUR and
            study the resulting balance distribution.

Price model: Geometric Brownian Motion on 1-minute candles.
  σ = 4% daily vol / sqrt(1440)  ≈ 0.001054 per candle.
"""

import time
import numpy as np

# ── GBM parameters ────────────────────────────────────────────────────────────
DAILY_VOL  = 0.04
SIGMA      = DAILY_VOL / np.sqrt(1440)   # per 1-min candle ≈ 0.001054
MAX_STEPS  = 3_000                        # cap per trade (~50 h)
N_SIMS     = 1_000_000
N_TRADES   = 50                           # trades per balance simulation
INITIAL    = 1_000.0                      # starting balance in EUR
SEED       = 42

# ── Common stop constants ──────────────────────────────────────────────────────
PROFIT_TARGET = 1.03
STOP_LOSS     = 0.99

# ── Strategy definitions ───────────────────────────────────────────────────────
STRATEGIES = [
    dict(name="Current    (trail=0.99, peak>buy)",
         trail_threshold=0.99, trail_activation=1.000),
    dict(name="Solution 1 (trail=0.98, wider)",
         trail_threshold=0.98, trail_activation=1.000),
    dict(name="Solution 2 (trail=0.99, peak>buy*1.015)",
         trail_threshold=0.99, trail_activation=1.015),
]

# ── Market scenarios ───────────────────────────────────────────────────────────
SCENARIOS = [
    dict(label="Bear   (μ=-0.0002/candle)", mu=-0.0002),
    dict(label="Neutral (μ=0)",              mu=0.0000),
    dict(label="Bull   (μ=+0.0002/candle)", mu=+0.0002),
]


# ─────────────────────────────────────────────────────────────────────────────
def simulate_single_trades(strategy: dict, mu: float,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Vectorised single-trade simulation for N_SIMS paths.
    Returns array of P&L % (float32).
    """
    trail_thr = strategy['trail_threshold']
    trail_act = strategy['trail_activation']
    drift     = mu - 0.5 * SIGMA ** 2

    prices      = np.ones(N_SIMS, dtype=np.float64)
    peaks       = np.ones(N_SIMS, dtype=np.float64)
    active      = np.ones(N_SIMS, dtype=bool)
    exit_price  = np.ones(N_SIMS, dtype=np.float64)
    exit_reason = np.full(N_SIMS, 3, dtype=np.int8)   # 3 = timeout

    for step in range(MAX_STEPS):
        if not active.any():
            break
        z          = rng.standard_normal(N_SIMS)
        new_prices = np.where(active, prices * np.exp(drift + SIGMA * z), prices)
        new_peaks  = np.where(active, np.maximum(peaks, new_prices), peaks)

        profit_hit = active & (new_prices >= PROFIT_TARGET)
        stop_hit   = active & (new_prices <  STOP_LOSS)
        trail_hit  = (active
                      & (new_peaks > trail_act)          # activation guard
                      & (new_prices < new_peaks * trail_thr))

        fired = profit_hit | stop_hit | trail_hit
        exit_reason[profit_hit]                              = 0
        exit_reason[stop_hit   & ~profit_hit]               = 1
        exit_reason[trail_hit  & ~profit_hit & ~stop_hit]   = 2
        exit_price[fired] = new_prices[fired]
        active[fired]     = False
        prices = new_prices
        peaks  = new_peaks

    exit_price[active] = prices[active]   # timeout exits
    return ((exit_price - 1.0) * 100.0).astype(np.float32), exit_reason


# ─────────────────────────────────────────────────────────────────────────────
def bootstrap_balance(pnl_pct: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """
    Sample N_SIMS sequences of N_TRADES trades (with replacement) and
    compound a 1,000 EUR balance. Returns final balances (float32).
    """
    factors  = (1.0 + pnl_pct / 100.0).astype(np.float32)
    # shape: (N_SIMS, N_TRADES)
    sampled  = rng.choice(factors, size=(N_SIMS, N_TRADES), replace=True)
    balances = INITIAL * sampled.prod(axis=1)
    return balances.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
def print_separator(char="─", width=66):
    print(char * width)


def print_header(text, char="═", width=66):
    print(char * width)
    print(f"  {text}")
    print(char * width)


def analyse_single_trade(pnl: np.ndarray, exit_reason: np.ndarray,
                          strategy_name: str, scenario_label: str) -> dict:
    EXIT_LABELS = {0: 'profit_target', 1: 'stop_loss',
                   2: 'trailing_stop',  3: 'timeout'}

    n      = len(pnl)
    counts = {k: int((exit_reason == k).sum()) for k in range(4)}
    wins   = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    win_rate = len(wins) / n

    avg_win  = float(wins.mean())   if len(wins)   > 0 else 0.0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0.0
    rr       = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    ev       = float(pnl.mean())
    kelly    = (win_rate - (1 - win_rate) / rr) if rr != float('inf') else win_rate

    # trailing-stop sub-stats
    ts = exit_reason == 2
    ts_avg = float(pnl[ts].mean()) if ts.sum() > 0 else 0.0
    ts_pos_pct = float((pnl[ts] > 0).mean() * 100) if ts.sum() > 0 else 0.0

    return dict(
        ev=ev, win_rate=win_rate*100, kelly=kelly*100,
        avg_win=avg_win, avg_loss=avg_loss, rr=rr,
        counts=counts, n=n,
        ts_n=int(ts.sum()), ts_avg=ts_avg, ts_pos_pct=ts_pos_pct,
        sl_n=counts[1], sl_avg=avg_loss if counts[1] > 0 else 0.0,
        pnl=pnl,
    )


def analyse_balance(balances: np.ndarray) -> dict:
    final = balances
    return dict(
        mean   = float(final.mean()),
        median = float(np.median(final)),
        std    = float(final.std()),
        p5     = float(np.percentile(final, 5)),
        p10    = float(np.percentile(final, 10)),
        p25    = float(np.percentile(final, 25)),
        p75    = float(np.percentile(final, 75)),
        p90    = float(np.percentile(final, 90)),
        p95    = float(np.percentile(final, 95)),
        p99    = float(np.percentile(final, 99)),
        pct_profit = float((final > INITIAL).mean() * 100),
        pct_loss50 = float((final < INITIAL * 0.5).mean() * 100),
    )


# ─────────────────────────────────────────────────────────────────────────────
def main():
    rng = np.random.default_rng(SEED)

    print_header(f"TradingBot Monte Carlo — {N_SIMS:,} simulations")
    print(f"  Strategies : {len(STRATEGIES)}")
    print(f"  Scenarios  : {len(SCENARIOS)}")
    print(f"  σ/candle   = {SIGMA:.6f}  (daily vol = {DAILY_VOL*100:.1f}%)")
    print(f"  Start bal  = {INITIAL:,.0f} EUR   over {N_TRADES} trades")
    print()

    # Store all results: results[scenario][strategy]
    all_single = {}   # single-trade stats
    all_balance = {}  # balance stats

    for sc in SCENARIOS:
        sc_label = sc['label']
        all_single[sc_label]  = {}
        all_balance[sc_label] = {}
        print(f">> Scenario: {sc_label}")
        for st in STRATEGIES:
            t0 = time.perf_counter()
            print(f"   Running '{st['name']}' ...", end=' ', flush=True)

            # Phase 1: single-trade simulation
            pnl, exit_r = simulate_single_trades(st, sc['mu'], rng)
            t1 = time.perf_counter()

            # Phase 2: bootstrap balance
            bals = bootstrap_balance(pnl, rng)
            t2 = time.perf_counter()

            single_stats  = analyse_single_trade(pnl, exit_r, st['name'], sc_label)
            balance_stats = analyse_balance(bals)

            all_single[sc_label][st['name']]  = single_stats
            all_balance[sc_label][st['name']] = balance_stats

            print(f"done  ({t2-t0:.1f}s)")

        print()

    # ── DETAILED REPORT ────────────────────────────────────────────────────
    EXIT_LABELS = {0: 'profit_target', 1: 'stop_loss',
                   2: 'trailing_stop',  3: 'timeout'}

    for sc in SCENARIOS:
        sc_label = sc['label']
        print()
        print_header(f"SCENARIO: {sc_label}")

        # Per-strategy detail
        for st in STRATEGIES:
            sn = st['name']
            s  = all_single[sc_label][sn]
            b  = all_balance[sc_label][sn]

            print()
            print(f"  ── Strategy: {sn}")
            print(f"     Exit distribution (single trade):")
            for k, lbl in EXIT_LABELS.items():
                pct = s['counts'][k] / N_SIMS * 100
                bar = '█' * int(pct / 2)
                print(f"       {lbl:<16} {s['counts'][k]:>8,}  ({pct:>6.2f}%)  {bar}")

            print(f"     P&L per trade:")
            print(f"       EV            : {s['ev']:>+9.4f}%")
            print(f"       Win rate      : {s['win_rate']:>9.2f}%")
            print(f"       Avg win       : {s['avg_win']:>+9.4f}%")
            print(f"       Avg loss      : {s['avg_loss']:>+9.4f}%")
            print(f"       R/R ratio     : {s['rr']:>9.4f}")
            print(f"       Kelly         : {s['kelly']:>+9.2f}%")
            print(f"     Trailing stop   : {s['ts_n']:>7,} fires  avg={s['ts_avg']:>+.4f}%  "
                  f"profitable: {s['ts_pos_pct']:.1f}%")
            print(f"     Balance after {N_TRADES} trades (start={INITIAL:,.0f} EUR):")
            print(f"       Mean          : {b['mean']:>10,.2f} EUR")
            print(f"       Median        : {b['median']:>10,.2f} EUR")
            print(f"       p5 / p95      : {b['p5']:>10,.2f} / {b['p95']:>10,.2f} EUR")
            print(f"       p10 / p90     : {b['p10']:>10,.2f} / {b['p90']:>10,.2f} EUR")
            print(f"       % above start : {b['pct_profit']:>9.2f}%")
            print(f"       % below 50%   : {b['pct_loss50']:>9.2f}%  (ruin risk)")

    # ── COMPARISON TABLES ──────────────────────────────────────────────────
    print()
    print_header("COMPARISON TABLE — Per-trade Expected Value", "═")
    hdr = f"  {'Strategy':<42}"
    for sc in SCENARIOS:
        hdr += f"  {'EV%':>7}"
    print(hdr)
    print_separator()
    for st in STRATEGIES:
        sn  = st['name']
        row = f"  {sn:<42}"
        for sc in SCENARIOS:
            ev = all_single[sc['label']][sn]['ev']
            row += f"  {ev:>+7.4f}"
        print(row)

    print()
    print_header("COMPARISON TABLE — Mean Final Balance (EUR, 50 trades)", "═")
    hdr = f"  {'Strategy':<42}"
    for sc in SCENARIOS:
        hdr += f"  {'Mean€':>9}"
    print(hdr)
    print_separator()
    for st in STRATEGIES:
        sn  = st['name']
        row = f"  {sn:<42}"
        for sc in SCENARIOS:
            m = all_balance[sc['label']][sn]['mean']
            row += f"  {m:>9,.1f}"
        print(row)

    print()
    print_header("COMPARISON TABLE — Median Final Balance (EUR, 50 trades)", "═")
    hdr = f"  {'Strategy':<42}"
    for sc in SCENARIOS:
        hdr += f"  {'Median€':>9}"
    print(hdr)
    print_separator()
    for st in STRATEGIES:
        sn  = st['name']
        row = f"  {sn:<42}"
        for sc in SCENARIOS:
            m = all_balance[sc['label']][sn]['median']
            row += f"  {m:>9,.1f}"
        print(row)

    print()
    print_header("COMPARISON TABLE — p5 Worst-Case Balance (EUR, 50 trades)", "═")
    hdr = f"  {'Strategy':<42}"
    for sc in SCENARIOS:
        hdr += f"  {'p5 EUR':>9}"
    print(hdr)
    print_separator()
    for st in STRATEGIES:
        sn  = st['name']
        row = f"  {sn:<42}"
        for sc in SCENARIOS:
            m = all_balance[sc['label']][sn]['p5']
            row += f"  {m:>9,.1f}"
        print(row)

    print()
    print_header("COMPARISON TABLE — % Simulations ending above 1,000 EUR", "═")
    hdr = f"  {'Strategy':<42}"
    for sc in SCENARIOS:
        hdr += f"  {'%>1000':>7}"
    print(hdr)
    print_separator()
    for st in STRATEGIES:
        sn  = st['name']
        row = f"  {sn:<42}"
        for sc in SCENARIOS:
            p = all_balance[sc['label']][sn]['pct_profit']
            row += f"  {p:>7.2f}"
        print(row)

    print()
    print_header("COMPARISON TABLE — Trailing Stop: % of exits that were profitable", "═")
    hdr = f"  {'Strategy':<42}"
    for sc in SCENARIOS:
        hdr += f"  {'TS+%':>7}"
    print(hdr)
    print_separator()
    for st in STRATEGIES:
        sn  = st['name']
        row = f"  {sn:<42}"
        for sc in SCENARIOS:
            p = all_single[sc['label']][sn]['ts_pos_pct']
            row += f"  {p:>7.1f}"
        print(row)

    print()
    print_header("SUMMARY & RECOMMENDATION", "═")
    print("""
  The tables above compare the three strategies across bear / neutral / bull
  markets on both per-trade profitability and compounded 50-trade performance.

  Key metrics to judge:
    EV%     – expected return per trade (higher is better)
    Median  – most likely balance after 50 trades
    p5      – worst-case balance (5th percentile); captures downside risk
    %>1000  – probability of being in profit after 50 trades
    TS+%    – % of trailing-stop exits that were profitable (quality of exits)
""")


if __name__ == '__main__':
    main()
