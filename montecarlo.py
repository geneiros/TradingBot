"""
Monte Carlo simulation for TradingBot sell-exit logic.

Mirrors the exact constants and conditions from bot.py:
  - Take profit  : price >= buy * 1.03
  - Stop-loss    : price < buy * 0.99
  - Trailing stop: peak > buy AND price < peak * 0.99

Each simulation starts at entry (price = 1.0) and runs until one of the
three conditions fires or the step limit is reached (timeout).

Price model: Geometric Brownian Motion (GBM) on 1-minute candles.
  σ_candle derived from daily volatility assumption.
  Three drift scenarios: bear, neutral, bull.
"""

import time
import numpy as np

# ── Bot constants (must mirror bot.py) ────────────────────────────────────────
PROFIT_TARGET  = 1.03
STOP_LOSS      = 0.99
TRAILING_STOP  = 0.99

# ── Simulation parameters ──────────────────────────────────────────────────────
N_SIMS      = 1_000_000
MAX_STEPS   = 3_000        # ~50 h of 1-min candles; plenty to exit
SEED        = 42

# GBM: σ per 1-min candle derived from daily vol (√1440 minutes/day)
DAILY_VOL   = 0.04         # 4 % daily σ (conservative crypto estimate)
SIGMA       = DAILY_VOL / np.sqrt(1440)   # ≈ 0.001054 per minute

EXIT_LABELS = {0: 'profit_target', 1: 'stop_loss', 2: 'trailing_stop', 3: 'timeout'}
EXIT_COLORS = {0: '✅', 1: '🔴', 2: '🟡', 3: '⏱️ '}


def run_scenario(mu: float, label: str, rng: np.random.Generator) -> dict:
    """Vectorised simulation for one drift scenario."""
    print(f"  Running '{label}' (mu={mu:+.6f}/candle) ...", flush=True)
    t0 = time.perf_counter()

    prices      = np.ones(N_SIMS)
    peaks       = np.ones(N_SIMS)
    active      = np.ones(N_SIMS, dtype=bool)
    exit_price  = np.ones(N_SIMS)
    exit_reason = np.full(N_SIMS, 3, dtype=np.int8)  # default = timeout
    exit_step   = np.full(N_SIMS, MAX_STEPS, dtype=np.int32)

    drift = mu - 0.5 * SIGMA ** 2          # log-space drift correction

    for step in range(MAX_STEPS):
        if not active.any():
            break

        # ── Price move (all sims, masked by active) ──────────────────────
        z = rng.standard_normal(N_SIMS)
        new_prices = np.where(active, prices * np.exp(drift + SIGMA * z), prices)
        new_peaks  = np.where(active, np.maximum(peaks, new_prices), peaks)

        # ── Exit conditions ────────────────────────────────────────────────
        profit_hit = active & (new_prices >= PROFIT_TARGET)
        stop_hit   = active & (new_prices <  STOP_LOSS)
        trail_hit  = active & (new_peaks > 1.0) & (new_prices < new_peaks * TRAILING_STOP)

        fired = profit_hit | stop_hit | trail_hit

        exit_reason[profit_hit]                          = 0
        exit_reason[stop_hit & ~profit_hit]              = 1
        exit_reason[trail_hit & ~profit_hit & ~stop_hit] = 2

        exit_price[fired] = new_prices[fired]
        exit_step[fired]  = step + 1
        active[fired]     = False

        prices = new_prices
        peaks  = new_peaks

    # Timeout: record last price
    exit_price[active] = prices[active]
    exit_step[active]  = MAX_STEPS

    elapsed = time.perf_counter() - t0

    return {
        'label':       label,
        'mu':          mu,
        'profits_pct': (exit_price - 1.0) * 100.0,
        'exit_reason': exit_reason,
        'exit_step':   exit_step,
        'elapsed':     elapsed,
    }


def analyse(result: dict) -> None:
    label       = result['label']
    profits     = result['profits_pct']
    exit_reason = result['exit_reason']
    exit_step   = result['exit_step']
    elapsed     = result['elapsed']

    n      = len(profits)
    counts = {k: int((exit_reason == k).sum()) for k in range(4)}
    wins   = profits[profits > 0]
    losses = profits[profits <= 0]

    print()
    print("=" * 62)
    print(f"  SCENARIO: {label}")
    print("=" * 62)
    print(f"  Simulations   : {n:>12,}")
    print(f"  Steps cap     : {MAX_STEPS:>12,}  (~{MAX_STEPS/1440:.1f} trading days)")
    print(f"  Compute time  : {elapsed:>11.2f} s")
    print()

    print("-- Exit distribution ------------------------------------------")
    for k, lbl in EXIT_LABELS.items():
        pct = counts[k] / n * 100
        bar = '#' * int(pct / 2)
        print(f"  {EXIT_COLORS[k]} {lbl:<16} {counts[k]:>8,}  ({pct:>6.2f}%)  {bar}")
    print()

    print("-- P&L statistics (% per trade) --------------------------------")
    print(f"  Mean return        : {profits.mean():>+9.4f}%")
    print(f"  Median return      : {np.median(profits):>+9.4f}%")
    print(f"  Std dev            : {profits.std():>9.4f}%")
    print(f"  Win rate (P&L > 0) : {len(wins)/n*100:>9.2f}%")
    print(f"  Profit-target hits : {counts[0]/n*100:>9.2f}%")
    print()

    print("-- Percentile table --------------------------------------------")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        v = np.percentile(profits, p)
        print(f"  p{p:<3}  {v:>+9.4f}%")
    print()

    print("-- Risk / reward -----------------------------------------------")
    avg_win  = wins.mean()   if len(wins)   > 0 else 0.0
    avg_loss = losses.mean() if len(losses) > 0 else 0.0
    rr_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    ev       = profits.mean()
    win_rate = len(wins) / n
    kelly    = (win_rate - (1 - win_rate) / rr_ratio) if rr_ratio != float('inf') else win_rate
    print(f"  Avg win            : {avg_win:>+9.4f}%")
    print(f"  Avg loss           : {avg_loss:>+9.4f}%")
    print(f"  Risk/reward ratio  : {rr_ratio:>9.4f}")
    print(f"  Expected value     : {ev:>+9.4f}%  per trade")
    print(f"  Kelly fraction     : {kelly*100:>+9.2f}%  of capital")
    print()

    print("-- Holding time (candles = 1-min bars) -------------------------")
    print(f"  Mean               : {exit_step.mean():>9.1f}")
    print(f"  Median             : {np.median(exit_step):>9.0f}")
    print(f"  p10                : {np.percentile(exit_step, 10):>9.0f}")
    print(f"  p90                : {np.percentile(exit_step, 90):>9.0f}")
    print()

    print("-- Tail risk ---------------------------------------------------")
    var99  = np.percentile(profits, 1)
    cvar99 = profits[profits <= var99].mean()
    print(f"  5th pct outcome    : {np.percentile(profits, 5):>+9.4f}%")
    print(f"  VaR  1%            : {var99:>+9.4f}%")
    print(f"  CVaR 1%            : {cvar99:>+9.4f}%  (avg beyond VaR)")
    print()

    print("-- Trailing stop effectiveness ---------------------------------")
    ts_mask = exit_reason == 2
    sl_mask = exit_reason == 1
    ts_n    = ts_mask.sum()
    sl_n    = sl_mask.sum()
    if ts_n > 0:
        ts_pos = (profits[ts_mask] > 0).sum()
        ts_avg = profits[ts_mask].mean()
        print(f"  Trailing stop fires : {ts_n:>7,}  avg={ts_avg:>+.4f}%  profitable: {ts_pos/ts_n*100:.1f}%")
    else:
        print(f"  Trailing stop fires :       0")
    if sl_n > 0:
        print(f"  Stop-loss fires     : {sl_n:>7,}  avg={profits[sl_mask].mean():>+.4f}%")
    else:
        print(f"  Stop-loss fires     :       0")
    print()


def main():
    print(f"TradingBot Monte Carlo  --  {N_SIMS:,} simulations per scenario")
    print(f"  PROFIT_TARGET={PROFIT_TARGET}  STOP_LOSS={STOP_LOSS}  TRAILING_STOP={TRAILING_STOP}")
    print(f"  sigma/candle={SIGMA:.6f}  (daily vol = {DAILY_VOL*100:.1f}%,  1-min candles)")
    print()

    rng = np.random.default_rng(SEED)

    scenarios = [
        (-0.0002, "Bear   (-0.02%/candle drift)"),
        ( 0.0000, "Neutral (no drift)"),
        ( 0.0002, "Bull   (+0.02%/candle drift)"),
    ]

    results = []
    for mu, label in scenarios:
        results.append(run_scenario(mu, label, rng))

    for r in results:
        analyse(r)

    # Cross-scenario summary
    print("=" * 62)
    print("  CROSS-SCENARIO SUMMARY")
    print("=" * 62)
    print(f"  {'Scenario':<35} {'EV %':>8}  {'Win%':>6}  {'Kelly%':>7}")
    print(f"  {'-'*35} {'-'*8}  {'-'*6}  {'-'*7}")
    for r in results:
        p        = r['profits_pct']
        wins     = p[p > 0]
        losses   = p[p <= 0]
        win_rate = len(wins) / len(p)
        avg_win  = wins.mean()   if len(wins)   > 0 else 0
        avg_loss = losses.mean() if len(losses) > 0 else 0
        rr       = abs(avg_win / avg_loss) if avg_loss != 0 else 1
        kelly    = win_rate - (1 - win_rate) / rr
        ev       = p.mean()
        print(f"  {r['label']:<35} {ev:>+8.4f}  {win_rate*100:>6.2f}  {kelly*100:>+7.2f}")
    print()


if __name__ == '__main__':
    main()
