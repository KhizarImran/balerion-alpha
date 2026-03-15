# ICT Silver Bullet Strategy

## Overview

An ICT (Inner Circle Trader) intraday strategy for **US30 (Dow Jones) on 1-minute data**.
The strategy hunts for short-term liquidity sweeps followed by a bullish or bearish
Fair Value Gap (FVG) pullback entry during three ICT killzone windows per day.

Both **long** (bullish) and **short** (bearish) sides are implemented.

See `tradinglogic.md` for the full rule set.

---

## Strategy Logic Summary

### Long entry (all must be true)
1. Rolling low (`liquidity_window` bars) has been swept within `sweep_lookback` bars
2. Rolling high has NOT been swept (bullish bias intact)
3. A bullish FVG exists: `high[N-2] < low[N]`
4. Price pulls back inside the FVG zone within `pullback_window` bars
5. Bar falls within one of the three killzone windows (Eastern Time)

**Stop Loss:** below swept liquidity low − `sl_buffer_pts`
**Take Profit:** `rr_ratio × SL distance` above entry

### Short entry (mirror of long)
1. Rolling high swept within `sweep_lookback` bars (bearish bias)
2. Rolling low NOT swept
3. Bearish FVG: `low[N-2] > high[N]`
4. Price pulls back into bearish FVG zone
5. Killzone session

**Stop Loss:** above swept liquidity high + `sl_buffer_pts`
**Take Profit:** `rr_ratio × SL distance` below entry

### Sessions (Eastern Time)

| Session | ET Hours |
|---|---|
| London | 03:00 – 03:59 |
| New York AM | 10:00 – 10:59 |
| New York PM | 14:00 – 14:59 |

One entry per session per calendar day. A per-session cumulative loss cap
(`session_cap_usd`) prevents further entries once the estimated loss exceeds
the threshold.

Time exit fallback: 52 bars if neither SL nor TP is hit.

---

## Optimised Parameters

Two-phase grid search (Phase 1: 10,752 + Phase 2: ~6,000 combinations).
These parameters are optimised for Sharpe Ratio on the in-sample window.

| Parameter | Value | Description |
|---|---|---|
| `rr_ratio` | `4.5` | TP = 4.5 × SL distance |
| `liquidity_window` | `150` | Rolling window for significant high/low levels |
| `sweep_lookback` | `210` | Bars to look back for sweep detection |
| `pullback_window` | `10` | Bars the FVG zone stays alive |
| `sl_buffer_pts` | `0.0` | Extra points beyond liquidity level for SL |
| `session_cap_usd` | `-$5,500` | Max cumulative estimated loss per session per day |

---

## Backtest Results (in-sample, long side only)

Period: 2025-11-11 to 2026-02-25 | Capital: $100,000 | Leverage: 100:1 | Lots: 10

| Metric | Value |
|---|---|
| Total Trades | 46 |
| Return on Margin | +15.85% |
| Sharpe Ratio | 5.81 |
| Win Rate | 58.7% |
| Max Drawdown | 0.04% |

> **Note:** In-sample results on a single 3.5-month window. No out-of-sample
> or walk-forward validation has been run. Short-side results will differ
> from long-side as a different bias (high sweep) is required.

---

## Files

| File | Purpose |
|---|---|
| `strategy.py` | Signal detection — `detect_setups()` returns `(sell_setups, buy_setups, [])` |
| `backtest.py` | vectorbt backtest + MLflow logging (experiment: `silver-bullet`) |
| `optimize.py` | Two-phase grid-search optimisation + HTML report (experiment: `silver-bullet-optimization`) |
| `tradinglogic.md` | Human-readable full strategy rules |
| `README.md` | This file |

---

## Usage

```bash
# Start MLflow server first (from mlflow-server sibling repo):
# docker compose up -d

# Visualise signals (last 5 days, saves HTML to reports/):
uv run python experiments/silver-bullet/strategy.py

# Full backtest + MLflow run:
uv run python experiments/silver-bullet/backtest.py

# Run both optimisation phases from scratch (~3-4 hours):
uv run python experiments/silver-bullet/optimize.py

# Rebuild optimisation HTML report from saved CSVs (no backtesting):
# Set REPORT_ONLY = True at the top of optimize.py, then:
uv run python experiments/silver-bullet/optimize.py
```

---

## Output

All generated files go to `experiments/silver-bullet/reports/` (gitignored).

| File | Contents |
|---|---|
| `reports/strategy_US30_1min.html` | Signal chart (last 5 days, both long + short setups) |
| `reports/{SYMBOL}_{timestamp}/backtest_report.txt` | Text summary of params and metrics |
| `reports/{SYMBOL}_{timestamp}/trades.csv` | Full trade records |
| `reports/{SYMBOL}_{timestamp}/equity.html` | Equity curve + drawdown chart |
| `reports/{SYMBOL}_{timestamp}/analytics.html` | 8-panel analytics dashboard |
| `reports/{SYMBOL}_{timestamp}/vbt_report.html` | Native vectorbt 7-panel plot |
| `reports/{SYMBOL}_{timestamp}/portfolio.pkl` | Serialised vbt.Portfolio object |
| `reports/silver_bullet_optimization_phase1.csv` | Phase 1 optimisation results |
| `reports/silver_bullet_optimization_phase2.csv` | Phase 2 optimisation results |
| `reports/silver_bullet_optimization_report.html` | Optimisation scatter + box plots |

---

## Leverage Model

US30 at ~47,000 points requires leverage for practical position sizing.
The backtest simulates 100:1 leverage by passing `init_cash = ACCOUNT_SIZE × LEVERAGE`
to vectorbt. All P&L figures are correct in absolute terms; return percentages
are reported as **return on margin** (`abs_pnl / ACCOUNT_SIZE`).

```
$100,000 margin × 100:1 leverage → $10,000,000 notional buying power
10 lots × $1/point/lot → $10 P&L per US30 point
```

---

## Architecture Notes

- `strategy.py` follows the canonical pattern from `experiments/STRATEGY_BACKTEST_PATTERN.md`
- `backtest.py` loads `strategy.py` via `importlib` — no package installation needed
- `stop_exit_price="StopMarket"` is used in vectorbt — fills at exact SL/TP price, not bar close
- `MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS=true` is set before any MLflow API call
- `optimize.py` uses the same `detect_setups()` interface as `backtest.py`
- Broker timestamps are in `Europe/Athens` server time; DST conversion to ET is
  handled via `pytz` in `strategy.py`
