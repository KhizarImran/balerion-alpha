# ICT Silver Bullet Strategy

## Overview

An ICT (Inner Circle Trader) concept-based intraday strategy for US30 (Dow Jones) on 1-minute data. The strategy hunts for short-term liquidity sweeps followed by a bullish Fair Value Gap (FVG) entry during three ICT killzone windows per day. Two-phase parameter optimisation was run across 16,000+ combinations to find the best configuration.

## Strategy Logic

### Entry Conditions (all must be true)

| # | Condition | Description |
|---|---|---|
| 1 | **Low swept** | A `liquidity_window`-bar rolling low has been taken out within the last `sweep_lookback` bars |
| 2 | **High not swept** | The `liquidity_window`-bar rolling high has NOT been swept — bullish bias retained |
| 3 | **Bullish FVG** | A fair value gap exists: `high[N-2] < low[N]` (two-bar upward imbalance) |
| 4 | **FVG pullback** | Current close is inside the FVG zone within `pullback_window` bars of the gap forming |
| 5 | **Session window** | Bar falls within one of the three ICT killzone hours (Eastern Time) |

One entry is allowed per session per calendar day (first qualifying bar wins).

### Sessions (Eastern Time)

| Session | ET Hours | Broker Server Hours* |
|---|---|---|
| London | 03:00 – 03:59 | 10:00 – 10:59 (winter) |
| New York AM | 10:00 – 10:59 | 17:00 – 17:59 (winter) |
| New York PM | 14:00 – 14:59 | 21:00 – 21:59 (winter) |

\* Broker data timestamps are in `Europe/Athens` server time (UTC+2 winter / UTC+3 summer). Timezone conversion is handled automatically via `pytz` to account for DST transitions in both the EU and US.

### Stop Loss & Take Profit

- **Stop Loss:** Below the swept liquidity low minus `sl_buffer_pts` points
- **Take Profit:** `rr_ratio × SL distance` above entry price
- **Time exit fallback:** Position closed after 52 bars if neither SL nor TP is hit

### Session P&L Cap

A per-session cumulative loss cap (`session_cap_usd`) prevents further entries in a session once estimated losses exceed the threshold. The cap assumes the worst case (SL hit) for each trade taken.

---

## Optimised Parameters

Two-phase grid search over 16,827 combinations (Phase 1: 10,752 + Phase 2: 6,075).

| Parameter | Optimised Value | Description |
|---|---|---|
| `rr_ratio` | `4.5` | Take profit = 4.5 × SL distance |
| `liquidity_window` | `150` bars | Rolling window for significant high/low levels |
| `sweep_lookback` | `210` bars | Bars to look back for sweep detection |
| `pullback_window` | `10` bars | Bars the FVG zone stays alive for re-entry |
| `sl_buffer_pts` | `0.0` pts | Extra points below liquidity low for SL |
| `session_cap_usd` | `-$5,500` | Max cumulative estimated loss per session per day |

---

## Backtest Results (in-sample)

Period: 2025-11-11 to 2026-02-25 | Capital: $100,000 | Leverage: 100:1 | Lots: 10

| Metric | Value |
|---|---|
| Total Trades | 46 |
| Return on Margin | +15.85% |
| Sharpe Ratio | 5.81 |
| Win Rate | 58.7% |
| Max Drawdown | 0.04% |

> **Note:** These are in-sample results on a single 3.5-month window. No out-of-sample or walk-forward validation has been run. Treat these figures as optimistic.

---

## Files

| File | Purpose |
|---|---|
| `strategy.py` | Signal generation + last-5-days Plotly visualisation chart |
| `backtest.py` | vectorbt backtest with optimised params — generates full HTML report |
| `optimize.py` | Two-phase grid-search optimisation (run from scratch, ~3.5 hours) |
| `optimize_phase2.py` | Phase 2 fine search only + HTML optimisation report (`REPORT_ONLY = True` to skip backtesting and rebuild HTML from saved CSVs) |
| `_check_results.py` | Utility: prints top-10 rows from phase 1 optimisation CSV |
| `_smoke_test.py` | Utility: smoke-tests 3 parameter combos end-to-end |

---

## Usage

```bash
# Visualise signals on last 5 trading days
uv run python experiments/silver-bullet/strategy.py

# Full backtest + combined HTML report (uses optimised params)
uv run python experiments/silver-bullet/backtest.py

# Run both optimisation phases from scratch (~3.5 hours)
uv run python experiments/silver-bullet/optimize.py

# Phase 2 only — reuse phase 1 CSV, run fine search
uv run python experiments/silver-bullet/optimize_phase2.py

# Rebuild optimisation HTML report without re-running backtests
# (set REPORT_ONLY = True at top of file first)
uv run python experiments/silver-bullet/optimize_phase2.py
```

---

## Output

All generated files go to `reports/` (gitignored — never commit):

| File | Contents |
|---|---|
| `reports/silver_bullet_US30_{start}_{end}.html` | Signal visualisation chart |
| `reports/silver_bullet_US30_backtest_{start}_{end}.html` | Full scrollable report: 6-panel performance chart, KPI cards, metrics table, last-20-trades table, last-5-trades candlestick detail |
| `reports/silver_bullet_optimization_phase1.csv` | Phase 1 results (10,752 rows) |
| `reports/silver_bullet_optimization_phase2.csv` | Phase 2 results (6,075 rows) |
| `reports/silver_bullet_optimization_report.html` | Optimisation scatter + box plots |

---

## Leverage Model

US30 at ~47,000 points makes direct position sizing impractical on a $100k account without leverage. The backtest simulates 100:1 leverage by passing `init_cash = real_margin × leverage` to vectorbt, giving it enough notional buying power to fill the desired lot size. All P&L figures are correct in absolute terms; return percentages are reported as **return on margin** (`abs_pnl / real_margin`), not return on notional.

```
$100,000 margin × 100:1 leverage → $10,000,000 notional
10 lots × $1/point/lot → $10 P&L per US30 point
Margin per lot ≈ $470  (well within $100k)
```

---

## Notes

- Data source: `../balerion-data/data/indices/us30_1m.parquet`
- Broker timestamps are in `Europe/Athens` server time — DST conversion handled automatically via `pytz`
- Signals are boolean pandas Series; lookahead bias avoided by using rolling windows and `.shift()` where needed
- The session P&L cap is not binding at the current optimised parameters (the strategy never accumulates enough loss in a single session to trigger it)
- Only long (bullish) entries are implemented; the symmetric short (bearish FVG after high sweep) is not yet coded
