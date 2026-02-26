# ICT Silver Bullet Strategy

## Overview

An ICT (Inner Circle Trader) concept-based intraday strategy for US30 (Dow Jones) on 1-minute data. The strategy hunts for short-term liquidity sweeps followed by a bullish Fair Value Gap (FVG) entry during the 10 AM New York session.

## Strategy Logic

### Entry Conditions (all must be true)

| # | Condition | Description |
|---|---|---|
| 1 | **Low swept** | A 240-bar rolling low has been taken out within the last 120 bars |
| 2 | **High not swept** | The 240-bar rolling high has NOT been swept — bullish bias retained |
| 3 | **Bullish FVG** | A fair value gap exists: `high[N-2] < low[N]` (two-bar imbalance to the upside) |
| 4 | **FVG pullback** | Current close is inside the FVG zone (top = `low[N]`, bottom = `high[N-2]`) within 15 bars of the gap forming |
| 5 | **Session** | Bar is at 10:00 AM EST (New York killzone) |

Only the **first qualifying bar per calendar day** is taken as an entry.

### Exit

Fixed **52-bar (52-minute) time exit** — the position is closed exactly 52 minutes after entry regardless of P&L.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `liquidity_window` | 240 | Rolling bars for significant high/low |
| `sweep_lookback` | 120 | Bars to look back for sweep detection |
| `pullback_window` | 15 | Bars to keep FVG zone alive for re-entry |
| `session_hour_est` | 10 | EST hour for session filter |

## Files

| File | Purpose |
|---|---|
| `strategy.py` | Signal generation + last-5-days Plotly chart |
| `backtest.py` | Full-year vectorbt backtest + scrollable HTML report |

## Usage

```bash
# Visualise signals (last 5 trading days)
uv run python experiments/silver-bullet/strategy.py

# Full backtest + combined HTML report
uv run python experiments/silver-bullet/backtest.py
```

## Output

Both scripts write outputs to `reports/` (gitignored):

- `reports/silver_bullet_US30_2025-11-11_2026-02-25.html` — signal chart
- `reports/silver_bullet_US30_backtest_2025-11-11_2026-02-25.html` — full scrollable report with 6 VectorBT charts + stats table

## Notes

- Data source: `../balerion-data/data/indices/us30_1m.parquet`
- All timestamps in the data are assumed to be UTC; EST = UTC-5 (no DST adjustment)
- No stop-loss or take-profit — exit is purely time-based (52 bars)
- One trade per day maximum
