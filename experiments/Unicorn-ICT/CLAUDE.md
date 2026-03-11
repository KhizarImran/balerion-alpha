# CLAUDE.md — Unicorn-ICT Experiment

Context file for Claude Code. Describes the current state, file structure, and what has been built so far in this experiment.

---

## Strategy Name

**Failed Order Block + Fair Value Gap** (formerly "ICT Unicorn Model")

---

## What This Strategy Does

Identifies institutional order blocks at major swing highs/lows that get fully violated (displaced through), then trades the retrace back into the Fair Value Gap left by that displacement move.

Full logic: see `tradinglogic.md`

---

## Files

| File | Purpose |
|---|---|
| `strategy.py` | Signal detection + Plotly visualisation chart |
| `backtest.py` | vectorbt backtest → MLflow logging |
| `tradinglogic.md` | Full trading logic documentation |
| `CLAUDE.md` | This file — progress and context for Claude |
| `reports/` | Generated outputs (gitignored) |

---

## Current Implementation State

### strategy.py

- Loads EURUSD 1H data via `utils.DataLoader`
- Detects major swing highs/lows using `smc.swing_highs_lows(swing_length=10)`
- **SELL setup**: last bearish candle before swing low = bullish OB → mitigated (close <= OB low within 120 bars) → bearish FVG overlapping OB during mitigation leg → entry when price retraces up into FVG
- **BUY setup**: mirror — last bullish candle before swing high = bearish OB → mitigated up → bullish FVG → entry on retrace down
- Entry = close of entry bar, SL = OB high/low, TP = 2R
- Produces a Plotly HTML chart with OB rectangles, FVG rectangles (extended to entry), SL/TP lines, entry markers
- Chart saved to `reports/failed_ob_fvg_EURUSD_1h.html`

### backtest.py

- Imports `detect_setups` and `load_ohlcv` directly from `strategy.py` via `importlib`
- Converts setups to per-bar vectorbt signal arrays (boolean entries + fractional SL/TP)
- Risk-based position sizing: 1% of $10,000 account per trade, lot size from SL distance
- Runs `vbt.Portfolio.from_signals()` with long + short simultaneously
- Equity re-anchored from notional ($1,000,000) to real margin ($10,000)
- Saves: `backtest_report.txt`, `trades.csv`, `equity.html`, `vbt_report.html`, `portfolio.pkl`
- Logs all params, metrics, and artifact folder to MLflow at `http://localhost:5000`
- MLflow experiment: `unicorn-ict`

---

## Latest Backtest Results

**Period**: 2025-11-18 to 2026-02-27 (3.5 months, 1,697 1H bars)

| Metric | Value |
|---|---|
| SELL setups detected | 26 |
| BUY setups detected | 20 |
| Valid trades (after SL validation) | 32 |
| Return on Margin | +2.52% |
| Abs P&L | +$251.54 |
| Sharpe Ratio | 0.47 |
| Sortino Ratio | 0.82 |
| Max Drawdown | -0.11% |
| Win Rate | 43.8% |
| Profit Factor | 1.08 |

Results are limited by data quantity — only 3.5 months available. Strategy needs 12–24 months for statistical significance.

---

## Config (current)

| Parameter | Value |
|---|---|
| Symbol | EURUSD |
| Timeframe | 1H |
| Swing length | 10 (major pivots only) |
| Mitigation window | 120 bars (5 days) |
| Account size | $10,000 |
| Leverage | 100:1 |
| Risk per trade | 1% |
| Fees | 0.7 pip/side |
| RR | 2:1 fixed |

---

## How to Run

```powershell
# Start MLflow server first (Docker must be running)
cd ..\mlflow-server
docker compose up -d

# Visualisation chart only
uv run python experiments/Unicorn-ICT/strategy.py

# Full backtest + MLflow logging
uv run python experiments/Unicorn-ICT/backtest.py
```

---

## Known Issues / Next Steps

- **Data is only 3.5 months** (Nov 2025 – Feb 2026) — need more history from MT5 for meaningful backtesting
- Win rate of 43.8% with 2R yields positive expectancy but sample size (32 trades) is too small to be conclusive
- No session filter applied — could add London/NY session filter to reduce noise trades
- No parameter optimisation done yet — swing_length and mitigation window are candidates
- Could explore partial TP / trailing stop as an alternative to fixed 2R
