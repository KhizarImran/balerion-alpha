# Strategy + Backtest Pattern

This document describes the two-script architecture used in every experiment in this repository. Any AI agent or developer can follow this pattern to build a new strategy from scratch.

The pattern separates concerns into two files:

| File | Responsibility |
|---|---|
| `strategy.py` | Signal detection, filtering, trade parameter calculation, visualisation |
| `backtest.py` | Vectorbt simulation, position sizing, metrics, MLflow logging |

`backtest.py` imports `strategy.py` at runtime using `importlib` — no package installation required. Any change to `strategy.py` is automatically picked up by `backtest.py` on the next run.

---

## Repository Layout

```
balerion-alpha/
├── experiments/
│   └── <strategy-name>/
│       ├── strategy.py          # signal detection + chart
│       ├── backtest.py          # vectorbt backtest + MLflow
│       ├── tradinglogic.md      # human-readable strategy rules
│       └── reports/             # generated outputs (gitignored)
├── utils/
│   ├── __init__.py              # exports DataLoader, build_equity_chart, build_analytics_chart
│   ├── data_loader.py           # loads Parquet OHLCV from ../balerion-data/
│   ├── backtest_charts.py       # reusable Plotly chart builders for vbt results
│   └── plotting.py              # strategy visualisation helpers
└── CLAUDE.md                    # top-level project context
```

Data lives in a sibling repository `../balerion-data/data/`. The `DataLoader` class resolves this path automatically.

---

## Running Order

```
MLflow server must be running first:
    cd ../mlflow-server && docker compose up -d

Visualise signals only (no backtest):
    uv run python experiments/<strategy-name>/strategy.py

Full backtest + MLflow logging:
    uv run python experiments/<strategy-name>/backtest.py
```

Always use `uv run python` — never bare `python`.

---

## Part 1: strategy.py

### Purpose

`strategy.py` does one job: given an OHLCV DataFrame, return a list of trade setups. Each setup is a dict containing the exact entry timestamp, entry price, stop loss price, and take profit price. It also produces a Plotly HTML chart for visual inspection.

### Boilerplate (copy this for every new strategy)

```python
import os
import sys
import io
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["SMC_CREDIT"] = "0"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from smartmoneyconcepts import smc
from utils import DataLoader
```

The three lines before imports are mandatory:
- `sys.path.insert` — makes `utils/` importable from the experiment subfolder
- UTF-8 fix — prevents crashes on Windows terminals (cp1252) when MLflow prints emoji
- `SMC_CREDIT=0` — silences the smartmoneyconcepts library credit banner

### Config Block

All tunable parameters sit at module level as uppercase constants. `backtest.py` overrides some of these (e.g. `RISK_REWARD`) when calling `detect_setups()` — the module-level values are only the defaults.

```python
SYMBOL = "EURUSD"
START_DATE = "2024-01-01"
END_DATE = "2026-03-13"
LOWER_TF = "1h"    # entry/signal timeframe — used by load_ohlcv and detect_setups
HIGHER_TF = "1D"   # confluence filter timeframe — used by _build_htf_fvgs
SWING_LENGTH = 10
SESSION_START = 7  # UK session start hour (inclusive)
SESSION_END = 18   # UK session end hour (exclusive)
RISK_REWARD = 2.0  # TP = RISK_REWARD * SL distance
MIN_FVG_PIPS = 5   # minimum FVG height in pips — quality filter

_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"}
PIP_SIZE = 0.01 if SYMBOL in _JPY_PAIRS else 0.0001
```

### Data Loading

The project supports two data sources, resolved automatically:

1. **Dukascopy Parquet files** — preferred. Files named `{symbol}_dukascopy_{tf}.parquet` in `../balerion-data/data/fx/`. Auto-detected at module load time.
2. **DataLoader fallback** — loads MT5 1-minute data and resamples.

```python
_DUKASCOPY_DIR = Path(__file__).resolve().parent.parent.parent.parent / "balerion-data" / "data" / "fx"
DUKASCOPY_FILES = {}
if _DUKASCOPY_DIR.exists():
    for f in _DUKASCOPY_DIR.glob("*_dukascopy_*.parquet"):
        symbol_from_file = f.stem.split("_dukascopy")[0].upper()
        DUKASCOPY_FILES[symbol_from_file] = f

def load_ohlcv(symbol, start, end, tf):
    if symbol in DUKASCOPY_FILES:
        return load_dukascopy(symbol, start, end)
    loader = DataLoader()
    df_1m = loader.load_fx(symbol, start_date=start, end_date=end)
    return loader.resample_ohlcv(df_1m, tf)
```

The returned DataFrame always has a `DatetimeIndex` (timezone-naive) and lowercase columns: `open`, `high`, `low`, `close`, `volume`.

### The Setup Dict

Every detected trade is stored as a plain dict. The exact keys required by `backtest.py` are:

```python
{
    "entry_ts":   pd.Timestamp,  # index label of the entry bar — used to place the signal
    "entry":      float,         # entry price (close of entry bar)
    "sl":         float,         # absolute stop loss price
    "tp":         float,         # absolute take profit price

    # optional — used only by build_chart(), not by backtest.py
    "ob_ts":      pd.Timestamp,
    "mit_ts":     pd.Timestamp,
    "fvg_ts":     pd.Timestamp,
    "ob_top":     float,
    "ob_bottom":  float,
    "fvg_top":    float,
    "fvg_bottom": float,
}
```

**Critical constraints**:
- For SELL (short) setups: `sl > entry` must be true or `backtest.py` will discard the setup
- For BUY (long) setups: `sl < entry` must be true or `backtest.py` will discard the setup
- `entry_ts` must exist in the OHLCV DataFrame index

### detect_setups()

The main function `backtest.py` imports. Signature:

```python
def detect_setups(
    df: pd.DataFrame,
    swing_length: int = SWING_LENGTH,
    risk_reward: float = RISK_REWARD,
) -> tuple[list, list, list]:
    ...
    return sell_setups, buy_setups, htf_fvgs
```

Returns a 3-tuple. `backtest.py` unpacks it as:

```python
sell_setups, buy_setups, _htf_fvgs = detect_setups(df, swing_length=SWING_LENGTH, risk_reward=RISK_REWARD)
```

The third element (`htf_fvgs`) is only used by `build_chart()`. `backtest.py` discards it with `_htf_fvgs`.

**Filter order inside `detect_setups()`** — apply these in sequence for each candidate setup. Each filter is a `continue` that skips to the next candidate:

1. **Structural conditions** — OB identification, mitigation window, FVG overlap
2. **FVG size filter** — `(fvg_top - fvg_bottom) >= MIN_FVG_PIPS * PIP_SIZE`
3. **Lookahead guard** — entry scan starts at `max(mit_bar + 1, fvg_bar + 2)`, never on an FVG-forming candle
4. **Session filter** — `SESSION_START <= entry_hour < SESSION_END`
5. **Multi-timeframe filter** — `_in_active_htf_fvg(entry_ts, entry_price, direction, htf_fvgs)`
6. **SL/TP calculation** — computed last, after all filters pass

### Multi-Timeframe Filter

The HTF FVG confluence filter is built by two private helpers:

```python
def _build_htf_fvgs(df_lower: pd.DataFrame, higher_tf: str = HIGHER_TF) -> list:
    """Resample to HIGHER_TF, detect FVGs, track each FVG's active window."""
    loader = DataLoader()
    df_htf = loader.resample_ohlcv(df_lower, higher_tf)
    fvg_df = smc.fvg(df_htf, join_consecutive=False)
    # for each FVG bar[i] (middle candle):
    #   active_from = htf_index[i + 2]   (right candle i+1 is still forming)
    #   active_until = first close that mitigates the zone, or pd.NaT if open
    ...
    return htf_fvgs  # list of dicts: direction, top, bottom, active_from, active_until

def _in_active_htf_fvg(entry_ts, entry_price, direction, htf_fvgs) -> bool:
    """Point-in-time lookup: is this entry inside a matching active HTF FVG?"""
    for fvg in htf_fvgs:
        if fvg["direction"] != direction: continue
        if entry_ts < fvg["active_from"]: continue
        if fvg["active_until"] is not pd.NaT and entry_ts >= fvg["active_until"]: continue
        if fvg["bottom"] <= entry_price <= fvg["top"]: return True
    return False
```

Called inside `detect_setups()` with `direction=-1` for SELL setups and `direction=1` for BUY setups.

### SL/TP Calculation

SL is placed at the extreme of the combined OB + FVG zone (not just the OB alone):

```python
# SELL: SL above the highest ceiling of OB or FVG
sl = max(ob_top, fvg_top)
sl_dist = abs(sl - entry_price)
tp = entry_price - risk_reward * sl_dist

# BUY: SL below the lowest floor of OB or FVG
sl = min(ob_bottom, fvg_bottom)
sl_dist = abs(entry_price - sl)
tp = entry_price + risk_reward * sl_dist
```

This gives the trade room to breathe through the full combined zone before being invalidated.

### build_chart()

Produces a 3-row Plotly HTML chart for visual inspection. Does not affect the backtest. Runs only when `strategy.py` is executed directly (`__main__`).

```
Row 1 (52%): Lower-TF candlesticks — OB rectangles, FVG rectangles, SL/TP lines, entry markers
Row 2 (13%): Lower-TF volume
Row 3 (35%): Higher-TF candlesticks — active HTF FVG zones (green=bullish, red=bearish)
```

### __main__ block

```python
if __name__ == "__main__":
    df = load_ohlcv(SYMBOL, START_DATE, END_DATE, LOWER_TF)
    sell_setups, buy_setups, htf_fvgs = detect_setups(df)
    fig = build_chart(df, sell_setups, buy_setups, htf_fvgs=htf_fvgs)
    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"strategy_{SYMBOL}_{LOWER_TF}.html"
    fig.write_html(str(out))
    fig.show()
```

---

## Part 2: backtest.py

### Purpose

`backtest.py` imports `detect_setups` and `load_ohlcv` from `strategy.py`, converts the setup dicts into vectorbt-compatible per-bar arrays, runs the simulation, computes metrics, saves artifacts locally, and logs everything to MLflow.

### Strategy Module Import

```python
import importlib.util
_strategy_path = Path(__file__).resolve().parent / "strategy.py"
_spec = importlib.util.spec_from_file_location("strategy", _strategy_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
detect_setups = _mod.detect_setups
load_ohlcv = _mod.load_ohlcv
```

This loads `strategy.py` as a module without it being a Python package. Any change to `strategy.py` is picked up automatically on the next `backtest.py` run — no re-installation needed.

### Config Block

`backtest.py` has its own config. Parameters here override the defaults in `strategy.py` by being passed explicitly to `detect_setups()`:

```python
SYMBOL = "EURUSD"
START_DATE = "2024-01-01"
END_DATE = "2026-03-13"
TIMEFRAME = "1h"          # must match LOWER_TF in strategy.py
SWING_LENGTH = 10
RISK_REWARD = 2.0         # passed to detect_setups() — overrides strategy.py default

ACCOUNT_SIZE = 10_000     # USD real margin capital
LEVERAGE = 100            # 1:100 FX retail leverage
RISK_PER_TRADE = 0.01     # 1% of account per trade

PIP = 0.0001              # change to 0.01 for JPY pairs
PIP_VALUE_PER_LOT = 10.0  # USD per pip per standard lot (adjust per symbol/account currency)
STD_LOT = 100_000         # base units per standard lot

# Fee model — see "Fee Modelling" section below for a full explanation
FX_FEES = 0.0             # disabled in vectorbt — use fixed fee subtracted post-simulation
FIXED_FEE_PER_TRADE = 8.0 # USD per round-trip (entry + exit); adjust per instrument
```

### setups_to_signal_arrays()

Converts the setup dicts into per-bar pandas Series that vectorbt can consume.

```python
def setups_to_signal_arrays(df, sell_setups, buy_setups) -> tuple:
    """
    Returns:
        short_entries  : boolean Series — True on SELL entry bars
        long_entries   : boolean Series — True on BUY entry bars
        short_sl_arr   : float Series   — fractional SL distance from entry (NaN elsewhere)
        short_tp_arr   : float Series   — fractional TP distance from entry (NaN elsewhere)
        long_sl_arr    : float Series
        long_tp_arr    : float Series
        lot_units      : float Series   — position size in base currency units
    """
```

**Fractional SL/TP**: vectorbt's `sl_stop` and `tp_stop` parameters take a fraction of the entry price, not an absolute price. The conversion:

```python
# SELL
sl_frac = abs(sl - entry) / entry   # e.g. sl=1.0850, entry=1.0800 → 0.00463
tp_frac = abs(entry - tp) / entry

# BUY
sl_frac = abs(entry - sl) / entry
tp_frac = abs(tp - entry) / entry
```

**Position sizing**: risk-based — the lot size is chosen so that hitting the SL costs exactly `RISK_PER_TRADE * ACCOUNT_SIZE`:

```python
risk_usd = ACCOUNT_SIZE * RISK_PER_TRADE          # e.g. $100
sl_pips = (sl_frac * entry) / PIP                 # SL distance in pips
lots = risk_usd / (sl_pips * PIP_VALUE_PER_LOT)   # standard lots
lot_units = lots * STD_LOT                        # base currency units (passed to vbt)
```

**Validation guards** — setups are silently dropped before writing to the arrays if:
- `entry_ts` is not in the DataFrame index
- `sl <= entry` for a SELL setup (SL must be above entry for a short)
- `sl >= entry` for a BUY setup (SL must be below entry for a long)
- `sl_frac` or `tp_frac` is NaN, zero, or negative

### run_backtest() — vectorbt Call

```python
vbt.settings.plotting["use_widgets"] = False  # required before saving HTML

combined_sl = long_sl.fillna(short_sl)  # do NOT fillna(0) — see Common Pitfalls
combined_tp = long_tp.fillna(short_tp)

pf = vbt.Portfolio.from_signals(
    close=df["close"],
    entries=long_entries,
    exits=pd.Series(False, index=df.index),   # exits handled by SL/TP only
    short_entries=short_entries,
    short_exits=pd.Series(False, index=df.index),
    sl_stop=combined_sl,
    tp_stop=combined_tp,
    stop_exit_price="StopMarket",   # fills at the exact SL/TP price, not bar close
    init_cash=ACCOUNT_SIZE,         # real margin capital only — NOT * LEVERAGE
    size=lot_units,
    size_type="amount",
    fees=FX_FEES,                   # 0.0 — fees subtracted manually post-simulation
    freq=TIMEFRAME,
    accumulate=False,
)
```

Key decisions:
- **`stop_exit_price="StopMarket"`** — exits at the exact stop level, not the bar close. If price gaps through the level, fills at next bar open. This is the realistic behaviour. The default `"Close"` would overstate losses.
- **`init_cash=ACCOUNT_SIZE`** — pass real margin capital only. Do **not** multiply by `LEVERAGE`. See "Fee Modelling" section for why.
- **`exits=False` / `short_exits=False`** — positions are never manually exited. The only exits are SL and TP stops.
- **`accumulate=False`** — no pyramiding. Only one position per direction at a time.
- **Combined SL/TP arrays** — `long_sl.fillna(short_sl)` merges both directions into a single array. NaN on non-signal bars is correct — do not fill with 0 (see Common Pitfalls).

### P&L Re-anchoring and Fee Subtraction

```python
stats = pf.stats()
total_trades = int(stats.get("Total Trades", 0))

gross_pnl  = pf.value().iloc[-1] - ACCOUNT_SIZE        # vectorbt gross result
total_fees = total_trades * FIXED_FEE_PER_TRADE         # fixed cost per round-trip
abs_pnl    = gross_pnl - total_fees                     # net P&L after fees

return_on_margin = (abs_pnl / ACCOUNT_SIZE) * 100
```

With `init_cash=ACCOUNT_SIZE`, the base is already in real capital terms — no leverage offset required.

### Metrics Extracted

```python
sharpe       = pf.sharpe_ratio()
sortino      = pf.sortino_ratio()
max_dd       = pf.max_drawdown() * 100
total_trades = pf.trades.count()
win_rate     = pf.trades.win_rate() * 100
profit_factor = pf.trades.profit_factor()
```

### Artifacts Saved

All outputs go into a timestamped subfolder: `reports/{SYMBOL}_{YYYYMMDD_HHMMSS}/`

| File | Content |
|---|---|
| `backtest_report.txt` | Text summary of all params and metrics |
| `trades.csv` | Full trade-by-trade records from `pf.trades.records_readable` |
| `equity.html` | 2-panel equity curve + drawdown (from `utils.build_equity_chart`) |
| `analytics.html` | 8-panel dashboard: P&L by weekday/hour/month, histogram, cumulative P&L, direction breakdown |
| `vbt_report.html` | Native vectorbt 7-panel plot |
| `portfolio.pkl` | Serialised `vbt.Portfolio` object for offline analysis |

The entire folder is then uploaded to MLflow in one call:

```python
mlflow.log_artifacts(str(reports_dir), artifact_path="outputs")
```

### MLflow Logging

```python
os.environ["MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS"] = "true"  # must be set before any mlflow call

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("my-experiment-name")

with mlflow.start_run(run_name=f"{SYMBOL}_{TIMEFRAME}_{START_DATE}_{END_DATE}"):
    mlflow.log_params({...})    # symbol, dates, account settings, strategy params
    mlflow.log_metrics({...})   # all numeric performance metrics
    mlflow.log_artifacts(str(reports_dir), artifact_path="outputs")
```

The MLflow server uses `--serve-artifacts` with proxy mode. Artifacts are uploaded over HTTP — they do not touch the local filesystem of the server. `MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS=true` must be set before any MLflow API call.

---

## Fee Modelling

### The Problem with Percentage Fees

vectorbt's `fees` parameter takes a **fraction of notional trade value** (e.g. `0.00008` = 0.008% per side). This sounds small but compounds badly with large lot sizes:

```
Example: 0.5 standard lots on EURUSD at 1.1000
Notional = 0.5 × 100,000 × 1.1000 = $55,000
Fee at 0.008% = $55,000 × 0.00008 = $4.40 per side = $8.80 round-trip
```

That is reasonable in this case, but the fee **scales with the lot size**. With risk-based sizing, the lot size varies per trade (larger lots when SL is tight, smaller when SL is wide). This means:

- Trades with tight SLs get oversized lots → disproportionately large fees → SL is not the actual worst case
- The percentage fee model rewards wide SLs (which are already less efficient)
- It introduces an artificial bias that does not reflect how retail brokers charge

Retail FX brokers charge **spread**, not a percentage. Spread cost per trade is approximately constant in dollar terms (it depends on pip size and current price, not the notional value of your position).

### The Correct Model: Fixed Fee Per Round-Trip

Subtract a fixed dollar amount per completed trade post-simulation:

```python
FX_FEES = 0.0                  # disabled in vectorbt
FIXED_FEE_PER_TRADE = 8.0      # USD per round-trip — set this per instrument

# After vbt.Portfolio.from_signals():
stats = pf.stats()
total_trades = int(stats.get("Total Trades", 0))
gross_pnl    = pf.value().iloc[-1] - ACCOUNT_SIZE
total_fees   = total_trades * FIXED_FEE_PER_TRADE
abs_pnl      = gross_pnl - total_fees
```

### Calibrating FIXED_FEE_PER_TRADE

Use the typical spread for the instrument and the average lot size from a test run:

```
fee_per_trade ≈ spread_pips × pip_value_per_lot × avg_lots × 2  (×2 for entry + exit)
```

| Instrument | Typical spread | Pip value (1 lot) | Conservative estimate |
|---|---|---|---|
| EURUSD | 0.5–1.0 pip | $10 | $8–$10 |
| USDJPY | 0.5–1.0 pip | $9.50 | $8–$10 |
| GBPUSD | 0.8–1.5 pip | $10 | $10–$15 |
| XAUUSD | 2–4 pip ($0.01) | $1 | $5–$10 |
| US30 | 2–5 points | $1 | $5–$10 |

Round up to be conservative. The goal is a realistic worst-case, not an optimistic best-case.

### Why Not `init_cash = ACCOUNT_SIZE * LEVERAGE`

A common mistake is passing `init_cash=ACCOUNT_SIZE * LEVERAGE` to give vectorbt "notional buying power". This inflates the starting cash 100× and causes all PnL figures to be 100× too large. The `size` parameter (in base currency units) already controls position size correctly — `init_cash` only needs to be large enough that vectorbt does not reject orders for insufficient cash. `ACCOUNT_SIZE` alone is sufficient because the lot sizes are computed from risk %, not from available cash.

```python
# WRONG — inflates all PnL by 100x
init_cash=ACCOUNT_SIZE * LEVERAGE   # $1,000,000

# CORRECT
init_cash=ACCOUNT_SIZE              # $10,000
```

---

## Data Contract Between the Two Scripts

`backtest.py` depends on exactly these things from `strategy.py`:

| Symbol | Type | Description |
|---|---|---|
| `detect_setups(df, swing_length, risk_reward)` | function | Returns `(sell_setups, buy_setups, htf_fvgs)` |
| `load_ohlcv(symbol, start, end, tf)` | function | Returns OHLCV DataFrame with DatetimeIndex |
| `sell_setups[i]["entry_ts"]` | pd.Timestamp | Entry bar index label |
| `sell_setups[i]["entry"]` | float | Entry price |
| `sell_setups[i]["sl"]` | float | Stop loss price (must be > entry for shorts) |
| `sell_setups[i]["tp"]` | float | Take profit price (must be < entry for shorts) |
| `buy_setups[i]["sl"]` | float | Stop loss price (must be < entry for longs) |
| `buy_setups[i]["tp"]` | float | Take profit price (must be > entry for longs) |

The additional keys in the setup dict (`ob_ts`, `fvg_ts`, etc.) are used only by `build_chart()` and are ignored by `backtest.py`.

---

## Replicating This Pattern for a New Strategy

1. Create `experiments/<new-strategy>/` directory
2. Copy the boilerplate from the config block and `load_ohlcv` function into a new `strategy.py`
3. Implement `detect_setups()` — it must return `(sell_setups, buy_setups, any_extra_data)`. The `sell_setups` and `buy_setups` lists must contain dicts with at minimum: `entry_ts`, `entry`, `sl`, `tp`
4. Implement `build_chart()` for visual inspection — this is optional for the backtest but essential for debugging the signal logic
5. Copy `backtest.py` from an existing experiment, update `SYMBOL`, `START_DATE`, `END_DATE`, `TIMEFRAME`, `RISK_REWARD`, and the MLflow experiment name. No other changes are needed if the setup dict contract is respected
6. Set `PIP` and `PIP_VALUE_PER_LOT` correctly for the instrument:
   - JPY pairs: `PIP = 0.01`, `PIP_VALUE_PER_LOT = 1000.0` (for USD account)
   - Gold (XAUUSD): `PIP = 0.01`, `PIP_VALUE_PER_LOT = 1.0`
   - Standard FX: `PIP = 0.0001`, `PIP_VALUE_PER_LOT = 10.0`

---

## Common Pitfalls

| Pitfall | Symptom | Fix |
|---|---|---|
| `init_cash=ACCOUNT_SIZE * LEVERAGE` | All PnL figures are 100× too large (e.g. +1,163% instead of +11.6%) | Use `init_cash=ACCOUNT_SIZE` only. Position size is already controlled by the `size` array |
| `fees=FX_FEES` with a percentage fee | Fee per trade scales with lot size, distorting results for risk-based sizing | Set `FX_FEES=0.0` and subtract `FIXED_FEE_PER_TRADE * total_trades` from gross PnL manually |
| `gross_pnl` base wrong after removing leverage | PnL still references the old `ACCOUNT_SIZE * LEVERAGE` offset | Use `gross_pnl = pf.value().iloc[-1] - ACCOUNT_SIZE` (no leverage multiplier) |
| `combined_sl.fillna(0)` on non-signal bars | Phantom stop-outs fire on every non-entry bar, collapsing the equity curve | Leave NaN on non-signal bars: `combined_sl = long_sl.fillna(short_sl)` — do not chain `.fillna(0)` |
| SL on wrong side of entry | Setup silently dropped with no error | Add `assert sl > entry` (SELL) or `assert sl < entry` (BUY) in `detect_setups()` during development |
| Entry fires on an FVG-forming candle | Lookahead bias — strategy cannot have known the FVG existed | Use `entry_start = max(mit_bar + 1, fvg_bar + 2)` — never `fvg_bar` or `fvg_bar + 1` |
| Lookahead bias in signal calculation | Unrealistically high win rate | Use `.shift(1)` when comparing current vs previous bar values |
| `stop_exit_price` defaulting to `"Close"` | SL/TP fill at bar close instead of stop level, overstating losses | Always pass `stop_exit_price="StopMarket"` explicitly |
| `use_widgets=True` crashes HTML save | `write_html()` raises an error or produces a blank file | Set `vbt.settings.plotting["use_widgets"] = False` before any `.plot()` or `.write_html()` call |
| MLflow artifacts not uploading | Silent failure or HTTP error during `log_artifacts` | Set `os.environ["MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS"] = "true"` before any `import mlflow` |
| Timezone mismatch with smc library | Errors or misaligned signals | Strip timezone from the DataFrame index: `df.index = df.index.tz_localize(None)` |
