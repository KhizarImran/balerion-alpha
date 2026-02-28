# CLAUDE.md — balerion-alpha

This file provides Claude Code with the context needed to work effectively in this repository.

---

## Project Overview

**balerion-alpha** is the research and backtesting layer (Layer 2) of the **Balerion** quantitative hedge fund system, developed by **Khizar Imran**. It depends on a sibling repository called **balerion-data** (Layer 1) which supplies raw OHLCV market data collected via MetaTrader 5 and stored as Parquet files.

The project focuses on strategy research, signal generation, visualization, and backtesting for FX pairs and indices.

---

## Architecture

```
balerion-data/ (sibling repo at ../balerion-data/)
  - MT5 live data collection
  - Parquet storage (OHLCV, 1-minute native)
  - Assets: EURUSD, USDJPY, GBPUSD, EURGBP, USDCAD, AUDNZD, US30, XAUUSD
      |
      | (data loading via utils/data_loader.py)
      v
balerion-alpha/ (this repo)
  - experiments/   → signal generation + Plotly visualization
  - backtesting/   → vectorbt backtesting scripts (planned)
  - utils/         → shared data loading + plotting library
  - reports/       → generated HTML charts and backtest reports (gitignored)
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Package Manager | `uv` (Astral) — always use `uv run python` |
| Backtesting Engine | vectorbt >= 0.26.0 |
| Data I/O | pandas, pyarrow (Parquet) |
| Visualization | Plotly (interactive HTML), matplotlib, seaborn |
| Numerical Computing | numpy, scipy |
| ML (available) | scikit-learn |
| Notebooks | Jupyter, IPython |
| Config | PyYAML |
| Code Quality | ruff (linter), black (formatter), mypy (type checking) |
| Testing | pytest |

---

## Directory Structure

```
balerion-alpha/
├── .agents/skills/vectorbt-expert/SKILL.md  # AI skill: comprehensive vectorbt + FX reference
├── backtesting/          # Standalone vectorbt scripts (currently empty)
├── config/               # Configuration files (currently empty)
├── experiments/          # Strategy experiments — the main working area
│   ├── sma_crossover/    # SMA crossover experiment
│   │   ├── strategy.py   # Signal generation + visualization
│   │   └── README.md
│   ├── silver-bullet/    # ICT Silver Bullet strategy (US30, 1-minute)
│   │   ├── strategy.py   # Signal generation + visualization
│   │   ├── backtest.py   # vectorbt backtest → reports/ (project root)
│   │   └── optimize.py   # Parameter optimization
│   └── Unicorn-ICT/      # ICT Unicorn Model (EURUSD, 1H)
│       ├── strategy.py   # Signal generation + visualization
│       ├── backtest.py   # vectorbt backtest → reports/{SYMBOL}_{timestamp}/
│       └── reports/      # Per-run timestamped output folders (gitignored)
├── notebooks/            # Jupyter notebooks (currently empty)
├── reports/              # Project-level generated outputs (gitignored — never commit)
├── src/balerion_alpha/   # Package scaffold (stub only — not used)
├── tests/                # Unit tests (currently empty)
├── utils/                # Core shared library
│   ├── __init__.py       # Exports: DataLoader, load_data, plot_* functions
│   ├── data_loader.py    # Loads Parquet data from ../balerion-data/data/
│   └── plotting.py       # Plotly dark-themed chart utilities
├── .gitignore            # Ignores reports/, experiments/**/reports/, .venv/, etc.
├── pyproject.toml        # Dependencies (uv)
├── uv.lock
└── skills-lock.json      # Locks vectorbt-expert AI skill
```

---

## Common Commands

```bash
# Install / sync dependencies
uv sync

# Run an experiment
uv run python experiments/sma_crossover/strategy.py

# Run a backtest script inside an experiment folder
uv run python experiments/<name>/backtest.py

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Format
uv run black .

# Type check
uv run mypy .
```

---

## Core Utilities (`utils/`)

### Data Loading

```python
from utils import load_data, DataLoader

# Quick one-liner
df = load_data('EURUSD', asset_type='fx', start_date='2024-01-01')
df = load_data('US30', asset_type='indices', start_date='2024-01-01')

# Full DataLoader class
loader = DataLoader()
df = loader.load_fx('EURUSD', start_date='2024-01-01', end_date='2025-01-01', timeframe='1m')
df = loader.load_index('US30', start_date='2024-01-01')
symbols = loader.get_available_fx_symbols()

# Resample OHLCV to a higher timeframe
df_hourly = loader.resample_ohlcv(df, '1H')
df_4h = loader.resample_ohlcv(df, '4H')
df_daily = loader.resample_ohlcv(df, '1D')
```

The data loader resolves `../balerion-data/data/` relative to project root. Files are named `{symbol_lower}_{timeframe}.parquet` (e.g. `eurusd_1m.parquet`).

OHLCV columns returned: `open`, `high`, `low`, `close`, `volume` (lowercase).

### Plotting

```python
from utils import plot_candlestick_with_signals, plot_strategy_performance, save_plot

# Candlestick chart with buy/sell markers and indicator overlays
fig = plot_candlestick_with_signals(
    df,
    title='EURUSD - My Strategy',
    buy_signals=df['buy_signal'],    # boolean Series
    sell_signals=df['sell_signal'],  # boolean Series
    indicators={
        'SMA 20': df['sma_20'],
        'SMA 50': df['sma_50'],
    },
    show_volume=True,
    height=800,
)
fig.show()

# Equity curve + drawdown chart
fig = plot_strategy_performance(equity_curve, trades, title='Backtest Results')

# Save to reports/
save_plot(fig, 'reports/my_strategy_EURUSD.html')   # HTML default
save_plot(fig, 'reports/my_strategy_EURUSD.png', format='png')
```

---

## Available Market Data

| Asset Class | Symbols | Native Timeframe |
|---|---|---|
| FX Pairs | EURUSD, USDJPY, GBPUSD, EURGBP, USDCAD, AUDNZD | 1 minute |
| Indices | US30 (Dow Jones), XAUUSD (Gold) | 1 minute |

Resample to any timeframe: `5m`, `15m`, `1H`, `4H`, `1D`, etc.

---

## Experiment Structure

Each experiment lives in `experiments/<strategy-name>/` and follows this structure:

```
experiments/my_strategy/
├── strategy.py    # Signal generation + Plotly visualization
├── backtest.py    # vectorbt backtesting (optional)
└── README.md      # Strategy documentation
```

### Template: `strategy.py`

```python
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from utils import load_data, plot_candlestick_with_signals


def calculate_signals(df: pd.DataFrame, **params) -> pd.DataFrame:
    df = df.copy()
    # Strategy logic here
    df['buy_signal'] = ...   # boolean
    df['sell_signal'] = ...  # boolean
    return df


def run_experiment(symbol='EURUSD', asset_type='fx', start_date='2024-01-01',
                   end_date=None, show_chart=True, save_chart=True, **params):
    df = load_data(symbol, asset_type=asset_type, start_date=start_date, end_date=end_date)
    df = calculate_signals(df, **params)

    fig = plot_candlestick_with_signals(
        df,
        title=f'{symbol} - My Strategy',
        buy_signals=df['buy_signal'],
        sell_signals=df['sell_signal'],
        indicators={'My Indicator': df['my_indicator']},
    )

    if save_chart:
        reports_dir = project_root / 'reports'
        reports_dir.mkdir(exist_ok=True)
        fig.write_html(str(reports_dir / f'my_strategy_{symbol}.html'))

    if show_chart:
        fig.show()

    return df, fig


if __name__ == '__main__':
    run_experiment()
```

### Template: `backtest.py`

```python
import sys
from pathlib import Path
from datetime import datetime
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import vectorbt as vbt
from utils import DataLoader
from strategy import calculate_signals

# --- Config ---
SYMBOL = 'EURUSD'
ACCOUNT_SIZE = 10_000       # real margin capital in USD
LEVERAGE = 100              # 1:100 retail FX leverage
RISK_PER_TRADE = 0.01       # 1% of account per trade
PIP = 0.0001                # pip size (0.01 for JPY pairs)
PIP_VALUE_PER_LOT = 10.0    # USD per pip per standard lot (EURUSD)
STD_LOT = 100_000           # units per standard lot
FX_FEES = 0.00007           # ~0.7 pip spread per side
FREQ = '1h'

# --- Load & resample ---
loader = DataLoader()
df_1m = loader.load_fx(SYMBOL, start_date='2024-01-01')
df = loader.resample_ohlcv(df_1m, '1H')
df = calculate_signals(df)
close = df['close']

# --- Risk-based lot sizing ---
# Lot units sized so SL hit = RISK_PER_TRADE * ACCOUNT_SIZE
# units = (account * risk) / (sl_pips * pip_value_per_lot)
# Pass init_cash = ACCOUNT_SIZE * LEVERAGE (notional buying power)

# --- Run backtest ---
# IMPORTANT: vbt defaults to FigureWidget which requires anywidget.
# Always set use_widgets=False before calling pf.plot() to save HTML.
vbt.settings.plotting['use_widgets'] = False

pf = vbt.Portfolio.from_signals(
    close,
    entries=df['buy_signal'],
    exits=df['sell_signal'],
    init_cash=ACCOUNT_SIZE * LEVERAGE,  # notional buying power
    size=10_000,                        # units — replace with per-bar sizing
    size_type='amount',
    fees=FX_FEES,
    sl_stop=0.003,                      # replace with per-bar sl_stop Series
    tp_stop=0.006,                      # replace with per-bar tp_stop Series
    freq=FREQ,
)

# Re-anchor stats to real margin capital
abs_pnl = pf.value().iloc[-1] - (ACCOUNT_SIZE * LEVERAGE)
return_on_margin = (abs_pnl / ACCOUNT_SIZE) * 100

stats = pf.stats()
stats['Start Value'] = ACCOUNT_SIZE
stats['End Value'] = ACCOUNT_SIZE + abs_pnl
stats['Total Return [%]'] = return_on_margin

print(stats)

# --- Save outputs to timestamped run folder ---
run_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
reports_dir = Path(__file__).resolve().parent / 'reports' / f'{SYMBOL}_{run_ts}'
reports_dir.mkdir(parents=True, exist_ok=True)

with open(reports_dir / 'backtest_report.txt', 'w') as f:
    f.write(str(stats))

pf.trades.records_readable.to_csv(reports_dir / 'trades.csv')

vbt_fig = pf.plot(subplots=['value', 'cum_returns', 'underwater', 'drawdowns', 'orders', 'trades', 'net_exposure'])
vbt_fig.write_html(str(reports_dir / 'vbt_report.html'))

pf.save(str(reports_dir / 'portfolio.pkl'))

print(f'Outputs saved -> {reports_dir}')
```

---

## vectorbt Reference

The `.agents/skills/vectorbt-expert/SKILL.md` file contains a 1,000+ line expert reference for vectorbt. It covers:

- All 4 Portfolio simulation modes (`from_signals`, `from_orders`, `from_order_func`, `from_holding`)
- Position sizing (`percent`, `value`, `targetpercent`, etc.)
- Stop loss, take profit, trailing stop parameters
- Parameter optimization (vectorized broadcasting + loop-based)
- Performance metrics and plotting
- 7 ready-to-use strategy templates

Key vectorbt patterns for this project:

```python
import vectorbt as vbt

# Basic signal-based backtest
pf = vbt.Portfolio.from_signals(
    close,
    entries=buy_signals,
    exits=sell_signals,
    init_cash=10_000,
    fees=0.0001,
    size=0.95,            # 95% of capital per trade
    size_type='percent',
    freq='1T',            # 1-minute native data; use '1H', '4H', '1D' after resampling
    min_size=0.01,
    direction='longonly',
)

print(pf.stats())
print(f"Total Return: {pf.total_return() * 100:.2f}%")
print(f"Sharpe Ratio: {pf.sharpe_ratio():.2f}")
print(f"Max Drawdown: {pf.max_drawdown() * 100:.2f}%")
print(f"Win Rate: {pf.trades.win_rate() * 100:.1f}%")
print(f"Total Trades: {pf.trades.count()}")

# Plot
fig = pf.plot(subplots=['value', 'underwater', 'cum_returns'])
fig.show()
```

---

## Reports Convention

All generated outputs go in `reports/` (gitignored — never commit these files).

Naming convention:
- Charts: `{strategy}_{symbol}_{params}.html`
- Backtest reports: `{strategy}_backtest_{timestamp}.txt`

Example: `reports/sma_crossover_EURUSD_20_50.html`

---

## Important Notes

- **Always use `uv run python`** — never bare `python` — to ensure the correct virtualenv is used.
- **`reports/` is gitignored** — never commit generated charts or backtest output files.
- **Data lives in `../balerion-data/`** — the data layer must be set up as a sibling directory.
- **Path bootstrap:** Every experiment script must add project root to `sys.path` to import `utils`. Use `project_root = Path(__file__).resolve().parent.parent.parent`.
- **No `.env` file needed** — the project currently has no secrets or environment variables.
- **No CI/CD** — this is a local research environment, not a deployed service.
- **Signals must be boolean pandas Series** with the same index as the price DataFrame.
- **Avoid lookahead bias** in signal calculations — use `.shift(1)` when comparing current vs previous values.
