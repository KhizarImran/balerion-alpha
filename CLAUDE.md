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

mlflow-server/ (sibling repo at ../mlflow-server/)
  - Docker-based MLflow tracking server (port 5000)
  - All backtest runs, metrics, params, and artifacts pushed here
  - Start with: docker compose up -d (from ../mlflow-server/)
  - UI at: http://localhost:5000
```

---

## Tech Stack

| Category | Technology |
|---|---|
| Language | Python 3.11 |
| Package Manager | `uv` (Astral) — always use `uv run python` |
| Backtesting Engine | vectorbt >= 0.26.0 |
| Experiment Tracking | MLflow >= 2.19.0 → `http://localhost:5000` |
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
│   ├── silver-bullet/    # ICT Silver Bullet strategy (US30, 1-minute)
│   │   ├── strategy.py   # Signal generation
│   │   ├── backtest.py   # vectorbt backtest → MLflow (experiment: silver-bullet)
│   │   └── optimize.py   # Parameter optimization → MLflow (experiment: silver-bullet-optimization)
│   └── Unicorn-ICT/      # ICT Unicorn Model (EURUSD, 1H)
│       ├── strategy.py   # Signal generation
│       └── backtest.py   # vectorbt backtest → MLflow (experiment: unicorn-ict)
├── notebooks/            # Jupyter notebooks (currently empty)
├── reports/              # Project-level generated outputs (gitignored — never commit)
├── src/balerion_alpha/   # Package scaffold stub
├── tests/                # Unit tests (currently empty)
├── utils/                # Core shared library
│   ├── __init__.py       # Exports: DataLoader, load_data, plot_* functions
│   ├── data_loader.py    # Loads Parquet data from ../balerion-data/data/
│   └── plotting.py       # Plotly dark-themed chart utilities
├── .gitignore
├── pyproject.toml        # Dependencies (uv)
├── uv.lock
└── skills-lock.json      # Locks vectorbt-expert AI skill
```

---

## MLflow Integration

All backtest scripts push runs to the MLflow server running at `http://localhost:5000`.

**The MLflow server must be running before executing any backtest.** Start it from `../mlflow-server/`:

```powershell
cd ../mlflow-server
docker compose up -d
```

### Experiments

| MLflow Experiment | Script |
|---|---|
| `unicorn-ict` | `experiments/Unicorn-ICT/backtest.py` |
| `silver-bullet` | `experiments/silver-bullet/backtest.py` |
| `silver-bullet-optimization` | `experiments/silver-bullet/optimize.py` |

### What gets logged

Each run logs:
- **Params**: symbol, dates, account size, leverage, risk settings, strategy parameters
- **Metrics**: Sharpe ratio, Sortino ratio, return on margin %, win rate %, max drawdown %, profit factor, total trades, abs P&L
- **Artifacts**: full output folder (`outputs/`) containing HTML charts, trades CSV, backtest report TXT, and portfolio PKL

### Template for new backtest scripts

```python
import os
import mlflow

os.environ["MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS"] = "true"

MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("my-experiment")

# ... run backtest ...

mlflow.start_run(run_name="descriptive_run_name")
mlflow.log_params({...})
mlflow.log_metrics({...})

# Save outputs to a local folder, then upload the whole folder
mlflow.log_artifacts(str(reports_dir), artifact_path="outputs")
mlflow.end_run()
```

### Important MLflow notes

- The server uses `--serve-artifacts` with `mlflow-artifacts:/` as the default root — artifacts are uploaded via HTTP proxy, not written directly to disk.
- `MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS=true` must be set in the client before calling any MLflow API.
- Use `mlflow.log_artifacts(folder)` (plural) to upload an entire directory, not `log_artifact` per file.
- Local report folders under `experiments/` are **not** kept — artifacts live exclusively in MLflow.

---

## Common Commands

```bash
# Install / sync dependencies
uv sync

# Run a backtest (MLflow server must be running first)
uv run python experiments/Unicorn-ICT/backtest.py
uv run python experiments/silver-bullet/backtest.py
uv run python experiments/silver-bullet/optimize.py

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
```

---

## Available Market Data

| Asset Class | Symbols | Native Timeframe |
|---|---|---|
| FX Pairs | EURUSD, USDJPY, GBPUSD, EURGBP, USDCAD, AUDNZD | 1 minute |
| Indices | US30 (Dow Jones), XAUUSD (Gold) | 1 minute |

Resample to any timeframe: `5m`, `15m`, `1H`, `4H`, `1D`, etc.

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

# IMPORTANT: always disable FigureWidget before saving HTML
vbt.settings.plotting['use_widgets'] = False

pf = vbt.Portfolio.from_signals(
    close,
    entries=buy_signals,
    exits=sell_signals,
    init_cash=ACCOUNT_SIZE * LEVERAGE,  # notional buying power
    size=lot_units,
    size_type='amount',
    fees=FX_FEES,
    sl_stop=sl_series,   # per-bar fractional Series
    tp_stop=tp_series,   # per-bar fractional Series
    freq='1h',
)

# Re-anchor stats to real margin capital
abs_pnl = pf.value().iloc[-1] - (ACCOUNT_SIZE * LEVERAGE)
return_on_margin = (abs_pnl / ACCOUNT_SIZE) * 100
```

---

## Important Notes

- **Always use `uv run python`** — never bare `python` — to ensure the correct virtualenv is used.
- **MLflow server must be running** before executing any backtest script. Start from `../mlflow-server/` with `docker compose up -d`.
- **Artifacts live in MLflow only** — local report folders under `experiments/` are not kept after a run.
- **`reports/` is gitignored** — never commit generated charts or backtest output files.
- **Data lives in `../balerion-data/`** — the data layer must be set up as a sibling directory.
- **Path bootstrap:** Every experiment script must add project root to `sys.path` to import `utils`. Use `project_root = Path(__file__).resolve().parent.parent.parent`.
- **Windows terminal encoding:** Scripts set `sys.stdout` to UTF-8 to handle MLflow's emoji output on cp1252 terminals.
- **No `.env` file needed** — no secrets or environment variables required beyond `MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS=true` (set inline in each script).
- **No CI/CD** — this is a local research environment, not a deployed service.
- **Signals must be boolean pandas Series** with the same index as the price DataFrame.
- **Avoid lookahead bias** in signal calculations — use `.shift(1)` when comparing current vs previous values.
