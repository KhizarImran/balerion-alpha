# balerion-alpha

Research and backtesting layer for the Balerion quantitative hedge fund.

## Overview

This repository contains strategy research, signal generation, and backtesting for the Balerion quant fund. It works in conjunction with the data layer (`balerion-data`) for market data and pushes all experiment results to a self-hosted MLflow server for tracking, comparison, and artifact storage.

## Architecture

```
┌─────────────────────────────────────────────┐
│         balerion-data (Layer 1)             │
│  - MT5 data collection                      │
│  - Parquet storage (OHLCV, 1-minute)        │
│  - FX pairs + indices                       │
└─────────────────┬───────────────────────────┘
                  │ Data Loading
                  ▼
┌─────────────────────────────────────────────┐
│       balerion-alpha (Layer 2)              │
│                                             │
│  ┌──────────────┐  ┌──────────────┐        │
│  │ Experiments  │  │  Backtests   │        │
│  │ - Strategies │  │ - vectorbt   │        │
│  │ - Signals    │  │ - Metrics    │        │
│  │ - Plotly viz │  │ - Artifacts  │        │
│  └──────────────┘  └──────┬───────┘        │
│                            │                │
└────────────────────────────┼────────────────┘
                             │ MLflow logging
                             ▼
┌─────────────────────────────────────────────┐
│       mlflow-server (../mlflow-server/)     │
│  - Docker container, port 5000              │
│  - SQLite backend + artifact proxy          │
│  - UI: http://localhost:5000                │
└─────────────────────────────────────────────┘
```

## Directory Structure

```
balerion-alpha/
├── experiments/
│   ├── silver-bullet/        # ICT Silver Bullet (US30, 1-minute)
│   │   ├── strategy.py       # Signal generation
│   │   ├── backtest.py       # Backtest → MLflow experiment: silver-bullet
│   │   └── optimize.py       # Optimization → MLflow experiment: silver-bullet-optimization
│   └── Unicorn-ICT/          # ICT Unicorn Model (EURUSD, 1H)
│       ├── strategy.py       # Signal generation
│       └── backtest.py       # Backtest → MLflow experiment: unicorn-ict
├── utils/
│   ├── data_loader.py        # Load data from balerion-data
│   ├── plotting.py           # Plotly visualization utilities
│   └── __init__.py
├── reports/                  # Generated outputs (gitignored)
├── src/balerion_alpha/       # Package scaffold stub
├── tests/
├── pyproject.toml            # Dependencies (uv)
└── README.md
```

## Prerequisites

- Python 3.11+
- [uv](https://astral.sh/uv) package manager
- `../balerion-data/` repository set up as a sibling directory
- `../mlflow-server/` running (Docker)

## Setup

```powershell
# Install uv (Windows)
irm https://astral.sh/uv/install.ps1 | iex

# Install dependencies
cd balerion-alpha
uv sync
```

## MLflow Server

All backtest runs are tracked in MLflow. The server lives in the sibling `mlflow-server/` directory and runs as a Docker container.

```powershell
# Start the server (run once, stays up)
cd ../mlflow-server
docker compose up -d

# Open the UI
# http://localhost:5000

# Stop the server
docker compose down
```

**The MLflow server must be running before executing any backtest script.**

### Experiments

| MLflow Experiment | Script |
|---|---|
| `unicorn-ict` | `experiments/Unicorn-ICT/backtest.py` |
| `silver-bullet` | `experiments/silver-bullet/backtest.py` |
| `silver-bullet-optimization` | `experiments/silver-bullet/optimize.py` |

### What gets tracked per run

- **Params**: symbol, date range, account size, leverage, risk settings, all strategy parameters
- **Metrics**: Sharpe ratio, Sortino ratio, return on margin %, win rate %, max drawdown %, profit factor, total trades, absolute P&L
- **Artifacts**: `outputs/` folder containing equity HTML chart, VBT report HTML, trades CSV, backtest report TXT, portfolio PKL

## Running Backtests

```powershell
# Unicorn ICT — EURUSD 1H
uv run python experiments/Unicorn-ICT/backtest.py

# Silver Bullet — US30 1-minute
uv run python experiments/silver-bullet/backtest.py

# Silver Bullet — parameter optimization (long-running)
uv run python experiments/silver-bullet/optimize.py
```

Results are viewable immediately at `http://localhost:5000`.

## Available Data

| Asset Class | Symbols | Native Timeframe |
|---|---|---|
| FX Pairs | EURUSD, USDJPY, GBPUSD, EURGBP, USDCAD, AUDNZD | 1 minute |
| Indices | US30 (Dow Jones), XAUUSD (Gold) | 1 minute |

Resample to any higher timeframe via `DataLoader.resample_ohlcv(df, '1H')`.

## Adding a New Experiment

1. Create `experiments/my_strategy/` with `strategy.py` and `backtest.py`
2. Add MLflow tracking to `backtest.py` (see `CLAUDE.md` for the template)
3. Set the experiment name with `mlflow.set_experiment("my-strategy")`
4. Run — the experiment appears automatically in the MLflow UI

## Key Dependencies

| Package | Purpose |
|---|---|
| `vectorbt` | Backtesting engine |
| `mlflow` | Experiment tracking + artifact storage |
| `pandas` / `pyarrow` | Data manipulation + Parquet I/O |
| `plotly` | Interactive HTML charts |
| `numpy` / `scipy` | Numerical computing |

## License

Private — All rights reserved.
