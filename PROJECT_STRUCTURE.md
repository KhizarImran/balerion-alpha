# Balerion Alpha - Project Structure

## Overview

Research and backtesting layer for the Balerion quant hedge fund.

## Directory Structure

```
balerion-alpha/
│
├── experiments/              # Strategy experiments
│   └── {strategy_name}/      # Each strategy in its own folder
│       ├── strategy.py       # Signal generation + visualization
│       ├── backtest.py       # Backtesting script (optional)
│       └── README.md         # Strategy documentation
│
├── reports/                  # ALL generated outputs
│   ├── {strategy}_{symbol}_{params}.html      # Charts
│   ├── {strategy}_backtest_report.txt         # Backtest results
│   └── README.md
│
├── utils/                    # Shared utilities
│   ├── data_loader.py        # Load data from balerion-data
│   ├── plotting.py           # Plotly visualization functions
│   └── __init__.py
│
├── config/                   # Configuration files
├── notebooks/                # Jupyter notebooks for exploration
├── tests/                    # Unit tests
│
├── .gitignore
├── pyproject.toml            # uv project configuration
├── README.md                 # Full documentation
├── QUICKSTART.md             # Quick start guide
└── PROJECT_STRUCTURE.md      # This file
```

## Workflow

### 1. Create Experiment

```bash
mkdir experiments/my_strategy
cd experiments/my_strategy
# Create strategy.py
```

### 2. Develop Strategy

In `strategy.py`:
- Load data from balerion-data using `utils.load_data()`
- Implement `calculate_signals()` function
- Visualize with `utils.plot_candlestick_with_signals()`
- Save chart to `reports/`

### 3. Backtest Strategy (Optional)

Create `backtest.py` in same folder:
- Import signals from `strategy.py`
- Run vectorbt backtest
- Save report to `reports/`

### 4. Analyze Results

- Charts in `reports/` are interactive HTML
- Backtest reports are text files with metrics
- Use notebooks for deeper analysis

## Key Principles

1. **Each experiment is self-contained** - All logic in its own folder
2. **All outputs go to reports/** - Charts and backtest results
3. **Shared utilities** - Data loading and plotting in `utils/`
4. **No backtesting framework** - Each experiment creates its own backtest script
5. **Visualization first** - Always plot signals before backtesting

## Example: SMA Crossover

```
experiments/sma_crossover/
├── strategy.py          # Calculates SMA signals, generates chart
├── backtest.py          # (You'll create) Runs backtest on signals
└── README.md

Output in reports/:
├── sma_crossover_EURUSD_20_50.html          # Interactive chart
└── sma_crossover_backtest_20260226.txt      # Backtest metrics
```

## Dependencies

- **vectorbt** - Backtesting
- **pandas/numpy** - Data manipulation
- **plotly** - Interactive charts
- **pyarrow** - Read parquet files
- **jupyter** - Notebooks

## Data Source

All market data comes from `balerion-data` repository:
- Location: `../balerion-data/data/`
- Format: Parquet files
- Assets: 6 FX pairs + 2 indices
- Timeframe: 1-minute OHLCV

## Git Workflow

- Experiment code is tracked (strategy.py, backtest.py)
- Reports are gitignored (generated outputs)
- Commit your strategies, not your results
