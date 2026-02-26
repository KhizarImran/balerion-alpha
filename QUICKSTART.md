# Quick Start Guide

## Installation

```bash
# Navigate to project
cd balerion-alpha

# Install dependencies
uv sync

# Activate virtual environment (optional, uv run handles this automatically)
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
```

## Running Your First Experiment

```bash
# Run the SMA crossover example
uv run python experiments/sma_crossover/strategy.py
```

This will:
1. Load EURUSD data from balerion-data
2. Calculate SMA crossover signals  
3. Open an interactive Plotly chart in your browser
4. Save the chart as `sma_crossover_EURUSD_20_50.html`

## Backtesting a Strategy

```python
# In a Python script or Jupyter notebook
from backtesting import backtest_strategy_module
from experiments.sma_crossover import strategy as sma

results = backtest_strategy_module(
    sma,
    symbol='EURUSD',
    start_date='2025-01-01',
    end_date='2025-02-26',
    initial_capital=10000.0,
    fast_period=20,
    slow_period=50
)

# View results
print(results['portfolio'].stats())
```

## Creating a New Strategy

1. Create a new folder: `experiments/my_strategy/`
2. Create `strategy.py` with:
   - `calculate_signals()` function
   - `run_experiment()` function
3. Test your strategy visually first
4. Then backtest with vectorbt

See `experiments/sma_crossover/` for a complete example.

## Project Structure

```
experiments/       # Your strategy experiments (standalone scripts)
backtesting/       # Backtesting engine (vectorbt)
utils/             # Data loader and plotting utilities
notebooks/         # Jupyter notebooks for exploration
config/            # Configuration files
tests/             # Tests
```

## Key Functions

### Load Data
```python
from utils import load_data
df = load_data('EURUSD', start_date='2024-01-01')
```

### Plot Signals
```python
from utils import plot_candlestick_with_signals

fig = plot_candlestick_with_signals(
    df,
    title='My Strategy',
    buy_signals=df['buy_signal'],
    sell_signals=df['sell_signal'],
    indicators={'SMA': df['sma']}
)
fig.show()
```

### Backtest
```python
from backtesting import backtest_from_signals_df

portfolio = backtest_from_signals_df(
    df,
    buy_signal_col='buy_signal',
    sell_signal_col='sell_signal',
    initial_capital=10000.0
)
```

## Tips

- Always visualize signals before backtesting
- Start with simple strategies
- Test on different timeframes
- Use notebooks for exploration
- Document your findings

## Next Steps

1. Explore the SMA crossover example
2. Try different parameters
3. Create your own strategy
4. Backtest and analyze results
5. Iterate and improve
