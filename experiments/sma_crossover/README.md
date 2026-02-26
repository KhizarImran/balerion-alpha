# SMA Crossover Strategy

## Overview
Classic simple moving average crossover strategy that generates buy/sell signals based on the crossing of two SMAs.

## Strategy Logic
- **Buy Signal**: Fast SMA crosses above slow SMA (bullish crossover)
- **Sell Signal**: Fast SMA crosses below slow SMA (bearish crossover)

## Parameters
- `fast_period`: Period for fast SMA (default: 20)
- `slow_period`: Period for slow SMA (default: 50)

## Usage

```python
# Run the experiment
python strategy.py
```

## Customization

Edit the parameters in `strategy.py`:

```python
df, fig = run_experiment(
    symbol='EURUSD',
    asset_type='fx',
    start_date='2025-01-01',
    end_date='2025-02-26',
    fast_period=20,
    slow_period=50,
    show_chart=True,
    save_chart=True
)
```

## Output
- Interactive HTML chart with candlesticks, SMAs, and buy/sell signals
- Signal statistics printed to console
