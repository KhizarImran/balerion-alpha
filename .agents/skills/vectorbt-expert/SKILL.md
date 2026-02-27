---
name: vectorbt-expert
description: VectorBT backtesting expert. Use when user asks to backtest strategies, create entry/exit signals, analyze portfolio performance, optimize parameters, fetch historical data, use VectorBT/vectorbt, compare strategies, position sizing, equity curves, drawdown charts, or trade analysis. Also triggers for FX lot sizes, pip-based stop loss/take profit, spread costs, or FX strategy backtesting.
user-invocable: false
---

# VectorBT Backtesting Expert Skill

## Environment

- Python with vectorbt, pandas, numpy, plotly
- Data source: `utils.data_loader` (balerion-data Parquet files — sibling repo at `../balerion-data/`)
- All scripts run via `uv run python` from the project root `balerion-alpha/`
- Reports saved to `reports/` directory (gitignored — never commit)
- Never use icons/emojis in code or output

## Project Path Bootstrap (Required in Every Script)

```python
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))
```

---

## Data Loading

### Data Source: balerion-data

All market data lives in the sibling repository `../balerion-data/` as Parquet files collected via MetaTrader 5. The file naming convention is `{symbol_lower}_{timeframe}.parquet`.

```
../balerion-data/data/
    fx/
        audnzd_1m.parquet
        eurgbp_1m.parquet
        eurusd_1m.parquet
        gbpusd_1m.parquet
        usdcad_1m.parquet
        usdjpy_1m.parquet
    indices/
        us30_1m.parquet
        xauusd_1m.parquet
```

All files are **1-minute native OHLCV** data. Resample to higher timeframes as needed.

### Available FX Pairs & Instruments

| Asset Class | Symbols | Parquet File | Native Timeframe |
|-------------|---------|-------------|-----------------|
| FX Pairs | EURUSD | `fx/eurusd_1m.parquet` | 1 minute |
| FX Pairs | USDJPY | `fx/usdjpy_1m.parquet` | 1 minute |
| FX Pairs | GBPUSD | `fx/gbpusd_1m.parquet` | 1 minute |
| FX Pairs | EURGBP | `fx/eurgbp_1m.parquet` | 1 minute |
| FX Pairs | USDCAD | `fx/usdcad_1m.parquet` | 1 minute |
| FX Pairs | AUDNZD | `fx/audnzd_1m.parquet` | 1 minute |
| Indices | US30 | `indices/us30_1m.parquet` | 1 minute |
| Indices | XAUUSD | `indices/xauusd_1m.parquet` | 1 minute |

OHLCV columns are lowercase: `open`, `high`, `low`, `close`, `volume`.
The DataFrame index is a `DatetimeIndex` named `timestamp`.

### Loading Data

```python
from utils import load_data, DataLoader

# --- Quick one-liner (most common) ---
df = load_data('EURUSD', asset_type='fx', start_date='2024-01-01')
df = load_data('EURUSD', asset_type='fx', start_date='2024-01-01', end_date='2024-12-31')
df = load_data('XAUUSD', asset_type='indices', start_date='2024-01-01')

# --- Full DataLoader class ---
loader = DataLoader()

# Load FX pair
df = loader.load_fx('EURUSD', start_date='2024-01-01', end_date='2025-01-01')

# Load index / commodity
df = loader.load_index('US30', start_date='2024-01-01')
df = loader.load_index('XAUUSD', start_date='2024-01-01')

# Load multiple symbols at once → returns dict {symbol: DataFrame}
data = loader.load_multiple(
    ['EURUSD', 'GBPUSD', 'USDJPY'],
    asset_type='fx',
    start_date='2024-01-01',
)
# Access: data['EURUSD'], data['GBPUSD'], etc.

# Discover available symbols
print(loader.get_available_fx_symbols())     # ['AUDNZD', 'EURGBP', 'EURUSD', ...]
print(loader.get_available_index_symbols())  # ['US30', 'XAUUSD']

# Inspect data coverage for a symbol
info = loader.get_data_info('EURUSD', asset_type='fx')
print(info)
# {'symbol': 'EURUSD', 'rows': ..., 'start_date': ..., 'end_date': ...,
#  'file_size_mb': ..., 'columns': ['open', 'high', 'low', 'close', 'volume']}
```

### Resampling to Higher Timeframes

All data is 1-minute native. Always resample before backtesting to avoid excessive bar counts:

```python
loader = DataLoader()
df_1m = loader.load_fx('EURUSD', start_date='2024-01-01')

df_5m  = loader.resample_ohlcv(df_1m, '5T')    #  5-minute bars
df_15m = loader.resample_ohlcv(df_1m, '15T')   # 15-minute bars
df_1h  = loader.resample_ohlcv(df_1m, '1H')    #  1-hour bars
df_4h  = loader.resample_ohlcv(df_1m, '4H')    #  4-hour bars
df_1d  = loader.resample_ohlcv(df_1m, '1D')    #  daily bars

close = df_1h['close']
high  = df_1h['high']
low   = df_1h['low']
```

The `resample_ohlcv` aggregation rules:
- `open` → first bar of period
- `high` → max of period
- `low` → min of period
- `close` → last bar of period
- `volume` → sum of period
- `spread` → mean of period (if present)
- `real_volume` → sum of period (if present)
- Incomplete (NaN) periods are dropped automatically.

### Freq Strings for vectorbt

| Timeframe | `freq=` string |
|-----------|---------------|
| 1 minute  | `'1T'`        |
| 5 minutes | `'5T'`        |
| 15 minutes| `'15T'`       |
| 1 hour    | `'1H'`        |
| 4 hours   | `'4H'`        |
| Daily     | `'1D'`        |

---

## FX Fundamentals for Backtesting

### Pip Values

A **pip** is the smallest standard price move for an FX pair.

| Pair      | Pip Size | Example: 50 pip SL |
|-----------|----------|--------------------|
| EURUSD    | 0.0001   | sl_stop = 50 * 0.0001 = 0.005 |
| GBPUSD    | 0.0001   | sl_stop = 0.005 |
| USDCAD    | 0.0001   | sl_stop = 0.005 |
| EURGBP    | 0.0001   | sl_stop = 0.005 |
| AUDNZD    | 0.0001   | sl_stop = 0.005 |
| USDJPY    | 0.01     | sl_stop = 50 * 0.01 = 0.5 |
| XAUUSD    | 0.01     | sl_stop = 50 * 0.01 = 0.5 |

**IMPORTANT**: vectorbt `sl_stop` and `tp_stop` are expressed as a **decimal fraction of price** (not pips directly). Convert pips to a fraction:

```python
# For a 4-decimal pair (EURUSD, GBPUSD, etc.)
PIP = 0.0001
SL_PIPS = 30
TP_PIPS = 60
entry_price = close.mean()  # approximate entry

sl_stop = (SL_PIPS * PIP) / entry_price   # e.g. 30 pips on 1.0800 = 0.00278
tp_stop = (TP_PIPS * PIP) / entry_price

# For a 2-decimal pair (USDJPY)
PIP_JPY = 0.01
SL_PIPS = 30
sl_stop = (SL_PIPS * PIP_JPY) / entry_price
```

### Lot Sizes

| Lot Type    | Units (base currency) | Typical Use |
|-------------|----------------------|-------------|
| Standard    | 100,000              | Professional / institutional |
| Mini        | 10,000               | Retail traders |
| Micro       | 1,000                | Small accounts / testing |
| Nano        | 100                  | Minimum position |

In vectorbt, "units" = the number of **base currency units** (contracts):

```python
# size_type="amount" with lot sizes
LOT = 100_000    # 1 standard lot
MINI_LOT = 10_000
MICRO_LOT = 1_000

pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    size=MICRO_LOT,          # Trade 1 micro lot per signal
    size_type='amount',
    init_cash=10_000,        # USD account
    fees=0.00005,            # ~0.5 pip spread on EURUSD (~$5 per standard lot)
    freq='1H',
)
```

### FX Fees / Spread

FX brokers charge via the **bid-ask spread** rather than commission. Typical values:

| Pair    | Typical Spread | As Decimal Fee (size_type='amount') |
|---------|---------------|--------------------------------------|
| EURUSD  | 0.5–1.5 pips  | 0.00005–0.00015 per unit             |
| GBPUSD  | 1–2 pips      | 0.0001–0.0002 per unit               |
| USDJPY  | 0.5–1.5 pips  | 0.005–0.015 per unit                 |
| XAUUSD  | 3–8 pips      | 0.03–0.08 per unit                   |

When using `size_type='percent'` or `size_type='value'`, model spread as a round-trip fraction:

```python
# 1 pip spread on EURUSD at price ~1.08, round-trip
fees = (1 * 0.0001) / 1.08   # ~0.000093 per side, or use 0.0001 as conservative estimate
```

---

## VectorBT Simulation Modes

### 1. from_signals (Signal-Based) — Most Common for FX

```python
import vectorbt as vbt
import numpy as np

pf = vbt.Portfolio.from_signals(
    close,                      # Price series
    entries,                    # Boolean Series — True = open long
    exits,                      # Boolean Series — True = close long
    init_cash=10_000,           # Account size in USD
    fees=0.00005,               # Spread cost per unit (0.5 pip on EURUSD)
    slippage=0.00002,           # Additional slippage
    size=10_000,                # 1 mini lot (10,000 units)
    size_type='amount',         # Fixed units per trade
    direction='longonly',       # longonly, shortonly, both
    freq='1H',                  # Data frequency
    sl_stop=0.00278,            # 30 pip SL on EURUSD (30 * 0.0001 / 1.08)
    tp_stop=0.00556,            # 60 pip TP on EURUSD (60 * 0.0001 / 1.08)
    accumulate=False,
)
```

### 2. from_orders (Rebalancing / Order-Based)

```python
pf = vbt.Portfolio.from_orders(
    close=close,
    size=0.95,                  # 95% of available cash
    size_type='percent',
    fees=0.0001,
    init_cash=10_000,
    freq='1H',
)
```

### 3. from_holding (Buy-and-Hold Benchmark)

```python
pf_benchmark = vbt.Portfolio.from_holding(close, init_cash=10_000, fees=0.0001)
```

---

## Stop Loss & Take Profit — FX Complete Reference

### Fixed SL/TP in Pips (Recommended for FX)

```python
import vectorbt as vbt

# Helper: convert pips to vectorbt sl_stop/tp_stop fraction
def pips_to_frac(pips, pip_size, entry_price):
    """Convert pip count to vectorbt stop fraction."""
    return (pips * pip_size) / entry_price

# --- EURUSD example ---
close = df['close']
entry_price_approx = close.mean()
PIP = 0.0001

sl_frac = pips_to_frac(30, PIP, entry_price_approx)   # 30 pip SL
tp_frac = pips_to_frac(60, PIP, entry_price_approx)   # 60 pip TP (2:1 RR)

pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    init_cash=10_000,
    size=10_000,               # 1 mini lot
    size_type='amount',
    fees=0.00005,
    sl_stop=sl_frac,
    tp_stop=tp_frac,
    freq='1H',
)
```

### Trailing Stop Loss

```python
# sl_trail: trails the highest price since entry, exits on pullback
pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    sl_trail=pips_to_frac(20, PIP, close.mean()),  # 20 pip trailing stop
    init_cash=10_000,
    size=10_000,
    size_type='amount',
    fees=0.00005,
    freq='1H',
)
```

### Per-Signal Dynamic SL/TP (Series-Based)

Pass a pandas Series of the same index to set a different SL/TP per bar:

```python
# ATR-based dynamic stop loss
import pandas_ta as pta

atr = pta.atr(df['high'], df['low'], df['close'], length=14)
sl_series = atr * 1.5 / close    # 1.5x ATR as fraction of price
tp_series = atr * 3.0 / close    # 3x ATR (2:1 RR)

pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    sl_stop=sl_series,
    tp_stop=tp_series,
    init_cash=10_000,
    size=10_000,
    size_type='amount',
    fees=0.00005,
    freq='1H',
)
```

### SL/TP Stop Parameters Reference

| Parameter  | Description |
|------------|-------------|
| `sl_stop`  | Fixed stop loss — exits if price falls this fraction below entry |
| `tp_stop`  | Fixed take profit — exits if price rises this fraction above entry |
| `sl_trail` | Trailing stop — tracks highest price since entry, exits on pullback |
| `ts_stop`  | Alias for `sl_trail` in some versions |

**Notes:**
- These work for `direction='longonly'`. For shorts, SL triggers above entry, TP below.
- SL/TP always override signal exits — whichever fires first wins.
- Use `stop_exit_price='close'` (default) or `'stop'` to choose fill price.

---

## FX Lot Sizing — Position Sizing Approaches

### Approach 1: Fixed Lot Size (amount)

Simple and realistic — each trade uses a fixed number of base currency units:

```python
# Trade 1 mini lot (10,000 units) every signal
pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    size=10_000,
    size_type='amount',
    init_cash=10_000,
    fees=0.00005,
    freq='1H',
)
```

### Approach 2: Risk-Based Lot Sizing (% risk per trade)

The professional approach — risk a fixed % of account per trade, sized by SL distance:

```python
import numpy as np

def compute_lot_size(account_equity, risk_pct, sl_pips, pip_value_per_lot, lot_units=100_000):
    """
    Compute lot size based on fixed % risk.

    Args:
        account_equity: Account balance in USD
        risk_pct: Risk per trade as decimal (0.01 = 1%)
        sl_pips: Stop loss distance in pips
        pip_value_per_lot: USD value of 1 pip for 1 standard lot
                           (for EURUSD ~$10, USDJPY ~$9, GBPUSD ~$10)
        lot_units: Units per standard lot (default 100,000)
    Returns:
        units: Number of base currency units to trade
    """
    risk_amount = account_equity * risk_pct
    lots = risk_amount / (sl_pips * pip_value_per_lot)
    units = lots * lot_units
    return units

# Example: 1% risk, 30 pip SL on EURUSD
account = 10_000
risk_pct = 0.01      # 1%
sl_pips = 30
pip_value = 10.0     # $10 per pip per standard lot on EURUSD

units = compute_lot_size(account, risk_pct, sl_pips, pip_value)
print(f"Trade size: {units:.0f} units ({units/100_000:.2f} standard lots)")
# => ~3,333 units = 0.03 standard lots
```

### Approach 3: Percent of Portfolio

Simple — invest a fraction of current equity:

```python
pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    size=0.95,              # 95% of available equity
    size_type='percent',
    init_cash=10_000,
    fees=0.0001,
    freq='1H',
)
```

### Position Sizing Reference Table

| `size_type` | `size=` meaning | Best For |
|-------------|-----------------|----------|
| `'amount'`  | Fixed units (e.g. 10,000 = 1 mini lot) | Realistic FX simulation |
| `'value'`   | Fixed cash per trade in account currency | Fixed USD exposure |
| `'percent'` | Fraction of current portfolio equity | Simple risk-adjusted |
| `'targetpercent'` | Rebalance to target weight | Multi-asset portfolios |

---

## Parameter Optimization

### Method 1: Broadcasting (Vectorized — VectorBT's Killer Feature)

Test thousands of SL/TP or indicator combinations simultaneously:

```python
import numpy as np

# Test 10 x 10 = 100 EMA pairs simultaneously
ema_fast = vbt.MA.run(close, window=np.arange(5, 15), ewm=True)
ema_slow = vbt.MA.run(close, window=np.arange(20, 55, 5), ewm=True)

entries = ema_fast.ma_crossed_above(ema_slow)
exits   = ema_fast.ma_crossed_below(ema_slow)

pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    init_cash=10_000,
    size=10_000,
    size_type='amount',
    fees=0.00005,
    freq='1H',
)

total_returns = pf.total_return()
best_idx = total_returns.idxmax()
print(f"Best fast EMA: {best_idx[0]}, slow EMA: {best_idx[1]}")
print(f"Best return: {total_returns.max():.2%}")
```

### Method 2: Loop-Based (for SL/TP Grid Search)

```python
results = []

sl_pip_grid = [15, 20, 25, 30, 40, 50]
rr_grid = [1.5, 2.0, 2.5, 3.0]   # Risk-reward ratios

for sl_pips in sl_pip_grid:
    for rr in rr_grid:
        tp_pips = sl_pips * rr
        sl_frac = (sl_pips * PIP) / close.mean()
        tp_frac = (tp_pips * PIP) / close.mean()

        pf = vbt.Portfolio.from_signals(
            close, entries, exits,
            init_cash=10_000,
            size=10_000,
            size_type='amount',
            fees=0.00005,
            sl_stop=sl_frac,
            tp_stop=tp_frac,
            freq='1H',
        )

        results.append({
            'sl_pips': sl_pips,
            'rr': rr,
            'tp_pips': tp_pips,
            'total_return': pf.total_return(),
            'sharpe': pf.sharpe_ratio(),
            'max_dd': pf.max_drawdown(),
            'win_rate': pf.trades.win_rate(),
            'trades': pf.trades.count(),
            'profit_factor': pf.trades.profit_factor(),
        })

results_df = pd.DataFrame(results)
best = results_df.loc[results_df['total_return'].idxmax()]
print(f"Best SL: {best['sl_pips']} pips, RR: {best['rr']:.1f}")
print(f"Return: {best['total_return']:.2%}, Sharpe: {best['sharpe']:.2f}")
```

---

## Performance Analysis

### Full Stats

```python
pf.stats()   # Complete performance summary
```

### FX-Relevant Metrics

```python
pf.total_return() * 100             # Total return %
pf.sharpe_ratio()                   # Sharpe ratio (annualized)
pf.sortino_ratio()                  # Sortino ratio
pf.calmar_ratio()                   # Calmar ratio (return / max DD)
pf.max_drawdown() * 100             # Maximum drawdown %
pf.trades.win_rate() * 100          # Win rate %
pf.trades.profit_factor()           # Profit factor (gross wins / gross losses)
pf.trades.count()                   # Total closed trades
pf.trades.records_readable          # Full trade log DataFrame
```

### Pip P&L per Trade

```python
trades = pf.trades.records_readable.copy()
trades['pnl_pips'] = (trades['Exit Price'] - trades['Entry Price']) / PIP
print(trades[['Entry Time', 'Exit Time', 'Entry Price', 'Exit Price', 'pnl_pips', 'PnL']].head(20))
print(f"\nAvg pip gain per trade: {trades['pnl_pips'].mean():.1f} pips")
print(f"Avg winning trade: {trades.loc[trades['pnl_pips'] > 0, 'pnl_pips'].mean():.1f} pips")
print(f"Avg losing trade:  {trades.loc[trades['pnl_pips'] < 0, 'pnl_pips'].mean():.1f} pips")
```

### Export Trade Log

```python
pf.trades.records_readable.to_csv('reports/trades.csv', index=False)
```

---

## Direction — Long, Short, Both

| Direction    | `direction=` | Behavior |
|--------------|-------------|----------|
| Long Only    | `'longonly'` | Buy on entry, sell on exit (default) |
| Short Only   | `'shortonly'` | Short on entry, cover on exit |
| Both         | `'both'`    | Flip between long and short |

### Long + Short Simultaneously

```python
pf = vbt.Portfolio.from_signals(
    close,
    entries=long_entries,
    exits=long_exits,
    short_entries=short_entries,
    short_exits=short_exits,
    init_cash=10_000,
    size=10_000,
    size_type='amount',
    fees=0.00005,
    sl_stop=sl_frac,
    tp_stop=tp_frac,
    freq='1H',
)
```

---

## Key Parameters Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `init_cash` | 100 | Starting capital in account currency (USD) |
| `fees` | 0 | Fee per unit traded (model spread as `pip_size / price`) |
| `fixed_fees` | 0 | Flat fee per trade in account currency |
| `slippage` | 0 | Additional slippage as fraction of price |
| `size` | np.inf | Position size (interpretation depends on `size_type`) |
| `size_type` | `'amount'` | How to interpret `size` |
| `direction` | `'longonly'` | Trade direction |
| `freq` | auto | Data frequency string (`'1T'`, `'1H'`, `'4H'`, `'1D'`) |
| `accumulate` | False | Allow pyramiding into existing positions |
| `sl_stop` | None | Stop loss as fraction of entry price |
| `tp_stop` | None | Take profit as fraction of entry price |
| `sl_trail` | None | Trailing stop as fraction (tracks high watermark) |
| `min_size` | 0 | Minimum order size |
| `size_granularity` | None | Round size to this increment |
| `stop_exit_price` | `'close'` | Fill price for SL/TP: `'close'` or `'stop'` |

## Plotting

### Built-in Plots

```python
fig = pf.plot()
fig.show()

fig = pf.plot(subplots=['value', 'underwater'])           # Equity + drawdown
fig.show()

fig = pf.plot(subplots=['value', 'underwater', 'cum_returns', 'trades'])
fig.show()
```

### Save to Reports

```python
from utils import save_plot

save_plot(fig, 'reports/my_strategy_EURUSD_1H.html')
```

### Full 7-Panel Plot

```python
fig = pf.plot(
    subplots=['value', 'underwater', 'drawdowns', 'orders', 'trades', 'net_exposure', 'cash'],
    make_subplots_kwargs=dict(
        rows=7, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.25, 0.12, 0.12, 0.16, 0.12, 0.12, 0.11],
    ),
    template='plotly_dark',
    title='FX Strategy Backtest Results',
)
fig.show()
```

### Pip P&L Distribution Chart

```python
import plotly.graph_objects as go

trades = pf.trades.records_readable.copy()
trades['pnl_pips'] = (trades['Exit Price'] - trades['Entry Price']) / PIP

fig = go.Figure()
fig.add_trace(go.Histogram(
    x=trades['pnl_pips'],
    nbinsx=40,
    marker_color=trades['pnl_pips'].apply(lambda x: 'green' if x > 0 else 'red'),
    name='Pip P&L Distribution',
))
fig.update_layout(
    title='Trade P&L Distribution (pips)',
    xaxis_title='Pips',
    yaxis_title='Count',
    template='plotly_dark',
)
fig.show()
```

---

## Common Patterns

### Avoid Lookahead Bias

Always use `.shift(1)` when comparing current bar value to a previous computation:

```python
# WRONG — uses current bar's SMA to generate a signal on the same bar
entries_bad = close > sma

# CORRECT — signal fires on the bar AFTER the condition is met
entries_good = close > sma.shift(1)
```

### Signal Deduplication (prevent stacking entries)

```python
# Ensure no duplicate consecutive entries/exits
entries = entries & ~entries.shift(1).fillna(False)
exits   = exits & ~exits.shift(1).fillna(False)
```

### Save / Load Portfolio

```python
pf.save('reports/my_backtest.pkl')
pf_loaded = vbt.Portfolio.load('reports/my_backtest.pkl')
```

### Benchmark Comparison (Buy and Hold)

```python
pf_bh = vbt.Portfolio.from_holding(close, init_cash=10_000, fees=0.0001, freq='1H')

stats = pd.concat([
    pf.stats().to_frame('Strategy'),
    pf_bh.stats().to_frame('Buy & Hold'),
], axis=1)
print(stats)
```

---

## Consecutive Wins/Losses Analysis

```python
def analyze_streaks(pf, pip_size=0.0001):
    trades_df = pf.trades.records_readable.copy()
    if len(trades_df) == 0:
        return {}

    trades_df['pnl_pips'] = (trades_df['Exit Price'] - trades_df['Entry Price']) / pip_size
    wins = (trades_df['pnl_pips'] > 0).tolist()

    cons_wins, cons_losses = [], []
    cw, cl = 0, 0
    for w in wins:
        if w:
            if cl > 0: cons_losses.append(cl); cl = 0
            cw += 1
        else:
            if cw > 0: cons_wins.append(cw); cw = 0
            cl += 1
    if cw > 0: cons_wins.append(cw)
    if cl > 0: cons_losses.append(cl)

    return {
        'max_consecutive_wins': max(cons_wins) if cons_wins else 0,
        'max_consecutive_losses': max(cons_losses) if cons_losses else 0,
        'avg_consecutive_wins': round(np.mean(cons_wins), 1) if cons_wins else 0,
        'avg_consecutive_losses': round(np.mean(cons_losses), 1) if cons_losses else 0,
        'avg_win_pips': round(trades_df.loc[trades_df['pnl_pips'] > 0, 'pnl_pips'].mean(), 1),
        'avg_loss_pips': round(trades_df.loc[trades_df['pnl_pips'] < 0, 'pnl_pips'].mean(), 1),
    }
```

---

## SL/TP Optimization Heatmap

```python
import plotly.graph_objects as go
import numpy as np

# After running grid search into results_df
pivot = results_df.pivot_table(
    values='total_return', index='rr', columns='sl_pips', aggfunc='first'
)

fig = go.Figure(data=go.Heatmap(
    z=pivot.values * 100,
    x=pivot.columns.astype(str) + ' pip SL',
    y=pivot.index.astype(str) + ':1 RR',
    colorscale='RdYlGn',
    text=np.round(pivot.values * 100, 1),
    texttemplate='%{text}%',
    textfont={'size': 10},
    colorbar=dict(title='Return %'),
))
fig.update_layout(
    title='SL / Risk-Reward Optimization — Total Return',
    xaxis_title='Stop Loss (pips)',
    yaxis_title='Risk:Reward Ratio',
    template='plotly_dark',
    height=600,
    width=800,
)
fig.show()
```
