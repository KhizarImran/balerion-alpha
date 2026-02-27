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

| Asset Class | Symbol | Parquet File | Native Timeframe |
|-------------|--------|-------------|-----------------|
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

# Load multiple symbols at once -> returns dict {symbol: DataFrame}
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
- `open` -> first bar of period
- `high` -> max of period
- `low` -> min of period
- `close` -> last bar of period
- `volume` -> sum of period
- `spread` -> mean of period (if present)
- `real_volume` -> sum of period (if present)
- Incomplete (NaN) periods are dropped automatically.

### Freq Strings for vectorbt

| Timeframe  | `freq=` string |
|------------|---------------|
| 1 minute   | `'1T'`        |
| 5 minutes  | `'5T'`        |
| 15 minutes | `'15T'`       |
| 1 hour     | `'1H'`        |
| 4 hours    | `'4H'`        |
| Daily      | `'1D'`        |

---

## FX Fundamentals for Backtesting

### Pip Values

A **pip** is the smallest standard price move for an FX pair.

| Pair   | Pip Size | Example: 30 pip SL as fraction |
|--------|----------|-------------------------------|
| EURUSD | 0.0001   | 30 * 0.0001 / 1.08 = 0.00278 |
| GBPUSD | 0.0001   | 30 * 0.0001 / 1.27 = 0.00236 |
| USDCAD | 0.0001   | 30 * 0.0001 / 1.36 = 0.00221 |
| EURGBP | 0.0001   | 30 * 0.0001 / 0.86 = 0.00349 |
| AUDNZD | 0.0001   | 30 * 0.0001 / 1.09 = 0.00275 |
| USDJPY | 0.01     | 30 * 0.01   / 150  = 0.00200 |
| XAUUSD | 0.01     | 30 * 0.01   / 2000 = 0.00015 |

**IMPORTANT**: vectorbt `sl_stop` and `tp_stop` are expressed as a **decimal fraction of entry price** (not pip counts). Always convert:

```python
def pips_to_frac(pips, pip_size, entry_price):
    """Convert pip count to vectorbt stop fraction."""
    return (pips * pip_size) / entry_price

PIP = 0.0001          # 4-decimal pairs (EURUSD, GBPUSD, EURGBP, USDCAD, AUDNZD)
PIP_JPY = 0.01        # 2-decimal pairs (USDJPY)
PIP_GOLD = 0.01       # XAUUSD

entry_price = close.mean()   # approximate; vectorbt uses actual entry price per trade

sl_frac = pips_to_frac(30, PIP, entry_price)    # 30 pip SL
tp_frac = pips_to_frac(60, PIP, entry_price)    # 60 pip TP (2:1 RR)
```

### Lot Sizes

| Lot Type | Units (base currency) | Typical Use |
|----------|-----------------------|-------------|
| Standard | 100,000 | Professional / institutional |
| Mini     | 10,000  | Retail traders |
| Micro    | 1,000   | Small accounts / testing |
| Nano     | 100     | Minimum position |

In vectorbt, `size` with `size_type='amount'` = number of **base currency units**:

```python
LOT      = 100_000   # 1 standard lot
MINI_LOT =  10_000   # 1 mini lot
MICRO_LOT =  1_000   # 1 micro lot
```

### FX Fees / Spread

FX brokers charge via the **bid-ask spread**. Model as a per-unit fee:

| Pair   | Typical Spread | Fee per unit (`size_type='amount'`) |
|--------|---------------|-------------------------------------|
| EURUSD | 0.5–1.5 pips  | 0.00005–0.00015                     |
| GBPUSD | 1–2 pips      | 0.0001–0.0002                       |
| USDJPY | 0.5–1.5 pips  | 0.005–0.015                         |
| XAUUSD | 3–8 pips      | 0.03–0.08                           |

When using `size_type='percent'`, model as a round-trip fraction of price:

```python
# 1 pip spread on EURUSD at ~1.08
fees = (1 * 0.0001) / 1.08   # ~0.000093; use 0.0001 as a conservative estimate
```

---

## VectorBT Simulation Modes

### 1. from_signals (Signal-Based) — Most Common for FX

Entry/exit boolean arrays. VectorBT processes signals sequentially — after entry, waits for exit before next entry (unless `accumulate=True`).

```python
import vectorbt as vbt

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
    sl_stop=0.00278,            # 30 pip SL on EURUSD
    tp_stop=0.00556,            # 60 pip TP on EURUSD
    accumulate=False,
)
```

### 2. from_orders (Order-Based) — Direct Orders

Provide explicit order arrays. Use for rebalancing or fixed-allocation strategies.

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

### 3. from_order_func (Custom Callback) — Most Powerful

Numba-compiled function called at each bar with full portfolio state access. Use for complex logic requiring dynamic position sizing or multi-condition order management.

### 4. from_holding (Buy-and-Hold Benchmark)

```python
pf_benchmark = vbt.Portfolio.from_holding(close, init_cash=10_000, fees=0.0001, freq='1H')
```

---

## Stop Loss & Take Profit — FX Complete Reference

### Fixed SL/TP in Pips

```python
import vectorbt as vbt

PIP = 0.0001
SL_PIPS = 30
TP_PIPS = 60
entry_price_approx = close.mean()

sl_frac = (SL_PIPS * PIP) / entry_price_approx
tp_frac = (TP_PIPS * PIP) / entry_price_approx

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
```

### Trailing Stop Loss

```python
# sl_trail: follows the highest price since entry, exits on pullback
pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    sl_trail=(20 * PIP) / close.mean(),   # 20 pip trailing stop
    init_cash=10_000,
    size=10_000,
    size_type='amount',
    fees=0.00005,
    freq='1H',
)
```

### Per-Signal Dynamic SL/TP (ATR-Based)

Pass a pandas Series of the same index to set a different stop per bar:

```python
import pandas as pd

# Manual ATR (no external dependency)
def calc_atr(high, low, close, period=14):
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low  - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()

atr = calc_atr(df['high'], df['low'], df['close'])
sl_series = (atr * 1.5) / close    # 1.5x ATR as fraction of price
tp_series = (atr * 3.0) / close    # 3x ATR (2:1 RR)

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

### SL/TP Parameters Reference

| Parameter    | Description |
|--------------|-------------|
| `sl_stop`    | Fixed stop loss — exits if price falls this fraction below entry |
| `tp_stop`    | Fixed take profit — exits if price rises this fraction above entry |
| `sl_trail`   | Trailing stop — tracks high watermark since entry, exits on pullback |
| `ts_stop`    | Alias for `sl_trail` in some vectorbt versions |

**Notes:**
- For `direction='longonly'`: SL fires below entry, TP fires above.
- For `direction='shortonly'`: SL fires above entry, TP fires below.
- SL/TP always override signal exits — whichever fires first wins.
- `stop_exit_price='close'` (default) fills at bar close; `'stop'` fills at exact stop level.

---

## FX Lot Sizing — Position Sizing Approaches

### Approach 1: Fixed Lot Size (`size_type='amount'`)

Most realistic for FX — each trade uses a fixed number of base currency units:

```python
pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    size=10_000,           # 1 mini lot
    size_type='amount',
    init_cash=10_000,
    fees=0.00005,
    sl_stop=sl_frac,
    tp_stop=tp_frac,
    freq='1H',
)
```

### Approach 2: Risk-Based Lot Sizing (% of account per trade)

The professional approach — size each trade so that the SL distance equals a fixed % of account equity:

```python
def compute_units(account_equity, risk_pct, sl_pips, pip_value_per_std_lot=10.0, lot_units=100_000):
    """
    Size a trade to risk a fixed % of account.

    Args:
        account_equity:      Account balance in USD
        risk_pct:            Fraction of account to risk (0.01 = 1%)
        sl_pips:             Stop loss distance in pips
        pip_value_per_std_lot: USD value of 1 pip on 1 standard lot
                              EURUSD ~$10, GBPUSD ~$10, USDJPY ~$9
        lot_units:           Units per standard lot (default 100,000)
    Returns:
        units to trade (float)
    """
    risk_amount = account_equity * risk_pct
    lots = risk_amount / (sl_pips * pip_value_per_std_lot)
    return lots * lot_units

# Example: $10,000 account, 1% risk, 30 pip SL on EURUSD
units = compute_units(10_000, 0.01, 30, pip_value_per_std_lot=10.0)
print(f"{units:.0f} units = {units/100_000:.3f} standard lots")
# => 3,333 units = 0.033 standard lots
```

### Approach 3: Percent of Portfolio (`size_type='percent'`)

```python
pf = vbt.Portfolio.from_signals(
    close, entries, exits,
    size=0.95,             # 95% of current equity
    size_type='percent',
    init_cash=10_000,
    fees=0.0001,
    freq='1H',
)
```

### Position Sizing Reference Table

| `size_type`         | `size=` meaning | Best For |
|---------------------|-----------------|----------|
| `'amount'`          | Fixed units (e.g. 10,000 = 1 mini lot) | Realistic FX simulation |
| `'value'`           | Fixed USD per trade | Fixed exposure |
| `'percent'`         | Fraction of current portfolio equity | Simple risk-adjusted |
| `'targetpercent'`   | Rebalance to target portfolio weight | Multi-asset portfolios |

---

## Parameter Optimization

### Method 1: Broadcasting (Vectorized — VectorBT's Killer Feature)

Test many indicator combinations simultaneously without loops:

```python
import numpy as np
import vectorbt as vbt

# Test 10 x 7 = 70 EMA pairs at once
fast_windows = np.arange(5, 15)        # 5–14
slow_windows = np.arange(20, 55, 5)   # 20, 25, 30, ..., 50

ema_fast = vbt.MA.run(close, window=fast_windows, ewm=True)
ema_slow  = vbt.MA.run(close, window=slow_windows, ewm=True)

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
print(f"Best fast: {best_idx[0]}, slow: {best_idx[1]}, return: {total_returns.max():.2%}")
```

### Method 2: Loop-Based (SL/TP Grid Search)

Use for grid searches over stop levels or risk:reward ratios:

```python
import pandas as pd

PIP = 0.0001
results = []

sl_pip_grid = [15, 20, 25, 30, 40, 50]
rr_grid     = [1.5, 2.0, 2.5, 3.0]

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
            'sl_pips':      sl_pips,
            'rr':           rr,
            'tp_pips':      tp_pips,
            'total_return': pf.total_return(),
            'sharpe':       pf.sharpe_ratio(),
            'max_dd':       pf.max_drawdown(),
            'win_rate':     pf.trades.win_rate(),
            'trades':       pf.trades.count(),
            'profit_factor':pf.trades.profit_factor(),
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
pf.total_return() * 100         # Total return %
pf.sharpe_ratio()               # Sharpe ratio (annualized)
pf.sortino_ratio()              # Sortino ratio
pf.calmar_ratio()               # Calmar ratio (return / max DD)
pf.max_drawdown() * 100         # Maximum drawdown %
pf.trades.win_rate() * 100      # Win rate %
pf.trades.profit_factor()       # Profit factor (gross wins / gross losses)
pf.trades.count()               # Total closed trades
pf.trades.records_readable      # Full trade log as DataFrame
```

### Pip P&L per Trade

```python
PIP = 0.0001   # adjust per pair

trades = pf.trades.records_readable.copy()
trades['pnl_pips'] = (trades['Exit Price'] - trades['Entry Price']) / PIP

print(trades[['Entry Time', 'Exit Time', 'Entry Price', 'Exit Price', 'pnl_pips', 'PnL']].head(20))
print(f"\nAvg pip gain:   {trades['pnl_pips'].mean():.1f} pips")
print(f"Avg win (pips): {trades.loc[trades['pnl_pips'] > 0, 'pnl_pips'].mean():.1f}")
print(f"Avg loss (pips):{trades.loc[trades['pnl_pips'] < 0, 'pnl_pips'].mean():.1f}")
```

### Export Trade Log

```python
pf.trades.records_readable.to_csv('reports/trades.csv', index=False)
```

---

## Direction — Long, Short, Both

| Direction  | `direction=`   | Behavior |
|------------|----------------|----------|
| Long Only  | `'longonly'`   | Buy on entry, sell on exit (default) |
| Short Only | `'shortonly'`  | Short on entry, cover on exit |
| Both       | `'both'`       | Flip between long and short |

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
# Note: direction= is ignored when short_entries/short_exits are provided
```

### Compare Long-Only vs Short-Only vs Both

```python
common = dict(init_cash=10_000, size=10_000, size_type='amount', fees=0.00005, freq='1H')

pf_long  = vbt.Portfolio.from_signals(close, entries=LE, exits=LX, direction='longonly',  **common)
pf_short = vbt.Portfolio.from_signals(close, short_entries=SE, short_exits=SX, direction='shortonly', **common)
pf_both  = vbt.Portfolio.from_signals(close, entries=LE, exits=LX, short_entries=SE, short_exits=SX, **common)

stats = pd.concat([
    pf_long.stats().to_frame('Long Only'),
    pf_short.stats().to_frame('Short Only'),
    pf_both.stats().to_frame('Both'),
], axis=1)
print(stats)
```

---

## Key Parameters Reference

| Parameter          | Default      | Description |
|--------------------|--------------|-------------|
| `init_cash`        | 100          | Starting capital in account currency (USD) |
| `fees`             | 0            | Fee per unit traded; model spread as `pip_size / price` |
| `fixed_fees`       | 0            | Flat fee per trade in account currency |
| `slippage`         | 0            | Additional slippage as fraction of price |
| `size`             | `np.inf`     | Position size (interpretation depends on `size_type`) |
| `size_type`        | `'amount'`   | How to interpret `size` |
| `direction`        | `'longonly'` | Trade direction |
| `freq`             | auto         | Data frequency string (`'1T'`, `'1H'`, `'4H'`, `'1D'`) |
| `accumulate`       | `False`      | Allow pyramiding into existing positions |
| `sl_stop`          | `None`       | Stop loss as fraction of entry price |
| `tp_stop`          | `None`       | Take profit as fraction of entry price |
| `sl_trail`         | `None`       | Trailing stop as fraction (tracks high watermark) |
| `min_size`         | 0            | Minimum order size |
| `size_granularity` | `None`       | Round size to this increment |
| `stop_exit_price`  | `'close'`    | Fill price for SL/TP: `'close'` or `'stop'` |

---

## Plotting

### Built-in Plots

```python
fig = pf.plot()
fig.show()

fig = pf.plot(subplots=['value', 'underwater'])
fig.show()

fig = pf.plot(subplots=['value', 'underwater', 'cum_returns', 'trades'])
fig.show()
```

### Available Subplots

```python
list(pf.subplots.keys())
# Common: 'value', 'cum_returns', 'underwater', 'drawdowns', 'trades', 'orders', 'net_exposure', 'cash'
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

### Save to Reports

```python
reports_dir = project_root / 'reports'
reports_dir.mkdir(exist_ok=True)
fig.write_html(str(reports_dir / 'my_strategy_EURUSD_1H.html'))
```

### Pip P&L Distribution Chart

```python
import plotly.graph_objects as go

trades = pf.trades.records_readable.copy()
trades['pnl_pips'] = (trades['Exit Price'] - trades['Entry Price']) / PIP

colors = trades['pnl_pips'].apply(lambda x: 'green' if x > 0 else 'red')
fig = go.Figure()
fig.add_trace(go.Histogram(x=trades['pnl_pips'], nbinsx=40, name='Pip P&L'))
fig.update_layout(
    title='Trade P&L Distribution (pips)',
    xaxis_title='Pips',
    yaxis_title='Count',
    template='plotly_dark',
)
fig.show()
```

### Equity vs Buy-and-Hold Chart

```python
from plotly.subplots import make_subplots

pf_bh = vbt.Portfolio.from_holding(close, init_cash=10_000, fees=0.0001, freq='1H')

cum_strat = pf.value() / pf.value().iloc[0] - 1
cum_bh    = pf_bh.value() / pf_bh.value().iloc[0] - 1
drawdown  = cum_strat / cum_strat.cummax() - 1

fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    row_heights=[0.65, 0.35], vertical_spacing=0.07)
fig.add_trace(go.Scatter(x=cum_strat.index, y=cum_strat, name='Strategy'), row=1, col=1)
fig.add_trace(go.Scatter(x=cum_bh.index,    y=cum_bh,    name='Buy & Hold'), row=1, col=1)
fig.add_trace(go.Scatter(x=drawdown.index,  y=drawdown,  name='Drawdown'), row=2, col=1)
fig.update_yaxes(tickformat='.1%', row=1, col=1)
fig.update_yaxes(tickformat='.1%', row=2, col=1)
fig.update_layout(title='Strategy vs Buy & Hold', template='plotly_dark')
fig.show()
```

---

## Common Patterns

### Avoid Lookahead Bias

Always use `.shift(1)` when referencing a computed value on the bar it is calculated:

```python
# WRONG — signal generated using current bar's indicator value
entries_bad  = close > sma

# CORRECT — signal fires on the bar AFTER the condition is met
entries_good = close > sma.shift(1)
```

### Signal Deduplication

Prevent stacking consecutive entries of the same direction:

```python
entries = entries & ~entries.shift(1).fillna(False)
exits   = exits   & ~exits.shift(1).fillna(False)
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
import numpy as np

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
        'max_consecutive_wins':   max(cons_wins)   if cons_wins   else 0,
        'max_consecutive_losses': max(cons_losses) if cons_losses else 0,
        'avg_consecutive_wins':   round(np.mean(cons_wins),   1) if cons_wins   else 0,
        'avg_consecutive_losses': round(np.mean(cons_losses), 1) if cons_losses else 0,
        'avg_win_pips':  round(trades_df.loc[trades_df['pnl_pips'] > 0, 'pnl_pips'].mean(), 1),
        'avg_loss_pips': round(trades_df.loc[trades_df['pnl_pips'] < 0, 'pnl_pips'].mean(), 1),
    }
```

---

## SL/TP Optimization Heatmap

```python
import plotly.graph_objects as go
import numpy as np

# After running loop-based grid search into results_df
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
