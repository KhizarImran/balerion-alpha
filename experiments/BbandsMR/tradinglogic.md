# BbandsMR — Bollinger Bands Mean Reversion

## Overview

BbandsMR is a mean reversion strategy on **EURGBP 1H**. It fades price extremes
by entering when the close moves outside the Bollinger Bands, expecting price to
revert back toward the midline (SMA). An ADX filter suppresses signals during
trending conditions, preventing the strategy from fading genuine breakouts.

---

## Instruments

| Field       | Value                |
|-------------|----------------------|
| Symbol      | EURGBP               |
| Timeframe   | 1H                   |
| Session     | 24h (no filter)      |
| Data source | Dukascopy 1H parquet |

---

## Indicators

| Indicator      | Parameters                     | Purpose                        |
|----------------|--------------------------------|--------------------------------|
| Bollinger Bands | 20-period, 2.0 std dev, close | Entry trigger + TP target      |
| ATR            | 14-period                      | SL distance                    |
| ADX            | 14-period                      | Trend filter (avoid breakouts) |

---

## Entry Rules

### Long Entry
1. Close of current bar is **below the lower Bollinger Band**
2. ADX < 25 (market is not trending strongly)
3. No position currently open

### Short Entry
1. Close of current bar is **above the upper Bollinger Band**
2. ADX < 25 (market is not trending strongly)
3. No position currently open

Entry price = close of the signal bar.

---

## Exit Rules

### Take Profit
- **TP = BB midline (SMA 20) at the signal bar**
- Price is expected to mean-revert from the extreme back to the moving average center.

### Stop Loss
- **SL = entry close ± (1.5 × ATR)**
- Long:  SL = close − 1.5 × ATR  (below entry)
- Short: SL = close + 1.5 × ATR  (above entry)
- Dynamic — adapts to current volatility regime.

### Minimum RR Filter
- Trade is only taken if `(TP distance) / (SL distance) >= 0.5`
- This rejects setups where the midline is too close to generate meaningful reward
  relative to the risk (e.g. entering just barely outside the band).

### No Time Exit
- Exits are SL or TP only — no time-based close.

---

## Deduplication

A new signal is not generated while a previous trade is still open. The
`detect_setups()` function forward-simulates each trade to its SL/TP before
resuming scanning. This prevents stacking multiple open positions.

---

## Risk Management (Backtest)

| Parameter       | Value  |
|-----------------|--------|
| Account size    | $10,000 |
| Risk per trade  | 1%     |
| Fixed fee       | $5.00 / trade |
| Leverage        | 100×   |

Lot size is calculated from:

```
risk_$     = account × risk_per_trade
sl_pips    = SL_distance / pip_size
lots       = risk_$ / (sl_pips × pip_value_per_lot)
```

EURGBP is GBP-quoted — the vectorbt quote currency correction applies:
`units = (lots × STD_LOT) / entry_price`

---

## Chart Layout

| Row | Contents                                                   |
|-----|------------------------------------------------------------|
| 1   | Candlesticks + BB upper/mid/lower + entry/TP/SL markers    |
| 2   | ADX with ADX=25 threshold dashed line                      |
| 3   | ATR                                                        |

- Green triangle-up = long entry
- Red triangle-down = short entry
- Yellow star = TP hit
- Red X = SL hit
- Dotted lines connect entry to exit

---

## Files

| File           | Purpose                                          |
|----------------|--------------------------------------------------|
| `strategy.py`  | Signal detection + Plotly chart                  |
| `backtest.py`  | vectorbt simulation + MLflow logging (planned)   |
| `tradinglogic.md` | This document                                 |
