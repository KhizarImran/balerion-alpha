# Failed OB + FVG — Trading Logic

## Overview

A **Failed Order Block** is a prior demand or supply zone that price blew through with displacement momentum, failing to hold as expected. The **Fair Value Gap (FVG)** left by that same displacement move marks the imbalance price is likely to retrace into. When price returns to that FVG, it provides a high-probability entry in the direction of the original displacement.

---

## SELL Setup

### Conditions (all must be true)

1. **Bullish OB at a major swing low**
   - Identify a confirmed swing low using `smc.swing_highs_lows(swing_length=10)` — major pivots only, 10 bars either side required for confirmation.
   - Walk backwards from the swing low to find the **last bearish candle** (close < open) before it. That candle is the **bullish order block** — its `[low, high]` is the OB zone.

2. **OB gets 100% mitigated (close through)**
   - A later candle **closes at or below the OB's low** within **120 bars (5 days)** of the swing low.
   - This confirms institutional demand at that level completely failed.

3. **Bearish FVG during the mitigation leg**
   - `smc.fvg()` detects a **bearish FVG** (FVG == -1) that formed between the OB bar and the mitigation bar.
   - The FVG zone must **overlap** the OB zone (zones share a price range).

4. **Price retraces up into the FVG**
   - After mitigation, price pulls back up and a candle's **high >= FVG bottom**.
   - That bar is the **entry bar**.

### Trade Parameters

| Parameter | Value |
|---|---|
| Entry | Close of the entry bar |
| Stop Loss | OB top (high of the original bullish OB candle) |
| Take Profit | Entry − 2 × (SL − Entry) &nbsp; *(2R below entry)* |

---

## BUY Setup (mirror)

1. **Bearish OB at a major swing high**
   - Last bullish candle (close > open) before the swing high.

2. **OB gets 100% mitigated**
   - A later candle **closes at or above the OB's high** within 120 bars.

3. **Bullish FVG during the mitigation leg**
   - Bullish FVG (FVG == 1) formed between OB bar and mitigation bar, overlapping OB zone.

4. **Price retraces down into the FVG**
   - Candle's **low <= FVG top** after mitigation → entry bar.

### Trade Parameters

| Parameter | Value |
|---|---|
| Entry | Close of the entry bar |
| Stop Loss | OB bottom (low of the original bearish OB candle) |
| Take Profit | Entry + 2 × (Entry − SL) &nbsp; *(2R above entry)* |

---

## Summary Table

| Element | SELL | BUY |
|---|---|---|
| Pivot type | Major swing low | Major swing high |
| OB candle | Last bearish candle before pivot | Last bullish candle before pivot |
| Mitigation | Close <= OB low within 120 bars | Close >= OB high within 120 bars |
| FVG direction | Bearish FVG overlapping OB | Bullish FVG overlapping OB |
| Entry trigger | High >= FVG bottom (retrace up) | Low <= FVG top (retrace down) |
| Stop Loss | OB high | OB low |
| Take Profit | 2R below entry | 2R above entry |

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| `SWING_LENGTH` | 10 | Bars required either side of pivot for confirmation — major highs/lows only |
| Mitigation window | 120 bars | Maximum bars after pivot for OB to be fully closed through |
| Risk per trade | 1% | Fraction of account equity risked per trade |
| Leverage | 100:1 | FX retail leverage |
| RR | 2:1 | Fixed take profit at 2× the SL distance |

---

## Risk Management

- **Position sizing**: lot size calculated so that hitting the SL costs exactly 1% of account equity.
  - `sl_pips = sl_distance / PIP`
  - `lots = (account * 0.01) / (sl_pips * pip_value_per_lot)`
- **Stop Loss**: placed at the OB candle's high (SELL) or low (BUY) — the level that, if reclaimed, invalidates the setup entirely.
- **Take Profit**: fixed 2R — no trailing, no partial exits.
- **No re-entry**: a setup is consumed on first touch of the FVG.

---

## Chart Visuals

| Element | Colour | Description |
|---|---|---|
| OB rectangle | Blue (SELL) / Red (BUY) | Spans OB formation bar → mitigation bar |
| FVG rectangle | Yellow-green dashed (SELL) / Orange dashed (BUY) | Spans FVG formation bar → entry touch bar |
| Entry marker | Red triangle-down (SELL) / Green triangle-up (BUY) | Entry bar |
| SL line | Red dotted | Horizontal from entry bar |
| TP line | Green dotted | Horizontal from entry bar |
