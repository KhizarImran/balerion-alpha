# Failed Order Block + Fair Value Gap — Trading Logic

## Overview

A **Failed Order Block** is a prior demand (bullish OB) or supply (bearish OB) zone that price blew straight through with momentum, failing to hold as expected. When a bullish OB at a swing low is fully breached, it signals that institutions did **not** defend it — a high-confidence bearish setup. The **Fair Value Gap (FVG)** left by the same displacement move is the imbalance price is likely to retrace into, providing the entry zone.

---

## Core Concept

When a bullish order block (demand zone at a swing low) gets 100% mitigated:

1. Price crashed through a zone where institutions had previously placed buy orders.
2. The violation confirms bearish control — no demand at that level.
3. The displacement that caused the violation leaves a **bearish FVG** — a 3-candle imbalance where price moved too fast for the market to fill both sides.
4. When price retraces up into that FVG, it is returning to an unmitigated imbalance that now sits inside or near a confirmed supply zone (the failed OB).
5. Entry is at the FVG retrace, stop above the original OB top (where the failed demand was), target 2R.

---

## Setup Structure (Bearish — SELL)

### Step 1 — Bullish OB forms at a swing low
`smc.ob()` identifies a bullish order block: the candle with the lowest low between the prior swing low and the bar where price first closes above that swing low. This is the demand zone that caused the prior up-move.

### Step 2 — OB gets fully mitigated (100% breached)
`smc.ob()` sets `breaker = True` on the OB when a subsequent candle's low wicks below the OB's bottom. This is the "failed demand" confirmation — price did not respect the zone.

`MitigatedIndex` = the bar where this first happened.

### Step 3 — Bearish FVG present
`smc.fvg()` detects a bearish FVG (3-candle imbalance) that formed after the OB bar. The displacement candles that drove through the OB typically leave this imbalance.

Proximity check: the FVG must sit within one OB height above the OB's top — confirming it is part of the same bearish displacement, not an unrelated imbalance.

### Step 4 — Zone armed
From the mitigation bar, an entry zone is armed:
- **Entry zone** = the FVG zone `[fvg_bot, fvg_top]`
- **SL** = `ob_top` (above the failed demand zone)
- Zone valid for `ZONE_TIMEOUT = 200` bars

### Step 5 — Entry trigger
When price retraces up into the FVG zone (`high >= fvg_bot`) → **SELL signal**

```
Entry  = close of signal bar
SL     = ob_top
TP     = entry − 2 × (SL − entry)  [2R]
```

---

## Bullish Setup — mirror

Failed bearish OB (supply zone at a swing high fully breached, `smc.ob() OB == -1`) + bullish FVG below → BUY on retrace down into the FVG zone. SL = ob_bot. TP = entry + 2R.

---

## Entry Rules Summary

| Element | Bearish Setup (SELL) | Bullish Setup (BUY) |
|---|---|---|
| OB type | Bullish OB (at swing low) | Bearish OB (at swing high) |
| Mitigation | Price wicks below OB bottom | Price wicks above OB top |
| FVG direction | Bearish FVG | Bullish FVG |
| FVG location | After OB bar, near/above OB | After OB bar, near/below OB |
| Entry trigger | High taps FVG bottom | Low taps FVG top |
| Stop Loss | Above OB top | Below OB bottom |
| Take Profit | 2R below entry | 2R above entry |

---

## Zone Validity

- Zones remain active for **200 bars** after the mitigation bar.
- A zone is consumed (removed) when price enters it and a signal fires.

---

## Parameters

| Parameter | Value | Description |
|---|---|---|
| `SWING_LENGTH` | 3 | Controls granularity of swing detection and OB formation |
| `ZONE_TIMEOUT` | 200 | Bars an armed zone remains valid |

---

## Implementation Notes

- **`smc.swing_highs_lows(swing_length=3)`** — smaller values detect more swing points and more OBs. Value of 5 is stricter (fewer, more significant OBs).
- **`smc.ob(close_mitigation=False)`** — wick below OB bottom (not close) counts as mitigation. This is the default and more sensitive.
- **Data requirement** — `smc.ob()` requires price to close beyond a full swing level before confirming an OB. This is a high-selectivity filter; a minimum of 12–24 months of data is recommended for statistical significance.

---

## Risk Management

- **Stop Loss**: Above the OB top (the failed demand zone high)
- **Target**: Fixed 2R from entry
- **Risk per trade**: 1% of account equity (configurable in backtest)
