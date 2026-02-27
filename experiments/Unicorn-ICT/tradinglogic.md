# ICT Unicorn Model — Trading Logic

## Overview

The Unicorn is an ICT (Inner Circle Trader) price action setup that combines a **liquidity sweep**, a **Fair Value Gap (FVG)**, and a **Breaker Block** into a single high-probability entry zone. The name reflects its rarity — all three elements must align.

---

## Core Concept

Price is engineered by institutions to **hunt liquidity** (stop losses sitting above swing highs or below swing lows) before reversing. The Unicorn captures the reversal move after that hunt by entering into the imbalance left behind by the displacement.

---

## Setup Structure (Step by Step)

### Step 1 — Liquidity Sweep
Price runs beyond a prior swing high (for a short setup) or swing low (for a long setup), triggering resting stop orders. This is the "hunt."

- Bullish: price sweeps below **equal lows** or a prior swing low
- Bearish: price sweeps above **equal highs** or a prior swing high

### Step 2 — Displacement
After the sweep, price reverses aggressively with strong impulsive candles in the opposite direction. This displacement:
- Breaks the prior structure (CHoCH — Change of Character)
- Leaves a **Fair Value Gap (FVG)** in its wake

### Step 3 — Fair Value Gap (FVG)
A 3-candle imbalance pattern formed during the displacement:
- **Bullish FVG**: Candle 1's high does not overlap with Candle 3's low — a gap exists between them
- **Bearish FVG**: Candle 1's low does not overlap with Candle 3's high

This gap represents inefficiency that price is likely to return to and fill (partially or fully).

### Step 4 — Breaker Block Confluence
The FVG must overlap with a **Breaker Block** — a prior bearish order block (for longs) or bullish order block (for shorts) that has since been violated by the displacement move. The broken order block "flips" polarity and becomes support/resistance.

The overlap between the FVG and the Breaker Block is the **Unicorn zone** — the entry area.

### Step 5 — Entry
Price retraces back into the Unicorn zone (FVG + Breaker overlap). Entry is taken as price taps into this zone.

---

## Entry Rules Summary

| Element       | Bullish Unicorn                          | Bearish Unicorn                          |
|---------------|------------------------------------------|------------------------------------------|
| Sweep         | Below a swing low / equal lows           | Above a swing high / equal highs         |
| Displacement  | Strong bullish move breaking structure   | Strong bearish move breaking structure   |
| FVG           | Bullish FVG left by the displacement     | Bearish FVG left by the displacement     |
| Breaker Block | Prior bearish OB, now acting as support  | Prior bullish OB, now acting as resistance |
| Entry         | Price retraces into FVG / Breaker zone   | Price retraces into FVG / Breaker zone   |
| Stop Loss     | Below the sweep low                      | Above the sweep high                     |
| Target        | Next draw on liquidity (swing high, EQH) | Next draw on liquidity (swing low, EQL)  |

---

## Invalidation

The setup is invalidated if:
- Price **closes below** the FVG / Breaker zone (for longs) without bouncing
- The displacement candles are weak (small bodies, large wicks) — indicates no real institutional commitment
- The retracement fully fills the FVG and continues beyond the Breaker Block

---

## Timeframe Usage

| Timeframe | Purpose                                      |
|-----------|----------------------------------------------|
| 4H / 1D   | Establish directional bias (HTF structure)   |
| 15m / 1H  | Identify the sweep and displacement          |
| 1m / 5m   | Pinpoint FVG entry and refine stop placement |

---

## Best Trading Sessions (FX)

- **London Open**: 02:00 – 05:00 AM New York time
- **New York Open**: 07:00 – 10:00 AM New York time

Setups occurring outside kill zones carry significantly lower probability.

---

## Risk Management

- **Stop Loss**: Beyond the liquidity sweep candle (gives the trade room)
- **Minimum R:R**: 2:1 — the setup should offer at least twice the risk as reward
- **Target**: Prior swing high/low, equal highs/lows, or an HTF order block acting as draw

---

## Visual Summary

```
Bearish OB (prior)
     |
     v
[Swing Low swept] <-- liquidity hunt
     |
     | <-- strong bullish displacement
     |       [FVG created here]
     |
[Price retraces into FVG + Breaker zone] <-- ENTRY
     |
     v
[Target: prior swing high / EQH]
```

---

## Notes

- The Unicorn is **rare by design** — not every FVG qualifies. Do not force the setup.
- The sweep must be a genuine stop hunt, not just a minor wick.
- Higher timeframe bias (bullish or bearish) must align with the direction of the trade.
- Works best on liquid FX pairs: EURUSD, GBPUSD, USDJPY, XAUUSD.
