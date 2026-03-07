# ICT Unicorn Model — Trading Logic

## Overview

The Unicorn is an ICT (Inner Circle Trader) price action setup built on two structural elements: a **Breaker Block** and a **Fair Value Gap (FVG)**. When these two zones overlap, the overlap becomes a high-probability entry area for a reversal trade.

---

## Core Concept

Institutions leave footprints when they reverse price: they first break a prior order block (creating a **Breaker Block**) and leave behind an imbalance in the candles (an **FVG**). When price retraces back into the area where these two zones overlap, it is returning to the scene of the original institutional move — a prime entry zone.

---

## Setup Structure (Step by Step)

### Step 1 — Identify a Breaker Block

A breaker block is a prior order block that has been broken by a displacement move, causing it to flip its role (support becomes resistance, or vice versa).

**Bearish Breaker Block:**
- Find the last bullish candle (close > open) that formed a **higher low** — i.e. its low is above the prior candle's low, acting as a local support.
- When a subsequent bar breaks **below** that prior low (a structure break downward), the original bullish candle is now a **bearish breaker block**.
- The breaker zone = full candle range of that bullish candle (high to low).
- Price is expected to retrace back up into this zone to provide a sell opportunity.

**Bullish Breaker Block (mirror):**
- Find the last bearish candle (close < open) that formed a **lower high**.
- When a subsequent bar breaks **above** that prior high (structure break upward), the bearish candle flips to a **bullish breaker block**.
- Zone = full candle range (high to low). Price retracing into it = buy opportunity.

---

### Step 2 — Identify a Fair Value Gap (FVG)

An FVG is a 3-candle price imbalance — a gap between candle 1 and candle 3 that candle 2 does not fill.

| Type | Condition | Zone |
|---|---|---|
| Bearish FVG | `low[i-2] > high[i]` | `(high[i], low[i-2])` |
| Bullish FVG | `high[i-2] < low[i]` | `(high[i-2], low[i])` |

FVGs represent price moving too fast — inefficiency that price tends to return to and fill.

---

### Step 3 — Find the Overlap (Unicorn Zone)

If any **active bearish FVG** overlaps with any **active bearish breaker block**, the overlap is the Unicorn entry zone for a short.

If any **active bullish FVG** overlaps with any **active bullish breaker block**, the overlap is the Unicorn entry zone for a long.

Both zones must be within `ZONE_TIMEOUT` (50) bars of formation to remain active.

---

### Step 4 — Entry

Signal fires when price **retraces into the overlap zone**:

- **SELL**: price high taps into the bearish overlap zone (retrace up into it)
- **BUY**: price low taps into the bullish overlap zone (retrace down into it)

One signal per zone — the zone is consumed on entry.

---

## Entry Rules Summary

| Element | Bullish Unicorn | Bearish Unicorn |
|---|---|---|
| Breaker Block | Last bearish candle before a higher-high break | Last bullish candle before a lower-low break |
| FVG | Bullish 3-candle imbalance | Bearish 3-candle imbalance |
| Entry Zone | Overlap of bullish breaker + bullish FVG | Overlap of bearish breaker + bearish FVG |
| Entry Trigger | Price retraces down into the overlap zone | Price retraces up into the overlap zone |
| Stop Loss | Below the bottom of the breaker block | Above the top of the breaker block |
| Target | 2R above entry | 2R below entry |

---

## Zone Validity

- Zones remain active for **50 bars** after formation.
- A zone is consumed (removed) when price enters it and a signal fires.
- A zone is also discarded when it expires (50-bar timeout).

---

## Timeframe Usage

| Timeframe | Purpose |
|---|---|
| 4H / 1D | Establish directional bias (HTF structure) |
| 1H | Primary signal timeframe (breakers + FVGs detected here) |
| 5m / 15m | Refine entry and stop placement |

---

## Best Trading Sessions (FX)

- **London Open**: 02:00 – 05:00 AM New York time
- **New York Open**: 07:00 – 10:00 AM New York time

---

## Risk Management

- **Stop Loss**: Beyond the breaker block candle range (high or low depending on direction)
- **Target**: Fixed 2R (configurable in backtest via `min_rr`)
- **Risk per trade**: 1% of account equity (configurable)
