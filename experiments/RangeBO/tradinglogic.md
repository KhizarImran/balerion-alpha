# Range Breakout — Trading Logic

## Overview

The Range Breakout (RangeBO) strategy trades the breakout of a defined Asian/pre-London session range on **USDJPY** at the **1-hour timeframe**. The strategy exploits the tendency for price to accumulate in a tight range during the overnight Asian session before directional momentum emerges at or around the London open.

**Long (buy) trades** are taken on upside breakouts.

**Short (sell) trades** are taken on downside breakouts.

---

## Core Concept

During the early hours of the trading day (00:00–07:00 UK time), USDJPY typically consolidates as liquidity is thin and no major session is fully active. This accumulation phase creates a range with clearly defined highs and lows. When price breaks out of this range at or after 07:00 UK time (London open), it signals the beginning of a directional move fuelled by institutional order flow entering the market.

Both sides of the range are traded — a close above the Range High triggers a long; a close below the Range Low triggers a short. Only **one trade per day** is taken: whichever breakout fires first.

---

## Setup Structure (Step by Step)

### Step 1 — Define the Range (00:00–07:00 UK Time)
Using 1-hour bars with timestamps in UK time (Europe/London), identify all bars whose open time falls within the window **00:00 (inclusive) to 07:00 (exclusive)**.

- **Range High**: the highest `high` across all bars in the window
- **Range Low**: the lowest `low` across all bars in the window

The range is locked at the close of the 06:00 bar (i.e., fully formed before the 07:00 bar opens). It is recalculated fresh for each trading day.

### Step 2 — Breakout Confirmation (07:00 onwards)
A valid breakout occurs on the first 1-hour bar (opening at or after 07:00 UK time) whose **close** exceeds either boundary of the range.

- **Bullish breakout**: bar close > Range High
- **Bearish breakout**: bar close < Range Low
- A wick through the boundary is not sufficient — the **close** must confirm the break.
- Only the **first** breakout per day is traded. Once a trade is triggered, no further entries are taken that day regardless of direction.

### Step 3 — Entry
Enter at the **close of the breakout bar**.

- **Long entry**: close of the first bar that closes above Range High
- **Short entry**: close of the first bar that closes below Range Low

### Step 4 — Stop Loss
- **Long SL**: Range Low — the opposite boundary of the accumulation range
- **Short SL**: Range High — the opposite boundary of the accumulation range

Stop distance (in pips) = Range High − Range Low (the full range width).

### Step 5 — Take Profit (1:2 R:R)
The take profit is set at a **1:2 reward-to-risk ratio** based on the full range width.

- **Long TP** = Entry price + 2 × (Entry price − Range Low)
- **Short TP** = Entry price − 2 × (Range High − Entry price)
- If the range is 30 pips wide, the target is 60 pips in the direction of the breakout.

### Step 6 — Auto Close at 16:00 UK Time
If the trade is still open at **16:00 UK time** (end of the New York/London overlap), it is closed at market regardless of outcome.

- Prevents trades from running into the illiquid late New York / Asian pre-session hours.
- Any open position at the 16:00 bar close is exited at that bar's close price.

---

## Entry Rules Summary

| Element        | Long (Buy)                                              | Short (Sell)                                             |
|----------------|---------------------------------------------------------|----------------------------------------------------------|
| Instrument     | USDJPY                                                  | USDJPY                                                   |
| Timeframe      | 1-hour bars                                             | 1-hour bars                                              |
| Range window   | 00:00–07:00 UK time (7 bars)                            | 00:00–07:00 UK time (7 bars)                             |
| Breakout       | First bar at/after 07:00 UK that **closes above** Range High | First bar at/after 07:00 UK that **closes below** Range Low |
| Entry          | Close of the breakout bar                               | Close of the breakout bar                                |
| Stop Loss      | Range Low                                               | Range High                                               |
| Take Profit    | Entry + 2 × (Entry − Range Low)  [1:2 R:R]             | Entry − 2 × (Range High − Entry)  [1:2 R:R]             |
| Auto close     | 16:00 UK time bar close if trade still open             | 16:00 UK time bar close if trade still open              |
| Max trades/day | 1 (first breakout direction wins)                       | 1 (first breakout direction wins)                        |

---

## Invalidation / No-Trade Conditions

The setup is skipped for that day if:

- The range is abnormally wide (e.g., due to a high-impact overnight news event) — optionally filtered by a maximum range size in pips.
- No bar closes outside either range boundary before 16:00 UK time on the same day.
- The range cannot be computed (e.g., missing data for the day).

---

## Risk Management

- **Stop Loss**: Opposite range boundary — a natural structural level, not an arbitrary pip distance.
- **Take Profit**: Fixed 1:2 R:R relative to the full range width.
- **Time stop**: Hard exit at 16:00 UK time — limits overnight gap risk and late-session chop exposure.
- **One trade per day**: Prevents overtrading on false breakout re-entries in the other direction.

---

## Timeframe & Session Context

| Session          | UK Time        | Role                                                  |
|------------------|----------------|-------------------------------------------------------|
| Asian / Off-peak | 00:00 – 07:00  | Range accumulation — define High & Low                |
| London Open      | 07:00 – 12:00  | Primary breakout window                               |
| NY / Overlap     | 12:00 – 16:00  | Trade runs / auto-close deadline                      |
| Post 16:00       | 16:00+         | No new trades; any open position closed at market     |

---

## Visual Summary

```
Price
  |
  |  LONG:  ════════════════════════ Range High  <-- close above → BUY entry
  |         |  Asian Range (00-07) |                  SL at Range Low
  |  SHORT: ════════════════════════ Range Low   <-- close below → SELL entry
  |                                                   SL at Range High
  |
  |  00:00               07:00         TP hit or 16:00 auto-close
  |----|----|----|----|----|----|----|----|----|----|----|----|---> Time (UK)
        <--- Range bars --->   ^ First breakout bar close = entry
```

---

## Notes

- All timestamps must be converted to **UK local time (Europe/London)** to correctly handle BST/GMT transitions.
- USDJPY pip size is 0.01 (2-decimal pair); pip value per standard lot ≈ $9.25–$10 depending on the USDJPY rate.
- If both a long and short breakout occur on the same day (e.g., a false break low followed by a strong break high), only the **first** breakout is traded.
- Works best on trending days where a clear directional bias exists from the London open. Choppy or news-driven days may produce false breakouts in either direction.
