# TrendBO — Trading Logic

## Concept

TrendBO is an intraday trend-following breakout strategy. It uses two EMAs to establish the macro trend direction, then waits for price to enter a compressed consolidation zone (low ADX + tight range relative to ATR), and only takes breakouts in the direction of the trend.

The core insight is: trending markets frequently pause to consolidate before continuing. If you can identify when a range is genuinely tight (not just random noise), and confirm the broader trend is intact, the breakout out of that range has a higher probability of following through in the trend direction.

---

## Indicators

| Indicator | Parameters | Purpose |
|---|---|---|
| EMA 50 | period=50 | Short-term trend direction |
| EMA 200 | period=200 | Long-term trend direction |
| ATR | period=14 | Volatility baseline |
| ADX | period=14 | Trend strength (low = ranging) |
| Rolling High/Low | period=20 | Consolidation zone boundaries |

---

## Trend Filter (EMA 50 / 200)

The 50 EMA vs 200 EMA crossover determines which direction trades are allowed:

- **Bullish trend**: `EMA_50 > EMA_200` → only LONG breakouts permitted
- **Bearish trend**: `EMA_50 < EMA_200` → only SHORT breakouts permitted

No trades are taken when the EMAs are within a narrow band of each other (flat / crossover noise). This prevents trading during trend transitions.

---

## Consolidation Detection

A bar is considered "in consolidation" when all three conditions hold simultaneously:

1. **ADX < 25** — directional movement is weak; the market is not trending
2. **Range compression** — the 20-bar rolling range (high - low) is less than `RANGE_ATR_MULT * ATR_14`. This means the range is tight *relative to* recent volatility. A range of 30 pips in a 50-pip ATR environment is different from a 30-pip range in a 10-pip ATR environment.
3. **Minimum range size** — the range must be at least `MIN_CONSOLIDATION_PIPS` wide to avoid degenerate flat/dead zones

A consolidation episode is confirmed after `CONSOL_MIN_BARS` consecutive bars satisfy conditions 1 and 2.

---

## Breakout Entry

A breakout is triggered on the **first bar** that closes **outside** the consolidation zone while the trend filter is aligned:

- **Long entry**: `close > rolling_high` AND `EMA_50 > EMA_200`
- **Short entry**: `close < rolling_low` AND `EMA_50 < EMA_200`

Additional quality filters applied to the breakout bar:

- **ATR expansion**: current ATR > `ATR_EXPAND_MULT * ATR_14.shift(1)` — volatility is expanding, confirming the breakout is not just drift
- **Session filter**: entries only allowed between `SESSION_START_HOUR` and `SESSION_END_HOUR` (London/NY overlap for USDJPY)
- **One trade per consolidation episode**: once a setup fires, that consolidation zone is consumed. A new consolidation must form before the next entry.

---

## Stop Loss

SL is placed at the **opposite side of the consolidation range**:

- Long: `SL = rolling_low at breakout bar`
- Short: `SL = rolling_high at breakout bar`

This gives the trade room to breathe — if price re-enters the consolidation zone, the thesis is invalidated.

Minimum SL distance is floored at `MIN_SL_PIPS` to prevent oversized lot calculations on thin consolidations.

---

## Take Profit

TP is set at a fixed risk:reward multiple of the SL distance:

```
TP = entry + RISK_REWARD * (entry - SL)   # long
TP = entry - RISK_REWARD * (SL - entry)   # short
```

Default `RISK_REWARD = 2.0` (1:2 R:R).

---

## Position Sizing

Risk-based sizing — each trade risks a fixed percentage of account capital:

```
risk_$ = ACCOUNT_SIZE * RISK_PER_TRADE
sl_pips = abs(entry - sl) / PIP_SIZE
lots = risk_$ / (sl_pips * PIP_VALUE_PER_LOT)
units = lots * STD_LOT
```

If the SL is very tight (< MIN_SL_PIPS), the SL distance is floored at MIN_SL_PIPS to prevent excessively large lot sizes.

---

## Session Filter

Intraday entries are restricted to the active trading window to avoid illiquid periods:

| Parameter | Value | Rationale |
|---|---|---|
| `SESSION_START_HOUR` | 7 | London open (UTC) |
| `SESSION_END_HOUR` | 20 | NY close (UTC) |

Bars outside this window are skipped for entries. Existing positions are not force-closed at session end (SL/TP handles exits).

---

## Exit Logic

Positions are exited by:
1. **Take profit** (`tp_stop` in vectorbt) — primary exit
2. **Stop loss** (`sl_stop` in vectorbt) — risk management exit
3. No time-based forced close (unlike RangeBO) — let SL/TP do the work

`stop_exit_price="StopMarket"` fills at the exact SL/TP level.

---

## Filter Summary (applied in order)

1. EMA trend alignment check
2. Consolidation episode active (ADX < 25 for N consecutive bars)
3. Range compression check (range / ATR < threshold)
4. Minimum consolidation width check
5. Session hour check
6. Breakout close beyond rolling high/low
7. ATR expansion on breakout bar
8. One setup per consolidation episode

---

## Parameters (defaults)

| Parameter | Default | Description |
|---|---|---|
| `EMA_FAST` | 50 | Fast EMA period |
| `EMA_SLOW` | 200 | Slow EMA period |
| `ATR_PERIOD` | 14 | ATR lookback |
| `ADX_PERIOD` | 14 | ADX lookback |
| `RANGE_PERIOD` | 20 | Rolling high/low lookback |
| `CONSOL_MIN_BARS` | 5 | Minimum consecutive ranging bars to confirm consolidation |
| `RANGE_ATR_MULT` | 4.0 | Max range/ATR ratio to qualify as compressed (20-bar range naturally sits at 3-5x ATR on 1H; < 4.0 = bottom ~29% = genuinely tight) |
| `ATR_EXPAND_MULT` | 1.0 | Min ATR on breakout bar vs ATR on prior bar |
| `MIN_CONSOLIDATION_PIPS` | 10 | Minimum consolidation width |
| `MIN_SL_PIPS` | 10 | SL distance floor for lot sizing |
| `RISK_REWARD` | 2.0 | TP = RR * SL distance |
| `SESSION_START_HOUR` | 7 | UTC hour — London open |
| `SESSION_END_HOUR` | 20 | UTC hour — NY close |
