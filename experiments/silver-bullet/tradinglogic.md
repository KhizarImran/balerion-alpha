# ICT Silver Bullet — Trading Logic

## Concept

The Silver Bullet is an ICT (Inner Circle Trader) intraday strategy. It waits
for a liquidity sweep — price taking out a significant high or low — to
establish a directional bias, then enters on a Fair Value Gap (FVG) pullback
during one of three daily killzone windows.

---

## Long Entry (Bullish)

All five conditions must be true on the entry bar.

| # | Condition | Detail |
|---|---|---|
| 1 | **Low swept** | The `liquidity_window`-bar rolling low has been touched within the last `sweep_lookback` bars. Bullish bias: liquidity below the market was hunted. |
| 2 | **High not swept** | The corresponding rolling high has NOT been swept. Bullish bias remains intact. |
| 3 | **Bullish FVG** | `high[N-2] < low[N]` — a two-bar upward imbalance (fair value gap). |
| 4 | **FVG pullback** | Current bar closes inside the FVG zone (`fvg_bottom <= close <= fvg_top`) within `pullback_window` bars of the FVG forming. |
| 5 | **Killzone session** | Bar falls within London (03:00–03:59 ET), NY AM (10:00–10:59 ET), or NY PM (14:00–14:59 ET). |

**Stop Loss:** Below the swept liquidity low minus `sl_buffer_pts`.

**Take Profit:** `rr_ratio × SL distance` above entry.

---

## Short Entry (Bearish)

Mirror of the long, inverted.

| # | Condition | Detail |
|---|---|---|
| 1 | **High swept** | The `liquidity_window`-bar rolling high has been touched within the last `sweep_lookback` bars. Bearish bias. |
| 2 | **Low not swept** | The corresponding rolling low has NOT been swept. Bearish bias intact. |
| 3 | **Bearish FVG** | `low[N-2] > high[N]` — a two-bar downward imbalance. |
| 4 | **FVG pullback** | Close is inside the bearish FVG zone (`fvg_bottom <= close <= fvg_top`) within `pullback_window` bars of the FVG forming. |
| 5 | **Killzone session** | Same three session windows as above. |

**Stop Loss:** Above the swept liquidity high plus `sl_buffer_pts`.

**Take Profit:** `rr_ratio × SL distance` below entry.

---

## Trade Management Rules

| Rule | Value |
|---|---|
| Entries per session per day | 1 (first qualifying bar wins) |
| Time exit fallback | 52 bars if SL/TP not hit |
| Session P&L cap | No further entries once cumulative estimated loss in a session exceeds `session_cap_usd` (conservative: assumes SL hit each trade) |

---

## Fair Value Gap Details

A **bullish FVG** is defined on bar N when:
```
high[N-2] < low[N]
```
- FVG zone: `bottom = high[N-2]`,  `top = low[N]`
- Zone is "alive" for `pullback_window` bars from bar N onward
- Entry triggers when price closes back inside the zone

A **bearish FVG** is defined on bar N when:
```
low[N-2] > high[N]
```
- FVG zone: `bottom = high[N]`,  `top = low[N-2]`
- Same forward-fill window applies

---

## Session Windows

All times in Eastern Time (ET). Broker data is in `Europe/Athens` server time
(UTC+2 winter / UTC+3 summer). DST conversion is handled automatically via
`pytz` in `strategy.py`.

| Session | ET Hours | Notes |
|---|---|---|
| London | 03:00 – 03:59 | London open killzone |
| New York AM | 10:00 – 10:59 | NY AM killzone (post-open) |
| New York PM | 14:00 – 14:59 | NY PM killzone (afternoon reversal) |

---

## SL/TP Calculation

```
# Long
sl     = liq_low - sl_buffer_pts
sl_dist = entry - sl
tp     = entry + rr_ratio * sl_dist

# Short
sl     = liq_high + sl_buffer_pts
sl_dist = sl - entry
tp     = entry - rr_ratio * sl_dist
```

`sl_buffer_pts` adds breathing room below/above the liquidity level.
Default (optimised): `sl_buffer_pts = 0.0`.

---

## Optimised Parameters (in-sample, 2025-11-11 to 2026-02-25)

| Parameter | Value | Description |
|---|---|---|
| `rr_ratio` | 4.5 | TP = 4.5 x SL distance |
| `liquidity_window` | 150 bars | Rolling window for significant highs/lows |
| `sweep_lookback` | 210 bars | Look-back for sweep detection |
| `pullback_window` | 10 bars | FVG zone lifetime |
| `sl_buffer_pts` | 0.0 | No extra buffer |
| `session_cap_usd` | -$5,500 | Per-session cumulative loss cap |

> These are in-sample results. No out-of-sample or walk-forward validation
> has been run. Treat the optimised values as a starting point, not ground
> truth.

---

## Lookahead Bias Notes

- Rolling `liquidity_high` and `liquidity_low` use `.rolling(window).max/min()`
  on the current and past `window` bars only — no future data.
- Sweep detection uses `.rolling(sweep_lookback).max()` on a boolean touch flag.
- FVG detection references `high[N-2]` vs `low[N]` using shifted series — no
  forward lookahead.
- The session filter uses timezone-converted timestamps of the current bar only.
- One entry per session per day is enforced by iterating chronologically and
  skipping a `(date, session)` key once used.
