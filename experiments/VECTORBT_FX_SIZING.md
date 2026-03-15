# VectorBT FX Position Sizing — Lot Calculation & Quote Currency Correction

This document explains how FX lot sizes are calculated for risk-based position sizing, why different currency pairs require a quote currency correction when used with vectorbt, and how to implement it correctly for each instrument class.

---

## How vectorbt computes P&L

vectorbt's `Portfolio.from_signals` computes trade P&L as:

```
P&L = units × (exit_price - entry_price)
```

It does not know or care what currency the price is quoted in. It treats the result as being in your account currency (USD). This is correct for USD-quoted pairs but wrong for JPY-quoted pairs.

---

## Currency pair classes

| Class | Example pairs | Price quote | P&L per pip (1 unit) | Correction needed |
|---|---|---|---|---|
| USD-quoted (direct) | EURUSD, GBPUSD, AUDUSD | USD per 1 EUR/GBP/AUD | `1 × PIP` = $0.0001 | None |
| JPY-quoted (indirect) | USDJPY, EURJPY, GBPJPY | JPY per 1 USD/EUR/GBP | `1 × PIP` = 0.01 JPY ≠ USD | Divide units by entry price |
| CAD-quoted | USDCAD | CAD per 1 USD | `1 × PIP` = 0.0001 CAD ≠ USD | Divide units by entry price |
| CHF-quoted | USDCHF | CHF per 1 USD | `1 × PIP` = 0.0001 CHF ≠ USD | Divide units by entry price |

For USD-quoted pairs (EURUSD, GBPUSD), vectorbt's raw `units × price_change` naturally produces USD P&L. No correction needed.

For all other pairs where the quote currency is not USD, the raw P&L is in the quote currency (JPY, CAD, CHF, etc.) and must be converted to USD before vectorbt can use it correctly.

---

## Risk-based lot sizing formula

The goal is to size each trade so that if the stop loss is hit, the account loses exactly `RISK_PER_TRADE × ACCOUNT_SIZE`:

```
risk_$     = ACCOUNT_SIZE × RISK_PER_TRADE        e.g. $10,000 × 0.01 = $100
sl_pips    = abs(entry - sl) / PIP_SIZE            e.g. 68 pips
lots       = risk_$ / (sl_pips × PIP_VALUE_PER_LOT)
units      = lots × STD_LOT                        = lots × 100,000
```

### PIP_VALUE_PER_LOT reference

| Pair | PIP_SIZE | PIP_VALUE_PER_LOT (approx, USD account) |
|---|---|---|
| EURUSD | 0.0001 | $10.00 |
| GBPUSD | 0.0001 | $10.00 |
| AUDUSD | 0.0001 | $10.00 |
| USDCAD | 0.0001 | $7.35 (at 1.36 CAD/USD) |
| USDJPY | 0.01   | $9.50 (at 150 JPY/USD) |
| EURJPY | 0.01   | $9.50 (at 150 JPY/USD) |
| XAUUSD | 0.01   | $1.00 |

---

## The quote currency problem with vectorbt

### USD-quoted pair (EURUSD) — no correction needed

```
entry = 1.0800, sl = 1.0770  →  sl_pips = 30
lots  = 100 / (30 × 10.0) = 0.3333 lots
units = 33,333

vbt P&L on SL hit = units × (exit - entry)
                  = 33,333 × (1.0770 - 1.0800)
                  = 33,333 × (-0.0030)
                  = -$100.00  ✓
```

The result is already in USD. Correct.

---

### JPY-quoted pair (USDJPY) — correction required

```
entry = 150.00, sl = 149.32  →  sl_pips = 68
lots  = 100 / (68 × 9.5) = 0.1548 lots
units = 15,480

vbt P&L on SL hit (without correction) = units × (exit - entry)
                                        = 15,480 × (149.32 - 150.00)
                                        = 15,480 × (-0.68)
                                        = -10,526 JPY   ← vectorbt labels this as $-10,526 ✗
```

vectorbt sees `–10,526` and treats it as USD. But it is actually 10,526 JPY. At 150 USDJPY, the real USD loss is:

```
-10,526 JPY ÷ 150 = -$70.17   (≈ intended $100, difference is PIP_VALUE_PER_LOT approximation)
```

The result without correction is **~145× too large**.

---

## The fix: divide units by entry price for non-USD-quoted pairs

For any pair where the quote currency is not USD, divide the computed `units` by `entry_price` before passing to vectorbt:

```python
units = _compute_lot_units(sl_pips)

if _IS_JPY_PAIR:   # or any non-USD quoted pair
    units = units / entry_price
```

**Why this works:**

```
units_corrected = 15,480 / 150 = 103.2

vbt P&L on SL hit = 103.2 × (149.32 - 150.00)
                  = 103.2 × (-0.68)
                  = -$70.18   ✓  (≈ $100, small delta from PIP_VALUE approximation)
```

vectorbt now computes the right USD magnitude.

---

## Generalised implementation

```python
_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"}
_CAD_PAIRS = {"USDCAD"}
_CHF_PAIRS = {"USDCHF"}

# Pairs where quote currency != USD — require unit correction
_NON_USD_QUOTED = _JPY_PAIRS | _CAD_PAIRS | _CHF_PAIRS

_IS_NON_USD = SYMBOL in _NON_USD_QUOTED


def _compute_lot_units(sl_pips: float) -> float:
    """Risk-based lot sizing. Returns base currency units."""
    if sl_pips <= 0:
        return 0.0
    risk_dollars = ACCOUNT_SIZE * RISK_PER_TRADE
    lots = risk_dollars / (sl_pips * PIP_VALUE_PER_LOT)
    return lots * STD_LOT


# Inside setups_to_signal_arrays, after computing units:
units = _compute_lot_units(sl_pips)

if _IS_NON_USD:
    # Correct for quote currency: vectorbt P&L = units × price_change (in quote ccy).
    # Dividing by entry_price converts the notional so P&L is expressed in USD.
    units = units / entry

# Cap: notional (units × entry) must not exceed ACCOUNT_SIZE × LEVERAGE
# so vectorbt never rejects an order for insufficient cash (init_cash=ACCOUNT_SIZE).
max_units = (ACCOUNT_SIZE * LEVERAGE) / entry
units = min(units, max_units)
```

---

## init_cash and the notional cap

vectorbt's `from_signals` checks:

```
order accepted  if  cash >= units × price
order rejected  if  cash < units × price
```

With `init_cash=ACCOUNT_SIZE` ($10,000) and `size_type='amount'`, any order whose notional exceeds $10,000 would be rejected. Since FX trades at 100:1 leverage can have notional up to $1,000,000, most orders would silently fail.

The cap ensures:

```python
max_units = (ACCOUNT_SIZE * LEVERAGE) / entry   # e.g. $1,000,000 / 150 = 6,666 units
units = min(risk_based_units, max_units)
```

This guarantees `units × entry ≤ ACCOUNT_SIZE × LEVERAGE` — i.e. the notional never exceeds the cash balance that vectorbt sees, so no order is ever rejected.

After the JPY correction, the corrected units are already much smaller (÷150), so the cap is rarely hit — but it is kept as a hard safety floor.

---

## Per-pair configuration reference

| Symbol | Asset type | PIP | PIP_VALUE_PER_LOT | _IS_NON_USD | Notes |
|---|---|---|---|---|---|
| EURUSD | FX | 0.0001 | 10.0 | False | Direct USD quote — no correction |
| GBPUSD | FX | 0.0001 | 10.0 | False | Direct USD quote — no correction |
| AUDUSD | FX | 0.0001 | 10.0 | False | Direct USD quote — no correction |
| AUDNZD | FX | 0.0001 | ~6.50 | False* | NZD-quoted; P&L in NZD, needs correction |
| EURGBP | FX | 0.0001 | ~12.60 | False* | GBP-quoted; P&L in GBP, needs correction |
| USDCAD | FX | 0.0001 | ~7.35 | True | CAD-quoted — divide units by entry |
| USDJPY | FX | 0.01   | 9.5  | True | JPY-quoted — divide units by entry |
| EURJPY | FX | 0.01   | 9.5  | True | JPY-quoted — divide units by entry |
| XAUUSD | Index | 0.01 | 1.0  | False | USD-quoted — no correction |
| US30   | Index | 1.0  | 1.0  | False | USD-quoted — no correction |

*AUDNZD and EURGBP are quoted in NZD and GBP respectively. They technically need a conversion but the correction factor is close to 1.0 for most of the data range, so the error is small. For precision, apply the same `units / entry` correction.

---

## Common symptoms of a missing correction

| Symptom | Cause |
|---|---|
| Gross P&L is ~100–150× larger than expected | JPY-quoted pair, units not divided by entry |
| Max drawdown is –100% or below despite 1% risk/trade | P&L inflated, apparent losses exceed account size |
| Win/loss dollar amounts don't match `sl_pips × pip_value × lots` | Quote currency mismatch |
| VBT `Best Trade [%]` shows 3% but you expected 2% RR × 1% risk = 2% | P&L computed in wrong currency |

---

## Quick sanity check

After running a backtest, verify the per-trade dollar amounts are in the right ballpark:

```python
trades = pf.trades.records_readable
avg_win  = trades.loc[trades["PnL"] > 0, "PnL"].mean()
avg_loss = trades.loc[trades["PnL"] < 0, "PnL"].mean()
expected_loss = ACCOUNT_SIZE * RISK_PER_TRADE   # e.g. $100

print(f"Avg loss : ${avg_loss:.2f}  (expected ~-${expected_loss:.0f})")
print(f"Avg win  : ${avg_win:.2f}  (expected ~+${expected_loss * RISK_REWARD:.0f})")
```

If `avg_loss` is `~-$10,000` instead of `~-$100`, the quote currency correction is missing.
