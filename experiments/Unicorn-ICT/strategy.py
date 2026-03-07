"""
ICT Unicorn Model — Strategy (Breaker Block + FVG Overlap)

CORRECT SEQUENCE — BEARISH SETUP (sell)
========================================

Step 1 — Swing High confirmed
  A pivot high forms: bar i is the highest high across (pivot_left + pivot_right + 1) bars.
  Confirmed at bar i+pivot_right (we wait for the right-side bars to close).

Step 2 — Order Block identified
  The last bullish candle (close > open) in the pivot_left bars LEADING INTO the
  pivot high bar. This is the candle that made the final push up — the "last up-close
  before the top". Its full range (high → low) is the raw Order Block zone.

Step 3 — Displacement FVG
  After the swing high, price drops. We look for a BEARISH FVG that forms on or after
  the swing high bar during the sell-off. A bearish FVG at bar j means:
      low[j-2] > high[j]   — a downward gap exists between candle j-2 and candle j
  The FVG zone: top = low[j-2],  bot = high[j]   (price is ABOVE this gap now)

Step 4 — Structure break confirms the Breaker
  The Order Block becomes a Breaker Block when price CLOSES BELOW the swing low
  that existed immediately before the swing high (i.e. the prior confirmed pivot low).
  This is the CHoCH — Change of Character.

Step 5 — Overlap = Entry Zone
  IF the displacement FVG (step 3) overlaps with the OB zone (step 2), the overlap
  is the Unicorn entry zone.

Step 6 — Entry signal
  Price retraces UP into the overlap zone (wick or close touches the bottom edge).
  → SELL signal. SL above the OB high. TP = 2R below entry.


CORRECT SEQUENCE — BULLISH SETUP (buy) — exact mirror
======================================================

Step 1 — Swing Low confirmed
Step 2 — Last BEARISH candle before the pivot low = Order Block
Step 3 — Bullish FVG forms on or after the swing low during the rally:
          high[j-2] < low[j]   — upward gap
          FVG zone: top = low[j],  bot = high[j-2]
Step 4 — Price closes ABOVE the swing high that existed before the swing low
Step 5 — FVG overlaps OB zone → overlap zone armed
Step 6 — Price retraces DOWN into the overlap zone → BUY signal


No lookahead bias. All pivots are confirmed with a right-side shift.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import DataLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PIVOT_LEFT  = 5   # bars to the left of pivot
PIVOT_RIGHT = 5   # bars to the right (confirmation lag)
ZONE_TIMEOUT = 50 # bars an armed zone stays valid


# ---------------------------------------------------------------------------
# Step 1 helper: compute confirmed pivot highs and lows
# ---------------------------------------------------------------------------

def _compute_pivots(df: pd.DataFrame, left: int, right: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Return arrays pivot_high and pivot_low.
    pivot_high[i] = the pivot high price confirmed at bar i (NaN if bar i is not a confirmation bar).
    The actual peak occurred at bar i - right (the center of the window), but we only
    mark it as confirmed once the right-side bars have closed — hence the shift(right).
    """
    n = left + right + 1
    roll_max = df["high"].rolling(window=n, center=True).max()
    roll_min = df["low"].rolling(window=n, center=True).min()
    h_vals = df["high"].to_numpy(dtype=float)
    l_vals = df["low"].to_numpy(dtype=float)
    raw_ph = np.where(h_vals == roll_max.to_numpy(dtype=float), h_vals, np.nan)
    raw_pl = np.where(l_vals == roll_min.to_numpy(dtype=float), l_vals, np.nan)
    ph: np.ndarray = pd.Series(raw_ph, index=df.index).shift(right).to_numpy()  # type: ignore[assignment]
    pl: np.ndarray = pd.Series(raw_pl, index=df.index).shift(right).to_numpy()  # type: ignore[assignment]
    return ph, pl


# ---------------------------------------------------------------------------
# Step 3 helper: find the first bearish/bullish FVG from a start bar onward
# ---------------------------------------------------------------------------

def _find_bear_fvg_after(
    highs: np.ndarray,
    lows: np.ndarray,
    start_bar: int,
    end_bar: int,
) -> tuple[float, float, int]:
    """
    Scan bars [start_bar, end_bar] for the FIRST bearish FVG.
    Bearish FVG at bar j: low[j-2] > high[j]
    Returns (fvg_top, fvg_bot, bar_index) or (nan, nan, -1) if none found.
    fvg_top = low[j-2],  fvg_bot = high[j]
    """
    for j in range(max(start_bar, 2), end_bar + 1):
        if lows[j - 2] > highs[j]:
            return float(lows[j - 2]), float(highs[j]), j
    return np.nan, np.nan, -1


def _find_bull_fvg_after(
    highs: np.ndarray,
    lows: np.ndarray,
    start_bar: int,
    end_bar: int,
) -> tuple[float, float, int]:
    """
    Scan bars [start_bar, end_bar] for the FIRST bullish FVG.
    Bullish FVG at bar j: high[j-2] < low[j]
    Returns (fvg_top, fvg_bot, bar_index) or (nan, nan, -1) if none found.
    fvg_top = low[j],  fvg_bot = high[j-2]
    """
    for j in range(max(start_bar, 2), end_bar + 1):
        if highs[j - 2] < lows[j]:
            return float(lows[j]), float(highs[j - 2]), j
    return np.nan, np.nan, -1


# ---------------------------------------------------------------------------
# Main detection: Breaker Block + FVG overlap
# ---------------------------------------------------------------------------

def detect_breaker_fvg_signals(
    df: pd.DataFrame,
    pivot_left: int = PIVOT_LEFT,
    pivot_right: int = PIVOT_RIGHT,
    zone_timeout: int = ZONE_TIMEOUT,
) -> pd.DataFrame:
    """
    Full Unicorn detection: swing high/low → OB candidate → displacement FVG →
    structure break (CHoCH) → overlap zone → retrace entry.

    See module docstring for the complete step-by-step logic.
    """
    ph, pl = _compute_pivots(df, pivot_left, pivot_right)

    df = df.copy()
    n = len(df)

    highs  = df["high"].values.astype(float)
    lows   = df["low"].values.astype(float)
    opens  = df["open"].values.astype(float)
    closes = df["close"].values.astype(float)

    # -----------------------------------------------------------------------
    # Output arrays
    # -----------------------------------------------------------------------
    bear_breaker_arr     = np.zeros(n, dtype=bool)
    bear_breaker_top_arr = np.full(n, np.nan)
    bear_breaker_bot_arr = np.full(n, np.nan)
    bear_breaker_ob_bar_arr = np.full(n, -1, dtype=int)  # positional index of the OB candle
    bear_fvg_arr         = np.zeros(n, dtype=bool)
    bear_fvg_top_arr     = np.full(n, np.nan)
    bear_fvg_bot_arr     = np.full(n, np.nan)

    bull_breaker_arr     = np.zeros(n, dtype=bool)
    bull_breaker_top_arr = np.full(n, np.nan)
    bull_breaker_bot_arr = np.full(n, np.nan)
    bull_breaker_ob_bar_arr = np.full(n, -1, dtype=int)  # positional index of the OB candle
    bull_fvg_arr         = np.zeros(n, dtype=bool)
    bull_fvg_top_arr     = np.full(n, np.nan)
    bull_fvg_bot_arr     = np.full(n, np.nan)

    sell_signal_arr   = np.zeros(n, dtype=bool)
    sell_zone_top_arr = np.full(n, np.nan)
    sell_zone_bot_arr = np.full(n, np.nan)
    sell_sl_arr       = np.full(n, np.nan)
    sell_tp_arr       = np.full(n, np.nan)
    sell_fvg_bar_arr  = np.full(n, -1, dtype=int)   # FVG bar index for this signal
    sell_ob_bar_arr   = np.full(n, -1, dtype=int)   # OB candle bar index for this signal

    buy_signal_arr   = np.zeros(n, dtype=bool)
    buy_zone_top_arr = np.full(n, np.nan)
    buy_zone_bot_arr = np.full(n, np.nan)
    buy_sl_arr       = np.full(n, np.nan)
    buy_tp_arr       = np.full(n, np.nan)
    buy_fvg_bar_arr  = np.full(n, -1, dtype=int)    # FVG bar index for this signal
    buy_ob_bar_arr   = np.full(n, -1, dtype=int)    # OB candle bar index for this signal

    # -----------------------------------------------------------------------
    # State trackers
    #
    # We process each confirmed pivot once. For each pivot high we:
    #   a) find the OB candle (last bullish candle before the pivot)
    #   b) record the prior pivot low as the CHoCH trigger level
    #   c) scan forward from the pivot high bar for a displacement FVG
    #   d) wait for price to close below the CHoCH level → breaker confirmed
    #   e) check FVG vs OB overlap → arm entry zone
    #   f) wait for price to retrace into overlap → signal
    #
    # We maintain a list of "pending" setups that are waiting for their
    # CHoCH to fire, and a list of "armed" zones waiting for price entry.
    # -----------------------------------------------------------------------

    # pending_bear: list of dicts, each representing a setup waiting for CHoCH
    #   keys: ob_high, ob_low, choch_level, pivot_bar, fvg_top, fvg_bot, fvg_bar
    # (fvg fields may be nan if not yet found — we keep scanning)
    pending_bear: list[dict] = []
    pending_bull: list[dict] = []

    # armed_bear / armed_bull: overlap zones waiting for price to retrace in
    #   keys: zone_top, zone_bot, sl, bar
    armed_bear: list[dict] = []
    armed_bull: list[dict] = []

    # Rolling tracker: most recent confirmed pivot low and pivot high prices
    # These are updated as we encounter pivot confirmations.
    last_conf_pivot_low  = np.nan  # the most recent confirmed pivot LOW price
    last_conf_pivot_high = np.nan  # the most recent confirmed pivot HIGH price

    for i in range(n):
        h = highs[i]
        l = lows[i]
        c = closes[i]

        # -------------------------------------------------------------------
        # A. Process any newly confirmed pivot HIGH at bar i
        # -------------------------------------------------------------------
        if not np.isnan(ph[i]):
            swing_high_price = ph[i]
            # The actual peak bar is i - pivot_right (before the right-side confirmation)
            peak_bar = i - pivot_right

            # Step 2: Find last bullish candle in the pivot_left bars before the peak
            ob_high, ob_low, ob_bar = np.nan, np.nan, -1
            scan_start = max(peak_bar - pivot_left, 0)
            for j in range(peak_bar - 1, scan_start - 1, -1):
                if closes[j] > opens[j]:
                    ob_high = highs[j]
                    ob_low  = lows[j]
                    ob_bar  = j
                    break

            if ob_bar < 0:
                # No bullish candle found in the window — skip this pivot
                last_conf_pivot_high = swing_high_price
                continue

            # CHoCH level: the most recent confirmed pivot LOW before this swing high
            choch_level = last_conf_pivot_low  # price must close below this to confirm breaker

            # Add to pending setups (FVG will be found as we scan forward)
            pending_bear.append({
                "ob_high":    ob_high,
                "ob_low":     ob_low,
                "ob_bar":     ob_bar,
                "choch_level": choch_level,
                "pivot_bar":  peak_bar,  # bar where the actual swing high peak sits
                "swing_high": swing_high_price,
                "fvg_top":    np.nan,
                "fvg_bot":    np.nan,
                "fvg_bar":    -1,
                "armed_bar":  i,  # bar when this setup was registered
            })

            last_conf_pivot_high = swing_high_price

        # -------------------------------------------------------------------
        # B. Process any newly confirmed pivot LOW at bar i
        # -------------------------------------------------------------------
        if not np.isnan(pl[i]):
            swing_low_price = pl[i]
            trough_bar = i - pivot_right

            # Step 2: Find last bearish candle in the pivot_left bars before the trough
            ob_high, ob_low, ob_bar = np.nan, np.nan, -1
            scan_start = max(trough_bar - pivot_left, 0)
            for j in range(trough_bar - 1, scan_start - 1, -1):
                if closes[j] < opens[j]:
                    ob_high = highs[j]
                    ob_low  = lows[j]
                    ob_bar  = j
                    break

            if ob_bar < 0:
                last_conf_pivot_low = swing_low_price
                continue

            choch_level = last_conf_pivot_high  # price must close above this

            pending_bull.append({
                "ob_high":    ob_high,
                "ob_low":     ob_low,
                "ob_bar":     ob_bar,
                "choch_level": choch_level,
                "pivot_bar":  trough_bar,
                "swing_low":  swing_low_price,
                "fvg_top":    np.nan,
                "fvg_bot":    np.nan,
                "fvg_bar":    -1,
                "armed_bar":  i,
            })

            last_conf_pivot_low = swing_low_price

        # -------------------------------------------------------------------
        # C. For each pending BEAR setup: try to find FVG and check CHoCH
        # -------------------------------------------------------------------
        still_pending_bear = []
        for setup in pending_bear:
            age = i - setup["armed_bar"]

            # Expire if too old
            if age > zone_timeout:
                continue

            # Try to find a bearish FVG if we haven't yet
            # Search from the bar AFTER the swing high peak
            if np.isnan(setup["fvg_top"]):
                fvg_start = setup["pivot_bar"] + 1
                fvg_top, fvg_bot, fvg_bar = _find_bear_fvg_after(
                    highs, lows, fvg_start, i
                )
                if fvg_bar >= 0:
                    setup["fvg_top"] = fvg_top
                    setup["fvg_bot"] = fvg_bot
                    setup["fvg_bar"] = fvg_bar
                    # Mark this FVG on the output arrays for visualisation
                    bear_fvg_arr[fvg_bar]     = True
                    bear_fvg_top_arr[fvg_bar] = fvg_top
                    bear_fvg_bot_arr[fvg_bar] = fvg_bot

            # Check CHoCH: close below the prior pivot low
            choch = setup["choch_level"]
            if not np.isnan(choch) and c < choch:
                # Breaker confirmed
                ob_h = setup["ob_high"]
                ob_l = setup["ob_low"]

                # Mark breaker on the bar where CHoCH fires
                bear_breaker_arr[i]        = True
                bear_breaker_top_arr[i]    = ob_h
                bear_breaker_bot_arr[i]    = ob_l
                bear_breaker_ob_bar_arr[i] = setup["ob_bar"]

                # Compute overlap with the displacement FVG (if one was found)
                if not np.isnan(setup["fvg_top"]):
                    fvg_top = setup["fvg_top"]
                    fvg_bot = setup["fvg_bot"]
                    # OB zone is (ob_l, ob_h); FVG zone is (fvg_bot, fvg_top)
                    overlap_top = min(ob_h,  fvg_top)
                    overlap_bot = max(ob_l,  fvg_bot)
                    if overlap_top > overlap_bot:
                        armed_bear.append({
                            "zone_top": overlap_top,
                            "zone_bot": overlap_bot,
                            "sl":       ob_h,
                            "bar":      i,
                            "fvg_bar":  setup["fvg_bar"],
                            "ob_bar":   setup["ob_bar"],
                        })

                # Setup consumed — do NOT keep in still_pending_bear
                continue

            still_pending_bear.append(setup)

        pending_bear = still_pending_bear

        # -------------------------------------------------------------------
        # D. For each pending BULL setup: try to find FVG and check CHoCH
        # -------------------------------------------------------------------
        still_pending_bull = []
        for setup in pending_bull:
            age = i - setup["armed_bar"]
            if age > zone_timeout:
                continue

            if np.isnan(setup["fvg_top"]):
                fvg_start = setup["pivot_bar"] + 1
                fvg_top, fvg_bot, fvg_bar = _find_bull_fvg_after(
                    highs, lows, fvg_start, i
                )
                if fvg_bar >= 0:
                    setup["fvg_top"] = fvg_top
                    setup["fvg_bot"] = fvg_bot
                    setup["fvg_bar"] = fvg_bar
                    bull_fvg_arr[fvg_bar]     = True
                    bull_fvg_top_arr[fvg_bar] = fvg_top
                    bull_fvg_bot_arr[fvg_bar] = fvg_bot

            choch = setup["choch_level"]
            if not np.isnan(choch) and c > choch:
                # Breaker confirmed
                ob_h = setup["ob_high"]
                ob_l = setup["ob_low"]

                bull_breaker_arr[i]        = True
                bull_breaker_top_arr[i]    = ob_h
                bull_breaker_bot_arr[i]    = ob_l
                bull_breaker_ob_bar_arr[i] = setup["ob_bar"]

                if not np.isnan(setup["fvg_top"]):
                    fvg_top = setup["fvg_top"]
                    fvg_bot = setup["fvg_bot"]
                    overlap_top = min(ob_h,  fvg_top)
                    overlap_bot = max(ob_l,  fvg_bot)
                    if overlap_top > overlap_bot:
                        armed_bull.append({
                            "zone_top": overlap_top,
                            "zone_bot": overlap_bot,
                            "sl":       ob_l,
                            "bar":      i,
                            "fvg_bar":  setup["fvg_bar"],
                            "ob_bar":   setup["ob_bar"],
                        })

                continue

            still_pending_bull.append(setup)

        pending_bull = still_pending_bull

        # -------------------------------------------------------------------
        # E. Expire old armed zones
        # -------------------------------------------------------------------
        armed_bear = [z for z in armed_bear if (i - z["bar"]) <= zone_timeout]
        armed_bull = [z for z in armed_bull if (i - z["bar"]) <= zone_timeout]

        # -------------------------------------------------------------------
        # F. Entry: price retraces into an armed zone
        # -------------------------------------------------------------------

        # SELL: high taps into the bearish overlap zone (retrace UP into it)
        hit_bear = []
        for z in armed_bear:
            if h >= z["zone_bot"]:  # wick reaches the bottom of the zone
                sell_signal_arr[i]   = True
                sell_zone_top_arr[i] = z["zone_top"]
                sell_zone_bot_arr[i] = z["zone_bot"]
                sl      = z["sl"]
                entry   = c
                sl_dist = abs(sl - entry)
                sell_sl_arr[i]      = sl
                sell_tp_arr[i]      = entry - 2.0 * sl_dist
                sell_fvg_bar_arr[i] = z["fvg_bar"]
                sell_ob_bar_arr[i]  = z["ob_bar"]
                hit_bear.append(z)
                break  # one signal per bar

        for z in hit_bear:
            armed_bear.remove(z)

        # BUY: low taps into the bullish overlap zone (retrace DOWN into it)
        hit_bull = []
        for z in armed_bull:
            if l <= z["zone_top"]:  # wick reaches the top of the zone
                buy_signal_arr[i]   = True
                buy_zone_top_arr[i] = z["zone_top"]
                buy_zone_bot_arr[i] = z["zone_bot"]
                sl      = z["sl"]
                entry   = c
                sl_dist = abs(entry - sl)
                buy_sl_arr[i]      = sl
                buy_tp_arr[i]      = entry + 2.0 * sl_dist
                buy_fvg_bar_arr[i] = z["fvg_bar"]
                buy_ob_bar_arr[i]  = z["ob_bar"]
                hit_bull.append(z)
                break

        for z in hit_bull:
            armed_bull.remove(z)

    # -----------------------------------------------------------------------
    # Store all output columns
    # -----------------------------------------------------------------------
    df["bear_breaker"]        = bear_breaker_arr
    df["bear_breaker_top"]    = bear_breaker_top_arr
    df["bear_breaker_bot"]    = bear_breaker_bot_arr
    df["bear_breaker_ob_bar"] = bear_breaker_ob_bar_arr
    df["bear_fvg"]         = bear_fvg_arr
    df["bear_fvg_top"]     = bear_fvg_top_arr
    df["bear_fvg_bot"]     = bear_fvg_bot_arr

    df["bull_breaker"]        = bull_breaker_arr
    df["bull_breaker_top"]    = bull_breaker_top_arr
    df["bull_breaker_bot"]    = bull_breaker_bot_arr
    df["bull_breaker_ob_bar"] = bull_breaker_ob_bar_arr
    df["bull_fvg"]         = bull_fvg_arr
    df["bull_fvg_top"]     = bull_fvg_top_arr
    df["bull_fvg_bot"]     = bull_fvg_bot_arr

    df["sell_signal"]   = sell_signal_arr
    df["sell_zone_top"] = sell_zone_top_arr
    df["sell_zone_bot"] = sell_zone_bot_arr
    df["sell_sl_price"] = sell_sl_arr
    df["sell_tp_price"] = sell_tp_arr
    df["sell_fvg_bar"]  = sell_fvg_bar_arr  # integer positional index of the FVG bar
    df["sell_ob_bar"]   = sell_ob_bar_arr   # integer positional index of the OB candle

    df["buy_signal"]   = buy_signal_arr
    df["buy_zone_top"] = buy_zone_top_arr
    df["buy_zone_bot"] = buy_zone_bot_arr
    df["buy_sl_price"] = buy_sl_arr
    df["buy_tp_price"] = buy_tp_arr
    df["buy_fvg_bar"]  = buy_fvg_bar_arr    # integer positional index of the FVG bar
    df["buy_ob_bar"]   = buy_ob_bar_arr     # integer positional index of the OB candle

    return df


# ---------------------------------------------------------------------------
# Master signal calculator
# ---------------------------------------------------------------------------

def calculate_signals(
    df: pd.DataFrame,
    pivot_left: int = PIVOT_LEFT,
    pivot_right: int = PIVOT_RIGHT,
    zone_timeout: int = ZONE_TIMEOUT,
) -> pd.DataFrame:
    return detect_breaker_fvg_signals(
        df,
        pivot_left=pivot_left,
        pivot_right=pivot_right,
        zone_timeout=zone_timeout,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _build_zone_shapes(df: pd.DataFrame) -> list:
    """
    Draw ONLY zones that produced a confirmed signal (i.e. had a Breaker + FVG overlap).

    For each signal bar:
      - Bearish breaker block rectangle : 10 candles wide from the CHoCH (breaker) bar
      - Bearish FVG rectangle           : 10 candles wide from the FVG bar
      - Overlap zone rectangle          : highlighted, from signal bar, 10 candles wide

    Everything drawn only when there is a real overlap — no orphan zones.
    """
    shapes = []
    bar_dur = pd.Timedelta(hours=1)
    idx = df.index  # DatetimeIndex — use positional lookup via iloc for FVG bar

    # ---- BEARISH signals ----
    for sig_pos in np.where(df["sell_signal"].to_numpy())[0]:  # type: ignore[arg-type]
        sig_ts = idx[sig_pos]
        row    = df.iloc[sig_pos]

        # Breaker block: rectangle starts at the actual OB candle bar
        ob_bar_pos = int(row["sell_ob_bar"])
        if ob_bar_pos >= 0:
            ob_ts  = idx[ob_bar_pos]
            ob_top = float(df.iloc[ob_bar_pos]["bear_breaker_top"])
            ob_bot = float(df.iloc[ob_bar_pos]["bear_breaker_bot"])
            # bear_breaker_top/bot are stored on the CHoCH bar, not the OB bar —
            # so search the CHoCH bar (bear_breaker=True) for the price levels
            # by walking forward from ob_bar_pos to find the matching choch bar
            brk_top, brk_bot = np.nan, np.nan
            for k in range(ob_bar_pos + 1, min(ob_bar_pos + 60, len(df))):
                if df.iloc[k]["bear_breaker"]:
                    brk_top = float(df.iloc[k]["bear_breaker_top"])
                    brk_bot = float(df.iloc[k]["bear_breaker_bot"])
                    break
            if np.isnan(brk_top):
                # fallback: use values stored on whatever bar has them
                brk_top = float(row["sell_zone_top"]) if not np.isnan(row["sell_zone_top"]) else np.nan
                brk_bot = float(row["sell_zone_bot"]) if not np.isnan(row["sell_zone_bot"]) else np.nan

            if not np.isnan(brk_top):
                shapes.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=ob_ts,
                    x1=sig_ts,  # ends when price taps in — breaker is consumed  # type: ignore[operator]
                    y0=brk_bot, y1=brk_top,
                    fillcolor="rgba(33,150,243,0.18)",
                    line=dict(color="rgba(33,150,243,0.9)", width=1),
                    layer="below",
                ))

        # FVG rectangle: 10 candles from the FVG bar
        fvg_bar_pos = int(row["sell_fvg_bar"])
        if fvg_bar_pos >= 0:
            fvg_ts  = idx[fvg_bar_pos]
            fvg_top = float(df.iloc[fvg_bar_pos]["bear_fvg_top"])
            fvg_bot = float(df.iloc[fvg_bar_pos]["bear_fvg_bot"])
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=fvg_ts,
                x1=fvg_ts + bar_dur * 10,  # type: ignore[operator]
                y0=fvg_bot, y1=fvg_top,
                fillcolor="rgba(239,83,80,0.12)",
                line=dict(color="rgba(239,83,80,0.6)", width=1, dash="dot"),
                layer="below",
            ))

        # Overlap zone: brighter, from signal bar, 10 candles wide
        ov_top = float(row["sell_zone_top"])
        ov_bot = float(row["sell_zone_bot"])
        if not (np.isnan(ov_top) or np.isnan(ov_bot)):
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=sig_ts,
                x1=sig_ts + bar_dur * 10,  # type: ignore[operator]
                y0=ov_bot, y1=ov_top,
                fillcolor="rgba(239,83,80,0.35)",
                line=dict(color="#ef5350", width=2),
                layer="below",
            ))

    # ---- BULLISH signals ----
    for sig_pos in np.where(df["buy_signal"].to_numpy())[0]:  # type: ignore[arg-type]
        sig_ts = idx[sig_pos]
        row    = df.iloc[sig_pos]

        # Breaker block: rectangle starts at the actual OB candle bar
        ob_bar_pos = int(row["buy_ob_bar"])
        if ob_bar_pos >= 0:
            ob_ts  = idx[ob_bar_pos]
            brk_top, brk_bot = np.nan, np.nan
            for k in range(ob_bar_pos + 1, min(ob_bar_pos + 60, len(df))):
                if df.iloc[k]["bull_breaker"]:
                    brk_top = float(df.iloc[k]["bull_breaker_top"])
                    brk_bot = float(df.iloc[k]["bull_breaker_bot"])
                    break

            if not np.isnan(brk_top):
                shapes.append(dict(
                    type="rect", xref="x", yref="y",
                    x0=ob_ts,
                    x1=sig_ts,  # ends when price taps in — breaker is consumed  # type: ignore[operator]
                    y0=brk_bot, y1=brk_top,
                    fillcolor="rgba(33,150,243,0.18)",
                    line=dict(color="rgba(33,150,243,0.9)", width=1),
                    layer="below",
                ))

        fvg_bar_pos = int(row["buy_fvg_bar"])
        if fvg_bar_pos >= 0:
            fvg_ts  = idx[fvg_bar_pos]
            fvg_top = float(df.iloc[fvg_bar_pos]["bull_fvg_top"])
            fvg_bot = float(df.iloc[fvg_bar_pos]["bull_fvg_bot"])
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=fvg_ts,
                x1=fvg_ts + bar_dur * 10,  # type: ignore[operator]
                y0=fvg_bot, y1=fvg_top,
                fillcolor="rgba(38,166,154,0.12)",
                line=dict(color="rgba(38,166,154,0.6)", width=1, dash="dot"),
                layer="below",
            ))

        ov_top = float(row["buy_zone_top"])
        ov_bot = float(row["buy_zone_bot"])
        if not (np.isnan(ov_top) or np.isnan(ov_bot)):
            shapes.append(dict(
                type="rect", xref="x", yref="y",
                x0=sig_ts,
                x1=sig_ts + bar_dur * 10,  # type: ignore[operator]
                y0=ov_bot, y1=ov_top,
                fillcolor="rgba(38,166,154,0.35)",
                line=dict(color="#26a69a", width=2),
                layer="below",
            ))

    return shapes


def plot_unicorn_chart(df: pd.DataFrame, symbol: str = "EURUSD") -> go.Figure:
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(f"{symbol} 1H — ICT Unicorn (Breaker + FVG Overlap)", "Volume"),
    )

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC",
        increasing_line_color="#26a69a", increasing_fillcolor="#26a69a",
        decreasing_line_color="#ef5350", decreasing_fillcolor="#ef5350",
    ), row=1, col=1)

    # Sell signals
    ss = df[df["sell_signal"]]
    if len(ss):
        fig.add_trace(go.Scatter(
            x=ss.index, y=ss["high"] * 1.0006,
            mode="markers", name="SELL Entry",
            marker=dict(symbol="triangle-down", size=16,
                        color="#f44336", line=dict(color="#b71c1c", width=2)),
        ), row=1, col=1)

    # Buy signals
    bs = df[df["buy_signal"]]
    if len(bs):
        fig.add_trace(go.Scatter(
            x=bs.index, y=bs["low"] * 0.9994,
            mode="markers", name="BUY Entry",
            marker=dict(symbol="triangle-up", size=16,
                        color="#00e676", line=dict(color="#1b5e20", width=2)),
        ), row=1, col=1)

    # Volume
    if "volume" in df.columns:
        vol_colors = [
            "#26a69a" if c >= o else "#ef5350"
            for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(go.Bar(
            x=df.index, y=df["volume"],
            name="Volume", marker_color=vol_colors, opacity=0.5,
        ), row=2, col=1)

    shapes = _build_zone_shapes(df)

    n_sell = int(df["sell_signal"].sum())
    n_buy  = int(df["buy_signal"].sum())
    n_bbk  = int(df["bear_breaker"].sum())
    n_buk  = int(df["bull_breaker"].sum())
    n_bfvg = int(df["bear_fvg"].sum())
    n_ufvg = int(df["bull_fvg"].sum())

    fig.update_layout(
        height=950,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=shapes,
        margin=dict(t=120),
        title=dict(
            text=(
                f"<b>{symbol} 1H — ICT Unicorn (Breaker + FVG Overlap)</b><br>"
                f"<sup>"
                f"Bear breakers: {n_bbk}  |  Bull breakers: {n_buk}  |  "
                f"Bear FVGs: {n_bfvg}  |  Bull FVGs: {n_ufvg}  |  "
                f"SELL signals: {n_sell}  |  BUY signals: {n_buy}"
                f"</sup>"
            ),
            x=0.5, xanchor="center",
            y=0.98, yanchor="top",
        ),
    )
    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    symbol: str = "EURUSD",
    start_date: str = "2025-10-01",
    end_date: str = "2026-02-27",
    pivot_left: int = PIVOT_LEFT,
    pivot_right: int = PIVOT_RIGHT,
    zone_timeout: int = ZONE_TIMEOUT,
    show_chart: bool = True,
    save_chart: bool = True,
) -> tuple[pd.DataFrame, go.Figure]:
    print(f"Loading {symbol} 1-minute data ({start_date} to {end_date})...")
    loader = DataLoader()
    df_1m = loader.load_fx(symbol, start_date=start_date, end_date=end_date)
    print(f"  Loaded {len(df_1m):,} 1-minute bars.")

    print("Resampling to 1H...")
    df = loader.resample_ohlcv(df_1m, "1h")
    print(f"  {len(df):,} hourly bars  ({df.index.min()} -> {df.index.max()}).")

    print(f"Detecting setups  (pivot L/R={pivot_left}/{pivot_right}, timeout={zone_timeout})...")
    df = calculate_signals(df, pivot_left=pivot_left, pivot_right=pivot_right,
                           zone_timeout=zone_timeout)

    print(f"  Bear FVGs (displacement) : {int(df['bear_fvg'].sum())}")
    print(f"  Bull FVGs (displacement) : {int(df['bull_fvg'].sum())}")
    print(f"  Bearish Breakers         : {int(df['bear_breaker'].sum())}")
    print(f"  Bullish Breakers         : {int(df['bull_breaker'].sum())}")
    print(f"  SELL signals             : {int(df['sell_signal'].sum())}")
    print(f"  BUY  signals             : {int(df['buy_signal'].sum())}")

    print("Building chart...")
    fig = plot_unicorn_chart(df, symbol=symbol)

    if save_chart:
        reports_dir = Path(__file__).resolve().parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        out = reports_dir / f"unicorn_ict_{symbol}_1H.html"
        fig.write_html(str(out))
        print(f"  Chart saved -> {out}")

    if show_chart:
        fig.show()

    return df, fig


if __name__ == "__main__":
    run_experiment()
