"""
ICT Unicorn Model — Strategy Visualization

Full Unicorn setup detection logic:

  BEARISH UNICORN (sell)
  ----------------------
  1. A confirmed swing high exists (left-side pivot, N bars of lower highs on each side).
  2. Price prints a HIGHER HIGH — takes out that swing high (manipulation sweep up).
  3. Within the next `choch_window` bars, price breaks BELOW the most recent confirmed
     swing low — this is the Change of Character (CHoCH), confirming a trend shift down.
  4. A bearish FVG formed during the displacement move that broke the swing low is
     recorded as the entry zone.
  5. On any subsequent bar, if price retraces INTO that bearish FVG (high >= fvg_bot)
     a SELL signal fires.  The FVG is then consumed (one signal per setup).

  BULLISH UNICORN (buy) — exact mirror
  -------------------------------------
  1. Confirmed swing low exists.
  2. Price prints a LOWER LOW — takes out that swing low (sweep down).
  3. Within `choch_window` bars, price breaks ABOVE the most recent confirmed swing high.
  4. A bullish FVG formed during the displacement up is recorded.
  5. Price retraces INTO the bullish FVG (low <= fvg_top) → BUY signal.

All detection is strictly left-looking (no lookahead bias).
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
# Pivot swing high / swing low  (left-side only — no lookahead)
# ---------------------------------------------------------------------------


def calculate_pivots(df: pd.DataFrame, left: int = 3, right: int = 3) -> pd.DataFrame:
    """
    Detect pivot swing highs and swing lows using a left+right bar window.

    A pivot high at bar i requires:
        df['high'][i] == max(df['high'][i-left : i+right+1])
    A pivot low at bar i requires:
        df['low'][i]  == min(df['low'][i-left  : i+right+1])

    Because we use center=True rolling this introduces a right-side look of
    `right` bars — we shift the result forward by `right` bars so that the
    pivot is only *confirmed* once the right-side bars have closed.
    This avoids lookahead: the pivot at bar i is known at bar i+right.

    Adds columns:
        pivot_high : float — confirmed pivot high price (NaN elsewhere)
        pivot_low  : float — confirmed pivot low price  (NaN elsewhere)
    """
    df = df.copy()
    n = left + right + 1

    roll_max = df["high"].rolling(window=n, center=True).max()
    roll_min = df["low"].rolling(window=n, center=True).min()

    raw_ph = np.where(df["high"] == roll_max, df["high"], np.nan)
    raw_pl = np.where(df["low"] == roll_min, df["low"], np.nan)

    # Shift forward by `right` so the signal only appears after right-side bars close
    df["pivot_high"] = pd.Series(raw_ph, index=df.index).shift(right)
    df["pivot_low"] = pd.Series(raw_pl, index=df.index).shift(right)

    return df


# ---------------------------------------------------------------------------
# FVG detection
# ---------------------------------------------------------------------------


def calculate_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect all Fair Value Gaps.

    Bullish FVG at bar i  : high[i-2] < low[i]   (gap between bar i-2 and bar i)
    Bearish FVG at bar i  : low[i-2]  > high[i]

    The FVG is attributed to bar i (the third candle), which is fully closed.
    No lookahead.

    Adds columns:
        bull_fvg, bear_fvg            : bool
        bull_fvg_top, bull_fvg_bot    : zone edges
        bear_fvg_top, bear_fvg_bot    : zone edges
    """
    df = df.copy()
    high = df["high"]
    low = df["low"]

    bull = high.shift(2) < low
    bear = low.shift(2) > high

    df["bull_fvg"] = bull.fillna(False)
    df["bear_fvg"] = bear.fillna(False)
    df["bull_fvg_top"] = np.where(df["bull_fvg"], low, np.nan)
    df["bull_fvg_bot"] = np.where(df["bull_fvg"], high.shift(2), np.nan)
    df["bear_fvg_top"] = np.where(df["bear_fvg"], low.shift(2), np.nan)
    df["bear_fvg_bot"] = np.where(df["bear_fvg"], high, np.nan)

    return df


# ---------------------------------------------------------------------------
# Unicorn setup state machine
# ---------------------------------------------------------------------------


def detect_unicorn_setups(
    df: pd.DataFrame,
    choch_window: int = 10,
    fvg_timeout: int = 20,
) -> pd.DataFrame:
    """
    Bar-by-bar state machine that detects the full Unicorn sequence and
    generates entry signals when price retraces into the setup FVG.

    Parameters
    ----------
    df           : DataFrame with pivot_high, pivot_low, bull_fvg / bear_fvg cols
    choch_window : Max bars allowed between the sweep and the CHoCH confirmation.
                   If CHoCH doesn't happen within this window the setup resets to
                   idle and the same bar is immediately re-evaluated for a new sweep.
    fvg_timeout  : Max bars an armed FVG zone waits for price to retrace into it.
                   If the retrace doesn't happen within this many bars the setup is
                   invalidated, state returns to idle, and new setups are sought.

    New columns added
    -----------------
    bear_sweep        : bool  — bar where a swing high was taken out (sweep up)
    bear_choch        : bool  — bar where price broke below swing low (CHoCH down)
    bear_fvg_armed    : bool  — a bearish Unicorn FVG is now active / waiting
    bear_fvg_arm_top  : float — top of armed bearish FVG entry zone
    bear_fvg_arm_bot  : float — bottom of armed bearish FVG entry zone
    sell_signal       : bool  — price has tapped into the armed bearish FVG

    bull_sweep        : bool
    bull_choch        : bool
    bull_fvg_armed    : bool
    bull_fvg_arm_top  : float
    bull_fvg_arm_bot  : float
    buy_signal        : bool
    """
    df = df.copy()
    n = len(df)

    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values

    ph = df["pivot_high"].values  # confirmed pivot highs (NaN if none)
    pl = df["pivot_low"].values  # confirmed pivot lows  (NaN if none)

    # FVG arrays
    bull_fvg = df["bull_fvg"].values
    bull_fvg_top = df["bull_fvg_top"].values
    bull_fvg_bot = df["bull_fvg_bot"].values
    bear_fvg = df["bear_fvg"].values
    bear_fvg_top = df["bear_fvg_top"].values
    bear_fvg_bot = df["bear_fvg_bot"].values

    # Output arrays
    bear_sweep_arr = np.zeros(n, dtype=bool)
    bear_choch_arr = np.zeros(n, dtype=bool)
    bear_fvg_armed_arr = np.zeros(n, dtype=bool)
    bear_fvg_arm_top_arr = np.full(n, np.nan)
    bear_fvg_arm_bot_arr = np.full(n, np.nan)
    sell_signal_arr = np.zeros(n, dtype=bool)
    sell_sl_price_arr = np.full(n, np.nan)  # SL price at sell signal bar
    sell_tp_price_arr = np.full(n, np.nan)  # TP price at sell signal bar

    bull_sweep_arr = np.zeros(n, dtype=bool)
    bull_choch_arr = np.zeros(n, dtype=bool)
    bull_fvg_armed_arr = np.zeros(n, dtype=bool)
    bull_fvg_arm_top_arr = np.full(n, np.nan)
    bull_fvg_arm_bot_arr = np.full(n, np.nan)
    buy_signal_arr = np.zeros(n, dtype=bool)
    buy_sl_price_arr = np.full(n, np.nan)  # SL price at buy signal bar
    buy_tp_price_arr = np.full(n, np.nan)  # TP price at buy signal bar

    # ---- rolling pivot trackers ----
    last_pivot_high = np.nan
    last_pivot_low = np.nan
    second_last_pivot_high = (
        np.nan
    )  # pivot high before the current last — used as TP for buys
    second_last_pivot_low = (
        np.nan
    )  # pivot low  before the current last — used as TP for sells

    # Bear state machine variables
    bear_state = "idle"  # idle | swept | armed
    bear_choch_level = np.nan  # pivot low to break for CHoCH
    bear_sweep_high = np.nan  # high of the sweep bar (SL goes above this)
    bear_tp_level = np.nan  # prior pivot low — the liquidity target below
    bear_sweep_bar = -1
    bear_arm_bar = -1  # bar when armed (for fvg_timeout)
    bear_fvg_top_val = np.nan
    bear_fvg_bot_val = np.nan

    # Bull state machine variables
    bull_state = "idle"
    bull_choch_level = np.nan  # pivot high to break for CHoCH
    bull_sweep_low = np.nan  # low of the sweep bar (SL goes below this)
    bull_tp_level = np.nan  # prior pivot high — the liquidity target above
    bull_sweep_bar = -1
    bull_arm_bar = -1
    bull_fvg_top_val = np.nan
    bull_fvg_bot_val = np.nan

    for i in range(n):
        h = highs[i]
        l = lows[i]
        c = closes[i]

        # --- Update rolling pivot trackers ---
        if not np.isnan(ph[i]):
            second_last_pivot_high = last_pivot_high
            last_pivot_high = ph[i]
        if not np.isnan(pl[i]):
            second_last_pivot_low = last_pivot_low
            last_pivot_low = pl[i]

        # ==================================================================
        # BEARISH UNICORN STATE MACHINE
        # ==================================================================

        # Timeout in swept state: reset to idle then fall through to re-check
        # this same bar for a new sweep (no `continue` — don't skip the bar).
        if bear_state == "swept" and (i - bear_sweep_bar) > choch_window:
            bear_state = "idle"

        # Timeout in armed state: FVG not filled within fvg_timeout bars —
        # invalidate and return to idle to look for a fresh setup.
        if bear_state == "armed" and (i - bear_arm_bar) > fvg_timeout:
            bear_state = "idle"

        if bear_state == "idle":
            if not np.isnan(last_pivot_high) and not np.isnan(last_pivot_low):
                if h > last_pivot_high:
                    bear_state = "swept"
                    bear_choch_level = last_pivot_low
                    bear_sweep_high = h
                    # TP: next liquidity pool below — the pivot low before the CHoCH level
                    bear_tp_level = second_last_pivot_low
                    bear_sweep_bar = i
                    bear_sweep_arr[i] = True

        elif bear_state == "swept":
            # CHoCH: close breaks below the swing low that existed before the sweep
            if c < bear_choch_level:
                bear_choch_arr[i] = True

                # Scan back for the nearest bearish FVG between sweep bar and now
                found_fvg = False
                for j in range(i, max(bear_sweep_bar - 1, -1), -1):
                    if bear_fvg[j]:
                        bear_fvg_top_val = bear_fvg_top[j]
                        bear_fvg_bot_val = bear_fvg_bot[j]
                        found_fvg = True
                        break

                if found_fvg:
                    bear_state = "armed"
                    bear_arm_bar = i
                    bear_fvg_armed_arr[i] = True
                    bear_fvg_arm_top_arr[i] = bear_fvg_top_val
                    bear_fvg_arm_bot_arr[i] = bear_fvg_bot_val
                else:
                    bear_state = "idle"

        elif bear_state == "armed":
            bear_fvg_armed_arr[i] = True
            bear_fvg_arm_top_arr[i] = bear_fvg_top_val
            bear_fvg_arm_bot_arr[i] = bear_fvg_bot_val

            # Entry: high retraces into the bearish FVG zone
            if h >= bear_fvg_bot_val:
                sell_signal_arr[i] = True
                # SL: above the sweep high; TP: prior pivot low (next liquidity target)
                sell_sl_price_arr[i] = bear_sweep_high
                sell_tp_price_arr[i] = (
                    bear_tp_level if not np.isnan(bear_tp_level) else bear_choch_level
                )
                bear_state = "idle"
            # Invalidate: price closes above the FVG top (zone fully violated)
            elif c > bear_fvg_top_val:
                bear_state = "idle"
            # Invalidate: price closes above the FVG top (zone fully violated)
            elif c > bear_fvg_top_val:
                bear_state = "idle"

        # ==================================================================
        # BULLISH UNICORN STATE MACHINE (mirror)
        # ==================================================================

        # Timeout in swept state: reset and fall through — don't skip with continue
        if bull_state == "swept" and (i - bull_sweep_bar) > choch_window:
            bull_state = "idle"

        # Timeout in armed state: FVG not filled within fvg_timeout bars
        if bull_state == "armed" and (i - bull_arm_bar) > fvg_timeout:
            bull_state = "idle"

        if bull_state == "idle":
            if not np.isnan(last_pivot_low) and not np.isnan(last_pivot_high):
                if l < last_pivot_low:
                    bull_state = "swept"
                    bull_choch_level = last_pivot_high
                    bull_sweep_low = l
                    # TP: next liquidity pool above — the pivot high before the CHoCH level
                    bull_tp_level = second_last_pivot_high
                    bull_sweep_bar = i
                    bull_sweep_arr[i] = True

        elif bull_state == "swept":
            if c > bull_choch_level:
                bull_choch_arr[i] = True

                found_fvg = False
                for j in range(i, max(bull_sweep_bar - 1, -1), -1):
                    if bull_fvg[j]:
                        bull_fvg_top_val = bull_fvg_top[j]
                        bull_fvg_bot_val = bull_fvg_bot[j]
                        found_fvg = True
                        break

                if found_fvg:
                    bull_state = "armed"
                    bull_arm_bar = i
                    bull_fvg_armed_arr[i] = True
                    bull_fvg_arm_top_arr[i] = bull_fvg_top_val
                    bull_fvg_arm_bot_arr[i] = bull_fvg_bot_val
                else:
                    bull_state = "idle"

        elif bull_state == "armed":
            bull_fvg_armed_arr[i] = True
            bull_fvg_arm_top_arr[i] = bull_fvg_top_val
            bull_fvg_arm_bot_arr[i] = bull_fvg_bot_val

            # Entry: low retraces into the bullish FVG zone
            if l <= bull_fvg_top_val:
                buy_signal_arr[i] = True
                # SL: below the sweep low; TP: prior pivot high (next liquidity target)
                buy_sl_price_arr[i] = bull_sweep_low
                buy_tp_price_arr[i] = (
                    bull_tp_level if not np.isnan(bull_tp_level) else bull_choch_level
                )
                bull_state = "idle"
            # Invalidate: price closes below the FVG bottom
            elif c < bull_fvg_bot_val:
                bull_state = "idle"
            # Invalidate: price closes below the FVG bottom
            elif c < bull_fvg_bot_val:
                bull_state = "idle"

    # Store results
    df["bear_sweep"] = bear_sweep_arr
    df["bear_choch"] = bear_choch_arr
    df["bear_fvg_armed"] = bear_fvg_armed_arr
    df["bear_fvg_arm_top"] = bear_fvg_arm_top_arr
    df["bear_fvg_arm_bot"] = bear_fvg_arm_bot_arr
    df["sell_signal"] = sell_signal_arr
    df["sell_sl_price"] = sell_sl_price_arr
    df["sell_tp_price"] = sell_tp_price_arr

    df["bull_sweep"] = bull_sweep_arr
    df["bull_choch"] = bull_choch_arr
    df["bull_fvg_armed"] = bull_fvg_armed_arr
    df["bull_fvg_arm_top"] = bull_fvg_arm_top_arr
    df["bull_fvg_arm_bot"] = bull_fvg_arm_bot_arr
    df["buy_signal"] = buy_signal_arr
    df["buy_sl_price"] = buy_sl_price_arr
    df["buy_tp_price"] = buy_tp_price_arr

    return df


# ---------------------------------------------------------------------------
# Master signal calculator
# ---------------------------------------------------------------------------


def calculate_signals(
    df: pd.DataFrame,
    pivot_left: int = 3,
    pivot_right: int = 3,
    choch_window: int = 10,
    fvg_timeout: int = 20,
) -> pd.DataFrame:
    df = calculate_pivots(df, left=pivot_left, right=pivot_right)
    df = calculate_fvg(df)
    df = detect_unicorn_setups(df, choch_window=choch_window, fvg_timeout=fvg_timeout)
    return df


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def build_armed_fvg_shapes(df: pd.DataFrame) -> list:
    """
    Draw the armed Unicorn FVG zones as shaded rectangles.
    Each zone is drawn from the bar it becomes armed to when it fires or dies.
    """
    shapes = []
    bar_dur = pd.Timedelta(hours=1)

    # --- Bearish armed zones ---
    in_zone = False
    zone_start = None
    zone_top = np.nan
    zone_bot = np.nan

    for ts, row in df.iterrows():
        if row["bear_fvg_armed"] and not in_zone:
            in_zone = True
            zone_start = ts
            zone_top = row["bear_fvg_arm_top"]
            zone_bot = row["bear_fvg_arm_bot"]
        elif not row["bear_fvg_armed"] and in_zone:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=zone_start,
                    x1=ts,
                    y0=zone_bot,
                    y1=zone_top,
                    fillcolor="rgba(239, 83, 80, 0.22)",
                    line=dict(color="rgba(239, 83, 80, 0.6)", width=1),
                    layer="below",
                )
            )
            in_zone = False
    # close any still-open zone at last bar
    if in_zone:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=zone_start,
                x1=df.index[-1] + bar_dur * 5,
                y0=zone_bot,
                y1=zone_top,
                fillcolor="rgba(239, 83, 80, 0.22)",
                line=dict(color="rgba(239, 83, 80, 0.6)", width=1),
                layer="below",
            )
        )

    # --- Bullish armed zones ---
    in_zone = False
    for ts, row in df.iterrows():
        if row["bull_fvg_armed"] and not in_zone:
            in_zone = True
            zone_start = ts
            zone_top = row["bull_fvg_arm_top"]
            zone_bot = row["bull_fvg_arm_bot"]
        elif not row["bull_fvg_armed"] and in_zone:
            shapes.append(
                dict(
                    type="rect",
                    xref="x",
                    yref="y",
                    x0=zone_start,
                    x1=ts,
                    y0=zone_bot,
                    y1=zone_top,
                    fillcolor="rgba(38, 166, 154, 0.22)",
                    line=dict(color="rgba(38, 166, 154, 0.6)", width=1),
                    layer="below",
                )
            )
            in_zone = False
    if in_zone:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=zone_start,
                x1=df.index[-1] + bar_dur * 5,
                y0=zone_bot,
                y1=zone_top,
                fillcolor="rgba(38, 166, 154, 0.22)",
                line=dict(color="rgba(38, 166, 154, 0.6)", width=1),
                layer="below",
            )
        )

    return shapes


def plot_unicorn_chart(df: pd.DataFrame, symbol: str = "EURUSD") -> go.Figure:
    """
    Full Unicorn chart:
      - Candlesticks
      - Pivot High / Low markers (liquidity pools)
      - Sweep markers (manipulation bars)
      - CHoCH markers (market structure shift bars)
      - Armed FVG zones (shaded — active entry windows)
      - Buy / Sell signal markers
      - Volume
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(f"{symbol} 1H — ICT Unicorn Setups", "Volume"),
    )

    # --- Candlesticks ---
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
        ),
        row=1,
        col=1,
    )

    # --- Pivot Highs (liquidity pools above) ---
    ph_bars = df[df["pivot_high"].notna()]
    if len(ph_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=ph_bars.index,
                y=ph_bars["pivot_high"] * 1.0003,
                mode="markers",
                name="Pivot High (BSL)",
                marker=dict(
                    symbol="line-ew",
                    size=10,
                    color="rgba(239, 83, 80, 0.5)",
                    line=dict(color="#ef5350", width=2),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Pivot Lows (liquidity pools below) ---
    pl_bars = df[df["pivot_low"].notna()]
    if len(pl_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=pl_bars.index,
                y=pl_bars["pivot_low"] * 0.9997,
                mode="markers",
                name="Pivot Low (SSL)",
                marker=dict(
                    symbol="line-ew",
                    size=10,
                    color="rgba(38, 166, 154, 0.5)",
                    line=dict(color="#26a69a", width=2),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Bearish sweeps (manipulation high) ---
    bs_bars = df[df["bear_sweep"]]
    if len(bs_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=bs_bars.index,
                y=bs_bars["high"] * 1.0005,
                mode="markers",
                name="Bear Sweep (HH)",
                marker=dict(
                    symbol="triangle-down",
                    size=13,
                    color="#ff5252",
                    line=dict(color="#b71c1c", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Bullish sweeps (manipulation low) ---
    lls_bars = df[df["bull_sweep"]]
    if len(lls_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=lls_bars.index,
                y=lls_bars["low"] * 0.9995,
                mode="markers",
                name="Bull Sweep (LL)",
                marker=dict(
                    symbol="triangle-up",
                    size=13,
                    color="#69f0ae",
                    line=dict(color="#1b5e20", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Bearish CHoCH bars ---
    bc_bars = df[df["bear_choch"]]
    if len(bc_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=bc_bars.index,
                y=bc_bars["low"] * 0.9993,
                mode="markers+text",
                name="Bear CHoCH",
                text=["CHoCH"] * len(bc_bars),
                textposition="bottom center",
                textfont=dict(size=9, color="#ff9100"),
                marker=dict(
                    symbol="x",
                    size=10,
                    color="#ff9100",
                    line=dict(color="#e65100", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Bullish CHoCH bars ---
    buc_bars = df[df["bull_choch"]]
    if len(buc_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=buc_bars.index,
                y=buc_bars["high"] * 1.0007,
                mode="markers+text",
                name="Bull CHoCH",
                text=["CHoCH"] * len(buc_bars),
                textposition="top center",
                textfont=dict(size=9, color="#40c4ff"),
                marker=dict(
                    symbol="x",
                    size=10,
                    color="#40c4ff",
                    line=dict(color="#01579b", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Sell signals ---
    sell_bars = df[df["sell_signal"]]
    if len(sell_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_bars.index,
                y=sell_bars["high"] * 1.0008,
                mode="markers",
                name="SELL Entry",
                marker=dict(
                    symbol="triangle-down",
                    size=16,
                    color="#f44336",
                    line=dict(color="#b71c1c", width=2),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Buy signals ---
    buy_bars = df[df["buy_signal"]]
    if len(buy_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_bars.index,
                y=buy_bars["low"] * 0.9992,
                mode="markers",
                name="BUY Entry",
                marker=dict(
                    symbol="triangle-up",
                    size=16,
                    color="#00e676",
                    line=dict(color="#1b5e20", width=2),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Volume ---
    if "volume" in df.columns:
        vol_colors = [
            "#26a69a" if c >= o else "#ef5350" for c, o in zip(df["close"], df["open"])
        ]
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                name="Volume",
                marker_color=vol_colors,
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

    # --- Armed FVG zone shapes ---
    shapes = build_armed_fvg_shapes(df)

    n_sells = int(df["sell_signal"].sum())
    n_buys = int(df["buy_signal"].sum())

    fig.update_layout(
        height=950,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=shapes,
        title=dict(
            text=(
                f"<b>{symbol} 1H — ICT Unicorn Model</b><br>"
                f"<sup>Sweep + CHoCH + FVG retrace logic  |  "
                f"Sell setups: {n_sells}  |  Buy setups: {n_buys}  |  "
                f"Red zone = armed bearish FVG  |  Green zone = armed bullish FVG</sup>"
            ),
            x=0.5,
            xanchor="center",
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
    asset_type: str = "fx",
    start_date: str = "2025-10-01",
    end_date: str = "2026-02-27",
    pivot_left: int = 3,
    pivot_right: int = 3,
    choch_window: int = 10,
    fvg_timeout: int = 20,
    show_chart: bool = True,
    save_chart: bool = True,
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Load EURUSD 1-minute data, resample to 1H, detect Unicorn setups, plot.
    """
    print(f"Loading {symbol} 1-minute data ({start_date} to {end_date})...")
    loader = DataLoader()
    df_1m = loader.load_fx(symbol, start_date=start_date, end_date=end_date)
    print(f"  Loaded {len(df_1m):,} 1-minute bars.")

    print("Resampling to 1H...")
    df = loader.resample_ohlcv(df_1m, "1h")
    print(f"  {len(df):,} hourly bars  ({df.index.min()} -> {df.index.max()}).")

    print("Detecting Unicorn setups...")
    df = calculate_signals(
        df,
        pivot_left=pivot_left,
        pivot_right=pivot_right,
        choch_window=choch_window,
        fvg_timeout=fvg_timeout,
    )

    n_ph = int(df["pivot_high"].notna().sum())
    n_pl = int(df["pivot_low"].notna().sum())
    n_bsw = int(df["bear_sweep"].sum())
    n_llsw = int(df["bull_sweep"].sum())
    n_bchoch = int(df["bear_choch"].sum())
    n_buchoch = int(df["bull_choch"].sum())
    n_sells = int(df["sell_signal"].sum())
    n_buys = int(df["buy_signal"].sum())

    print(f"  Pivot Highs (BSL)  : {n_ph}")
    print(f"  Pivot Lows  (SSL)  : {n_pl}")
    print(f"  Bear Sweeps (HH)   : {n_bsw}")
    print(f"  Bull Sweeps (LL)   : {n_llsw}")
    print(f"  Bear CHoCH         : {n_bchoch}")
    print(f"  Bull CHoCH         : {n_buchoch}")
    print(f"  SELL signals       : {n_sells}")
    print(f"  BUY  signals       : {n_buys}")

    print("Building chart...")
    fig = plot_unicorn_chart(df, symbol=symbol)

    if save_chart:
        reports_dir = Path(__file__).resolve().parent / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_path = reports_dir / f"unicorn_ict_{symbol}_1H.html"
        fig.write_html(str(output_path))
        print(f"  Chart saved -> {output_path}")

    if show_chart:
        fig.show()

    return df, fig


if __name__ == "__main__":
    run_experiment()
