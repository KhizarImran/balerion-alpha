"""
ICT Silver Bullet Strategy — Signal Generation & Visualisation

Strategy:
  Asset:    US30 (Dow Jones), 1-minute data
  Sessions: Three ICT killzone windows (Eastern Time):
              London       03:00 - 03:59 ET
              New York AM  10:00 - 10:59 ET
              New York PM  14:00 - 14:59 ET

  Long entry conditions (all must be true):
    1. A rolling `liquidity_window`-bar LOW has been swept within the last
       `sweep_lookback` bars (bullish bias — stops below were hunted).
    2. The corresponding HIGH has NOT been swept (bias intact).
    3. A bullish FVG exists: high[N-2] < low[N] (two-bar upward imbalance).
    4. Price pulls back inside the FVG zone within `pullback_window` bars.
    5. Bar falls within one of the three killzone windows.

  Short entry conditions (mirror of long, inverted):
    1. A rolling `liquidity_window`-bar HIGH has been swept.
    2. The corresponding LOW has NOT been swept (bearish bias intact).
    3. A bearish FVG exists: low[N-2] > high[N] (two-bar downward imbalance).
    4. Price pulls back inside the bearish FVG zone within `pullback_window` bars.
    5. Bar falls within one of the three killzone windows.

  One entry per session per calendar day (first qualifying bar wins).
  A per-session cumulative loss cap (`session_cap_usd`) halts further
  entries in a session once the estimated loss exceeds the threshold.

  Stop Loss:   Long  — below swept liquidity low minus `sl_buffer_pts`
               Short — above swept liquidity high plus `sl_buffer_pts`
  Take Profit: `rr_ratio` x SL distance (long: above entry, short: below)
  Time exit:   52 bars if neither SL nor TP is hit.
"""

import os
import sys
import io
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["SMC_CREDIT"] = "0"

import numpy as np
import pandas as pd
import pytz
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "US30"
START_DATE = "2025-11-11"
END_DATE = "2026-02-25"
TIMEFRAME = "1min"

LIQUIDITY_WINDOW = 150  # rolling window (bars) for significant high/low levels
SWEEP_LOOKBACK = 50  # bars to look back for sweep detection
PULLBACK_WINDOW = 10  # bars the FVG zone stays alive for pullback entry
RR_RATIO = 2.0  # take profit = RR_RATIO x SL distance
SL_BUFFER_PTS = 0.0  # extra points beyond liquidity level for SL
SESSION_CAP_USD = -5_500.0  # max cumulative estimated loss per session per day

LOT_SIZE = 10  # lots per trade — used only for session cap estimation


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_ohlcv(
    symbol: str = SYMBOL,
    start: str = START_DATE,
    end: str = END_DATE,
    tf: str = TIMEFRAME,
) -> tuple:
    """
    Load OHLCV data for an index symbol, preferring Dukascopy data where
    available and falling back to MT5 data otherwise.

    Resolution order for 1-minute data:
      1. {symbol}_dukascopy_1m.parquet  — Dukascopy tick-aggregated 1m (preferred,
         timestamps are UTC)
      2. {symbol}_1m.parquet            — MT5 1-minute (fallback, timestamps are
         Europe/Athens broker server time)

    For non-1m timeframes the MT5 file is loaded and resampled.

    Returns:
        (df, source_tz) where source_tz is 'utc' or 'athens' — passed through
        to detect_setups() so the session filter uses the correct conversion.
    """
    loader = DataLoader()

    if tf in ("1min", "1m"):
        # Try Dukascopy 1m first (UTC timestamps)
        try:
            df = loader.load_index(
                symbol, start_date=start, end_date=end, timeframe="dukascopy_1m"
            )
            return df, "utc"
        except FileNotFoundError:
            pass
        # Fallback: MT5 1m (Athens server time)
        df = loader.load_index(symbol, start_date=start, end_date=end, timeframe="1m")
        return df, "athens"

    # Non-1m: load MT5 1m and resample (Athens server time)
    df = loader.load_index(symbol, start_date=start, end_date=end, timeframe="1m")
    return loader.resample_ohlcv(df, tf), "athens"


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

_ATHENS_TZ = pytz.timezone("Europe/Athens")  # MT5 broker server: UTC+2 / UTC+3
_UTC_TZ = pytz.timezone("UTC")  # Dukascopy timestamps are UTC
_EASTERN_TZ = pytz.timezone("America/New_York")

_KILLZONE_HOURS_ET = {3, 10, 14}  # London 03, NY AM 10, NY PM 14

_SESSION_NAMES = {3: "london", 10: "ny_am", 14: "ny_pm"}


def _get_et_hours(index: pd.DatetimeIndex, source_tz: str = "utc") -> np.ndarray:
    """
    Convert a naive DatetimeIndex to Eastern Time hours.

    Args:
        index:     Timezone-naive DatetimeIndex.
        source_tz: 'utc'    — Dukascopy data (timestamps are UTC)
                   'athens' — MT5 broker server time (Europe/Athens)
    """
    tz = _UTC_TZ if source_tz == "utc" else _ATHENS_TZ
    localised = index.tz_localize(tz, ambiguous="infer", nonexistent="shift_forward")
    eastern_idx = localised.tz_convert(_EASTERN_TZ)
    return eastern_idx.hour.to_numpy()


# ---------------------------------------------------------------------------
# Core signal detection
# ---------------------------------------------------------------------------


def detect_setups(
    df: pd.DataFrame,
    liquidity_window: int = LIQUIDITY_WINDOW,
    sweep_lookback: int = SWEEP_LOOKBACK,
    pullback_window: int = PULLBACK_WINDOW,
    rr_ratio: float = RR_RATIO,
    sl_buffer_pts: float = SL_BUFFER_PTS,
    session_cap_usd: float = SESSION_CAP_USD,
    lot_size: int = LOT_SIZE,
    source_tz: str = "utc",
) -> tuple[list, list, list]:
    """
    Detect ICT Silver Bullet long and short setups.

    Args:
        df:               OHLCV DataFrame with naive DatetimeIndex in broker
                          server time (Europe/Athens).
        liquidity_window: Rolling window (bars) for significant high/low levels.
        sweep_lookback:   Bars to look back for sweep detection.
        pullback_window:  Bars the FVG zone stays alive for pullback entry.
        rr_ratio:         Take profit = rr_ratio x SL distance.
        sl_buffer_pts:    Extra points beyond liquidity level for SL.
        session_cap_usd:  Max cumulative estimated loss per session per day.
        lot_size:         Lots per trade (for session cap estimation).

    Returns:
        (sell_setups, buy_setups, [])
        sell_setups and buy_setups are lists of dicts with keys:
            entry_ts   : pd.Timestamp   — index label of entry bar
            entry      : float          — entry price (close of entry bar)
            sl         : float          — stop loss price
            tp         : float          — take profit price
            direction  : str            — 'long' or 'short'
            session    : str            — 'london' | 'ny_am' | 'ny_pm'
            sl_dist    : float          — absolute SL distance in points
        The third element is an empty list (no HTF FVG filter for this strategy).
    """
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    idx = df.index
    n = len(df)

    # ------------------------------------------------------------------
    # 1. Rolling liquidity levels
    # ------------------------------------------------------------------
    liq_high = pd.Series(high).rolling(liquidity_window, min_periods=1).max().to_numpy()
    liq_low = pd.Series(low).rolling(liquidity_window, min_periods=1).min().to_numpy()

    # ------------------------------------------------------------------
    # 2. Sweep detection
    #    A level is swept when price touches it within the lookback window.
    # ------------------------------------------------------------------
    high_touched = high >= liq_high
    low_touched = low <= liq_low

    high_swept = (
        pd.Series(high_touched.astype(float))
        .rolling(sweep_lookback, min_periods=1)
        .max()
        .astype(bool)
        .to_numpy()
    )
    low_swept = (
        pd.Series(low_touched.astype(float))
        .rolling(sweep_lookback, min_periods=1)
        .max()
        .astype(bool)
        .to_numpy()
    )

    # ------------------------------------------------------------------
    # 3. Bullish FVG: high[N-2] < low[N] (two-bar upward imbalance)
    #    Zone: bottom = high[N-2], top = low[N], labelled on bar N.
    #    Zone is alive for pullback_window bars from bar N onward.
    # ------------------------------------------------------------------
    bull_fvg_top = np.full(n, np.nan)
    bull_fvg_bot = np.full(n, np.nan)
    for i in range(2, n):
        if high[i - 2] < low[i]:
            bull_fvg_top[i] = low[i]
            bull_fvg_bot[i] = high[i - 2]

    # Forward-fill each FVG zone for pullback_window bars
    bull_top_ff = np.full(n, np.nan)
    bull_bot_ff = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(bull_fvg_top[i]):
            end = min(n, i + pullback_window + 1)
            for j in range(i, end):
                if np.isnan(bull_top_ff[j]):
                    bull_top_ff[j] = bull_fvg_top[i]
                    bull_bot_ff[j] = bull_fvg_bot[i]

    # Price inside bullish FVG
    in_bull_fvg = (close <= bull_top_ff) & (close >= bull_bot_ff)

    # ------------------------------------------------------------------
    # 4. Bearish FVG: low[N-2] > high[N] (two-bar downward imbalance)
    #    Zone: top = low[N-2], bottom = high[N], labelled on bar N.
    # ------------------------------------------------------------------
    bear_fvg_top = np.full(n, np.nan)
    bear_fvg_bot = np.full(n, np.nan)
    for i in range(2, n):
        if low[i - 2] > high[i]:
            bear_fvg_top[i] = low[i - 2]
            bear_fvg_bot[i] = high[i]

    bear_top_ff = np.full(n, np.nan)
    bear_bot_ff = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(bear_fvg_top[i]):
            end = min(n, i + pullback_window + 1)
            for j in range(i, end):
                if np.isnan(bear_top_ff[j]):
                    bear_top_ff[j] = bear_fvg_top[i]
                    bear_bot_ff[j] = bear_fvg_bot[i]

    # Price inside bearish FVG
    in_bear_fvg = (close >= bear_bot_ff) & (close <= bear_top_ff)

    # ------------------------------------------------------------------
    # 5. Session windows — DST-aware conversion to Eastern Time
    # ------------------------------------------------------------------
    et_hour = _get_et_hours(idx, source_tz=source_tz)
    in_killzone = np.isin(et_hour, list(_KILLZONE_HOURS_ET))

    # ------------------------------------------------------------------
    # 6. Raw setup flags (before one-per-session-per-day deduplication)
    #    Long:  low swept, high NOT swept, in bullish FVG, in killzone
    #    Short: high swept, low NOT swept, in bearish FVG, in killzone
    # ------------------------------------------------------------------
    raw_long = low_swept & ~high_swept & in_bull_fvg & in_killzone
    raw_short = high_swept & ~low_swept & in_bear_fvg & in_killzone

    # ------------------------------------------------------------------
    # 7. Deduplicate to one entry per session per day + session cap
    # ------------------------------------------------------------------
    dates = np.array([t.date() for t in idx])
    session_labels = np.array(
        [_SESSION_NAMES.get(h, "") for h in et_hour], dtype=object
    )

    def _collect_setups(raw_mask: np.ndarray, direction: str) -> list:
        """
        Walk through raw_mask chronologically, pick the first qualifying
        bar per (date, session) group, apply session P&L cap.

        For long:  SL below swept liquidity low - sl_buffer_pts
        For short: SL above swept liquidity high + sl_buffer_pts
        """
        setups = []
        seen: dict = {}  # (date, session) -> session_pnl_so_far

        for i in range(n):
            if not raw_mask[i]:
                continue
            sess = session_labels[i]
            if sess == "":
                continue

            key = (dates[i], sess)
            if key in seen:
                continue  # already took an entry this session today

            # Session cap check
            sess_pnl_key = f"{dates[i]}_{sess}"
            if not hasattr(_collect_setups, "_pnl"):
                _collect_setups._pnl = {}  # type: ignore[attr-defined]
            current_pnl = _collect_setups._pnl.get(sess_pnl_key, 0.0)  # type: ignore[attr-defined]
            if current_pnl <= session_cap_usd:
                seen[key] = True
                continue

            entry_price = close[i]

            if direction == "long":
                sl = liq_low[i] - sl_buffer_pts
                sl_dist = entry_price - sl
            else:
                sl = liq_high[i] + sl_buffer_pts
                sl_dist = sl - entry_price

            if sl_dist <= 0:
                continue

            tp = (
                entry_price + rr_ratio * sl_dist
                if direction == "long"
                else entry_price - rr_ratio * sl_dist
            )

            # Estimate session loss (conservative: assume SL hit)
            estimated_loss = -sl_dist * lot_size
            _collect_setups._pnl[sess_pnl_key] = current_pnl + estimated_loss  # type: ignore[attr-defined]

            seen[key] = True
            setups.append(
                {
                    "entry_ts": idx[i],
                    "entry": float(entry_price),
                    "sl": float(sl),
                    "tp": float(tp),
                    "direction": direction,
                    "session": sess,
                    "sl_dist": float(sl_dist),
                    # FVG zone info (used by build_chart, ignored by backtest)
                    "fvg_top": float(bull_top_ff[i])
                    if direction == "long"
                    else float(bear_top_ff[i]),
                    "fvg_bottom": float(bull_bot_ff[i])
                    if direction == "long"
                    else float(bear_bot_ff[i]),
                    "liq_level": float(liq_low[i])
                    if direction == "long"
                    else float(liq_high[i]),
                }
            )

        # Reset the per-call state so it doesn't bleed between calls
        if hasattr(_collect_setups, "_pnl"):
            del _collect_setups._pnl  # type: ignore[attr-defined]

        return setups

    buy_setups = _collect_setups(raw_long, "long")
    sell_setups = _collect_setups(raw_short, "short")

    # Package liquidity data for build_chart() — not used by backtest.py
    liquidity = {
        "liq_high": pd.Series(liq_high, index=idx),
        "liq_low": pd.Series(liq_low, index=idx),
        # True on bars where price actually touched/swept the level
        "high_touched": pd.Series(high_touched, index=idx),
        "low_touched": pd.Series(low_touched, index=idx),
    }

    print(f"  Setups detected: {len(buy_setups)} long, {len(sell_setups)} short")
    return sell_setups, buy_setups, liquidity


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def build_chart(
    df: pd.DataFrame,
    sell_setups: list,
    buy_setups: list,
    liquidity: dict | None = None,
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    display_days: int = 5,
    source_tz: str = "utc",
) -> go.Figure:
    """
    Build a Plotly candlestick chart with:
      - Rolling liquidity high/low step lines (orange / teal)
      - Diamond markers where price actually swept each level
      - Bullish/bearish FVG zones (shaded rectangles)
      - Entry markers (triangle-up/down) and time-exit markers (x)
      - SL/TP dashed lines per trade
      - Full legend

    Args:
        liquidity: Dict returned as third element of detect_setups():
                     'liq_high'     : pd.Series — rolling N-bar high
                     'liq_low'      : pd.Series — rolling N-bar low
                     'high_touched' : pd.Series[bool] — bars where high was swept
                     'low_touched'  : pd.Series[bool] — bars where low was swept
                   Pass None (or omit) to skip liquidity overlays.

    Returns a go.Figure.
    """
    import vectorbt as vbt

    vbt.settings.plotting["use_widgets"] = False

    # Limit display to last `display_days` trading days
    unique_dates = sorted(set(df.index.date))
    cutoff = pd.Timestamp(unique_dates[max(0, len(unique_dates) - display_days)])
    dv = df[df.index >= cutoff]
    idx_list = dv.index.tolist()

    TIME_EXIT_BARS = 52

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.85, 0.15],
        vertical_spacing=0.03,
        subplot_titles=[
            f"{symbol} {timeframe} — ICT Silver Bullet (last {display_days} days)",
            "Volume",
        ],
    )

    # -- Candlesticks --
    fig.add_trace(
        go.Candlestick(
            x=idx_list,
            open=dv["open"].tolist(),
            high=dv["high"].tolist(),
            low=dv["low"].tolist(),
            close=dv["close"].tolist(),
            name="Price",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # -- Liquidity levels --
    if liquidity:
        liq_high_v = liquidity["liq_high"].reindex(dv.index)
        liq_low_v = liquidity["liq_low"].reindex(dv.index)

        # Rolling liquidity high — orange step line
        fig.add_trace(
            go.Scatter(
                x=idx_list,
                y=liq_high_v.tolist(),
                mode="lines",
                line=dict(color="rgba(255,160,0,0.55)", width=1, shape="hv"),
                name="Liq high",
                legendgroup="liq_high",
                hovertemplate="Liq high: %{y:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # Rolling liquidity low — teal step line
        fig.add_trace(
            go.Scatter(
                x=idx_list,
                y=liq_low_v.tolist(),
                mode="lines",
                line=dict(color="rgba(0,188,212,0.55)", width=1, shape="hv"),
                name="Liq low",
                legendgroup="liq_low",
                hovertemplate="Liq low: %{y:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # -- Killzone background shading --
    # London 03, NY AM 10, NY PM 14 ET — semi-transparent yellow band spanning
    # the full chart height. Uses xref="x" + yref="paper" so one shape covers
    # both rows without needing row/col arguments.
    et_hours_dv = _get_et_hours(dv.index, source_tz=source_tz)
    in_kz = np.isin(et_hours_dv, list(_KILLZONE_HOURS_ET))

    kz_legend_added = False
    i = 0
    while i < len(dv):
        if in_kz[i]:
            j = i
            while j < len(dv) and in_kz[j]:
                j += 1
            x0 = dv.index[i]
            x1 = dv.index[j - 1]
            fig.add_shape(
                type="rect",
                xref="x",
                yref="paper",
                x0=x0,
                x1=x1,
                y0=0,
                y1=1,
                fillcolor="rgba(255,235,59,0.07)",
                line=dict(width=0),
                layer="below",
            )
            if not kz_legend_added:
                fig.add_trace(
                    go.Scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        marker=dict(
                            symbol="square", size=12, color="rgba(255,235,59,0.4)"
                        ),
                        name="Killzone window",
                        legendgroup="killzone",
                    ),
                    row=1,
                    col=1,
                )
                kz_legend_added = True
            i = j
        else:
            i += 1

    # -- Volume --
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350" for c, o in zip(dv["close"], dv["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=idx_list,
            y=dv["volume"].tolist(),
            name="Volume",
            marker_color=vol_colors,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # -- Legend dummy traces (shapes can't appear in legend) --
    _DUMMY = dict(x=[None], y=[None], mode="markers")
    fig.add_trace(
        go.Scatter(
            **_DUMMY,
            marker=dict(
                symbol="triangle-up",
                size=12,
                color="#26a69a",
                line=dict(width=1, color="white"),
            ),
            name="Long entry",
            legendgroup="long_entry",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            **_DUMMY,
            marker=dict(
                symbol="triangle-down",
                size=12,
                color="#ef5350",
                line=dict(width=1, color="white"),
            ),
            name="Short entry",
            legendgroup="short_entry",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            **_DUMMY,
            marker=dict(symbol="x", size=10, color="white", line=dict(width=2)),
            name="Time exit (52 bars)",
            legendgroup="time_exit",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#26a69a", width=1, dash="dash"),
            name="Take profit",
            legendgroup="tp",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#ef5350", width=1, dash="dash"),
            name="Stop loss",
            legendgroup="sl",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="square", size=10, color="rgba(38,166,154,0.35)"),
            name="Bull FVG zone",
            legendgroup="bull_fvg",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(symbol="square", size=10, color="rgba(239,83,80,0.35)"),
            name="Bear FVG zone",
            legendgroup="bear_fvg",
        ),
        row=1,
        col=1,
    )

    # -- Helper: resolve timestamp to display-window index position --
    def _in_view(ts: pd.Timestamp) -> bool:
        return dv.index[0] <= ts <= dv.index[-1]

    def _time_exit_ts(entry_ts: pd.Timestamp) -> pd.Timestamp | None:
        """Return the timestamp 52 bars after entry, or None if out of window."""
        if entry_ts not in dv.index:
            return None
        pos = dv.index.get_loc(entry_ts)
        exit_pos = min(pos + TIME_EXIT_BARS, len(dv) - 1)
        return dv.index[exit_pos]

    def _line_end(ts: pd.Timestamp, bars: int = 60) -> pd.Timestamp:
        """Return timestamp `bars` ahead of ts, clamped to end of display window."""
        if ts not in dv.index:
            return dv.index[-1]
        pos = dv.index.get_loc(ts)
        return dv.index[min(pos + bars, len(dv) - 1)]

    # -- Long setups --
    long_exit_xs: list = []
    long_exit_ys: list = []

    for s in buy_setups:
        ts = s["entry_ts"]
        if not _in_view(ts):
            continue
        entry = s["entry"]
        sl = s["sl"]
        tp = s["tp"]
        x1 = _line_end(ts, 60)

        # FVG zone
        fig.add_shape(
            type="rect",
            x0=ts,
            x1=x1,
            y0=s["fvg_bottom"],
            y1=s["fvg_top"],
            fillcolor="rgba(38,166,154,0.15)",
            line=dict(color="rgba(38,166,154,0.5)", width=1),
            row=1,
            col=1,
        )
        # SL line
        fig.add_shape(
            type="line",
            x0=ts,
            x1=x1,
            y0=sl,
            y1=sl,
            line=dict(color="#ef5350", width=1, dash="dash"),
            row=1,
            col=1,
        )
        # TP line
        fig.add_shape(
            type="line",
            x0=ts,
            x1=x1,
            y0=tp,
            y1=tp,
            line=dict(color="#26a69a", width=1, dash="dash"),
            row=1,
            col=1,
        )
        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[ts],
                y=[entry],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    size=12,
                    color="#26a69a",
                    line=dict(width=1, color="white"),
                ),
                name="Long entry",
                legendgroup="long_entry",
                showlegend=False,
                hovertemplate=f"Long entry<br>Price: {entry:.1f}<br>SL: {sl:.1f}<br>TP: {tp:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # Time exit marker
        exit_ts = _time_exit_ts(ts)
        if exit_ts is not None:
            exit_price = dv.loc[exit_ts, "close"]
            long_exit_xs.append(exit_ts)
            long_exit_ys.append(exit_price)

    # Batch all long time-exit markers into one trace
    if long_exit_xs:
        fig.add_trace(
            go.Scatter(
                x=long_exit_xs,
                y=long_exit_ys,
                mode="markers",
                marker=dict(symbol="x", size=10, color="white", line=dict(width=2)),
                name="Time exit (52 bars)",
                legendgroup="time_exit",
                showlegend=False,
                hovertemplate="Time exit<br>Price: %{y:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # -- Short setups --
    short_exit_xs: list = []
    short_exit_ys: list = []

    for s in sell_setups:
        ts = s["entry_ts"]
        if not _in_view(ts):
            continue
        entry = s["entry"]
        sl = s["sl"]
        tp = s["tp"]
        x1 = _line_end(ts, 60)

        # FVG zone
        fig.add_shape(
            type="rect",
            x0=ts,
            x1=x1,
            y0=s["fvg_bottom"],
            y1=s["fvg_top"],
            fillcolor="rgba(239,83,80,0.15)",
            line=dict(color="rgba(239,83,80,0.5)", width=1),
            row=1,
            col=1,
        )
        # SL line
        fig.add_shape(
            type="line",
            x0=ts,
            x1=x1,
            y0=sl,
            y1=sl,
            line=dict(color="#ef5350", width=1, dash="dash"),
            row=1,
            col=1,
        )
        # TP line
        fig.add_shape(
            type="line",
            x0=ts,
            x1=x1,
            y0=tp,
            y1=tp,
            line=dict(color="#26a69a", width=1, dash="dash"),
            row=1,
            col=1,
        )
        # Entry marker
        fig.add_trace(
            go.Scatter(
                x=[ts],
                y=[entry],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=12,
                    color="#ef5350",
                    line=dict(width=1, color="white"),
                ),
                name="Short entry",
                legendgroup="short_entry",
                showlegend=False,
                hovertemplate=f"Short entry<br>Price: {entry:.1f}<br>SL: {sl:.1f}<br>TP: {tp:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        # Time exit marker
        exit_ts = _time_exit_ts(ts)
        if exit_ts is not None:
            exit_price = dv.loc[exit_ts, "close"]
            short_exit_xs.append(exit_ts)
            short_exit_ys.append(exit_price)

    if short_exit_xs:
        fig.add_trace(
            go.Scatter(
                x=short_exit_xs,
                y=short_exit_ys,
                mode="markers",
                marker=dict(symbol="x", size=10, color="white", line=dict(width=2)),
                name="Time exit (52 bars)",
                legendgroup="time_exit",
                showlegend=False,
                hovertemplate="Time exit<br>Price: %{y:.1f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    # -- Layout --
    fig.update_layout(
        height=960,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0.4)",
            bordercolor="rgba(255,255,255,0.2)",
            borderwidth=1,
            font=dict(size=12),
        ),
        title=dict(
            text=(
                f"{symbol} — ICT Silver Bullet  |  "
                f"LiqWin={LIQUIDITY_WINDOW}  Sweep={SWEEP_LOOKBACK}  "
                f"PBwin={PULLBACK_WINDOW}  RR={RR_RATIO}"
            ),
            x=0.5,
            xanchor="center",
            font=dict(size=15),
        ),
    )
    # Build rangebreaks from actual gaps in the data — only periods where
    # data is genuinely absent (weekends, holidays) are collapsed.
    # Finds every consecutive pair of bars whose gap > 1.5× the typical
    # 1-minute interval, then emits one bounds=[gap_end, gap_start] per gap.
    ts_arr = dv.index.sort_values().to_numpy()
    deltas = np.diff(ts_arr).astype("timedelta64[ms]").astype(float)
    # Only collapse gaps of 30 minutes or more — ignores tiny data dropouts
    # while catching the weekend (~2 days) and the Dukascopy daily break (~1h46m)
    thirty_min_ms = 30 * 60 * 1000
    rangebreaks = []
    for i in np.where(deltas >= thirty_min_ms)[0]:
        gap_end = pd.Timestamp(ts_arr[i])
        gap_start = pd.Timestamp(ts_arr[i + 1])
        rangebreaks.append(dict(bounds=[gap_end.isoformat(), gap_start.isoformat()]))

    # Simpler and more robust: build (end_of_data, start_of_next_data) pairs directly
    rangebreaks = []
    ts_arr = dv.index.sort_values().to_numpy()
    deltas = np.diff(ts_arr).astype("timedelta64[ms]").astype(float)
    normal_ms = float(pd.Series(deltas).mode().iloc[0])
    for i in np.where(deltas > normal_ms * 1.5)[0]:
        gap_end = pd.Timestamp(ts_arr[i])
        gap_start = pd.Timestamp(ts_arr[i + 1])
        rangebreaks.append(dict(bounds=[gap_end.isoformat(), gap_start.isoformat()]))

    fig.update_xaxes(rangeslider_visible=False, rangebreaks=rangebreaks, row=1, col=1)
    fig.update_xaxes(rangeslider_visible=False, rangebreaks=rangebreaks, row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("ICT SILVER BULLET — SIGNAL VISUALISATION")
    print("=" * 70)

    print(f"\nLoading {SYMBOL} data ({START_DATE} to {END_DATE})...")
    df, source_tz = load_ohlcv(SYMBOL, START_DATE, END_DATE, TIMEFRAME)
    print(
        f"  {len(df):,} rows  [{df.index.min()} -> {df.index.max()}]  (source_tz={source_tz})"
    )

    sell_setups, buy_setups, liquidity = detect_setups(df, source_tz=source_tz)
    print(f"  Long setups  : {len(buy_setups)}")
    print(f"  Short setups : {len(sell_setups)}")

    fig = build_chart(
        df,
        sell_setups,
        buy_setups,
        liquidity=liquidity,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
    )

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"strategy_{SYMBOL}_{TIMEFRAME}.html"
    fig.write_html(str(out))
    print(f"\n  Chart saved -> {out}")
    fig.show()
