"""
Range Breakout — Strategy Signal Detection + Visualisation

Detects the Asian/pre-London range (00:00-07:00 UK time) on USDJPY 1H,
identifies bullish and bearish breakout entries, and produces an interactive
Plotly candlestick chart showing:

  - Range High / Range Low shaded zones per day
  - BUY entry markers (close above Range High)
  - SELL entry markers (close below Range Low)
  - TP markers (1:2 R:R exits)
  - SL markers (stop-loss exits)
  - Time-exit markers (16:00 UK auto-close)

The chart is saved as an HTML artifact.

Conforms to STRATEGY_BACKTEST_PATTERN.md:
  - load_ohlcv(symbol, start, end, tf) -> pd.DataFrame  (tz-naive, lowercase cols)
  - detect_setups(df, ...) -> (sell_setups, buy_setups, extra)
  - Each setup dict has at minimum: entry_ts, entry, sl, tp

Usage:
    uv run python experiments/RangeBO/strategy.py
"""

import sys
import io
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
os.environ["SMC_CREDIT"] = "0"

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL = "USDJPY"
START_DATE = "2022-01-01"
END_DATE = "2026-03-05"
LOWER_TF = "1h"

RANGE_START_HOUR = 3  # 00:00 UK inclusive
RANGE_END_HOUR = 7  # 07:00 UK exclusive (range locked after 06:00 bar close)
TRADE_CLOSE_HOUR = 16  # 16:00 UK hard exit
RISK_REWARD = 2.0  # TP = RISK_REWARD * SL distance
TIMEZONE = "Europe/London"

MIN_RANGE_PIPS = 15  # skip thin/dead Asian sessions (range too narrow)
MAX_RANGE_PIPS = 100  # skip blown-out news/event days (range too wide)

_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"}
PIP_SIZE = 0.01 if SYMBOL in _JPY_PAIRS else 0.0001


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


_BALERION_DATA = (
    Path(__file__).resolve().parent.parent.parent.parent / "balerion-data" / "data"
)


def load_ohlcv(
    symbol: str = SYMBOL,
    start: str = START_DATE,
    end: str = END_DATE,
    tf: str = LOWER_TF,
) -> pd.DataFrame:
    """
    Load OHLCV data for the given symbol and date range.

    Preference order:
      1. Dukascopy 1H parquet  — {balerion-data}/data/fx/{symbol_lower}/{symbol_lower}_dukascopy_1h.parquet
      2. DataLoader fallback   — MT5 1m resampled to tf

    Returns a DataFrame with:
      - tz-naive DatetimeIndex (UTC stripped)
      - lowercase columns: open, high, low, close, volume
    """
    sym_lower = symbol.lower()

    # 1. Try Dukascopy 1H file
    duka_path = _BALERION_DATA / "fx" / sym_lower / f"{sym_lower}_dukascopy_1h.parquet"
    if duka_path.exists():
        df = pd.read_parquet(duka_path)
        # timestamp is a column, not the index
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df.index = df.index.tz_localize(None)  # strip UTC, keep tz-naive
        df = df[["open", "high", "low", "close", "volume"]]
        # Date filter
        if start:
            df = df[df.index >= pd.Timestamp(start)]
        if end:
            df = df[df.index <= pd.Timestamp(end)]
        return df

    # 2. Fallback: DataLoader (MT5 1m resampled)
    loader = DataLoader()
    df_1m = loader.load_fx(symbol, start_date=start, end_date=end)
    if tf in ("1m", "1min"):
        df = df_1m
    else:
        df = loader.resample_ohlcv(df_1m, tf)

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def detect_setups(
    df: pd.DataFrame,
    risk_reward: float = RISK_REWARD,
    range_start_hour: int = RANGE_START_HOUR,
    range_end_hour: int = RANGE_END_HOUR,
    trade_close_hour: int = TRADE_CLOSE_HOUR,
    timezone: str = TIMEZONE,
    min_range_pips: float = MIN_RANGE_PIPS,
    max_range_pips: float = MAX_RANGE_PIPS,
) -> tuple:
    """
    Detect Asian range breakout setups from a 1H OHLCV DataFrame.

    For each trading day:
      1. Build the range from bars 00:00-06:59 UK time
      2. Scan bars from range_end_hour onwards for the first breakout
      3. Long  entry when close > range_high (SL at range_low)
      4. Short entry when close < range_low  (SL at range_high)
      5. TP at risk_reward * SL distance

    Parameters
    ----------
    df : pd.DataFrame
        1H OHLCV with a tz-naive DatetimeIndex (UTC).

    Returns
    -------
    sell_setups : list of dicts  (entry_ts, entry, sl, tp, range_high, range_low, direction)
    buy_setups  : list of dicts  (entry_ts, entry, sl, tp, range_high, range_low, direction)
    extra       : dict           (day_ranges — list of per-day range dicts for charting)
    """
    df = df.copy()

    # Attach UK hour and date — localise to UTC first so tz_convert works
    df_utc = df.copy()
    df_utc.index = df_utc.index.tz_localize("UTC")
    df_uk = df_utc.copy()
    df_uk.index = df_uk.index.tz_convert(timezone)

    df["uk_hour"] = df_uk.index.hour
    df["uk_date"] = df_uk.index.normalize().tz_localize(None)

    sell_setups = []
    buy_setups = []
    day_ranges = []  # for build_chart()

    uk_dates = df["uk_date"].unique()

    for uk_day in uk_dates:
        day_mask = df["uk_date"] == uk_day

        # Range bars: 00:00-06:59 UK
        range_mask = (
            day_mask
            & (df["uk_hour"] >= range_start_hour)
            & (df["uk_hour"] < range_end_hour)
        )
        if range_mask.sum() == 0:
            continue

        r_high = df.loc[range_mask, "high"].max()
        r_low = df.loc[range_mask, "low"].min()

        # --- Range width filters ---
        range_width = r_high - r_low
        range_width_pips = range_width / PIP_SIZE

        if range_width_pips < min_range_pips:
            continue  # thin/dead session — no conviction

        if range_width_pips > max_range_pips:
            continue  # blown-out news day — unreliable breakout

        range_bars = df.index[range_mask]
        day_ranges.append(
            {
                "uk_day": uk_day,
                "r_high": r_high,
                "r_low": r_low,
                "range_start_ts": range_bars[0],
                "range_end_ts": range_bars[-1],
            }
        )

        # Trading bars: 07:00-15:59 UK (one trade per day max)
        trade_mask = (
            day_mask
            & (df["uk_hour"] >= range_end_hour)
            & (df["uk_hour"] < trade_close_hour)
        )
        trade_bars = df.index[trade_mask]

        for ts in trade_bars:
            bar = df.loc[ts]
            c = bar["close"]

            is_long = c > r_high
            is_short = c < r_low

            if not is_long and not is_short:
                continue

            direction = "long" if is_long else "short"
            entry = c
            sl = r_low if direction == "long" else r_high

            # Floor SL distance at full range width — prevents oversized lots
            # when the entry is only a few pips beyond the boundary.
            sl_dist = max(abs(entry - sl), range_width)

            if sl_dist <= 0:
                break  # degenerate day — skip remaining bars

            tp = (
                entry + risk_reward * sl_dist
                if direction == "long"
                else entry - risk_reward * sl_dist
            )

            setup = {
                "entry_ts": ts,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "range_high": r_high,
                "range_low": r_low,
                "direction": direction,
            }

            if direction == "long":
                buy_setups.append(setup)
            else:
                sell_setups.append(setup)

            break  # one trade per day

    return sell_setups, buy_setups, {"day_ranges": day_ranges}


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def build_chart(
    df: pd.DataFrame,
    sell_setups: list,
    buy_setups: list,
    extra: dict = None,
    symbol: str = SYMBOL,
    timezone: str = TIMEZONE,
) -> go.Figure:
    """
    Build a single Plotly candlestick figure showing range zones and trade markers.

    Includes:
      - Candlestick OHLC
      - Daily Asian range shaded rectangles
      - Range High / Range Low dashed lines
      - BUY / SELL entry markers
      - TP / SL / Time-exit markers  (simulated forward scan)
      - Trade connector lines
      - Volume subplot
    """
    if extra is None:
        extra = {}

    day_ranges = extra.get("day_ranges", [])

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22],
        subplot_titles=(f"{symbol} 1H — Asian Range Breakout", "Volume"),
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
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # --- Volume ---
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
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # --- Per-day range shapes ---
    shapes = []

    # Re-derive UK hours for shape building if not already on df
    if "uk_hour" not in df.columns:
        df_utc = df.copy()
        df_utc.index = df_utc.index.tz_localize("UTC")
        df_uk = df_utc.tz_convert(timezone)
        df["uk_hour"] = df_uk.index.hour

    for dr in day_ranges:
        r_high = dr["r_high"]
        r_low = dr["r_low"]
        x0 = dr["range_start_ts"]
        x1 = dr["range_end_ts"] + pd.Timedelta(hours=1)

        # Find end of trading day to extend the level lines
        day_bars = df[df.index.date == pd.Timestamp(dr["uk_day"]).date()]
        close_bars = day_bars[day_bars["uk_hour"] == TRADE_CLOSE_HOUR]
        x_end = (
            close_bars.index[-1] + pd.Timedelta(hours=1)
            if len(close_bars) > 0
            else (
                day_bars.index[-1] + pd.Timedelta(hours=1) if len(day_bars) > 0 else x1
            )
        )
        x_start = day_bars.index[0] if len(day_bars) > 0 else x0

        # Shaded accumulation zone
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                x1=x1,
                y0=r_low,
                y1=r_high,
                fillcolor="rgba(120, 120, 140, 0.12)",
                line=dict(color="rgba(180,180,200,0.3)", width=1),
                layer="below",
            )
        )

        # Range High dashed line
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=x_start,
                x1=x_end,
                y0=r_high,
                y1=r_high,
                line=dict(color="rgba(239, 83, 80, 0.5)", width=1, dash="dot"),
                layer="below",
            )
        )

        # Range Low dashed line
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=x_start,
                x1=x_end,
                y0=r_low,
                y1=r_low,
                line=dict(color="rgba(38, 166, 154, 0.5)", width=1, dash="dot"),
                layer="below",
            )
        )

    # --- Entry markers ---
    all_setups = [(s, "short") for s in sell_setups] + [(s, "long") for s in buy_setups]

    buy_x = [s["entry_ts"] for s, d in all_setups if d == "long"]
    buy_y = [s["entry"] for s, d in all_setups if d == "long"]
    sell_x = [s["entry_ts"] for s, d in all_setups if d == "short"]
    sell_y = [s["entry"] for s, d in all_setups if d == "short"]

    if buy_x:
        fig.add_trace(
            go.Scatter(
                x=buy_x,
                y=buy_y,
                mode="markers",
                name="BUY Entry",
                marker=dict(
                    symbol="triangle-up",
                    size=14,
                    color="#00e676",
                    line=dict(color="#1b5e20", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    if sell_x:
        fig.add_trace(
            go.Scatter(
                x=sell_x,
                y=sell_y,
                mode="markers",
                name="SELL Entry",
                marker=dict(
                    symbol="triangle-down",
                    size=14,
                    color="#f44336",
                    line=dict(color="#b71c1c", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Exit markers (simulated forward scan for charting purposes) ---
    # Re-attach UK hour for exit scan
    if "uk_hour" not in df.columns:
        df_utc2 = df.copy()
        df_utc2.index = df_utc2.index.tz_localize("UTC")
        df_uk2 = df_utc2.tz_convert(timezone)
        df["uk_hour"] = df_uk2.index.hour

    if "uk_date" not in df.columns:
        df_utc3 = df.copy()
        df_utc3.index = df_utc3.index.tz_localize("UTC")
        df_uk3 = df_utc3.tz_convert(timezone)
        df["uk_date"] = df_uk3.index.normalize().tz_localize(None)

    tp_x, tp_y = [], []
    sl_x, sl_y = [], []
    time_x, time_y = [], []

    for setup, direction in all_setups:
        ts = setup["entry_ts"]
        entry = setup["entry"]
        sl_px = setup["sl"]
        tp_px = setup["tp"]
        uk_day = df.loc[ts, "uk_date"] if ts in df.index else None

        if uk_day is None:
            continue

        day_mask = df["uk_date"] == uk_day
        post_mask = day_mask & (df.index > ts) & (df["uk_hour"] <= TRADE_CLOSE_HOUR)
        post_bars = df.index[post_mask]

        exit_ts = None
        exit_px = None
        exit_type = "time"

        for future_ts in post_bars:
            fb = df.loc[future_ts]
            fh = fb["high"]
            fl = fb["low"]
            fc = fb["close"]
            fhour = fb["uk_hour"]

            if fhour == TRADE_CLOSE_HOUR:
                exit_type = "time"
                exit_ts = future_ts
                exit_px = fc
                break

            if direction == "long":
                if fl <= sl_px:
                    exit_type = "sl"
                    exit_ts = future_ts
                    exit_px = sl_px
                    break
                if fh >= tp_px:
                    exit_type = "tp"
                    exit_ts = future_ts
                    exit_px = tp_px
                    break
            else:
                if fh >= sl_px:
                    exit_type = "sl"
                    exit_ts = future_ts
                    exit_px = sl_px
                    break
                if fl <= tp_px:
                    exit_type = "tp"
                    exit_ts = future_ts
                    exit_px = tp_px
                    break

        if exit_ts is None:
            exit_type = "time"
            exit_ts = ts
            exit_px = entry

        if exit_type == "tp":
            tp_x.append(exit_ts)
            tp_y.append(exit_px)
        elif exit_type == "sl":
            sl_x.append(exit_ts)
            sl_y.append(exit_px)
        else:
            time_x.append(exit_ts)
            time_y.append(exit_px)

        # Connector line
        color = (
            "#ffeb3b"
            if exit_type == "tp"
            else ("#ff5252" if exit_type == "sl" else "#b0bec5")
        )
        fig.add_trace(
            go.Scatter(
                x=[ts, exit_ts],
                y=[entry, exit_px],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

    if tp_x:
        fig.add_trace(
            go.Scatter(
                x=tp_x,
                y=tp_y,
                mode="markers",
                name="TP Hit",
                marker=dict(
                    symbol="star",
                    size=12,
                    color="#ffeb3b",
                    line=dict(color="#f57f17", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    if sl_x:
        fig.add_trace(
            go.Scatter(
                x=sl_x,
                y=sl_y,
                mode="markers",
                name="SL Hit",
                marker=dict(
                    symbol="x",
                    size=11,
                    color="#ff5252",
                    line=dict(color="#b71c1c", width=2),
                ),
            ),
            row=1,
            col=1,
        )

    if time_x:
        fig.add_trace(
            go.Scatter(
                x=time_x,
                y=time_y,
                mode="markers",
                name="Time Exit (16:00)",
                marker=dict(
                    symbol="circle",
                    size=9,
                    color="#b0bec5",
                    line=dict(color="#546e7a", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Summary stats for title ---
    n_trades = len(all_setups)
    n_tp = len(tp_x)
    n_sl = len(sl_x)
    n_time = len(time_x)
    win_rate = (n_tp / n_trades * 100) if n_trades > 0 else 0.0

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
                f"<b>{symbol} 1H — Asian Range Breakout</b><br>"
                f"<sup>Trades: {n_trades}  |  TP: {n_tp}  |  SL: {n_sl}  |  "
                f"Time exit: {n_time}  |  Win rate: {win_rate:.1f}%  |  "
                f"Grey zone = Asian range  |  "
                f"<span style='color:#ef5350'>--- Range High</span>  "
                f"<span style='color:#26a69a'>--- Range Low</span></sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
    )
    fig.update_xaxes(title_text="Date (UTC)", row=2, col=1)
    fig.update_yaxes(title_text="Price (JPY)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import vectorbt as vbt

    vbt.settings.plotting["use_widgets"] = False

    print(f"Loading {SYMBOL} {LOWER_TF} data ({START_DATE} to {END_DATE})...")
    df = load_ohlcv(SYMBOL, START_DATE, END_DATE, LOWER_TF)
    print(f"  {len(df):,} bars  [{df.index.min()} -> {df.index.max()}]")

    print("Detecting setups...")
    sell_setups, buy_setups, extra = detect_setups(df)
    print(f"  Long  setups : {len(buy_setups)}")
    print(f"  Short setups : {len(sell_setups)}")

    print("Building chart...")
    fig = build_chart(df, sell_setups, buy_setups, extra=extra, symbol=SYMBOL)

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"strategy_{SYMBOL}_{LOWER_TF}.html"
    fig.write_html(str(out))
    print(f"  Chart saved -> {out}")
    fig.show()
