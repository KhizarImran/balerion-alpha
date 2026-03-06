"""
Range Breakout — Strategy Visualisation

Detects the Asian/pre-London range (00:00–07:00 UK time) on USDJPY 1H,
identifies bullish and bearish breakout entries, and produces an interactive
Plotly candlestick chart showing:

  - Range High / Range Low shaded zones per day
  - BUY entry markers (close above Range High)
  - SELL entry markers (close below Range Low)
  - TP markers (1:2 R:R exits)
  - SL markers (stop-loss exits)
  - Time-exit markers (16:00 UK auto-close)

The chart is saved as an HTML artifact and pushed to MLflow.

Usage:
    uv run python experiments/RangeBO/strategy.py
"""

import sys
import io
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Fix Windows cp1252 terminal encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import mlflow

from utils import DataLoader

# ---------------------------------------------------------------------------
# MLflow setup
# ---------------------------------------------------------------------------
os.environ["MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS"] = "true"

MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("rangebo")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL = "USDJPY"
START_DATE = "2024-01-01"
END_DATE = "2026-03-05"

RANGE_START_HOUR = 0   # 00:00 UK inclusive
RANGE_END_HOUR = 7     # 07:00 UK exclusive  (range locked after 06:00 bar close)
TRADE_CLOSE_HOUR = 16  # 16:00 UK hard exit
RR = 2.0               # reward-to-risk ratio
TIMEZONE = "Europe/London"

# How many days to display per chart page (keeps HTML file manageable)
DAYS_PER_CHART = 30

# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a 1H OHLCV DataFrame (UTC-aware index), compute the Asian range
    per day and detect breakout entries for both long and short directions.

    New columns added
    -----------------
    uk_hour        : int   — bar open hour in UK local time
    range_high     : float — daily range high (NaN outside 07:00+ bars)
    range_low      : float — daily range low  (NaN outside 07:00+ bars)
    buy_signal     : bool  — bar that first closes above range_high after 07:00
    sell_signal    : bool  — bar that first closes below range_low  after 07:00
    entry_price    : float — close of the entry bar
    sl_price       : float — stop-loss price
    tp_price       : float — take-profit price (1:2 R:R)
    exit_type      : str   — 'tp' | 'sl' | 'time' | '' — how the trade closed
    exit_bar       : Timestamp — index of the exit bar
    exit_price     : float — price at exit
    trade_pnl_pips : float — signed P&L in pips (pip = 0.01 for JPY pairs)
    """
    df = df.copy()

    # Convert UTC index to UK local time for session logic
    df_uk = df.copy()
    df_uk.index = df_uk.index.tz_convert(TIMEZONE)
    df["uk_hour"] = df_uk.index.hour
    df["uk_date"] = df_uk.index.normalize()  # date in UK time

    # Output columns
    df["range_high"] = np.nan
    df["range_low"] = np.nan
    df["buy_signal"] = False
    df["sell_signal"] = False
    df["entry_price"] = np.nan
    df["sl_price"] = np.nan
    df["tp_price"] = np.nan
    df["exit_type"] = ""
    df["exit_bar"] = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ms, UTC]")
    df["exit_price"] = np.nan
    df["trade_pnl_pips"] = np.nan

    pip = 0.01  # USDJPY

    uk_dates = df["uk_date"].unique()

    for uk_day in uk_dates:
        day_mask = df["uk_date"] == uk_day

        # Range bars: hours 00..06 UK
        range_mask = day_mask & (df["uk_hour"] >= RANGE_START_HOUR) & (df["uk_hour"] < RANGE_END_HOUR)
        if range_mask.sum() == 0:
            continue

        r_high = df.loc[range_mask, "high"].max()
        r_low = df.loc[range_mask, "low"].min()

        # Broadcast range levels onto all post-07:00 bars of the same day
        # (for plotting); we store on every bar so chart shapes can read them
        df.loc[day_mask, "range_high"] = r_high
        df.loc[day_mask, "range_low"] = r_low

        # Trading bars: hours 07..15 UK (16:00 bar is auto-close, not entry)
        trade_mask = day_mask & (df["uk_hour"] >= RANGE_END_HOUR) & (df["uk_hour"] < TRADE_CLOSE_HOUR)
        trade_bars = df.index[trade_mask]

        trade_taken = False

        for ts in trade_bars:
            if trade_taken:
                break

            bar = df.loc[ts]
            c = bar["close"]
            h = bar["high"]
            l = bar["low"]

            is_long = c > r_high
            is_short = c < r_low

            if not is_long and not is_short:
                continue

            # --- Entry ---
            direction = "long" if is_long else "short"
            entry = c
            sl = r_low if direction == "long" else r_high
            sl_dist = abs(entry - sl)

            if sl_dist <= 0:
                continue

            tp = entry + RR * sl_dist if direction == "long" else entry - RR * sl_dist

            df.loc[ts, "buy_signal"] = direction == "long"
            df.loc[ts, "sell_signal"] = direction == "short"
            df.loc[ts, "entry_price"] = entry
            df.loc[ts, "sl_price"] = sl
            df.loc[ts, "tp_price"] = tp
            trade_taken = True

            # --- Simulate forward to find exit ---
            # Bars from the next bar until 16:00 UK on the same day
            post_mask = (
                day_mask
                & (df.index > ts)
                & (df["uk_hour"] <= TRADE_CLOSE_HOUR)
            )
            post_bars = df.index[post_mask]

            exit_type = "time"
            exit_ts = None
            exit_px = None

            for future_ts in post_bars:
                fb = df.loc[future_ts]
                fh = fb["high"]
                fl = fb["low"]
                fc = fb["close"]
                fhour = fb["uk_hour"]

                # Time exit at 16:00 bar close (regardless of TP/SL on that bar)
                if fhour == TRADE_CLOSE_HOUR:
                    exit_type = "time"
                    exit_ts = future_ts
                    exit_px = fc
                    break

                if direction == "long":
                    # SL hit first (conservative: check SL before TP within bar)
                    if fl <= sl:
                        exit_type = "sl"
                        exit_ts = future_ts
                        exit_px = sl
                        break
                    if fh >= tp:
                        exit_type = "tp"
                        exit_ts = future_ts
                        exit_px = tp
                        break
                else:  # short
                    if fh >= sl:
                        exit_type = "sl"
                        exit_ts = future_ts
                        exit_px = sl
                        break
                    if fl <= tp:
                        exit_type = "tp"
                        exit_ts = future_ts
                        exit_px = tp
                        break

            # If no forward bars found (e.g. data ends), mark as time exit on entry bar
            if exit_ts is None:
                exit_type = "time"
                exit_ts = ts
                exit_px = entry

            pnl_pips = (
                (exit_px - entry) / pip if direction == "long"
                else (entry - exit_px) / pip
            )

            df.loc[ts, "exit_type"] = exit_type
            df.loc[ts, "exit_bar"] = exit_ts
            df.loc[ts, "exit_price"] = exit_px
            df.loc[ts, "trade_pnl_pips"] = pnl_pips

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def build_chart(df: pd.DataFrame, symbol: str, start: str, end: str) -> go.Figure:
    """
    Build a single Plotly candlestick figure for the full date window.

    Includes:
      - Candlestick OHLC
      - Daily range shaded rectangles (grey, behind candles)
      - Range High / Range Low dashed horizontal lines per day
      - BUY / SELL entry markers
      - TP / SL / Time-exit markers
      - Volume subplot
    """
    # Slice to requested window
    mask = (df.index >= pd.Timestamp(start, tz="UTC")) & (df.index <= pd.Timestamp(end, tz="UTC"))
    d = df[mask].copy()

    if len(d) == 0:
        raise ValueError(f"No data between {start} and {end}")

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
            x=d.index,
            open=d["open"],
            high=d["high"],
            low=d["low"],
            close=d["close"],
            name="OHLC",
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # --- Volume ---
    vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(d["close"], d["open"])
    ]
    fig.add_trace(
        go.Bar(
            x=d.index,
            y=d["volume"],
            name="Volume",
            marker_color=vol_colors,
            opacity=0.5,
            showlegend=False,
        ),
        row=2, col=1,
    )

    # --- Per-day range shapes & level lines ---
    shapes = []
    range_high_x, range_high_y = [], []
    range_low_x, range_low_y = [], []

    uk_dates = d["uk_date"].unique()

    for uk_day in uk_dates:
        day_mask = d["uk_date"] == uk_day
        day_bars = d[day_mask]

        r_high = day_bars["range_high"].dropna().iloc[0] if day_bars["range_high"].notna().any() else None
        r_low = day_bars["range_low"].dropna().iloc[0] if day_bars["range_low"].notna().any() else None

        if r_high is None or r_low is None:
            continue

        # Range bars (00:00–06:00) x-span for the shaded box
        range_bars = day_bars[
            (day_bars["uk_hour"] >= RANGE_START_HOUR) & (day_bars["uk_hour"] < RANGE_END_HOUR)
        ]
        if len(range_bars) == 0:
            continue

        x0 = range_bars.index[0]
        x1 = range_bars.index[-1] + pd.Timedelta(hours=1)

        # Shaded accumulation zone
        shapes.append(dict(
            type="rect",
            xref="x", yref="y",
            x0=x0, x1=x1,
            y0=r_low, y1=r_high,
            fillcolor="rgba(120, 120, 140, 0.12)",
            line=dict(color="rgba(180,180,200,0.3)", width=1),
            layer="below",
        ))

        # Range High dashed line across full trading day
        all_day_bars = day_bars
        x_start = all_day_bars.index[0]
        # Extend line to 16:00 UK bar
        close_bars = day_bars[day_bars["uk_hour"] == TRADE_CLOSE_HOUR]
        x_end = (
            close_bars.index[-1] + pd.Timedelta(hours=1)
            if len(close_bars) > 0
            else all_day_bars.index[-1] + pd.Timedelta(hours=1)
        )

        shapes.append(dict(
            type="line",
            xref="x", yref="y",
            x0=x_start, x1=x_end,
            y0=r_high, y1=r_high,
            line=dict(color="rgba(239, 83, 80, 0.5)", width=1, dash="dot"),
            layer="below",
        ))
        shapes.append(dict(
            type="line",
            xref="x", yref="y",
            x0=x_start, x1=x_end,
            y0=r_low, y1=r_low,
            line=dict(color="rgba(38, 166, 154, 0.5)", width=1, dash="dot"),
            layer="below",
        ))

    # --- Entry markers ---
    buy_bars = d[d["buy_signal"]]
    sell_bars = d[d["sell_signal"]]

    if len(buy_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_bars.index,
                y=buy_bars["entry_price"],
                mode="markers",
                name="BUY Entry",
                marker=dict(symbol="triangle-up", size=14, color="#00e676",
                            line=dict(color="#1b5e20", width=1)),
            ),
            row=1, col=1,
        )

    if len(sell_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_bars.index,
                y=sell_bars["entry_price"],
                mode="markers",
                name="SELL Entry",
                marker=dict(symbol="triangle-down", size=14, color="#f44336",
                            line=dict(color="#b71c1c", width=1)),
            ),
            row=1, col=1,
        )

    # --- Exit markers (TP / SL / Time) — plot on the exit bar ---
    all_trades = d[d["buy_signal"] | d["sell_signal"]].copy()

    tp_exit_x, tp_exit_y = [], []
    sl_exit_x, sl_exit_y = [], []
    time_exit_x, time_exit_y = [], []

    for _, row in all_trades.iterrows():
        exit_ts = row["exit_bar"]
        exit_px = row["exit_price"]
        etype = row["exit_type"]

        if pd.isna(exit_ts) or pd.isna(exit_px):
            continue

        if etype == "tp":
            tp_exit_x.append(exit_ts)
            tp_exit_y.append(exit_px)
        elif etype == "sl":
            sl_exit_x.append(exit_ts)
            sl_exit_y.append(exit_px)
        else:
            time_exit_x.append(exit_ts)
            time_exit_y.append(exit_px)

    if tp_exit_x:
        fig.add_trace(
            go.Scatter(
                x=tp_exit_x, y=tp_exit_y,
                mode="markers",
                name="TP Hit",
                marker=dict(symbol="star", size=12, color="#ffeb3b",
                            line=dict(color="#f57f17", width=1)),
            ),
            row=1, col=1,
        )

    if sl_exit_x:
        fig.add_trace(
            go.Scatter(
                x=sl_exit_x, y=sl_exit_y,
                mode="markers",
                name="SL Hit",
                marker=dict(symbol="x", size=11, color="#ff5252",
                            line=dict(color="#b71c1c", width=2)),
            ),
            row=1, col=1,
        )

    if time_exit_x:
        fig.add_trace(
            go.Scatter(
                x=time_exit_x, y=time_exit_y,
                mode="markers",
                name="Time Exit (16:00)",
                marker=dict(symbol="circle", size=9, color="#b0bec5",
                            line=dict(color="#546e7a", width=1)),
            ),
            row=1, col=1,
        )

    # --- Trade connector lines (entry → exit) ---
    for _, row in all_trades.iterrows():
        exit_ts = row["exit_bar"]
        exit_px = row["exit_price"]
        entry_ts = row.name
        entry_px = row["entry_price"]

        if pd.isna(exit_ts) or pd.isna(exit_px):
            continue

        etype = row["exit_type"]
        color = "#ffeb3b" if etype == "tp" else ("#ff5252" if etype == "sl" else "#b0bec5")

        fig.add_trace(
            go.Scatter(
                x=[entry_ts, exit_ts],
                y=[entry_px, exit_px],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )

    # --- Stats annotation ---
    trade_rows = d[d["buy_signal"] | d["sell_signal"]]
    n_trades = len(trade_rows)
    n_tp = (trade_rows["exit_type"] == "tp").sum()
    n_sl = (trade_rows["exit_type"] == "sl").sum()
    n_time = (trade_rows["exit_type"] == "time").sum()
    total_pips = trade_rows["trade_pnl_pips"].sum()
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
                f"<b>{symbol} 1H — Asian Range Breakout</b>  "
                f"<sup>{start} to {end}</sup><br>"
                f"<sup>Trades: {n_trades}  |  TP: {n_tp}  |  SL: {n_sl}  |  "
                f"Time exit: {n_time}  |  Win rate: {win_rate:.1f}%  |  "
                f"Total P&L: {total_pips:+.1f} pips  |  "
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
# Main experiment runner
# ---------------------------------------------------------------------------


def run_experiment(
    symbol: str = SYMBOL,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    show_chart: bool = True,
    save_outputs: bool = True,
):
    print(f"Loading {symbol} 1H data ({start_date} to {end_date})...")
    loader = DataLoader()
    # Load without date filter to avoid tz-naive vs tz-aware comparison in data_loader
    df = loader.load_fx(symbol, timeframe="1h")

    # Ensure UTC-aware index
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    # Now slice with tz-aware timestamps
    df = df[
        (df.index >= pd.Timestamp(start_date, tz="UTC")) &
        (df.index <= pd.Timestamp(end_date, tz="UTC"))
    ]

    print(f"  Loaded {len(df):,} 1H bars  ({df.index.min()} -> {df.index.max()})")

    print("Detecting range breakout signals...")
    df = calculate_signals(df)

    trade_rows = df[df["buy_signal"] | df["sell_signal"]]
    n_trades = len(trade_rows)
    n_long = int(df["buy_signal"].sum())
    n_short = int(df["sell_signal"].sum())
    n_tp = (trade_rows["exit_type"] == "tp").sum()
    n_sl = (trade_rows["exit_type"] == "sl").sum()
    n_time = (trade_rows["exit_type"] == "time").sum()
    total_pips = trade_rows["trade_pnl_pips"].sum()
    win_rate = (n_tp / n_trades * 100) if n_trades > 0 else 0.0

    print(f"  Total trades     : {n_trades}  (long: {n_long}, short: {n_short})")
    print(f"  TP exits         : {n_tp}")
    print(f"  SL exits         : {n_sl}")
    print(f"  Time exits       : {n_time}")
    print(f"  Win rate         : {win_rate:.1f}%")
    print(f"  Total P&L        : {total_pips:+.1f} pips")

    # MLflow run
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(__file__).resolve().parent / "reports" / f"{symbol}_{run_ts}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    mlflow.start_run(run_name=f"{symbol}_1H_{start_date}_{end_date}")
    mlflow.log_params({
        "symbol": symbol,
        "start_date": start_date,
        "end_date": end_date,
        "range_start_hour_uk": RANGE_START_HOUR,
        "range_end_hour_uk": RANGE_END_HOUR,
        "trade_close_hour_uk": TRADE_CLOSE_HOUR,
        "rr": RR,
        "timezone": TIMEZONE,
    })
    mlflow.log_metrics({
        "total_trades": float(n_trades),
        "n_long": float(n_long),
        "n_short": float(n_short),
        "tp_exits": float(n_tp),
        "sl_exits": float(n_sl),
        "time_exits": float(n_time),
        "win_rate_pct": float(win_rate),
        "total_pnl_pips": float(total_pips),
    })

    # Build chart for the full date range
    print("Building chart...")
    vbt_settings_ok = True
    try:
        import vectorbt as vbt
        vbt.settings.plotting["use_widgets"] = False
    except ImportError:
        vbt_settings_ok = False

    fig = build_chart(df, symbol=symbol, start=start_date, end=end_date)

    if save_outputs:
        chart_path = reports_dir / "rangebo_chart.html"
        fig.write_html(str(chart_path))
        print(f"  Chart saved    -> {chart_path}")

        # Trades summary CSV
        trades_csv = trade_rows[[
            "entry_price", "sl_price", "tp_price",
            "exit_type", "exit_bar", "exit_price", "trade_pnl_pips",
            "buy_signal", "sell_signal",
        ]].copy()
        trades_path = reports_dir / "trades.csv"
        trades_csv.to_csv(trades_path)
        print(f"  Trades saved   -> {trades_path}")

        mlflow.log_artifacts(str(reports_dir), artifact_path="outputs")
        print(f"  MLflow run logged -> {MLFLOW_TRACKING_URI}")

    mlflow.end_run()

    if show_chart:
        fig.show()

    return df, fig


if __name__ == "__main__":
    run_experiment()
