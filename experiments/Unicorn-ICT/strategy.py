"""
ICT Failed OB + FVG — strategy.py

SELL SETUP
----------
1. Bullish OB = last bearish candle (close < open) before a confirmed swing LOW.
2. That OB gets mitigated: price closes BELOW the OB's low within 24 bars.
   The mitigation leg is a displacement move DOWN.
3. A bearish FVG forms during that mitigation leg and overlaps the OB zone.
   (bearish FVG = gap down: high[i-1] > low[i+1])
4. Price retraces back UP into the FVG zone — that bar is the SELL entry.

BUY SETUP (mirror)
------------------
1. Bearish OB = last bullish candle (close > open) before a confirmed swing HIGH.
2. Mitigated: price closes ABOVE the OB's high within 24 bars (displacement UP).
3. A bullish FVG forms during that leg and overlaps the OB zone.
4. Price retraces back DOWN into the FVG zone — BUY entry.

Chart visuals (only for setups that have a confirmed entry):
  Blue  OB rectangle  (ob_ts → mit_ts)
  Yellow-green FVG rectangle (fvg_ts → entry_ts)    ← SELL setups
  Red   triangle-down marker at entry bar

  Red   OB rectangle  (ob_ts → mit_ts)
  Orange FVG rectangle (fvg_ts → entry_ts)           ← BUY setups
  Green triangle-up marker at entry bar

Run:
    uv run python experiments/Unicorn-ICT/strategy.py
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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from smartmoneyconcepts import smc

from utils import DataLoader

SYMBOL = "GBPUSD"
START_DATE = "2025-11-18"
END_DATE = "2026-03-13"
TIMEFRAME = "1h"
SWING_LENGTH = 10

# Path to the Dukascopy 1H parquet (already 1H — no resampling needed)
DUKASCOPY_FILES = {
    "GBPUSD": Path(__file__).resolve().parent.parent.parent.parent
    / "balerion-data"
    / "data"
    / "fx"
    / "gbpusd_dukascopy_1h.parquet",
}


def load_dukascopy(symbol: str, start: str = None, end: str = None) -> pd.DataFrame:
    """
    Load a Dukascopy 1H parquet file.
    The file has 'timestamp' as a plain column (not the index) with UTC tz.
    Returns a DatetimeIndex DataFrame with lowercase OHLCV columns.
    """
    path = DUKASCOPY_FILES[symbol]
    df = pd.read_parquet(path)

    # set timestamp as index, strip timezone for consistency with smc library
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.set_index("timestamp")
    df.index.name = "timestamp"
    df = df[["open", "high", "low", "close", "volume"]]

    if start:
        df = df[df.index >= pd.Timestamp(start)]
    if end:
        df = df[df.index <= pd.Timestamp(end)]

    df = df.dropna()
    print(f"Loaded {len(df):,} 1h bars  ({df.index[0]} -> {df.index[-1]})")
    return df


def load_ohlcv(symbol, start, end, tf):
    """Load OHLCV — uses Dukascopy file if available, else falls back to DataLoader."""
    if symbol in DUKASCOPY_FILES:
        return load_dukascopy(symbol, start, end)
    loader = DataLoader()
    df_1m = loader.load_fx(symbol, start_date=start, end_date=end)
    df = loader.resample_ohlcv(df_1m, tf)
    print(f"Loaded {len(df):,} {tf} bars  ({df.index[0]} -> {df.index[-1]})")
    return df


def _first_true_after(arr: np.ndarray, start: int, end: int) -> int:
    """Return index of first True in arr[start:end], or -1 if none."""
    hits = np.where(arr[start:end])[0]
    return int(hits[0]) + start if len(hits) else -1


def detect_setups(df: pd.DataFrame, swing_length: int = SWING_LENGTH):
    """
    Returns:
        sell_setups : list of dicts  — bullish OB mitigated down + bearish FVG + retrace up
        buy_setups  : list of dicts  — bearish OB mitigated up   + bullish FVG + retrace down

    Vectorised numpy forward scans — fast enough for 150k+ bars.
    """
    import time

    t0 = time.time()

    swing_df = smc.swing_highs_lows(df, swing_length=swing_length)
    fvg_df = smc.fvg(df, join_consecutive=False)
    print(f"  smc computed in {time.time() - t0:.1f}s")

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)

    # boolean arrays for fast numpy scans
    is_bear_candle = closes < opens  # bearish candle mask
    is_bull_candle = closes > opens  # bullish candle mask

    swing_hl = swing_df["HighLow"].to_numpy()
    swing_high_bars = np.where(swing_hl == 1)[0]
    swing_low_bars = np.where(swing_hl == -1)[0]

    fvg_type = fvg_df["FVG"].to_numpy()
    fvg_tops = fvg_df["Top"].to_numpy()
    fvg_bots = fvg_df["Bottom"].to_numpy()

    bull_fvg_bars = np.where(fvg_type == 1)[0]
    bear_fvg_bars = np.where(fvg_type == -1)[0]

    MIT_WINDOW = 121  # 120 bars = 5 days

    sell_setups = []
    buy_setups = []

    # ── SELL: bullish OB at swing low, mitigated down, bearish FVG, retrace up ──
    for sl_bar in swing_low_bars:
        # last bearish candle at or before the swing low — numpy argmax trick
        # np.where on reversed slice, take first hit
        cands = np.where(is_bear_candle[: sl_bar + 1])[0]
        if len(cands) == 0:
            continue
        ob_bar = int(cands[-1])
        ob_top = highs[ob_bar]
        ob_bottom = lows[ob_bar]

        # mitigation: first close <= ob_bottom within MIT_WINDOW bars after sl_bar
        end = min(sl_bar + MIT_WINDOW, n)
        mit_idx = _first_true_after(closes[:end] <= ob_bottom, sl_bar + 1, end)
        if mit_idx == -1:
            continue
        mit_bar = mit_idx

        # bearish FVG between ob_bar and mit_bar overlapping OB zone
        window = bear_fvg_bars[(bear_fvg_bars >= ob_bar) & (bear_fvg_bars <= mit_bar)]
        matched = None
        for f in window:
            ft = float(fvg_tops[f])
            fb = float(fvg_bots[f])
            if min(ob_top, ft) > max(ob_bottom, fb):
                matched = (int(f), fb, ft)
                break
        if matched is None:
            continue
        fvg_bar, fvg_bot, fvg_t = matched

        # entry: first bar after mit_bar where high >= fvg_bot
        entry_idx = _first_true_after(highs >= fvg_bot, mit_bar + 1, n)
        if entry_idx == -1:
            continue
        entry_bar = entry_idx
        entry_price = closes[entry_bar]
        sl = ob_top
        sl_dist = abs(sl - entry_price)
        tp = entry_price - 2.0 * sl_dist

        sell_setups.append(
            dict(
                ob_ts=df.index[ob_bar],
                mit_ts=df.index[mit_bar],
                fvg_ts=df.index[fvg_bar],
                entry_ts=df.index[entry_bar],
                ob_top=ob_top,
                ob_bottom=ob_bottom,
                fvg_top=fvg_t,
                fvg_bottom=fvg_bot,
                entry=entry_price,
                sl=sl,
                tp=tp,
            )
        )

    # ── BUY: bearish OB at swing high, mitigated up, bullish FVG, retrace down ──
    for sh_bar in swing_high_bars:
        # last bullish candle at or before the swing high
        cands = np.where(is_bull_candle[: sh_bar + 1])[0]
        if len(cands) == 0:
            continue
        ob_bar = int(cands[-1])
        ob_top = highs[ob_bar]
        ob_bottom = lows[ob_bar]

        # mitigation: first close >= ob_top within MIT_WINDOW bars after sh_bar
        end = min(sh_bar + MIT_WINDOW, n)
        mit_idx = _first_true_after(closes[:end] >= ob_top, sh_bar + 1, end)
        if mit_idx == -1:
            continue
        mit_bar = mit_idx

        # bullish FVG between ob_bar and mit_bar overlapping OB zone
        window = bull_fvg_bars[(bull_fvg_bars >= ob_bar) & (bull_fvg_bars <= mit_bar)]
        matched = None
        for f in window:
            ft = float(fvg_tops[f])
            fb = float(fvg_bots[f])
            if min(ob_top, ft) > max(ob_bottom, fb):
                matched = (int(f), fb, ft)
                break
        if matched is None:
            continue
        fvg_bar, fvg_bot, fvg_t = matched

        # entry: first bar after mit_bar where low <= fvg_t
        entry_idx = _first_true_after(lows <= fvg_t, mit_bar + 1, n)
        if entry_idx == -1:
            continue
        entry_bar = entry_idx
        entry_price = closes[entry_bar]
        sl = ob_bottom
        sl_dist = abs(entry_price - sl)
        tp = entry_price + 2.0 * sl_dist

        buy_setups.append(
            dict(
                ob_ts=df.index[ob_bar],
                mit_ts=df.index[mit_bar],
                fvg_ts=df.index[fvg_bar],
                entry_ts=df.index[entry_bar],
                ob_top=ob_top,
                ob_bottom=ob_bottom,
                fvg_top=fvg_t,
                fvg_bottom=fvg_bot,
                entry=entry_price,
                sl=sl,
                tp=tp,
            )
        )

    print(f"SELL setups : {len(sell_setups)}")
    print(f"BUY  setups : {len(buy_setups)}")
    return sell_setups, buy_setups


def build_chart(df, sell_setups, buy_setups, symbol=SYMBOL):
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.80, 0.20],
        subplot_titles=(f"{symbol} {TIMEFRAME.upper()} — Failed OB + FVG", "Volume"),
    )

    # candlesticks
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

    # volume
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

    shapes = []

    # ── SELL setups ────────────────────────────────────────────────────────────
    bar_width = pd.Timedelta(hours=1)  # 1H bars — used to extend SL/TP lines

    for s in sell_setups:
        # OB — blue rectangle
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                layer="below",
                x0=s["ob_ts"],
                x1=s["mit_ts"],
                y0=s["ob_bottom"],
                y1=s["ob_top"],
                fillcolor="rgba(33, 150, 243, 0.15)",
                line=dict(color="rgba(33, 150, 243, 0.80)", width=1),
            )
        )
        # FVG — yellow-green dashed, from formation bar to entry touch
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                layer="below",
                x0=s["fvg_ts"],
                x1=s["entry_ts"],
                y0=s["fvg_bottom"],
                y1=s["fvg_top"],
                fillcolor="rgba(205, 220, 57, 0.25)",
                line=dict(color="rgba(205, 220, 57, 0.90)", width=1, dash="dash"),
            )
        )
        # SL line — red dotted
        sl_end = s["entry_ts"] + bar_width * 30
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=s["entry_ts"],
                x1=sl_end,
                y0=s["sl"],
                y1=s["sl"],
                line=dict(color="rgba(244, 67, 54, 0.8)", width=1, dash="dot"),
            )
        )
        # TP line — green dotted
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=s["entry_ts"],
                x1=sl_end,
                y0=s["tp"],
                y1=s["tp"],
                line=dict(color="rgba(0, 230, 118, 0.8)", width=1, dash="dot"),
            )
        )

    # SELL entry markers
    if sell_setups:
        fig.add_trace(
            go.Scatter(
                x=[s["entry_ts"] for s in sell_setups],
                y=[df.loc[s["entry_ts"], "high"] * 1.0003 for s in sell_setups],
                mode="markers",
                name="SELL entry",
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

    # ── BUY setups ─────────────────────────────────────────────────────────────
    for s in buy_setups:
        # OB — red rectangle
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                layer="below",
                x0=s["ob_ts"],
                x1=s["mit_ts"],
                y0=s["ob_bottom"],
                y1=s["ob_top"],
                fillcolor="rgba(239, 83, 80, 0.15)",
                line=dict(color="rgba(239, 83, 80, 0.80)", width=1),
            )
        )
        # FVG — orange dashed, from formation bar to entry touch
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                layer="below",
                x0=s["fvg_ts"],
                x1=s["entry_ts"],
                y0=s["fvg_bottom"],
                y1=s["fvg_top"],
                fillcolor="rgba(255, 152, 0, 0.25)",
                line=dict(color="rgba(255, 152, 0, 0.90)", width=1, dash="dash"),
            )
        )
        # SL line — red dotted
        sl_end = s["entry_ts"] + bar_width * 30
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=s["entry_ts"],
                x1=sl_end,
                y0=s["sl"],
                y1=s["sl"],
                line=dict(color="rgba(244, 67, 54, 0.8)", width=1, dash="dot"),
            )
        )
        # TP line — green dotted
        shapes.append(
            dict(
                type="line",
                xref="x",
                yref="y",
                x0=s["entry_ts"],
                x1=sl_end,
                y0=s["tp"],
                y1=s["tp"],
                line=dict(color="rgba(0, 230, 118, 0.8)", width=1, dash="dot"),
            )
        )

    # BUY entry markers
    if buy_setups:
        fig.add_trace(
            go.Scatter(
                x=[s["entry_ts"] for s in buy_setups],
                y=[df.loc[s["entry_ts"], "low"] * 0.9997 for s in buy_setups],
                mode="markers",
                name="BUY entry",
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

    fig.update_layout(
        height=900,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=True,
        shapes=shapes,
        margin=dict(t=110),
        title=dict(
            text=(
                f"<b>{symbol} {TIMEFRAME.upper()} — Failed OB + FVG</b><br>"
                f"<sup>"
                f"SELL setups: {len(sell_setups)}  |  "
                f"BUY setups: {len(buy_setups)}  |  "
                f"swing_length={SWING_LENGTH}"
                f"</sup>"
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    return fig


if __name__ == "__main__":
    df = load_ohlcv(SYMBOL, START_DATE, END_DATE, TIMEFRAME)
    sell_setups, buy_setups = detect_setups(df)

    fig = build_chart(df, sell_setups, buy_setups)

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"failed_ob_fvg_{SYMBOL}_{TIMEFRAME}.html"
    fig.write_html(str(out))
    print(f"Chart saved -> {out}")
    fig.show()
