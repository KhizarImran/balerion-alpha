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

SYMBOL = "EURUSD"
START_DATE = "2025-11-18"
END_DATE = "2026-02-27"
TIMEFRAME = "1h"
SWING_LENGTH = 10


def load_ohlcv(symbol, start, end, tf):
    loader = DataLoader()
    df_1m = loader.load_fx(symbol, start_date=start, end_date=end)
    df = loader.resample_ohlcv(df_1m, tf)
    print(f"Loaded {len(df):,} {tf} bars  ({df.index[0]} -> {df.index[-1]})")
    return df


def detect_setups(df: pd.DataFrame, swing_length: int = SWING_LENGTH):
    """
    Returns:
        sell_setups : list of dicts  — bullish OB mitigated down + bearish FVG + retrace up
        buy_setups  : list of dicts  — bearish OB mitigated up   + bullish FVG + retrace down

    Each dict keys:
        ob_ts, mit_ts           — OB rectangle span
        fvg_ts, entry_ts        — FVG rectangle span (fvg formation → entry touch)
        ob_top, ob_bottom       — OB zone prices
        fvg_top, fvg_bottom     — FVG zone prices
        entry_ts                — Timestamp of entry bar (None if not yet touched)
    """
    swing_df = smc.swing_highs_lows(df, swing_length=swing_length)
    fvg_df = smc.fvg(df, join_consecutive=False)

    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()
    opens = df["open"].to_numpy()
    closes = df["close"].to_numpy()
    n = len(df)

    swing_hl = swing_df["HighLow"].to_numpy()
    swing_high_bars = np.where(swing_hl == 1)[0]
    swing_low_bars = np.where(swing_hl == -1)[0]

    fvg_type = fvg_df["FVG"].to_numpy()
    fvg_tops = fvg_df["Top"].to_numpy()
    fvg_bots = fvg_df["Bottom"].to_numpy()

    bull_fvg_bars = np.where(fvg_type == 1)[0]
    bear_fvg_bars = np.where(fvg_type == -1)[0]

    sell_setups = []
    buy_setups = []

    # ── SELL: bullish OB at swing low, mitigated down, bearish FVG, retrace up ──
    for sl_bar in swing_low_bars:
        # last bearish candle at or before the swing low = bullish OB
        ob_bar = None
        for k in range(sl_bar, -1, -1):
            if closes[k] < opens[k]:
                ob_bar = k
                break
        if ob_bar is None:
            continue

        ob_top = highs[ob_bar]
        ob_bottom = lows[ob_bar]

        # mitigation: close <= OB bottom within 120 bars (5 days)
        mit_bar = None
        for k in range(sl_bar + 1, min(sl_bar + 121, n)):
            if closes[k] <= ob_bottom:
                mit_bar = k
                break
        if mit_bar is None:
            continue

        # bearish FVG between ob_bar and mit_bar overlapping OB zone
        window = bear_fvg_bars[(bear_fvg_bars >= ob_bar) & (bear_fvg_bars <= mit_bar)]
        matched = None
        for f in window:
            ft = float(fvg_tops[f])
            fb = float(fvg_bots[f])
            if min(ob_top, ft) > max(ob_bottom, fb):  # zones overlap
                matched = (int(f), fb, ft)
                break
        if matched is None:
            continue

        fvg_bar, fvg_bot, fvg_t = matched

        # entry: price retraces UP into the bearish FVG (high >= fvg_bottom)
        entry_bar = None
        for k in range(mit_bar + 1, n):
            if highs[k] >= fvg_bot:
                entry_bar = k
                break
        if entry_bar is None:
            continue

        entry_price = closes[entry_bar]
        sl = ob_top  # SL above the mitigated OB high
        sl_dist = abs(sl - entry_price)
        tp = entry_price - 2.0 * sl_dist  # 2R below entry

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
        # last bullish candle at or before the swing high = bearish OB
        ob_bar = None
        for k in range(sh_bar, -1, -1):
            if closes[k] > opens[k]:
                ob_bar = k
                break
        if ob_bar is None:
            continue

        ob_top = highs[ob_bar]
        ob_bottom = lows[ob_bar]

        # mitigation: close >= OB top within 120 bars (5 days)
        mit_bar = None
        for k in range(sh_bar + 1, min(sh_bar + 121, n)):
            if closes[k] >= ob_top:
                mit_bar = k
                break
        if mit_bar is None:
            continue

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

        # entry: price retraces DOWN into the bullish FVG (low <= fvg_top)
        entry_bar = None
        for k in range(mit_bar + 1, n):
            if lows[k] <= fvg_t:
                entry_bar = k
                break
        if entry_bar is None:
            continue

        entry_price = closes[entry_bar]
        sl = ob_bottom  # SL below the mitigated OB low
        sl_dist = abs(entry_price - sl)
        tp = entry_price + 2.0 * sl_dist  # 2R above entry

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
