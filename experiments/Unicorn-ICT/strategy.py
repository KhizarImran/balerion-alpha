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

# -- Parameters -----------------------------

SYMBOL = "EURUSD"
START_DATE = "2025-11-18"
END_DATE = "2026-03-13"
LOWER_TF = "1h"  # entry/signal timeframe
HIGHER_TF = "1D"  # confluence FVG filter timeframe
SWING_LENGTH = 10
SESSION_START = 7  # UK session start hour (inclusive), 07:00
SESSION_END = 18  # UK session end hour (exclusive), up to 17:59
RISK_REWARD = 4.0  # take profit multiplier (e.g. 2.0 = 2R, 3.0 = 3R)
MIN_FVG_PIPS = 5  # minimum FVG height in pips — filters out micro/noise gaps

# Pip size per symbol: JPY pairs use 0.01, everything else 0.0001
_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"}
PIP_SIZE = 0.01 if SYMBOL in _JPY_PAIRS else 0.0001

# --------------------------------------------

# Auto-detect Dukascopy files: scan balerion-data for *_dukascopy_*.parquet files
_DUKASCOPY_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "balerion-data"
    / "data"
    / "fx"
)
DUKASCOPY_FILES = {}
if _DUKASCOPY_DIR.exists():
    for f in _DUKASCOPY_DIR.glob("*_dukascopy_*.parquet"):
        # Extract symbol from filename: e.g., "gbpusd_dukascopy_1h.parquet" -> "GBPUSD"
        symbol_from_file = f.stem.split("_dukascopy")[0].upper()
        DUKASCOPY_FILES[symbol_from_file] = f

if DUKASCOPY_FILES:
    print(f"Auto-detected Dukascopy files: {list(DUKASCOPY_FILES.keys())}")


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


def _build_htf_fvgs(df_lower: pd.DataFrame, higher_tf: str = HIGHER_TF) -> list:
    """
    Resample lower-TF data to the higher timeframe, detect FVGs, track mitigation.

    An HTF FVG at bar[i] (middle candle) is formed by bars i-1, i, i+1.
    It becomes active from HTF bar i+2 onwards (the first bar after the
    right-forming candle — same lookahead rule as the lower-TF entry logic).

    A bearish FVG is mitigated when an HTF bar CLOSES >= fvg_top.
    A bullish FVG is mitigated when an HTF bar CLOSES <= fvg_bottom.

    Returns a list of dicts:
        direction    : 1 (bullish) or -1 (bearish)
        top          : float
        bottom       : float
        active_from  : pd.Timestamp — start of HTF bar i+2
        active_until : pd.Timestamp or pd.NaT — start of mitigation bar (NaT = unmitigated)
    """
    from utils import DataLoader

    loader = DataLoader()
    df_htf = loader.resample_ohlcv(df_lower, higher_tf)

    fvg_df = smc.fvg(df_htf, join_consecutive=False)

    htf_closes = df_htf["close"].to_numpy()
    htf_index = df_htf.index
    n_htf = len(df_htf)

    fvg_type = fvg_df["FVG"].to_numpy()
    fvg_tops = fvg_df["Top"].to_numpy()
    fvg_bots = fvg_df["Bottom"].to_numpy()

    htf_fvgs = []

    for i in range(n_htf):
        d = float(fvg_type[i]) if not np.isnan(fvg_type[i]) else 0.0
        if d == 0.0:
            continue

        direction = int(d)  # 1 or -1
        top = float(fvg_tops[i])
        bot = float(fvg_bots[i])

        # active from the start of HTF bar i+2
        active_from_idx = i + 2
        if active_from_idx >= n_htf:
            continue  # not enough bars to ever trade this FVG
        active_from = htf_index[active_from_idx]

        # scan for mitigation: from i+2 onwards
        active_until = pd.NaT
        for j in range(active_from_idx, n_htf):
            c = htf_closes[j]
            if direction == -1 and c >= top:  # bearish FVG mitigated
                active_until = htf_index[j]
                break
            elif direction == 1 and c <= bot:  # bullish FVG mitigated
                active_until = htf_index[j]
                break

        htf_fvgs.append(
            dict(
                direction=direction,
                top=top,
                bottom=bot,
                active_from=active_from,
                active_until=active_until,
            )
        )

    print(
        f"  {higher_tf} FVGs: {sum(1 for f in htf_fvgs if f['direction'] == -1)} bearish, "
        f"{sum(1 for f in htf_fvgs if f['direction'] == 1)} bullish"
    )
    return htf_fvgs


def _in_active_htf_fvg(
    entry_ts: pd.Timestamp,
    entry_price: float,
    direction: int,
    htf_fvgs: list,
) -> bool:
    """
    Return True if entry_ts / entry_price falls inside an active HTF FVG
    that matches the given direction (1=bullish for BUY, -1=bearish for SELL).
    """
    for fvg in htf_fvgs:
        if fvg["direction"] != direction:
            continue
        if entry_ts < fvg["active_from"]:
            continue
        if fvg["active_until"] is not pd.NaT and entry_ts >= fvg["active_until"]:
            continue
        if fvg["bottom"] <= entry_price <= fvg["top"]:
            return True
    return False


def detect_setups(
    df: pd.DataFrame, swing_length: int = SWING_LENGTH, risk_reward: float = RISK_REWARD
):
    """
    Returns:
        sell_setups : list of dicts  — bullish OB mitigated down + bearish FVG + retrace up
        buy_setups  : list of dicts  — bearish OB mitigated up   + bullish FVG + retrace down
        daily_fvgs  : list of dicts  — active daily FVG zones used for MTF filter

    Vectorised numpy forward scans — fast enough for 150k+ bars.
    """
    import time

    t0 = time.time()

    swing_df = smc.swing_highs_lows(df, swing_length=swing_length)
    fvg_df = smc.fvg(df, join_consecutive=False)
    print(f"  smc computed in {time.time() - t0:.1f}s")

    # build active HTF FVG list for multi-timeframe confluence filter
    htf_fvgs = _build_htf_fvgs(df)

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

    MIT_WINDOW = 48  # 120 bars = 5 days

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

        # FVG size filter: skip micro gaps below MIN_FVG_PIPS
        if (fvg_t - fvg_bot) < MIN_FVG_PIPS * PIP_SIZE:
            continue

        # entry: first bar AFTER the right candle of the FVG (fvg_bar+1 is still a
        # forming candle — earliest valid entry is fvg_bar+2, the 4th candle)
        entry_start = max(mit_bar + 1, fvg_bar + 2)
        entry_idx = _first_true_after(highs >= fvg_bot, entry_start, n)
        if entry_idx == -1:
            continue
        entry_bar = entry_idx
        entry_price = closes[entry_bar]

        # session filter: only trade during UK hours (SESSION_START to SESSION_END)
        entry_hour = df.index[entry_bar].hour  # type: ignore[union-attr]
        if not (SESSION_START <= entry_hour < SESSION_END):
            continue

        # multi-timeframe filter: entry must be inside an active HTF bearish FVG
        if not _in_active_htf_fvg(df.index[entry_bar], entry_price, -1, htf_fvgs):
            continue

        # SL above the highest boundary of the OB/FVG combined zone — gives the
        # trade room through both zones before being invalidated
        sl = max(ob_top, fvg_t)
        sl_dist = abs(sl - entry_price)
        tp = entry_price - risk_reward * sl_dist

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

        # FVG size filter: skip micro gaps below MIN_FVG_PIPS
        if (fvg_t - fvg_bot) < MIN_FVG_PIPS * PIP_SIZE:
            continue

        # entry: first bar AFTER the right candle of the FVG (fvg_bar+1 is still a
        # forming candle — earliest valid entry is fvg_bar+2, the 4th candle)
        entry_start = max(mit_bar + 1, fvg_bar + 2)
        entry_idx = _first_true_after(lows <= fvg_t, entry_start, n)
        if entry_idx == -1:
            continue
        entry_bar = entry_idx
        entry_price = closes[entry_bar]

        # session filter: only trade during UK hours (SESSION_START to SESSION_END)
        entry_hour = df.index[entry_bar].hour
        if not (SESSION_START <= entry_hour < SESSION_END):
            continue

        # multi-timeframe filter: entry must be inside an active HTF bullish FVG
        if not _in_active_htf_fvg(df.index[entry_bar], entry_price, 1, htf_fvgs):
            continue

        # SL below the lowest boundary of the OB/FVG combined zone — gives the
        # trade room through both zones before being invalidated
        sl = min(ob_bottom, fvg_bot)
        sl_dist = abs(entry_price - sl)
        tp = entry_price + risk_reward * sl_dist

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
    return sell_setups, buy_setups, htf_fvgs


def build_chart(
    df,
    sell_setups,
    buy_setups,
    htf_fvgs=None,
    symbol=SYMBOL,
    lower_tf=LOWER_TF,
    higher_tf=HIGHER_TF,
):
    """
    3-row chart:
      Row 1 — Lower-TF candlesticks with OB/FVG/SL/TP overlays and entry markers
      Row 2 — Lower-TF volume
      Row 3 — Higher-TF candlesticks with all active HTF FVG rectangles
    """
    from utils import DataLoader

    loader = DataLoader()
    df_htf = loader.resample_ohlcv(df, higher_tf)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.52, 0.13, 0.35],
        subplot_titles=(
            f"{symbol} {lower_tf.upper()} — Failed OB + FVG",
            f"Volume ({lower_tf.upper()})",
            f"{symbol} {higher_tf.upper()} — FVG Zones",
        ),
    )

    # ── Row 1: 1H candlesticks ──────────────────────────────────────────────────
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC 1H",
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
        ),
        row=1,
        col=1,
    )

    # ── Row 2: 1H volume ────────────────────────────────────────────────────────
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

    # ── Row 3: Higher-TF candlesticks ──────────────────────────────────────────
    htf_vol_colors = [
        "#26a69a" if c >= o else "#ef5350"
        for c, o in zip(df_htf["close"], df_htf["open"])
    ]
    fig.add_trace(
        go.Candlestick(
            x=df_htf.index,
            open=df_htf["open"],
            high=df_htf["high"],
            low=df_htf["low"],
            close=df_htf["close"],
            name=f"OHLC {higher_tf.upper()}",
            increasing_line_color="#26a69a",
            increasing_fillcolor="#26a69a",
            decreasing_line_color="#ef5350",
            decreasing_fillcolor="#ef5350",
        ),
        row=3,
        col=1,
    )

    shapes = []
    bar_width_1h = pd.Timedelta(hours=1)
    bar_width_1d = pd.Timedelta(days=1)

    # ── SELL setups on row 1 ────────────────────────────────────────────────────
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
        # FVG — yellow-green dashed
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
        sl_end = s["entry_ts"] + bar_width_1h * 30
        # SL line
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
        # TP line
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

    # ── BUY setups on row 1 ─────────────────────────────────────────────────────
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
        # FVG — orange dashed
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
        sl_end = s["entry_ts"] + bar_width_1h * 30
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

    # ── HTF FVG rectangles on row 3 ────────────────────────────────────────────
    # Each FVG spans from active_from to active_until (or end of data if unmitigated)
    data_end = df_htf.index[-1] + bar_width_1d * 3  # extend slightly past last bar
    if htf_fvgs:
        for fvg in htf_fvgs:
            x1 = fvg["active_until"] if fvg["active_until"] is not pd.NaT else data_end
            if fvg["direction"] == -1:
                # bearish FVG — red zone
                fill = "rgba(239, 83, 80, 0.20)"
                border = "rgba(239, 83, 80, 0.80)"
                label_color = "#ef5350"
            else:
                # bullish FVG — green zone
                fill = "rgba(38, 166, 154, 0.20)"
                border = "rgba(38, 166, 154, 0.80)"
                label_color = "#26a69a"

            shapes.append(
                dict(
                    type="rect",
                    xref="x3",  # row 3 x-axis
                    yref="y3",  # row 3 y-axis
                    layer="below",
                    x0=fvg["active_from"],
                    x1=x1,
                    y0=fvg["bottom"],
                    y1=fvg["top"],
                    fillcolor=fill,
                    line=dict(color=border, width=1, dash="dash"),
                )
            )
            # small label at the left edge of each FVG zone
            direction_label = "BFVG" if fvg["direction"] == 1 else "SFVG"
            fig.add_trace(
                go.Scatter(
                    x=[fvg["active_from"]],
                    y=[(fvg["top"] + fvg["bottom"]) / 2],
                    mode="text",
                    text=[direction_label],
                    textfont=dict(color=label_color, size=10),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=3,
                col=1,
            )

    fig.update_layout(
        height=1200,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis2_rangeslider_visible=False,
        xaxis3_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=True,
        shapes=shapes,
        margin=dict(t=110),
        title=dict(
            text=(
                f"<b>{symbol} — Failed OB + FVG  |  {higher_tf.upper()} FVG Confluence</b><br>"
                f"<sup>"
                f"SELL setups: {len(sell_setups)}  |  "
                f"BUY setups: {len(buy_setups)}  |  "
                f"swing_length={SWING_LENGTH}  |  "
                f"{higher_tf.upper()} FVGs: {sum(1 for f in (htf_fvgs or []) if f['direction'] == -1)} bearish, "
                f"{sum(1 for f in (htf_fvgs or []) if f['direction'] == 1)} bullish"
                f"</sup>"
            ),
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
        ),
    )
    fig.update_yaxes(title_text=f"Price ({lower_tf.upper()})", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_yaxes(title_text=f"Price ({higher_tf.upper()})", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)

    return fig


if __name__ == "__main__":
    df = load_ohlcv(SYMBOL, START_DATE, END_DATE, LOWER_TF)
    sell_setups, buy_setups, htf_fvgs = detect_setups(df)

    fig = build_chart(df, sell_setups, buy_setups, htf_fvgs=htf_fvgs)

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"failed_ob_fvg_{SYMBOL}_{LOWER_TF}.html"
    fig.write_html(str(out))
    print(f"Chart saved -> {out}")
    fig.show()
