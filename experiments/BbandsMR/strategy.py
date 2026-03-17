"""
BbandsMR — Bollinger Bands Mean Reversion Signal Detection + Visualisation

Intraday mean reversion strategy on EURGBP 1H:

  1. Bollinger Bands   : 20-period, 2.0 std dev on close
  2. Entry trigger     : close < lower band (long) or close > upper band (short)
  3. ADX filter        : ADX < 25 — only trade in ranging / non-trending conditions
  4. EMA 200 filter    : only trade in the direction that closes the gap to EMA200.
                         If close is far ABOVE EMA200 -> short only (snap back down).
                         If close is far BELOW EMA200 -> long only (snap back up).
                         "Far" is defined by EMA_DIST_THRESHOLD (fraction of price).
                         Setups near EMA200 (within threshold) are skipped — no
                         strong directional pull, so mean reversion edge is weaker.
  5. TP                : BB midline (SMA 20) at the signal bar
  6. SL                : entry close +/- 1.5 x ATR
  7. Min RR filter     : skip setups where (TP distance / SL distance) < MIN_RR
  8. Deduplication     : one trade at a time — forward-simulate each trade to
                         its exit before scanning for the next signal

Conforms to STRATEGY_BACKTEST_PATTERN.md:
  - load_ohlcv(symbol, start, end, tf) -> pd.DataFrame  (tz-naive, lowercase cols)
  - detect_setups(df, ...) -> (sell_setups, buy_setups, extra)
  - Each setup dict contains at minimum: entry_ts, entry, sl, tp

Usage:
    uv run python experiments/BbandsMR/strategy.py
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
import talib
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import DataLoader

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL = "EURGBP"
START_DATE = "2024-03-17"  # 2 years of Dukascopy 1H data
END_DATE = None  # today
TIMEFRAME = "1H"

# Bollinger Bands
BB_PERIOD = 20
BB_STD = 2.0

# Indicators
ATR_PERIOD = 14
ADX_PERIOD = 14

# Filter
ADX_THRESHOLD = 25  # ADX below this = market is ranging / non-trending

# EMA 200 directional filter
EMA_PERIOD = 200  # slow EMA — defines the macro mean price level
EMA_DIST_THRESHOLD = 0.002  # 0.2% of price (~17 pips on EURGBP ~0.85)
# if |close - ema200| / ema200 >= threshold:
#   above -> short only (revert down toward EMA)
#   below -> long  only (revert up   toward EMA)
# if within threshold: skip (no strong directional pull)

# Trade parameters
SL_ATR_MULT = 1.5  # SL = entry +/- SL_ATR_MULT * ATR
MIN_RR = 0.5  # minimum (TP distance / SL distance) to accept a setup

_GBP_PAIRS = {"EURGBP", "GBPUSD", "GBPJPY", "GBPCHF", "GBPCAD", "GBPAUD", "GBPNZD"}
PIP_SIZE = 0.0001  # EURGBP is a 4-decimal pair


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
    tf: str = TIMEFRAME,
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
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        df.index = df.index.tz_localize(None)
        df = df[["open", "high", "low", "close", "volume"]]
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
# Indicator helpers — all via ta-lib
# ---------------------------------------------------------------------------


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach all indicator columns to a copy of the DataFrame using ta-lib.

    Indicators added:
      bb_upper    — upper Bollinger Band
      bb_mid      — middle band (SMA BB_PERIOD)
      bb_lower    — lower Bollinger Band
      atr         — ATR(high, low, close, ATR_PERIOD)
      adx         — ADX(high, low, close, ADX_PERIOD)
      ema200      — EMA(close, EMA_PERIOD) — macro mean / directional filter
      ema_dist    — (close - ema200) / ema200 — signed fractional distance from EMA
    """
    df = df.copy()

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    upper, mid, lower = talib.BBANDS(
        close,
        timeperiod=BB_PERIOD,
        nbdevup=BB_STD,
        nbdevdn=BB_STD,
        matype=0,  # SMA
    )

    df["bb_upper"] = upper
    df["bb_mid"] = mid
    df["bb_lower"] = lower
    df["atr"] = talib.ATR(high, low, close, timeperiod=ATR_PERIOD)
    df["adx"] = talib.ADX(high, low, close, timeperiod=ADX_PERIOD)
    df["ema200"] = talib.EMA(close, timeperiod=EMA_PERIOD)
    # Signed fractional distance: positive = above EMA, negative = below EMA
    df["ema_dist"] = (df["close"] - df["ema200"]) / df["ema200"]

    return df


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def detect_setups(
    df: pd.DataFrame,
    bb_period: int = BB_PERIOD,
    bb_std: float = BB_STD,
    adx_threshold: float = ADX_THRESHOLD,
    sl_atr_mult: float = SL_ATR_MULT,
    min_rr: float = MIN_RR,
    ema_period: int = EMA_PERIOD,
    ema_dist_threshold: float = EMA_DIST_THRESHOLD,
) -> tuple:
    """
    Detect Bollinger Bands mean reversion setups with EMA200 directional filter.

    Entry:
      - close outside BB band AND ADX < adx_threshold (ranging market)
      - EMA200 directional filter:
          * close is far ABOVE EMA200 (ema_dist >= +threshold) -> short only
          * close is far BELOW EMA200 (ema_dist <= -threshold) -> long  only
          * close near EMA200 (|ema_dist| < threshold)         -> skip (no edge)
    TP:    BB midline at signal bar
    SL:    entry +/- sl_atr_mult * ATR

    One trade at a time — forward-simulates each setup to its exit (SL or TP)
    before scanning for the next signal.

    Parameters
    ----------
    df : pd.DataFrame
        1H OHLCV with tz-naive DatetimeIndex.
    ema_dist_threshold : float
        Minimum fractional distance from EMA200 required to take a trade.
        E.g. 0.002 = 0.2% (~17 pips on EURGBP ~0.85). Trades within this
        band around EMA200 are skipped.

    Returns
    -------
    sell_setups : list of dicts
    buy_setups  : list of dicts
    extra       : dict with 'df_indicators' for charting
    """
    df = _add_indicators(df.copy())

    sell_setups = []
    buy_setups = []

    # Warm-up: EMA200 needs 200 bars; ADX needs ~28; BBands needs 20.
    warmup = EMA_PERIOD + 5

    i = warmup
    while i < len(df):
        row = df.iloc[i]
        ts = df.index[i]

        close = row["close"]
        bb_upper = row["bb_upper"]
        bb_lower = row["bb_lower"]
        bb_mid = row["bb_mid"]
        atr = row["atr"]
        adx = row["adx"]
        ema200 = row["ema200"]
        ema_dist = row["ema_dist"]

        # Skip if any indicator is NaN
        if any(
            pd.isna(v) for v in [bb_upper, bb_lower, bb_mid, atr, adx, ema200, ema_dist]
        ):
            i += 1
            continue

        # EMA200 directional filter:
        # ema_dist > 0  -> price is above EMA200  -> only short (close the gap downward)
        # ema_dist < 0  -> price is below EMA200  -> only long  (close the gap upward)
        # |ema_dist| < threshold -> near EMA200 -> skip (no clear mean-reversion pull)
        long_allowed = ema_dist <= -ema_dist_threshold  # price stretched below EMA
        short_allowed = ema_dist >= ema_dist_threshold  # price stretched above EMA

        # ADX pre-check — applies to both directions
        if adx >= adx_threshold:
            i += 1
            continue

        # --- Long setup: close < lower band, price stretched below EMA200 ---
        if close < bb_lower and long_allowed:
            entry = close
            sl = entry - sl_atr_mult * atr
            tp = bb_mid

            sl_dist = entry - sl
            tp_dist = tp - entry

            if sl_dist <= 0 or tp_dist <= 0:
                i += 1
                continue
            if tp_dist / sl_dist < min_rr:
                i += 1
                continue

            exit_ts, exit_price, exit_type, exit_idx = _simulate_exit(
                df, i, entry, sl, tp, "long"
            )

            setup = {
                "entry_ts": ts,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_mid": bb_mid,
                "atr": atr,
                "adx": adx,
                "ema200": ema200,
                "ema_dist": ema_dist,
                "direction": "long",
                "exit_ts": exit_ts,
                "exit_price": exit_price,
                "exit_type": exit_type,
            }
            buy_setups.append(setup)

            i = exit_idx + 1 if exit_idx is not None else i + 1
            continue

        # --- Short setup: close > upper band, price stretched above EMA200 ---
        if close > bb_upper and short_allowed:
            entry = close
            sl = entry + sl_atr_mult * atr
            tp = bb_mid

            sl_dist = sl - entry
            tp_dist = entry - tp

            if sl_dist <= 0 or tp_dist <= 0:
                i += 1
                continue
            if tp_dist / sl_dist < min_rr:
                i += 1
                continue

            exit_ts, exit_price, exit_type, exit_idx = _simulate_exit(
                df, i, entry, sl, tp, "short"
            )

            setup = {
                "entry_ts": ts,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "bb_upper": bb_upper,
                "bb_lower": bb_lower,
                "bb_mid": bb_mid,
                "atr": atr,
                "adx": adx,
                "ema200": ema200,
                "ema_dist": ema_dist,
                "direction": "short",
                "exit_ts": exit_ts,
                "exit_price": exit_price,
                "exit_type": exit_type,
            }
            sell_setups.append(setup)

            i = exit_idx + 1 if exit_idx is not None else i + 1
            continue

        i += 1

    extra = {
        "df_indicators": df,
    }
    return sell_setups, buy_setups, extra


def _simulate_exit(
    df: pd.DataFrame,
    entry_idx: int,
    entry: float,
    sl: float,
    tp: float,
    direction: str,
) -> tuple:
    """
    Forward-scan bars after entry_idx until SL or TP is hit.

    Returns (exit_ts, exit_price, exit_type, exit_idx).
    exit_type is 'tp', 'sl', or 'open' (if no exit found in remaining data).
    exit_idx is None if trade is still open at end of data.
    """
    for j in range(entry_idx + 1, len(df)):
        bar_high = df["high"].iloc[j]
        bar_low = df["low"].iloc[j]
        bar_ts = df.index[j]

        if direction == "long":
            if bar_low <= sl:
                return bar_ts, sl, "sl", j
            if bar_high >= tp:
                return bar_ts, tp, "tp", j
        else:
            if bar_high >= sl:
                return bar_ts, sl, "sl", j
            if bar_low <= tp:
                return bar_ts, tp, "tp", j

    # Trade still open at end of data
    return None, None, "open", None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def build_chart(
    df: pd.DataFrame,
    sell_setups: list,
    buy_setups: list,
    extra: dict = None,
    symbol: str = SYMBOL,
) -> go.Figure:
    """
    Build a 3-row Plotly chart:
      Row 1 (55%): Candlesticks + Bollinger Bands + entry/exit markers
      Row 2 (25%): ADX indicator with threshold line
      Row 3 (20%): ATR indicator
    """
    if extra is None:
        extra = {}

    df_ind = extra.get("df_indicators", df)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            f"{symbol} 1H — BbandsMR (BB{BB_PERIOD}/{BB_STD} + ADX + EMA{EMA_PERIOD} filter)",
            f"ADX ({ADX_PERIOD}) — threshold {ADX_THRESHOLD}",
            f"ATR ({ATR_PERIOD})",
        ),
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

    # --- Bollinger Bands ---
    if "bb_upper" in df_ind.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ind.index,
                y=df_ind["bb_upper"],
                name=f"BB Upper ({BB_PERIOD}, {BB_STD})",
                line=dict(color="rgba(100, 160, 255, 0.6)", width=1, dash="dot"),
            ),
            row=1,
            col=1,
        )

    if "bb_lower" in df_ind.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ind.index,
                y=df_ind["bb_lower"],
                name=f"BB Lower ({BB_PERIOD}, {BB_STD})",
                line=dict(color="rgba(100, 160, 255, 0.6)", width=1, dash="dot"),
                fill="tonexty",
                fillcolor="rgba(100, 160, 255, 0.04)",
            ),
            row=1,
            col=1,
        )

    if "bb_mid" in df_ind.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ind.index,
                y=df_ind["bb_mid"],
                name=f"BB Mid / SMA {BB_PERIOD}",
                line=dict(color="rgba(180, 180, 255, 0.5)", width=1),
            ),
            row=1,
            col=1,
        )

    # --- EMA 200 ---
    if "ema200" in df_ind.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ind.index,
                y=df_ind["ema200"],
                name=f"EMA {EMA_PERIOD}",
                line=dict(color="#ffb300", width=1.8),
                opacity=0.9,
            ),
            row=1,
            col=1,
        )

    # --- ADX ---
    if "adx" in df_ind.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ind.index,
                y=df_ind["adx"],
                name=f"ADX {ADX_PERIOD}",
                line=dict(color="#ce93d8", width=1.2),
            ),
            row=2,
            col=1,
        )
        fig.add_hline(
            y=ADX_THRESHOLD,
            line=dict(color="rgba(255,255,255,0.25)", width=1, dash="dot"),
            row=2,
            col=1,
        )

    # --- ATR ---
    if "atr" in df_ind.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ind.index,
                y=df_ind["atr"],
                name=f"ATR {ATR_PERIOD}",
                line=dict(color="#80cbc4", width=1.2),
                fill="tozeroy",
                fillcolor="rgba(128,203,196,0.08)",
            ),
            row=3,
            col=1,
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
                    size=13,
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
                    size=13,
                    color="#f44336",
                    line=dict(color="#b71c1c", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Exit markers + dotted trade lines ---
    tp_x, tp_y = [], []
    sl_x, sl_y = [], []

    for setup, direction in all_setups:
        entry_ts = setup["entry_ts"]
        exit_ts = setup.get("exit_ts")
        exit_price = setup.get("exit_price")
        exit_type = setup.get("exit_type", "open")

        if exit_ts is None or exit_price is None:
            continue

        color = "#ffeb3b" if exit_type == "tp" else "#ff5252"

        # Dotted line from entry to exit
        fig.add_trace(
            go.Scatter(
                x=[entry_ts, exit_ts],
                y=[setup["entry"], exit_price],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )

        if exit_type == "tp":
            tp_x.append(exit_ts)
            tp_y.append(exit_price)
        else:
            sl_x.append(exit_ts)
            sl_y.append(exit_price)

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

    # --- Summary stats ---
    n_trades = len(all_setups)
    n_tp = len(tp_x)
    n_sl = len(sl_x)
    n_open = n_trades - n_tp - n_sl
    win_rate = (n_tp / (n_tp + n_sl) * 100) if (n_tp + n_sl) > 0 else 0.0

    # Hide weekend gaps (FX market closed Sat / most of Sun)
    _rangebreaks = [
        dict(bounds=["sat", "sun"]),
        dict(
            bounds=["sun", "mon"],
            pattern="hour",
            values=list(range(0, 21)),
        ),
    ]

    fig.update_layout(
        height=1100,
        margin=dict(t=120),
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(
            text=(
                f"<b>{symbol} 1H — BbandsMR</b><br>"
                f"<span style='font-size:13px; color:#aaaaaa;'>"
                f"Trades: {n_trades}  |  TP: {n_tp}  |  SL: {n_sl}  |  "
                f"Open: {n_open}  |  Win rate (closed): {win_rate:.1f}%"
                f"</span><br>"
                f"<span style='font-size:12px; color:#888888;'>"
                f"BB({BB_PERIOD}, {BB_STD})  &nbsp;|&nbsp;  ADX &lt; {ADX_THRESHOLD}"
                f"  &nbsp;|&nbsp;  EMA{EMA_PERIOD} filter (dist &ge; {EMA_DIST_THRESHOLD:.1%})"
                f"  &nbsp;|&nbsp;  SL = {SL_ATR_MULT}x ATR"
                f"</span>"
            ),
            x=0.5,
            xanchor="center",
            y=0.98,
            yanchor="top",
        ),
    )

    fig.update_yaxes(title_text="Price (GBP)", row=1, col=1)
    fig.update_yaxes(title_text="ADX", row=2, col=1)
    fig.update_yaxes(title_text="ATR", row=3, col=1)
    fig.update_xaxes(title_text="Date (UTC)", row=3, col=1)

    for axis in ("xaxis", "xaxis2", "xaxis3"):
        fig.update_layout(**{axis: dict(rangebreaks=_rangebreaks)})

    return fig


# ---------------------------------------------------------------------------
# __main__
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import vectorbt as vbt

    vbt.settings.plotting["use_widgets"] = False

    print(f"Loading {SYMBOL} {TIMEFRAME} data ({START_DATE} to {END_DATE})...")
    df = load_ohlcv(SYMBOL, START_DATE, END_DATE, TIMEFRAME)
    print(f"  {len(df):,} bars  [{df.index.min()} -> {df.index.max()}]")

    print("Detecting setups...")
    sell_setups, buy_setups, extra = detect_setups(df)
    n_total = len(sell_setups) + len(buy_setups)
    print(f"  Long  setups : {len(buy_setups)}")
    print(f"  Short setups : {len(sell_setups)}")
    print(f"  Total        : {n_total}")

    if n_total > 0:
        closed = [s for s in sell_setups + buy_setups if s["exit_type"] != "open"]
        tp_hits = sum(1 for s in closed if s["exit_type"] == "tp")
        sl_hits = sum(1 for s in closed if s["exit_type"] == "sl")
        win_rate = tp_hits / len(closed) * 100 if closed else 0.0
        print(f"  TP hits      : {tp_hits}")
        print(f"  SL hits      : {sl_hits}")
        print(f"  Win rate     : {win_rate:.1f}%")

    print("Building chart...")
    fig = build_chart(df, sell_setups, buy_setups, extra=extra, symbol=SYMBOL)

    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"strategy_{SYMBOL}_{TIMEFRAME}.html"
    fig.write_html(str(out))
    print(f"  Chart saved -> {out}")
    fig.show()
