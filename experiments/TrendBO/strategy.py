"""
TrendBO — Strategy Signal Detection + Visualisation

Intraday trend-following breakout strategy on USDJPY 1H:

  1. EMA trend filter   : 50 EMA vs 200 EMA establishes macro direction
                         (bullish = EMA50 > EMA200, bearish = EMA50 < EMA200)
  2. Consolidation      : price is ranging when ADX < ADX_THRESHOLD for at least
                         CONSOL_MIN_BARS consecutive bars AND the 20-bar rolling
                         range is tight relative to ATR (range / ATR < RANGE_ATR_MULT)
  3. Breakout entry     : first close above rolling high (long) or below rolling low
                         (short) in the direction of the EMA trend, with ATR expanding
  4. SL                 : opposite side of the consolidation range
  5. TP                 : RISK_REWARD * SL distance

Conforms to STRATEGY_BACKTEST_PATTERN.md:
  - load_ohlcv(symbol, start, end, tf) -> pd.DataFrame  (tz-naive, lowercase cols)
  - detect_setups(df, ...) -> (sell_setups, buy_setups, extra)
  - Each setup dict contains at minimum: entry_ts, entry, sl, tp

Usage:
    uv run python experiments/TrendBO/strategy.py
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
SYMBOL = "USDJPY"
START_DATE = "2025-01-01"
END_DATE = "2026-03-05"
LOWER_TF = "1h"

# Trend filter
EMA_FAST = 50
EMA_SLOW = 200

# Volatility / range
ATR_PERIOD = 14
ADX_PERIOD = 14
RANGE_PERIOD = 20  # rolling lookback for high/low consolidation zone

# Consolidation detection
CONSOL_MIN_BARS = 5  # consecutive bars below ADX threshold to confirm ranging
ADX_THRESHOLD = 25  # ADX below this = market is ranging
RANGE_ATR_MULT = 4.0  # range must be < RANGE_ATR_MULT * ATR to be "compressed"
# The 20-bar rolling range on 1H data is naturally 3-5x the 14-bar ATR.
# < 4.0 puts us in the bottom ~29% of range/ATR — genuinely compressed.
MIN_CONSOLIDATION_PIPS = 10  # minimum consolidation width (pips) to avoid dead zones
ATR_EXPAND_MULT = 1.0  # breakout bar ATR must be >= ATR_EXPAND_MULT * prior ATR

# Trade parameters
RISK_REWARD = 2.0
MIN_SL_PIPS = 10  # SL distance floor for position sizing
SESSION_START_HOUR = 7  # UTC — London open
SESSION_END_HOUR = 20  # UTC — NY close

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
      ema_fast    — EMA(close, EMA_FAST)       via talib.EMA
      ema_slow    — EMA(close, EMA_SLOW)        via talib.EMA
      atr         — ATR(high, low, close, ATR_PERIOD)  via talib.ATR
      adx         — ADX(high, low, close, ADX_PERIOD)  via talib.ADX
      roll_high   — rolling max of high over RANGE_PERIOD bars
      roll_low    — rolling min of low  over RANGE_PERIOD bars
      roll_range  — roll_high - roll_low
    """
    df = df.copy()

    # ta-lib requires plain numpy float64 arrays
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    df["ema_fast"] = talib.EMA(close, timeperiod=EMA_FAST)
    df["ema_slow"] = talib.EMA(close, timeperiod=EMA_SLOW)
    df["atr"] = talib.ATR(high, low, close, timeperiod=ATR_PERIOD)
    df["adx"] = talib.ADX(high, low, close, timeperiod=ADX_PERIOD)

    # Rolling range — pandas rolling is fine here (no ta-lib equivalent needed)
    df["roll_high"] = df["high"].rolling(RANGE_PERIOD).max()
    df["roll_low"] = df["low"].rolling(RANGE_PERIOD).min()
    df["roll_range"] = df["roll_high"] - df["roll_low"]

    return df


# ---------------------------------------------------------------------------
# Consolidation episode tracker
# ---------------------------------------------------------------------------


def _find_consolidation_episodes(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series of integer episode IDs (0 = not in consolidation).
    A new episode starts after CONSOL_MIN_BARS consecutive bars with:
      - ADX < ADX_THRESHOLD
      - roll_range < RANGE_ATR_MULT * ATR
      - roll_range >= MIN_CONSOLIDATION_PIPS * PIP_SIZE
    Episode ends when any condition breaks.
    """
    adx = df["adx"].values
    roll_range = df["roll_range"].values
    atr = df["atr"].values

    min_range_abs = MIN_CONSOLIDATION_PIPS * PIP_SIZE

    episodes = np.zeros(len(df), dtype=int)
    episode_id = 0
    consec = 0
    in_episode = False

    for i in range(len(df)):
        ranging = (
            not np.isnan(adx[i])
            and adx[i] < ADX_THRESHOLD
            and not np.isnan(roll_range[i])
            and not np.isnan(atr[i])
            and atr[i] > 0
            and roll_range[i] < RANGE_ATR_MULT * atr[i]
            and roll_range[i] >= min_range_abs
        )

        if ranging:
            consec += 1
            if consec >= CONSOL_MIN_BARS:
                if not in_episode:
                    episode_id += 1
                    in_episode = True
                episodes[i] = episode_id
        else:
            consec = 0
            if in_episode:
                in_episode = False

    return pd.Series(episodes, index=df.index, name="consol_episode")


# ---------------------------------------------------------------------------
# Signal detection
# ---------------------------------------------------------------------------


def detect_setups(
    df: pd.DataFrame,
    risk_reward: float = RISK_REWARD,
    ema_fast: int = EMA_FAST,
    ema_slow: int = EMA_SLOW,
    adx_threshold: float = ADX_THRESHOLD,
    range_atr_mult: float = RANGE_ATR_MULT,
    consol_min_bars: int = CONSOL_MIN_BARS,
    min_consolidation_pips: float = MIN_CONSOLIDATION_PIPS,
    min_sl_pips: float = MIN_SL_PIPS,
    session_start: int = SESSION_START_HOUR,
    session_end: int = SESSION_END_HOUR,
) -> tuple:
    """
    Detect EMA trend-aligned consolidation breakout setups.

    Parameters
    ----------
    df : pd.DataFrame
        1H OHLCV with tz-naive DatetimeIndex.

    Returns
    -------
    sell_setups : list of dicts
    buy_setups  : list of dicts
    extra       : dict — consolidation zone metadata for charting
    """
    df = _add_indicators(df.copy())
    df["consol_episode"] = _find_consolidation_episodes(df)

    sell_setups = []
    buy_setups = []
    consol_zones = []  # for charting: {episode_id, start_ts, end_ts, high, low}

    # Track which episodes have already produced a setup (one per episode)
    used_episodes: set = set()

    # Track zone boundaries per episode for charting
    episode_meta: dict = {}

    for i in range(EMA_SLOW + ADX_PERIOD + RANGE_PERIOD + 1, len(df)):
        ts = df.index[i]
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        episode = int(row["consol_episode"])

        # Update zone metadata for charting
        if episode > 0:
            if episode not in episode_meta:
                episode_meta[episode] = {
                    "start_ts": ts,
                    "end_ts": ts,
                    "high": row["roll_high"],
                    "low": row["roll_low"],
                    "episode_id": episode,
                }
            else:
                episode_meta[episode]["end_ts"] = ts
                episode_meta[episode]["high"] = max(
                    episode_meta[episode]["high"], row["roll_high"]
                )
                episode_meta[episode]["low"] = min(
                    episode_meta[episode]["low"], row["roll_low"]
                )

        # --- Breakout detection ---
        # Previous bar must have been in an active consolidation
        prev_episode = int(prev["consol_episode"])
        if prev_episode == 0:
            continue
        if prev_episode in used_episodes:
            continue

        # Session filter
        hour = ts.hour
        if not (session_start <= hour < session_end):
            continue

        close = row["close"]
        prev_roll_high = prev["roll_high"]
        prev_roll_low = prev["roll_low"]

        if pd.isna(prev_roll_high) or pd.isna(prev_roll_low):
            continue

        # ATR expansion check on breakout bar
        if pd.isna(row["atr"]) or pd.isna(prev["atr"]) or prev["atr"] <= 0:
            continue
        atr_expanding = row["atr"] >= ATR_EXPAND_MULT * prev["atr"]
        if not atr_expanding:
            continue

        # EMA trend direction
        ema_fast_val = row["ema_fast"]
        ema_slow_val = row["ema_slow"]
        if pd.isna(ema_fast_val) or pd.isna(ema_slow_val):
            continue

        bullish_trend = ema_fast_val > ema_slow_val
        bearish_trend = ema_fast_val < ema_slow_val

        # Long breakout
        if bullish_trend and close > prev_roll_high:
            entry = close
            sl_raw = prev_roll_low
            sl_dist = max(abs(entry - sl_raw), min_sl_pips * PIP_SIZE)
            sl = entry - sl_dist
            tp = entry + risk_reward * sl_dist

            if sl >= entry:
                continue

            setup = {
                "entry_ts": ts,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "consol_high": prev_roll_high,
                "consol_low": prev_roll_low,
                "direction": "long",
                "episode_id": prev_episode,
                "ema_fast": ema_fast_val,
                "ema_slow": ema_slow_val,
                "adx": row["adx"],
                "atr": row["atr"],
            }
            buy_setups.append(setup)
            used_episodes.add(prev_episode)
            continue

        # Short breakout
        if bearish_trend and close < prev_roll_low:
            entry = close
            sl_raw = prev_roll_high
            sl_dist = max(abs(sl_raw - entry), min_sl_pips * PIP_SIZE)
            sl = entry + sl_dist
            tp = entry - risk_reward * sl_dist

            if sl <= entry:
                continue

            setup = {
                "entry_ts": ts,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "consol_high": prev_roll_high,
                "consol_low": prev_roll_low,
                "direction": "short",
                "episode_id": prev_episode,
                "ema_fast": ema_fast_val,
                "ema_slow": ema_slow_val,
                "adx": row["adx"],
                "atr": row["atr"],
            }
            sell_setups.append(setup)
            used_episodes.add(prev_episode)

    consol_zones = list(episode_meta.values())

    extra = {
        "consol_zones": consol_zones,
        "df_indicators": df,
    }
    return sell_setups, buy_setups, extra


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
      Row 1 (55%): Candlesticks + EMAs + consolidation zones + entry/exit markers
      Row 2 (25%): ADX indicator with threshold line
      Row 3 (20%): ATR indicator
    """
    if extra is None:
        extra = {}

    df_ind = extra.get("df_indicators", df)
    consol_zones = extra.get("consol_zones", [])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.55, 0.25, 0.20],
        subplot_titles=(
            f"{symbol} 1H — TrendBO (EMA50/200 + Consolidation Breakout)",
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

    # --- EMA 50 ---
    if "ema_fast" in df_ind.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ind.index,
                y=df_ind["ema_fast"],
                name=f"EMA {EMA_FAST}",
                line=dict(color="#ffb300", width=1.5),
                opacity=0.85,
            ),
            row=1,
            col=1,
        )

    # --- EMA 200 ---
    if "ema_slow" in df_ind.columns:
        fig.add_trace(
            go.Scatter(
                x=df_ind.index,
                y=df_ind["ema_slow"],
                name=f"EMA {EMA_SLOW}",
                line=dict(color="#42a5f5", width=1.5),
                opacity=0.85,
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
        # Threshold line
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

    # --- Consolidation zone rectangles ---
    shapes = []
    for zone in consol_zones:
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=zone["start_ts"],
                x1=zone["end_ts"] + pd.Timedelta(hours=1),
                y0=zone["low"],
                y1=zone["high"],
                fillcolor="rgba(100, 100, 180, 0.10)",
                line=dict(color="rgba(150,150,220,0.30)", width=1),
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

    # --- Exit markers (simulated forward scan for charting) ---
    tp_x, tp_y = [], []
    sl_x, sl_y = [], []

    for setup, direction in all_setups:
        ts = setup["entry_ts"]
        if ts not in df.index:
            continue
        idx_pos = df.index.get_loc(ts)
        entry = setup["entry"]
        sl_px = setup["sl"]
        tp_px = setup["tp"]

        exit_ts = None
        exit_px = None
        exit_type = None

        for j in range(idx_pos + 1, len(df)):
            fh = df["high"].iloc[j]
            fl = df["low"].iloc[j]
            future_ts = df.index[j]

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

        if exit_ts is not None:
            color = "#ffeb3b" if exit_type == "tp" else "#ff5252"
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
            if exit_type == "tp":
                tp_x.append(exit_ts)
                tp_y.append(exit_px)
            else:
                sl_x.append(exit_ts)
                sl_y.append(exit_px)

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
    win_rate = (n_tp / n_trades * 100) if n_trades > 0 else 0.0

    # Hide weekend gaps and the FX Sunday open gap (Sat 00:00 – Sun 21:00 UTC).
    # Plotly rangebreaks removes these intervals from the x-axis so the chart
    # looks continuous with no blank stretches.
    _rangebreaks = [
        dict(bounds=["sat", "sun"]),  # full Saturday
        dict(
            bounds=["sun", "mon"],  # Sunday until 21:00 UTC (FX opens ~22:00)
            pattern="hour",
            values=list(range(0, 21)),
        ),
    ]

    fig.update_layout(
        height=1050,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=shapes,
        title=dict(
            text=(
                f"<b>{symbol} 1H — TrendBO</b><br>"
                f"<sup>Setups: {n_trades}  |  TP: {n_tp}  |  SL: {n_sl}  |  "
                f"Win rate: {win_rate:.1f}%  |  "
                f"<span style='color:#ffb300'>EMA{EMA_FAST}</span>  "
                f"<span style='color:#42a5f5'>EMA{EMA_SLOW}</span>  |  "
                f"Blue zones = consolidation</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
    )
    fig.update_yaxes(title_text="Price (JPY)", row=1, col=1)
    fig.update_yaxes(title_text="ADX", row=2, col=1)
    fig.update_yaxes(title_text="ATR", row=3, col=1)
    fig.update_xaxes(title_text="Date (UTC)", row=3, col=1)
    # Apply rangebreaks to all three shared x-axes
    for axis in ("xaxis", "xaxis2", "xaxis3"):
        fig.update_layout(**{axis: dict(rangebreaks=_rangebreaks)})

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
