"""
ICT Unicorn Model — Strategy Visualization

Loads EURUSD 1-minute data from balerion-data, resamples to 1H, detects
Unicorn setup components (FVG + liquidity sweeps), and plots an interactive
candlestick chart saved to experiments/Unicorn-ICT/reports/.

Setup components detected and visualised:
  - Bullish Fair Value Gaps  (shaded green zones)
  - Bearish Fair Value Gaps  (shaded red zones)
  - Swing High / Swing Low liquidity levels (horizontal lines)
  - Placeholder buy/sell signals at FVG midpoints (future: add Breaker Block filter)
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import DataLoader


# ---------------------------------------------------------------------------
# Signal / indicator calculations
# ---------------------------------------------------------------------------


def calculate_fvg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect Fair Value Gaps (FVGs) on the DataFrame.

    A Bullish FVG exists when:
        candle[i-2].high < candle[i].low   (gap between candle 1 and candle 3)

    A Bearish FVG exists when:
        candle[i-2].low > candle[i].high

    Adds columns:
        bull_fvg      : bool — bullish FVG on this bar (the middle candle)
        bear_fvg      : bool — bearish FVG on this bar
        bull_fvg_top  : upper edge of bullish FVG
        bull_fvg_bot  : lower edge of bullish FVG
        bear_fvg_top  : upper edge of bearish FVG
        bear_fvg_bot  : lower edge of bearish FVG
    """
    df = df.copy()

    high = df["high"]
    low = df["low"]

    # Bullish FVG: prior candle's high < next candle's low
    bull_fvg = (high.shift(2) < low) & (low.shift(2).notna())
    # Bearish FVG: prior candle's low > next candle's high
    bear_fvg = (low.shift(2) > high) & (high.shift(2).notna())

    df["bull_fvg"] = bull_fvg.fillna(False)
    df["bear_fvg"] = bear_fvg.fillna(False)

    # FVG boundaries — gap is between candle[i-2].high and candle[i].low (bullish)
    df["bull_fvg_top"] = np.where(df["bull_fvg"], low, np.nan)
    df["bull_fvg_bot"] = np.where(df["bull_fvg"], high.shift(2), np.nan)
    df["bear_fvg_top"] = np.where(df["bear_fvg"], low.shift(2), np.nan)
    df["bear_fvg_bot"] = np.where(df["bear_fvg"], high, np.nan)

    return df


def calculate_swing_levels(df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
    """
    Identify swing highs and swing lows as liquidity levels.

    A swing high is a bar whose high is the highest over
    [i - lookback : i + lookback].  Swing low is the mirror.

    Adds columns:
        swing_high : float (NaN where not a swing high)
        swing_low  : float (NaN where not a swing low)
    """
    df = df.copy()
    n = lookback

    rolling_max = df["high"].rolling(window=2 * n + 1, center=True).max()
    rolling_min = df["low"].rolling(window=2 * n + 1, center=True).min()

    df["swing_high"] = np.where(df["high"] == rolling_max, df["high"], np.nan)
    df["swing_low"] = np.where(df["low"] == rolling_min, df["low"], np.nan)

    return df


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine FVG detection and swing level detection.

    Placeholder entry signals:
      - buy_signal  : bullish FVG detected (price returned to gap area implied)
      - sell_signal : bearish FVG detected

    In a full Unicorn implementation these would require a prior liquidity sweep
    + Breaker Block confluence check — that logic is documented in tradinglogic.md
    and will be layered in the backtest.py.
    """
    df = calculate_fvg(df)
    df = calculate_swing_levels(df)

    # Simple placeholder signals: mark FVG formation bars
    df["buy_signal"] = df["bull_fvg"]
    df["sell_signal"] = df["bear_fvg"]

    return df


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def build_fvg_shapes(df: pd.DataFrame, max_fvgs: int = 30) -> list:
    """
    Build Plotly shape rectangles for the most recent FVG zones.

    Each FVG is drawn as a semi-transparent rectangle spanning from the
    formation bar to the next 20 bars (approximate forward projection).
    """
    shapes = []
    bar_duration = pd.Timedelta(hours=1)
    projection = bar_duration * 20

    # Bullish FVGs
    bull_bars = df[df["bull_fvg"]].tail(max_fvgs)
    for ts, row in bull_bars.iterrows():
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=ts,
                x1=ts + projection,
                y0=row["bull_fvg_bot"],
                y1=row["bull_fvg_top"],
                fillcolor="rgba(38, 166, 154, 0.18)",
                line=dict(width=0),
                layer="below",
            )
        )

    # Bearish FVGs
    bear_bars = df[df["bear_fvg"]].tail(max_fvgs)
    for ts, row in bear_bars.iterrows():
        shapes.append(
            dict(
                type="rect",
                xref="x",
                yref="y",
                x0=ts,
                x1=ts + projection,
                y0=row["bear_fvg_bot"],
                y1=row["bear_fvg_top"],
                fillcolor="rgba(239, 83, 80, 0.18)",
                line=dict(width=0),
                layer="below",
            )
        )

    return shapes


def plot_unicorn_chart(df: pd.DataFrame, symbol: str = "EURUSD") -> go.Figure:
    """
    Build the full Unicorn strategy candlestick chart with:
      - OHLC candlesticks
      - FVG zones (shaded rectangles)
      - Swing High / Swing Low markers
      - Buy / Sell signal markers
      - Volume subplot
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=(f"{symbol} 1H — ICT Unicorn Setup Components", "Volume"),
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
            decreasing_line_color="#ef5350",
            increasing_fillcolor="#26a69a",
            decreasing_fillcolor="#ef5350",
        ),
        row=1,
        col=1,
    )

    # --- Swing Highs ---
    swing_highs = df[df["swing_high"].notna()]
    if len(swing_highs) > 0:
        fig.add_trace(
            go.Scatter(
                x=swing_highs.index,
                y=swing_highs["swing_high"] * 1.0003,
                mode="markers",
                name="Swing High (Liquidity)",
                marker=dict(
                    symbol="triangle-down",
                    size=8,
                    color="rgba(239, 83, 80, 0.7)",
                    line=dict(color="#ef5350", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    # --- Swing Lows ---
    swing_lows = df[df["swing_low"].notna()]
    if len(swing_lows) > 0:
        fig.add_trace(
            go.Scatter(
                x=swing_lows.index,
                y=swing_lows["swing_low"] * 0.9997,
                mode="markers",
                name="Swing Low (Liquidity)",
                marker=dict(
                    symbol="triangle-up",
                    size=8,
                    color="rgba(38, 166, 154, 0.7)",
                    line=dict(color="#26a69a", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    # --- FVG signal markers (placeholder buy/sell) ---
    bull_fvg_bars = df[df["buy_signal"]]
    if len(bull_fvg_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=bull_fvg_bars.index,
                y=bull_fvg_bars["low"] * 0.9994,
                mode="markers",
                name="Bullish FVG",
                marker=dict(
                    symbol="diamond",
                    size=9,
                    color="#00e5ff",
                    line=dict(color="#0288d1", width=1),
                ),
            ),
            row=1,
            col=1,
        )

    bear_fvg_bars = df[df["sell_signal"]]
    if len(bear_fvg_bars) > 0:
        fig.add_trace(
            go.Scatter(
                x=bear_fvg_bars.index,
                y=bear_fvg_bars["high"] * 1.0006,
                mode="markers",
                name="Bearish FVG",
                marker=dict(
                    symbol="diamond",
                    size=9,
                    color="#ff9100",
                    line=dict(color="#e65100", width=1),
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

    # --- FVG zone shapes ---
    shapes = build_fvg_shapes(df)

    # --- Layout ---
    fig.update_layout(
        height=900,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=shapes,
        title=dict(
            text=(
                f"<b>{symbol} 1H — ICT Unicorn Model</b><br>"
                "<sup>Green zones = Bullish FVG | Red zones = Bearish FVG | "
                "Triangles = Swing Liquidity Levels</sup>"
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
    show_chart: bool = True,
    save_chart: bool = True,
) -> tuple[pd.DataFrame, go.Figure]:
    """
    Load data, calculate Unicorn components, and generate chart.

    Args:
        symbol     : Trading symbol
        asset_type : 'fx' or 'indices'
        start_date : Start date (YYYY-MM-DD)
        end_date   : End date (YYYY-MM-DD)
        show_chart : Open chart in browser
        save_chart : Save HTML to experiments/Unicorn-ICT/reports/
    """
    print(f"Loading {symbol} 1-minute data ({start_date} to {end_date})...")
    loader = DataLoader()
    df_1m = loader.load_fx(symbol, start_date=start_date, end_date=end_date)
    print(f"  Loaded {len(df_1m):,} 1-minute bars.")

    print("Resampling to 1H...")
    df = loader.resample_ohlcv(df_1m, "1h")
    print(
        f"  Resampled to {len(df):,} hourly bars "
        f"({df.index.min()} -> {df.index.max()})."
    )

    print("Calculating ICT Unicorn components...")
    df = calculate_signals(df)

    n_bull = df["bull_fvg"].sum()
    n_bear = df["bear_fvg"].sum()
    n_sh = df["swing_high"].notna().sum()
    n_sl = df["swing_low"].notna().sum()
    print(f"  Bullish FVGs   : {n_bull}")
    print(f"  Bearish FVGs   : {n_bear}")
    print(f"  Swing Highs    : {n_sh}")
    print(f"  Swing Lows     : {n_sl}")

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
