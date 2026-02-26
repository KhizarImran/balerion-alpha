"""
Simple Moving Average Crossover Strategy

This experiment tests a classic SMA crossover strategy:
- Buy signal: Fast SMA crosses above slow SMA
- Sell signal: Fast SMA crosses below slow SMA
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from utils import load_data, plot_candlestick_with_signals


def calculate_signals(
    df: pd.DataFrame, fast_period: int = 20, slow_period: int = 50
) -> pd.DataFrame:
    """
    Calculate SMA crossover signals.

    Args:
        df: DataFrame with OHLCV data
        fast_period: Fast SMA period
        slow_period: Slow SMA period

    Returns:
        DataFrame with signals and indicators
    """
    # Make a copy to avoid modifying original
    df = df.copy()

    # Calculate SMAs
    df["SMA_fast"] = df["close"].rolling(window=fast_period).mean()
    df["SMA_slow"] = df["close"].rolling(window=slow_period).mean()

    # Generate signals
    df["position"] = 0
    df.loc[df["SMA_fast"] > df["SMA_slow"], "position"] = 1  # Long
    df.loc[df["SMA_fast"] < df["SMA_slow"], "position"] = -1  # Short or exit

    # Identify crossover points (buy/sell signals)
    df["signal"] = df["position"].diff()

    df["buy_signal"] = df["signal"] == 2  # Change from -1 to 1 or 0 to 1
    df["sell_signal"] = df["signal"] == -2  # Change from 1 to -1

    # Alternative: only mark entry signals
    df["buy_signal"] = (df["SMA_fast"] > df["SMA_slow"]) & (
        df["SMA_fast"].shift(1) <= df["SMA_slow"].shift(1)
    )
    df["sell_signal"] = (df["SMA_fast"] < df["SMA_slow"]) & (
        df["SMA_fast"].shift(1) >= df["SMA_slow"].shift(1)
    )

    return df


def run_experiment(
    symbol: str = "EURUSD",
    asset_type: str = "fx",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    fast_period: int = 20,
    slow_period: int = 50,
    show_chart: bool = True,
    save_chart: bool = False,
):
    """
    Run the SMA crossover experiment.

    Args:
        symbol: Trading symbol
        asset_type: 'fx' or 'indices'
        start_date: Start date for analysis
        end_date: End date for analysis
        fast_period: Fast SMA period
        slow_period: Slow SMA period
        show_chart: Whether to display the chart
        save_chart: Whether to save the chart to file
    """
    print(f"Loading data for {symbol}...")
    df = load_data(
        symbol, asset_type=asset_type, start_date=start_date, end_date=end_date
    )

    # Resample to higher timeframe if needed (optional)
    # df = df.resample('1H').agg({
    #     'open': 'first',
    #     'high': 'max',
    #     'low': 'min',
    #     'close': 'last',
    #     'volume': 'sum'
    # }).dropna()

    print(f"Data loaded: {len(df)} rows from {df.index.min()} to {df.index.max()}")

    print(
        f"\nCalculating SMA crossover signals (fast={fast_period}, slow={slow_period})..."
    )
    df = calculate_signals(df, fast_period, slow_period)

    # Count signals
    n_buys = df["buy_signal"].sum()
    n_sells = df["sell_signal"].sum()
    print(f"Buy signals: {n_buys}")
    print(f"Sell signals: {n_sells}")

    # Calculate basic statistics
    print("\n--- Signal Statistics ---")
    if n_buys > 0:
        print(f"Total signals: {n_buys + n_sells}")
        print(f"Average bars between signals: {len(df) / (n_buys + n_sells):.1f}")

    # Plot the strategy
    print("\nGenerating visualization...")
    fig = plot_candlestick_with_signals(
        df,
        title=f"{symbol} - SMA Crossover (Fast={fast_period}, Slow={slow_period})",
        buy_signals=df["buy_signal"],
        sell_signals=df["sell_signal"],
        indicators={
            f"SMA {fast_period}": df["SMA_fast"],
            f"SMA {slow_period}": df["SMA_slow"],
        },
        show_volume=True,
    )

    if save_chart:
        # Save to reports directory
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_file = (
            reports_dir / f"sma_crossover_{symbol}_{fast_period}_{slow_period}.html"
        )
        fig.write_html(str(output_file))
        print(f"Chart saved to: {output_file}")

    if show_chart:
        fig.show()

    return df, fig


if __name__ == "__main__":
    # Run the experiment with default parameters
    df, fig = run_experiment(
        symbol="EURUSD",
        asset_type="fx",
        start_date="2025-11-18",
        end_date="2026-02-25",
        fast_period=20,
        slow_period=50,
        show_chart=True,
        save_chart=True,
    )

    # You can also run parameter sweep
    # for fast in [10, 20, 30]:
    #     for slow in [40, 50, 60]:
    #         if slow > fast:
    #             print(f"\n{'='*60}")
    #             print(f"Testing: Fast={fast}, Slow={slow}")
    #             print(f"{'='*60}")
    #             run_experiment(fast_period=fast, slow_period=slow, show_chart=False)
