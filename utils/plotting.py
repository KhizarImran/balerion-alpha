"""
Plotting utilities for visualizing strategies on candlestick charts.

Provides functions to create interactive candlestick charts with signal overlays
using plotly.
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List, Tuple


def plot_candlestick_with_signals(
    df: pd.DataFrame,
    title: str = "Strategy Visualization",
    buy_signals: Optional[pd.Series] = None,
    sell_signals: Optional[pd.Series] = None,
    indicators: Optional[dict] = None,
    show_volume: bool = True,
    height: int = 800,
) -> go.Figure:
    """
    Create an interactive candlestick chart with trading signals and indicators.

    Args:
        df: DataFrame with OHLCV data (timestamp index)
        title: Chart title
        buy_signals: Boolean series indicating buy signals
        sell_signals: Boolean series indicating sell signals
        indicators: Dictionary of indicator name -> series to plot
                   Example: {'SMA_20': sma_20_series, 'SMA_50': sma_50_series}
        show_volume: Whether to show volume subplot
        height: Chart height in pixels

    Returns:
        Plotly figure object

    Example:
        >>> df = load_data('EURUSD', start_date='2024-01-01')
        >>> df['SMA_20'] = df['close'].rolling(20).mean()
        >>> buy_signals = (df['close'] > df['SMA_20'])
        >>> fig = plot_candlestick_with_signals(
        ...     df,
        ...     title='EURUSD SMA Strategy',
        ...     buy_signals=buy_signals,
        ...     indicators={'SMA_20': df['SMA_20']}
        ... )
        >>> fig.show()
    """
    # Create subplots
    rows = 2 if show_volume else 1
    row_heights = [0.7, 0.3] if show_volume else [1.0]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=(title, "Volume") if show_volume else (title,),
    )

    # Add candlestick
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
        ),
        row=1,
        col=1,
    )

    # Add indicators
    if indicators:
        colors = ["blue", "orange", "purple", "brown", "pink", "gray"]
        for idx, (name, series) in enumerate(indicators.items()):
            color = colors[idx % len(colors)]
            fig.add_trace(
                go.Scatter(
                    x=series.index,
                    y=series,
                    name=name,
                    line=dict(color=color, width=1.5),
                    opacity=0.8,
                ),
                row=1,
                col=1,
            )

    # Add buy signals
    if buy_signals is not None:
        buy_points = df[buy_signals]
        if len(buy_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=buy_points.index,
                    y=buy_points["low"] * 0.999,  # Slightly below the low
                    mode="markers",
                    name="Buy Signal",
                    marker=dict(
                        symbol="triangle-up",
                        size=12,
                        color="green",
                        line=dict(color="darkgreen", width=1),
                    ),
                ),
                row=1,
                col=1,
            )

    # Add sell signals
    if sell_signals is not None:
        sell_points = df[sell_signals]
        if len(sell_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=sell_points.index,
                    y=sell_points["high"] * 1.001,  # Slightly above the high
                    mode="markers",
                    name="Sell Signal",
                    marker=dict(
                        symbol="triangle-down",
                        size=12,
                        color="red",
                        line=dict(color="darkred", width=1),
                    ),
                ),
                row=1,
                col=1,
            )

    # Add volume bars
    if show_volume and "volume" in df.columns:
        colors_volume = [
            "red" if close < open else "green"
            for close, open in zip(df["close"], df["open"])
        ]

        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df["volume"],
                name="Volume",
                marker_color=colors_volume,
                opacity=0.5,
            ),
            row=2,
            col=1,
        )

    # Update layout
    fig.update_layout(
        height=height,
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        template="plotly_dark",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    # Update axes
    fig.update_xaxes(title_text="Date", row=rows, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    if show_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


def plot_strategy_performance(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
    title: str = "Strategy Performance",
) -> go.Figure:
    """
    Plot equity curve and drawdown.

    Args:
        equity_curve: Series with equity values over time
        trades: DataFrame with trade information (optional)
        title: Chart title

    Returns:
        Plotly figure object
    """
    # Calculate drawdown
    running_max = equity_curve.expanding().max()
    drawdown = (equity_curve - running_max) / running_max * 100

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(title, "Drawdown %"),
        row_heights=[0.7, 0.3],
    )

    # Add equity curve
    fig.add_trace(
        go.Scatter(
            x=equity_curve.index,
            y=equity_curve,
            name="Equity",
            line=dict(color="#26a69a", width=2),
        ),
        row=1,
        col=1,
    )

    # Add drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown,
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#ef5350", width=1),
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        height=600, hovermode="x unified", template="plotly_dark", showlegend=True
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=2, col=1)

    return fig


def plot_multiple_indicators(
    df: pd.DataFrame, indicators: dict, title: str = "Technical Indicators"
) -> go.Figure:
    """
    Plot multiple indicators in separate subplots.

    Args:
        df: DataFrame with price data
        indicators: Dictionary of indicator_name -> series
        title: Chart title

    Returns:
        Plotly figure object
    """
    n_indicators = len(indicators)

    fig = make_subplots(
        rows=n_indicators + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        subplot_titles=[title] + list(indicators.keys()),
    )

    # Add price chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    # Add indicators
    for idx, (name, series) in enumerate(indicators.items(), start=2):
        fig.add_trace(
            go.Scatter(x=series.index, y=series, name=name, line=dict(width=2)),
            row=idx,
            col=1,
        )

    fig.update_layout(
        height=300 * (n_indicators + 1),
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
    )

    return fig


def save_plot(fig: go.Figure, filename: str, format: str = "html"):
    """
    Save plotly figure to file.

    Args:
        fig: Plotly figure object
        filename: Output filename
        format: Output format ('html', 'png', 'jpg', 'svg')
    """
    if format == "html":
        fig.write_html(filename)
    else:
        fig.write_image(filename, format=format)

    print(f"Plot saved to: {filename}")
