"""
Reusable Plotly chart builders for vectorbt backtest results.

Functions
---------
build_equity_chart(pf, symbol, account_size, leverage)
    Two-panel equity curve + drawdown with win/loss dots.

build_analytics_chart(pf, symbol, account_size, leverage)
    Eight-panel analytics dashboard:
      1. Equity curve + win/loss dots
      2. Drawdown %
      3. Net P&L by day of week
      4. Net P&L by entry hour
      5. Net P&L by month
      6. Trade P&L distribution (histogram)
      7. Cumulative P&L per trade (step line)
      8. Win / loss count by direction (Long vs Short)
"""

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def _reanchor_equity(pf, account_size: float, leverage: float):
    """
    Shift vectorbt equity from notional scale to real-margin scale.

    Returns (equity, drawdown_pct, abs_pnl).
    """
    equity_notional = pf.value()
    init_notional = account_size * leverage
    equity = equity_notional - init_notional + account_size
    dd = (equity - equity.cummax()) / equity.cummax() * 100
    abs_pnl = equity.iloc[-1] - account_size
    return equity, dd, abs_pnl


def build_equity_chart(
    pf, symbol: str, account_size: float, leverage: float
) -> go.Figure:
    """
    Two-panel chart: equity curve (real margin $) + drawdown %.

    Win/loss entry dots are plotted on the equity curve.

    Parameters
    ----------
    pf          : vectorbt Portfolio object
    symbol      : ticker string used in the title
    account_size: real margin capital in USD
    leverage    : leverage multiplier (e.g. 100 for 100:1)

    Returns
    -------
    go.Figure
    """
    equity, dd, abs_pnl = _reanchor_equity(pf, account_size, leverage)
    trades_df = pf.trades.records_readable.copy()

    eq_range = equity.max() - equity.min()
    pad = max(eq_range * 0.10, 200)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=(f"{symbol} — Equity Curve", "Drawdown %"),
    )

    # equity line
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity,
            name="Equity",
            line=dict(color="#26a69a", width=2),
            fill="tozeroy",
            fillcolor="rgba(38,166,154,0.08)",
        ),
        row=1,
        col=1,
    )

    # win / loss dots
    if len(trades_df):
        entry_times = pd.to_datetime(trades_df["Entry Timestamp"].values)
        entry_equity = equity.reindex(entry_times, method="nearest").values
        win_mask = (trades_df["PnL"] > 0).values
        loss_mask = ~win_mask

        if win_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=entry_times[win_mask],
                    y=entry_equity[win_mask],
                    mode="markers",
                    name="Win",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color="#00e676",
                        line=dict(color="#1b5e20", width=1),
                    ),
                ),
                row=1,
                col=1,
            )
        if loss_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=entry_times[loss_mask],
                    y=entry_equity[loss_mask],
                    mode="markers",
                    name="Loss",
                    marker=dict(
                        symbol="circle",
                        size=8,
                        color="#f44336",
                        line=dict(color="#b71c1c", width=1),
                    ),
                ),
                row=1,
                col=1,
            )

    # drawdown
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            name="Drawdown",
            line=dict(color="#ef5350", width=1),
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.15)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    rom = (abs_pnl / account_size) * 100
    total_trades = pf.trades.count()
    win_rate = pf.trades.win_rate() * 100 if total_trades > 0 else 0.0

    fig.update_layout(
        height=700,
        template="plotly_dark",
        hovermode="x unified",
        title=dict(
            text=(
                f"<b>{symbol} — Equity Curve</b><br>"
                f"<sup>Return on Margin: {rom:+.2f}%  |  "
                f"Sharpe: {pf.sharpe_ratio():.2f}  |  "
                f"Win Rate: {win_rate:.1f}%  |  "
                f"Trades: {total_trades}  |  "
                f"Max DD: {dd.min():.2f}%</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
    )
    fig.update_yaxes(
        title_text="Portfolio Value ($)",
        row=1,
        col=1,
        range=[equity.min() - pad, equity.max() + pad],
    )
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def build_analytics_chart(
    pf, symbol: str, account_size: float, leverage: float
) -> go.Figure:
    """
    Eight-panel analytics dashboard in a single HTML figure.

    Panels
    ------
    1. Equity curve (real margin $) + win/loss entry dots
    2. Drawdown %
    3. Net P&L by day of week (Mon–Sun bar chart)
    4. Net P&L by entry hour (bar chart)
    5. Net P&L by month (bar chart)
    6. Trade P&L distribution (histogram)
    7. Cumulative P&L per trade (step line)
    8. Win / loss count by direction — Long vs Short (grouped bar)

    Parameters
    ----------
    pf          : vectorbt Portfolio object
    symbol      : ticker string used in the title
    account_size: real margin capital in USD
    leverage    : leverage multiplier (e.g. 100 for 100:1)

    Returns
    -------
    go.Figure
    """
    trades_df = pf.trades.records_readable.copy()
    equity, dd, abs_pnl = _reanchor_equity(pf, account_size, leverage)

    eq_range = equity.max() - equity.min()
    pad = max(eq_range * 0.10, 200)
    y_min = equity.min() - pad
    y_max = equity.max() + pad

    DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    MONTH_NAMES = [
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    has_trades = len(trades_df) > 0

    if has_trades:
        entry_dt = pd.to_datetime(trades_df["Entry Timestamp"])
        trades_df = trades_df.copy()
        trades_df["dow"] = entry_dt.dt.dayofweek  # 0 = Mon
        trades_df["hour"] = entry_dt.dt.hour
        trades_df["month"] = entry_dt.dt.month
        trades_df["direction"] = trades_df["Direction"].str.strip()

        dow_pnl = trades_df.groupby("dow")["PnL"].sum().reindex(range(7), fill_value=0)
        hour_pnl = trades_df.groupby("hour")["PnL"].sum().sort_index()
        month_pnl = (
            trades_df.groupby("month")["PnL"].sum().reindex(range(1, 13), fill_value=0)
        )
        cum_pnl = trades_df["PnL"].cumsum().reset_index(drop=True)

        dir_counts = (
            trades_df.assign(
                outcome=trades_df["PnL"].gt(0).map({True: "Win", False: "Loss"})
            )
            .groupby(["direction", "outcome"])
            .size()
            .unstack(fill_value=0)
        )

    subplot_titles = (
        f"{symbol} — Equity Curve",
        "Drawdown %",
        "Net P&L by Day of Week",
        "Net P&L by Entry Hour",
        "Net P&L by Month",
        "Trade P&L Distribution",
        "Cumulative P&L per Trade",
        "Win / Loss Count by Direction",
    )

    fig = make_subplots(
        rows=8,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.045,
        row_heights=[0.18, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        subplot_titles=subplot_titles,
    )

    # ── Panel 1: equity curve ─────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity,
            name="Equity",
            line=dict(color="#26a69a", width=2),
            fill="tozeroy",
            fillcolor="rgba(38,166,154,0.08)",
        ),
        row=1,
        col=1,
    )

    if has_trades:
        entry_times = pd.to_datetime(trades_df["Entry Timestamp"].values)
        entry_equity = equity.reindex(entry_times, method="nearest").values
        win_mask = (trades_df["PnL"] > 0).values
        loss_mask = ~win_mask

        if win_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=entry_times[win_mask],
                    y=entry_equity[win_mask],
                    mode="markers",
                    name="Win",
                    marker=dict(
                        symbol="circle",
                        size=7,
                        color="#00e676",
                        line=dict(color="#1b5e20", width=1),
                    ),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )
        if loss_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=entry_times[loss_mask],
                    y=entry_equity[loss_mask],
                    mode="markers",
                    name="Loss",
                    marker=dict(
                        symbol="circle",
                        size=7,
                        color="#f44336",
                        line=dict(color="#b71c1c", width=1),
                    ),
                    showlegend=True,
                ),
                row=1,
                col=1,
            )

    # ── Panel 2: drawdown ─────────────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            name="Drawdown %",
            line=dict(color="#ef5350", width=1),
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.15)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # ── Panel 3: P&L by day of week ───────────────────────────────────────────
    if has_trades:
        dow_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in dow_pnl.values]
        fig.add_trace(
            go.Bar(
                x=[DAY_NAMES[i] for i in dow_pnl.index],
                y=dow_pnl.values,
                name="P&L by DoW",
                marker_color=dow_colors,
                showlegend=False,
            ),
            row=3,
            col=1,
        )

    # ── Panel 4: P&L by entry hour ────────────────────────────────────────────
    if has_trades:
        hour_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in hour_pnl.values]
        fig.add_trace(
            go.Bar(
                x=[f"{h:02d}:00" for h in hour_pnl.index],
                y=hour_pnl.values,
                name="P&L by Hour",
                marker_color=hour_colors,
                showlegend=False,
            ),
            row=4,
            col=1,
        )

    # ── Panel 5: P&L by month ─────────────────────────────────────────────────
    if has_trades:
        month_colors = ["#26a69a" if v >= 0 else "#ef5350" for v in month_pnl.values]
        fig.add_trace(
            go.Bar(
                x=[MONTH_NAMES[m - 1] for m in month_pnl.index],
                y=month_pnl.values,
                name="P&L by Month",
                marker_color=month_colors,
                showlegend=False,
            ),
            row=5,
            col=1,
        )

    # ── Panel 6: P&L distribution histogram ──────────────────────────────────
    if has_trades:
        fig.add_trace(
            go.Histogram(
                x=trades_df["PnL"],
                nbinsx=30,
                name="P&L Dist",
                marker_color="#5c6bc0",
                opacity=0.85,
                showlegend=False,
            ),
            row=6,
            col=1,
        )
        # zero reference line
        fig.add_trace(
            go.Scatter(
                x=[0, 0],
                y=[0, max(trades_df["PnL"].value_counts().max(), 1)],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.4)", width=1, dash="dot"),
                showlegend=False,
                name="zero",
            ),
            row=6,
            col=1,
        )

    # ── Panel 7: cumulative P&L per trade ─────────────────────────────────────
    if has_trades:
        cum_color = "#26a69a" if cum_pnl.iloc[-1] >= 0 else "#ef5350"
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(cum_pnl) + 1)),
                y=cum_pnl.values,
                mode="lines+markers",
                name="Cum P&L",
                line=dict(color=cum_color, width=2, shape="hv"),
                marker=dict(size=5),
                showlegend=False,
            ),
            row=7,
            col=1,
        )
        # breakeven reference
        fig.add_trace(
            go.Scatter(
                x=[1, len(cum_pnl)],
                y=[0, 0],
                mode="lines",
                line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dot"),
                showlegend=False,
                name="breakeven",
            ),
            row=7,
            col=1,
        )

    # ── Panel 8: win / loss count by direction ────────────────────────────────
    if has_trades and len(dir_counts) > 0:
        directions = dir_counts.index.tolist()
        for outcome, color in [("Win", "#00e676"), ("Loss", "#f44336")]:
            if outcome in dir_counts.columns:
                fig.add_trace(
                    go.Bar(
                        name=outcome,
                        x=directions,
                        y=dir_counts[outcome].values,
                        marker_color=color,
                        showlegend=True,
                    ),
                    row=8,
                    col=1,
                )

    # ── global layout ─────────────────────────────────────────────────────────
    rom = (abs_pnl / account_size) * 100
    total_trades = pf.trades.count()
    win_rate = pf.trades.win_rate() * 100 if total_trades > 0 else 0.0

    fig.update_layout(
        height=2200,
        template="plotly_dark",
        hovermode="x unified",
        barmode="group",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1),
        title=dict(
            text=(
                f"<b>{symbol} — Strategy Analytics Dashboard</b><br>"
                f"<sup>"
                f"Return on Margin: {rom:+.2f}%  |  "
                f"Abs P&L: ${abs_pnl:+,.2f}  |  "
                f"Sharpe: {pf.sharpe_ratio():.2f}  |  "
                f"Sortino: {pf.sortino_ratio():.2f}  |  "
                f"Win Rate: {win_rate:.1f}%  |  "
                f"Trades: {total_trades}  |  "
                f"Max DD: {dd.min():.2f}%  |  "
                f"Profit Factor: {pf.trades.profit_factor():.2f}"
                f"</sup>"
            ),
            x=0.5,
            xanchor="center",
            y=0.99,
            yanchor="top",
        ),
        margin=dict(t=120, b=40, l=60, r=40),
    )

    fig.update_yaxes(title_text="Portfolio ($)", row=1, col=1, range=[y_min, y_max])
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=3, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=4, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=5, col=1)
    fig.update_yaxes(title_text="Count", row=6, col=1)
    fig.update_yaxes(title_text="Cum P&L ($)", row=7, col=1)
    fig.update_yaxes(title_text="Trades", row=8, col=1)
    fig.update_xaxes(title_text="Trade #", row=7, col=1)

    return fig
