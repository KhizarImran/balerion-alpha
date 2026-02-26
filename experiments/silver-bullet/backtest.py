"""
ICT Silver Bullet Strategy — VectorBT Backtest + Combined HTML Report

Generates a single scrollable HTML file containing:
  - 6 VectorBT performance charts (3x2 grid)
  - Full statistics table with key metrics
  - Trades table

Run:
    uv run python experiments/silver-bullet/backtest.py
"""

import sys
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import vectorbt as vbt
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import load_data
from strategy import calculate_signals

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "US30"
ASSET_TYPE = "indices"
START_DATE = "2025-11-11"
END_DATE = "2026-02-25"
INIT_CASH = 100_000
FEES = 0.0001  # 0.01% — representative for CFD/index spread
FREQ = "1min"

# Position sizing with leverage
# -----------------------------------------------------------------------
# US30 is priced at ~47,000 points. Without leverage, 1 unit costs ~$47k,
# so a $100k account can only hold ~2 units — not realistic for CFD trading.
#
# Solution: simulate leverage by passing init_cash = real_margin * LEVERAGE
# to vectorbt. This gives vbt enough "notional" buying power to fill the
# desired lot size. All absolute P&L figures are correct; the return%
# reported by vbt is based on the inflated notional — use RETURN ON MARGIN
# (abs_pnl / INIT_CASH) for the real account return.
#
# 10 lots at 100:1 leverage on a $100k margin account:
#   Margin required per lot = ~$47,000 / 100 = $470
#   Total margin for 10 lots = $4,700  (well within $100k)
#   P&L per point = $10  (10 lots × $1/point/lot)
# -----------------------------------------------------------------------
LOT_SIZE = 10  # number of units (lots) per trade  ($1/pt per lot)
SIZE_TYPE = "amount"
LEVERAGE = 100  # 100:1 — standard retail CFD leverage for indices

# Risk management — optimised parameters (from optimize_phase2.py)
RR_RATIO = 4.5  # Take profit = RR_RATIO × SL distance
SL_BUFFER = 0.0  # Extra points below liquidity low for stop loss
SESSION_CAP = -5_500.0  # Max cumulative estimated loss per session per day ($)

# Signal parameters — optimised
LIQUIDITY_WINDOW = 150  # Rolling window for significant high/low levels
SWEEP_LOOKBACK = 210  # Bars to look back for sweep detection
PULLBACK_WINDOW = 10  # Bars FVG zone stays alive for re-entry

# ---------------------------------------------------------------------------
# Load data & calculate signals
# ---------------------------------------------------------------------------

print("=" * 70)
print("SILVER BULLET STRATEGY — VECTORBT BACKTEST")
print("=" * 70)

print(f"\nLoading {SYMBOL} data ({START_DATE} to {END_DATE})...")
df = load_data(SYMBOL, asset_type=ASSET_TYPE, start_date=START_DATE, end_date=END_DATE)
print(f"  {len(df):,} rows  [{df.index.min()} -> {df.index.max()}]")

print("Calculating signals...")
df = calculate_signals(
    df,
    rr_ratio=RR_RATIO,
    sl_buffer_pts=SL_BUFFER,
    session_cap_usd=SESSION_CAP,
    lot_size=LOT_SIZE,
    liquidity_window=LIQUIDITY_WINDOW,
    sweep_lookback=SWEEP_LOOKBACK,
    pullback_window=PULLBACK_WINDOW,
)

entries = df["buy_signal_daily"]
exits = df["sell_signal"]

print(f"  Entry signals : {int(entries.sum())}")
print(f"  Exit signals  : {int(exits.sum())}")

close = df["close"]

# ---------------------------------------------------------------------------
# Run backtest
# ---------------------------------------------------------------------------

print("\nRunning backtest...")
pf = vbt.Portfolio.from_signals(
    close=close,
    entries=entries,
    exits=exits,
    init_cash=INIT_CASH * LEVERAGE,  # notional buying power = margin * leverage
    size=LOT_SIZE,
    size_type=SIZE_TYPE,
    sl_stop=df["sl_stop_frac"],  # per-trade SL fraction (fraction below entry price)
    tp_stop=df["tp_stop_frac"],  # per-trade TP fraction (fraction above entry price)
    fees=FEES,
    freq=FREQ,
)

stats = pf.stats()
trades = pf.trades.records_readable

# Return on margin = abs P&L / real margin capital (not inflated notional)
abs_pnl = pf.value().iloc[-1] - (INIT_CASH * LEVERAGE)
return_on_margin = abs_pnl / INIT_CASH * 100

print(f"  Trades completed   : {len(trades)}")
print(f"  Abs P&L            : ${abs_pnl:,.2f}")
print(
    f"  Return on Margin   : {return_on_margin:+.2f}%  (on ${INIT_CASH:,} at {LEVERAGE}:1)"
)
print(f"  Win Rate           : {stats['Win Rate [%]']:.1f}%")
print(f"  Sharpe Ratio       : {stats['Sharpe Ratio']:.2f}")
print(f"  Max Drawdown       : {stats['Max Drawdown [%]']:.2f}%")

# ---------------------------------------------------------------------------
# Build combined 6-chart Plotly figure
# (built from portfolio data series directly — avoids FigureWidget / anywidget)
# ---------------------------------------------------------------------------

print("\nBuilding combined chart...")

# Derive time series from portfolio
# Re-anchor equity curve to real margin capital (INIT_CASH) rather than
# the inflated notional (INIT_CASH * LEVERAGE) used inside vbt.
notional_value = pf.value()  # vbt notional equity
portfolio_value = (
    notional_value - (INIT_CASH * LEVERAGE) + INIT_CASH
)  # real margin equity
cum_returns = (portfolio_value / portfolio_value.iloc[0] - 1) * 100  # %
rolling_max = portfolio_value.cummax()
drawdown_pct = (portfolio_value - rolling_max) / rolling_max * 100  # % (negative)

# Per-trade data
trade_pnl = trades["PnL"] if "PnL" in trades.columns else pd.Series(dtype=float)
trade_times = (
    trades["Entry Timestamp"]
    if "Entry Timestamp" in trades.columns
    else (
        trades["Entry Time"] if "Entry Time" in trades.columns else pd.Series(dtype=str)
    )
)
trade_returns = (
    trades["Return [%]"]
    if "Return [%]" in trades.columns
    else (trades["Return"] if "Return" in trades.columns else pd.Series(dtype=float))
)

# Net exposure (1 when in a position, 0 otherwise)
net_exposure = pf.net_exposure()

fig = make_subplots(
    rows=3,
    cols=2,
    subplot_titles=(
        "Portfolio Value Over Time",
        "Cumulative Returns (%)",
        "Drawdown (%)",
        "Underwater Equity Curve (%)",
        "Trade Return Distribution (%)",
        "Net Exposure Over Time",
    ),
    vertical_spacing=0.10,
    horizontal_spacing=0.08,
    specs=[
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
    ],
)

print("  Adding Portfolio Value...")
fig.add_trace(
    go.Scatter(
        x=portfolio_value.index.tolist(),
        y=portfolio_value.tolist(),
        mode="lines",
        name="Portfolio Value",
        line=dict(color="#667eea", width=2),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=portfolio_value.index.tolist(),
        y=[INIT_CASH] * len(portfolio_value),
        mode="lines",
        name="Initial Cash",
        line=dict(color="#aaa", width=1, dash="dash"),
    ),
    row=1,
    col=1,
)

print("  Adding Cumulative Returns...")
fig.add_trace(
    go.Scatter(
        x=portfolio_value.index.tolist(),
        y=cum_returns.tolist(),
        mode="lines",
        name="Cum. Return",
        line=dict(color="#27ae60", width=2),
        fill="tozeroy",
        fillcolor="rgba(39,174,96,0.1)",
    ),
    row=1,
    col=2,
)

print("  Adding Drawdown...")
fig.add_trace(
    go.Scatter(
        x=drawdown_pct.index.tolist(),
        y=drawdown_pct.tolist(),
        mode="lines",
        name="Drawdown",
        line=dict(color="#e74c3c", width=1),
        fill="tozeroy",
        fillcolor="rgba(231,76,60,0.15)",
    ),
    row=2,
    col=1,
)

print("  Adding Underwater...")
fig.add_trace(
    go.Scatter(
        x=drawdown_pct.index.tolist(),
        y=drawdown_pct.tolist(),
        mode="lines",
        name="Underwater",
        line=dict(color="#c0392b", width=2),
        fill="tozeroy",
        fillcolor="rgba(192,57,43,0.2)",
    ),
    row=2,
    col=2,
)

print("  Adding Trade Return Distribution...")
if len(trade_returns) > 0:
    tr_list = trade_returns.tolist()
    colors = ["green" if r >= 0 else "red" for r in tr_list]
    fig.add_trace(
        go.Bar(
            x=list(range(1, len(tr_list) + 1)),
            y=tr_list,
            name="Trade Return %",
            marker_color=colors,
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)

print("  Adding Net Exposure...")
fig.add_trace(
    go.Scatter(
        x=net_exposure.index.tolist(),
        y=net_exposure.tolist(),
        mode="lines",
        name="Net Exposure",
        line=dict(color="#8e44ad", width=1),
        fill="tozeroy",
        fillcolor="rgba(142,68,173,0.15)",
    ),
    row=3,
    col=2,
)

fig.update_layout(
    title=dict(
        text="Silver Bullet Strategy (ICT) — VectorBT Performance Analysis",
        font=dict(size=22, color="#2c3e50"),
        x=0.5,
        xanchor="center",
    ),
    height=1800,
    showlegend=True,
    template="plotly_white",
    hovermode="x unified",
)

fig.update_xaxes(title_text="Time", row=3, col=1)
fig.update_xaxes(title_text="Trade #", row=3, col=1)
fig.update_xaxes(title_text="Time", row=3, col=2)
fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=2)
fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
fig.update_yaxes(title_text="Underwater (%)", row=2, col=2)
fig.update_yaxes(title_text="Return (%)", row=3, col=1)
fig.update_yaxes(title_text="Net Exposure", row=3, col=2)

# ---------------------------------------------------------------------------
# Statistics HTML block
# ---------------------------------------------------------------------------

expectancy = stats.get("Expectancy", 0)
expectancy_color = "green" if expectancy > 0 else "red"
return_on_margin_pct = return_on_margin  # already computed above

# Build trades table rows (last 20 trades)
trade_rows_html = ""
display_trades = trades.tail(20) if len(trades) > 0 else trades
for _, t in display_trades.iterrows():
    pnl_val = t.get("PnL", t.get("Return", 0))
    pnl_color = "green" if pnl_val >= 0 else "red"
    try:
        entry_str = str(t.get("Entry Timestamp", t.get("Entry Time", "")))[:16]
        exit_str = str(t.get("Exit Timestamp", t.get("Exit Time", "")))[:16]
        ret_val = t.get("Return [%]", t.get("Return", 0))
        trade_rows_html += f"""
            <tr style="border-bottom: 1px solid #eee;">
                <td style="padding:6px 10px;">{entry_str}</td>
                <td style="padding:6px 10px;">{exit_str}</td>
                <td style="padding:6px 10px; text-align:right; color:{pnl_color};">${pnl_val:,.2f}</td>
                <td style="padding:6px 10px; text-align:right; color:{pnl_color};">{ret_val:.2f}%</td>
            </tr>"""
    except Exception:
        pass

stats_html = f"""
<div style="padding:30px; background:linear-gradient(135deg,#667eea 0%,#764ba2 100%); color:white;">
    <h2 style="text-align:center; margin-bottom:30px;">Performance Statistics</h2>

    <!-- KPI cards -->
    <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:16px; margin-bottom:30px;">
        <div style="background:rgba(255,255,255,.2); padding:20px; border-radius:10px; text-align:center;">
            <div style="font-size:11px; opacity:.9; text-transform:uppercase; letter-spacing:1px;">Return on Margin</div>
            <div style="font-size:28px; font-weight:bold;">{return_on_margin_pct:+.2f}%</div>
            <div style="font-size:10px; opacity:.7;">${abs_pnl:,.0f} on ${INIT_CASH:,} @ {LEVERAGE}:1</div>
        </div>
        <div style="background:rgba(255,255,255,.2); padding:20px; border-radius:10px; text-align:center;">
            <div style="font-size:11px; opacity:.9; text-transform:uppercase; letter-spacing:1px;">Win Rate</div>
            <div style="font-size:28px; font-weight:bold;">{stats["Win Rate [%]"]:.1f}%</div>
        </div>
        <div style="background:rgba(255,255,255,.2); padding:20px; border-radius:10px; text-align:center;">
            <div style="font-size:11px; opacity:.9; text-transform:uppercase; letter-spacing:1px;">Sharpe Ratio</div>
            <div style="font-size:28px; font-weight:bold;">{stats["Sharpe Ratio"]:.2f}</div>
        </div>
        <div style="background:rgba(255,255,255,.2); padding:20px; border-radius:10px; text-align:center;">
            <div style="font-size:11px; opacity:.9; text-transform:uppercase; letter-spacing:1px;">Profit Factor</div>
            <div style="font-size:28px; font-weight:bold;">{stats.get("Profit Factor", float("nan")):.2f}</div>
        </div>
        <div style="background:rgba(255,255,255,.2); padding:20px; border-radius:10px; text-align:center;">
            <div style="font-size:11px; opacity:.9; text-transform:uppercase; letter-spacing:1px;">Max Drawdown</div>
            <div style="font-size:28px; font-weight:bold;">{stats["Max Drawdown [%]"]:.2f}%</div>
        </div>
        <div style="background:rgba(255,255,255,.2); padding:20px; border-radius:10px; text-align:center;">
            <div style="font-size:11px; opacity:.9; text-transform:uppercase; letter-spacing:1px;">Total Trades</div>
            <div style="font-size:28px; font-weight:bold;">{int(stats["Total Trades"])}</div>
        </div>
    </div>

    <!-- Detailed metrics table -->
    <div style="background:rgba(255,255,255,.95); color:#2c3e50; padding:25px; border-radius:10px; margin-bottom:20px;">
        <h3 style="margin-top:0;">Detailed Metrics</h3>
        <table style="width:100%; border-collapse:collapse;">
            <tr style="border-bottom:1px solid #ddd;">
                <td style="padding:8px 10px;"><strong>Start Value</strong></td>
                <td style="padding:8px 10px; text-align:right;">${INIT_CASH:,.2f}</td>
                <td style="padding:8px 10px;"><strong>End Value</strong></td>
                <td style="padding:8px 10px; text-align:right;">${INIT_CASH + abs_pnl:,.2f}</td>
            </tr>
            <tr style="border-bottom:1px solid #ddd;">
                <td style="padding:8px 10px;"><strong>Best Trade</strong></td>
                <td style="padding:8px 10px; text-align:right; color:green;">{stats["Best Trade [%]"]:.2f}%</td>
                <td style="padding:8px 10px;"><strong>Worst Trade</strong></td>
                <td style="padding:8px 10px; text-align:right; color:red;">{stats["Worst Trade [%]"]:.2f}%</td>
            </tr>
            <tr style="border-bottom:1px solid #ddd;">
                <td style="padding:8px 10px;"><strong>Avg Winning Trade</strong></td>
                <td style="padding:8px 10px; text-align:right; color:green;">{stats["Avg Winning Trade [%]"]:.2f}%</td>
                <td style="padding:8px 10px;"><strong>Avg Losing Trade</strong></td>
                <td style="padding:8px 10px; text-align:right; color:red;">{stats["Avg Losing Trade [%]"]:.2f}%</td>
            </tr>
            <tr style="border-bottom:1px solid #ddd;">
                <td style="padding:8px 10px;"><strong>Expectancy</strong></td>
                <td style="padding:8px 10px; text-align:right; color:{expectancy_color};">${expectancy:,.2f}</td>
                <td style="padding:8px 10px;"><strong>Total Fees Paid</strong></td>
                <td style="padding:8px 10px; text-align:right;">${stats["Total Fees Paid"]:,.2f}</td>
            </tr>
            <tr style="border-bottom:1px solid #ddd;">
                <td style="padding:8px 10px;"><strong>Sortino Ratio</strong></td>
                <td style="padding:8px 10px; text-align:right;">{stats["Sortino Ratio"]:.2f}</td>
                <td style="padding:8px 10px;"><strong>Max DD Duration</strong></td>
                <td style="padding:8px 10px; text-align:right;">{stats["Max Drawdown Duration"]}</td>
            </tr>
        </table>
    </div>

    <!-- Recent trades -->
    <div style="background:rgba(255,255,255,.95); color:#2c3e50; padding:25px; border-radius:10px;">
        <h3 style="margin-top:0;">Recent Trades (last 20)</h3>
        <table style="width:100%; border-collapse:collapse; font-size:13px;">
            <thead>
                <tr style="background:#667eea; color:white;">
                    <th style="padding:8px 10px; text-align:left;">Entry</th>
                    <th style="padding:8px 10px; text-align:left;">Exit</th>
                    <th style="padding:8px 10px; text-align:right;">PnL ($)</th>
                    <th style="padding:8px 10px; text-align:right;">Return (%)</th>
                </tr>
            </thead>
            <tbody>
                {trade_rows_html if trade_rows_html else '<tr><td colspan="4" style="padding:10px; text-align:center;">No closed trades</td></tr>'}
            </tbody>
        </table>
    </div>

    <div style="text-align:center; margin-top:20px; opacity:.8; font-size:12px;">
        {len(df):,} candles &nbsp;|&nbsp;
        {df.index[0].date()} to {df.index[-1].date()} &nbsp;|&nbsp;
        {pd.Series(df.index.date).nunique()} trading days
    </div>
</div>
"""

# ---------------------------------------------------------------------------
# Recent trades candlestick chart (last 5 closed trades)
# Each panel shows ±60 bars around the entry, with SL/TP lines marked.
# ---------------------------------------------------------------------------

print("  Building recent-trades chart...")

import plotly.io as pio

pio.json.config.default_engine = "json"

# Identify entry/exit column names (vectorbt names vary slightly by version)
entry_col = "Entry Timestamp" if "Entry Timestamp" in trades.columns else "Entry Time"
exit_col = "Exit Timestamp" if "Exit Timestamp" in trades.columns else "Exit Time"

recent_trades = trades.tail(5).reset_index(drop=True)
n_recent = len(recent_trades)

CONTEXT_BARS = 80  # bars of price data shown either side of entry

trade_fig_titles = []
for _, t in recent_trades.iterrows():
    entry_ts = pd.Timestamp(t[entry_col])
    pnl_val = t.get("PnL", 0)
    outcome = "WIN" if pnl_val >= 0 else "LOSS"
    trade_fig_titles.append(
        f"Trade {_ + 1}  |  Entry {str(entry_ts)[:16]}  |  PnL ${pnl_val:,.2f}  [{outcome}]"
    )

trade_fig = make_subplots(
    rows=n_recent,
    cols=1,
    subplot_titles=trade_fig_titles,
    vertical_spacing=0.06,
)

for idx, (_, t) in enumerate(recent_trades.iterrows(), start=1):
    entry_ts = pd.Timestamp(t[entry_col])
    exit_ts = pd.Timestamp(t[exit_col])

    # Window of price data around entry
    entry_pos = (
        df.index.get_loc(entry_ts)
        if entry_ts in df.index
        else df.index.searchsorted(entry_ts)
    )
    win_start = max(0, entry_pos - CONTEXT_BARS)
    win_end = min(len(df), entry_pos + CONTEXT_BARS + 1)
    chunk = df.iloc[win_start:win_end]

    trade_fig.add_trace(
        go.Candlestick(
            x=chunk.index.tolist(),
            open=chunk["open"].tolist(),
            high=chunk["high"].tolist(),
            low=chunk["low"].tolist(),
            close=chunk["close"].tolist(),
            name=f"Price",
            showlegend=(idx == 1),
            increasing_line_color="#27ae60",
            decreasing_line_color="#e74c3c",
        ),
        row=idx,
        col=1,
    )

    # Entry marker
    entry_close = (
        df.loc[entry_ts, "close"]
        if entry_ts in df.index
        else chunk["close"].iloc[CONTEXT_BARS]
    )
    trade_fig.add_trace(
        go.Scatter(
            x=[entry_ts],
            y=[entry_close],
            mode="markers",
            marker=dict(
                symbol="triangle-up",
                size=14,
                color="#2980b9",
                line=dict(width=1, color="white"),
            ),
            name="Entry",
            showlegend=(idx == 1),
        ),
        row=idx,
        col=1,
    )

    # Exit marker
    exit_close = df.loc[exit_ts, "close"] if exit_ts in df.index else None
    if exit_close is not None:
        pnl_val = t.get("PnL", 0)
        exit_color = "#27ae60" if pnl_val >= 0 else "#e74c3c"
        trade_fig.add_trace(
            go.Scatter(
                x=[exit_ts],
                y=[exit_close],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    size=14,
                    color=exit_color,
                    line=dict(width=1, color="white"),
                ),
                name="Exit",
                showlegend=(idx == 1),
            ),
            row=idx,
            col=1,
        )

    # SL line
    sl_price = (
        df.loc[entry_ts, "sl_price"]
        if entry_ts in df.index and "sl_price" in df.columns
        else None
    )
    if sl_price is not None and not pd.isna(sl_price):
        trade_fig.add_shape(
            type="line",
            x0=chunk.index[0],
            x1=chunk.index[-1],
            y0=sl_price,
            y1=sl_price,
            line=dict(color="#e74c3c", width=1.5, dash="dash"),
            row=idx,
            col=1,
        )
        trade_fig.add_annotation(
            x=chunk.index[-1],
            y=sl_price,
            text=f"SL {sl_price:.0f}",
            showarrow=False,
            font=dict(size=10, color="#e74c3c"),
            xanchor="right",
            row=idx,
            col=1,
        )

    # TP line
    tp_price = (
        df.loc[entry_ts, "tp_price"]
        if entry_ts in df.index and "tp_price" in df.columns
        else None
    )
    if tp_price is not None and not pd.isna(tp_price):
        trade_fig.add_shape(
            type="line",
            x0=chunk.index[0],
            x1=chunk.index[-1],
            y0=tp_price,
            y1=tp_price,
            line=dict(color="#27ae60", width=1.5, dash="dash"),
            row=idx,
            col=1,
        )
        trade_fig.add_annotation(
            x=chunk.index[-1],
            y=tp_price,
            text=f"TP {tp_price:.0f}",
            showarrow=False,
            font=dict(size=10, color="#27ae60"),
            xanchor="right",
            row=idx,
            col=1,
        )

trade_fig.update_layout(
    title=dict(
        text="Last 5 Trades — Candlestick Detail (entry/exit/SL/TP)",
        font=dict(size=18, color="#2c3e50"),
        x=0.5,
        xanchor="center",
    ),
    height=380 * n_recent,
    template="plotly_white",
    showlegend=True,
    xaxis_rangeslider_visible=False,
)
for i in range(1, n_recent + 1):
    trade_fig.update_xaxes(rangeslider_visible=False, row=i, col=1)

recent_trades_html = trade_fig.to_html(include_plotlyjs=False, full_html=False)

# ---------------------------------------------------------------------------
# Assemble full HTML
# ---------------------------------------------------------------------------

charts_html = fig.to_html(include_plotlyjs=False, full_html=False)

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Silver Bullet Strategy — VectorBT Report</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: #f0f2f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 40px 20px;
            text-align: center;
        }}
        .header h1 {{ font-size: 32px; margin-bottom: 8px; }}
        .header p  {{ font-size: 14px; opacity: 0.85; }}
        .content {{
            max-width: 1800px;
            margin: 0 auto;
            background: white;
        }}
        .charts {{ padding: 20px; }}
        .section-title {{
            font-size: 20px;
            font-weight: 600;
            color: #2c3e50;
            padding: 24px 20px 8px;
            border-top: 3px solid #667eea;
            margin-top: 8px;
        }}
        .footer {{
            background: #2c3e50;
            color: rgba(255,255,255,.7);
            padding: 18px;
            text-align: center;
            font-size: 12px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Silver Bullet Strategy (ICT)</h1>
        <p>
            Asset: {SYMBOL} &nbsp;|&nbsp;
            Period: {START_DATE} to {END_DATE} &nbsp;|&nbsp;
            Capital: ${INIT_CASH:,} &nbsp;|&nbsp;
            Fees: {FEES * 100:.2f}% &nbsp;|&nbsp;
            RR {RR_RATIO}:1 &nbsp;|&nbsp;
            LiqWin {LIQUIDITY_WINDOW} &nbsp;|&nbsp;
            Sweep {SWEEP_LOOKBACK} &nbsp;|&nbsp;
            PBwin {PULLBACK_WINDOW}
        </p>
    </div>

    <div class="content">
        <div class="section-title">Performance Overview</div>
        <div class="charts">
            {charts_html}
        </div>

        <div class="section-title">Last 5 Trades — Candlestick Detail</div>
        <div class="charts">
            {recent_trades_html}
        </div>

        {stats_html}
    </div>

    <div class="footer">
        Generated with VectorBT &nbsp;|&nbsp;
        {len(trades)} trades analysed &nbsp;|&nbsp;
        Data: {df.index[0].date()} to {df.index[-1].date()}
    </div>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Save report
# ---------------------------------------------------------------------------

reports_dir = project_root / "reports"
reports_dir.mkdir(exist_ok=True)
output_file = (
    reports_dir / f"silver_bullet_{SYMBOL}_backtest_{START_DATE}_{END_DATE}.html"
)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(full_html)

file_size_mb = os.path.getsize(output_file) / 1024 / 1024

print("\n" + "=" * 70)
print("BACKTEST REPORT COMPLETE")
print("=" * 70)
print(f"\n  Report saved -> {output_file}")
print(f"  File size    : {file_size_mb:.1f} MB")
print(f"\nKey Metrics:")
print(f"  Abs P&L            : ${abs_pnl:,.2f}")
print(
    f"  Return on Margin   : {return_on_margin:+.2f}%  ({LOT_SIZE} lots, {LEVERAGE}:1 leverage, ${INIT_CASH:,} margin)"
)
print(f"  Win Rate           : {stats['Win Rate [%]']:.1f}%")
print(f"  Sharpe Ratio       : {stats['Sharpe Ratio']:.2f}")
print(f"  Max Drawdown       : {stats['Max Drawdown [%]']:.2f}%")
print(f"  Total Trades       : {int(stats['Total Trades'])}")
