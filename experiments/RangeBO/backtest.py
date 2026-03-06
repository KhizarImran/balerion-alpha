"""
Range Breakout — VectorBT Backtest

Trades the Asian range breakout (00:00–07:00 UK) on USDJPY 1H:
  - Long  when first bar at/after 07:00 UK closes above the Range High
  - Short when first bar at/after 07:00 UK closes below the Range Low
  - SL    at the opposite range boundary
  - TP    at 1:2 R:R  (range width x2 in breakout direction)
  - Hard exit at 16:00 UK regardless of outcome

Risk management:
  - 1% of account risked per trade
  - Lot size computed from SL distance so each SL hit = exactly 1% loss
  - 1:100 leverage (notional = account_size * leverage passed to vectorbt)
  - Fees: 1.5 pip spread per side (conservative for USDJPY retail)

Outputs saved to experiments/RangeBO/reports/{SYMBOL}_{YYYYMMDD_HHMMSS}/:
  - backtest_report.txt
  - trades.csv
  - equity.html
  - vbt_report.html
  - portfolio.pkl

All outputs are pushed to MLflow experiment "rangebo-backtest".

Usage:
    uv run python experiments/RangeBO/backtest.py
"""

import sys
import io
import os
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import mlflow

from utils import DataLoader

# Import signal logic from strategy.py in the same folder
import importlib.util
_strategy_path = Path(__file__).resolve().parent / "strategy.py"
_spec = importlib.util.spec_from_file_location("rangebo_strategy", _strategy_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
calculate_signals = _mod.calculate_signals

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
os.environ["MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS"] = "true"
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("rangebo-backtest")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SYMBOL         = "USDJPY"
ASSET_TYPE     = "fx"
START_DATE     = "2024-01-01"
END_DATE       = "2026-03-05"
TIMEZONE       = "Europe/London"
TRADE_CLOSE_HOUR = 16          # 16:00 UK hard exit hour

# Account
ACCOUNT_SIZE   = 10_000        # real margin capital (USD)
LEVERAGE       = 100           # 1:100
RISK_PER_TRADE = 0.01          # 1% of account per trade

# USDJPY instrument constants
PIP            = 0.01          # pip size for JPY pairs
PIP_VALUE_PER_LOT = 9.5        # approx USD per pip on 1 standard lot at ~150 USD/JPY
STD_LOT        = 100_000       # units per standard lot

FX_FEES        = 0.015         # 1.5 pip spread per side (fee as pip-equivalent fraction used below)
FREQ           = "1h"

RR             = 2.0           # reward-to-risk (must match strategy.py)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compute_lot_units(
    account_equity: float,
    sl_pips: float,
    risk_pct: float = RISK_PER_TRADE,
    pip_value_per_lot: float = PIP_VALUE_PER_LOT,
    std_lot: float = STD_LOT,
) -> float:
    """
    Size a trade so hitting SL loses exactly risk_pct * account_equity.

        risk_$ = account_equity * risk_pct
        lots   = risk_$ / (sl_pips * pip_value_per_lot)
        units  = lots * std_lot
    """
    if sl_pips <= 0:
        return 0.0
    risk_dollars = account_equity * risk_pct
    lots = risk_dollars / (sl_pips * pip_value_per_lot)
    return lots * std_lot


def build_vbt_inputs(df: pd.DataFrame) -> dict:
    """
    From the signal DataFrame produced by calculate_signals(), build all
    arrays that vectorbt needs:

      long_entries  / short_entries  : bool Series
      long_exits    / short_exits    : bool Series  (16:00 UK time exit)
      sl_series                      : fractional stop-loss per bar
      tp_series                      : fractional take-profit per bar
      size_series                    : lot units per bar

    Returns a dict of all these Series plus summary stats.
    """
    close = df["close"]
    n = len(df)

    long_entries  = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    sl_series     = pd.Series(0.0,   index=df.index)
    tp_series     = pd.Series(0.0,   index=df.index)
    size_series   = pd.Series(0.0,   index=df.index)

    # 16:00 UK bar — used as the time-stop exit signal
    # vectorbt exits the position on the first True in exits after entry
    df_uk_hour = df["uk_hour"]
    time_exit_mask = df_uk_hour == TRADE_CLOSE_HOUR
    long_exits  = pd.Series(time_exit_mask.values, index=df.index)
    short_exits = pd.Series(time_exit_mask.values, index=df.index)

    skipped = 0
    for ts, row in df[df["buy_signal"] | df["sell_signal"]].iterrows():
        entry  = row["entry_price"]
        sl_px  = row["sl_price"]
        tp_px  = row["tp_price"]
        is_long = bool(row["buy_signal"])

        if pd.isna(entry) or pd.isna(sl_px) or pd.isna(tp_px):
            skipped += 1
            continue

        sl_dist = abs(entry - sl_px)
        tp_dist = abs(tp_px - entry)

        if sl_dist <= 0:
            skipped += 1
            continue

        sl_frac = sl_dist / entry
        tp_frac = tp_dist / entry

        sl_pips = sl_dist / PIP
        units   = compute_lot_units(ACCOUNT_SIZE, sl_pips)

        if is_long:
            long_entries.loc[ts]  = True
        else:
            short_entries.loc[ts] = True

        sl_series.loc[ts]   = sl_frac
        tp_series.loc[ts]   = tp_frac
        size_series.loc[ts] = units

    return dict(
        long_entries  = long_entries,
        short_entries = short_entries,
        long_exits    = long_exits,
        short_exits   = short_exits,
        sl_series     = sl_series,
        tp_series     = tp_series,
        size_series   = size_series,
        skipped       = skipped,
    )


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------

def run_backtest(
    symbol: str        = SYMBOL,
    start_date: str    = START_DATE,
    end_date: str      = END_DATE,
    account_size: float = ACCOUNT_SIZE,
    leverage: int      = LEVERAGE,
    risk_per_trade: float = RISK_PER_TRADE,
    save_outputs: bool = True,
    show_chart: bool   = True,
):
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading {symbol} 1H data ({start_date} to {end_date})...")
    loader = DataLoader()
    df = loader.load_fx(symbol, timeframe="1h")

    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")

    df = df[
        (df.index >= pd.Timestamp(start_date, tz="UTC")) &
        (df.index <= pd.Timestamp(end_date,   tz="UTC"))
    ]
    print(f"  Loaded {len(df):,} 1H bars  ({df.index.min()} -> {df.index.max()})")

    # ------------------------------------------------------------------
    # 2. Signal detection
    # ------------------------------------------------------------------
    print("Detecting range breakout signals...")
    df = calculate_signals(df)

    n_long  = int(df["buy_signal"].sum())
    n_short = int(df["sell_signal"].sum())
    print(f"  Long entries  : {n_long}")
    print(f"  Short entries : {n_short}")

    # ------------------------------------------------------------------
    # 3. Build vectorbt inputs
    # ------------------------------------------------------------------
    vbt_in = build_vbt_inputs(df)
    n_long_vbt  = int(vbt_in["long_entries"].sum())
    n_short_vbt = int(vbt_in["short_entries"].sum())
    print(f"  After input build: {n_long_vbt} long, {n_short_vbt} short  "
          f"(skipped: {vbt_in['skipped']})")

    if n_long_vbt + n_short_vbt == 0:
        print("No valid trades. Exiting.")
        return None, None

    close = df["close"]

    # Fees: convert 1.5 pip spread to a per-unit fraction of price
    # fees = pip_amount / entry_price  — use mean price as proxy
    fee_frac = (FX_FEES * PIP) / close.mean()

    mean_sl_pips = (
        vbt_in["sl_series"][vbt_in["sl_series"] > 0]
        .mul(close)           # back to price units
        .div(PIP)             # to pips
        .mean()
    )
    mean_units = vbt_in["size_series"][vbt_in["size_series"] > 0].mean()
    print(f"  Mean SL distance   : {mean_sl_pips:.1f} pips")
    print(f"  Mean lot size      : {mean_units / STD_LOT:.4f} lots  ({mean_units:,.0f} units)")
    print(f"  Risk per trade     : ${account_size * risk_per_trade:,.0f}  "
          f"({risk_per_trade * 100:.1f}% of ${account_size:,})")

    # ------------------------------------------------------------------
    # 4. Run vectorbt simulation
    # ------------------------------------------------------------------
    print("Running vectorbt backtest...")
    vbt.settings.plotting["use_widgets"] = False

    pf = vbt.Portfolio.from_signals(
        close         = close,
        entries       = vbt_in["long_entries"],
        exits         = vbt_in["long_exits"],
        short_entries = vbt_in["short_entries"],
        short_exits   = vbt_in["short_exits"],
        sl_stop       = vbt_in["sl_series"],
        tp_stop       = vbt_in["tp_series"],
        size          = vbt_in["size_series"],
        size_type     = "amount",
        init_cash     = account_size * leverage,   # notional buying power
        fees          = fee_frac,
        freq          = FREQ,
        accumulate    = False,
    )

    # Re-anchor equity to real margin capital
    abs_pnl          = pf.value().iloc[-1] - (account_size * leverage)
    return_on_margin = (abs_pnl / account_size) * 100

    # ------------------------------------------------------------------
    # 5. Print results
    # ------------------------------------------------------------------
    stats = pf.stats()
    stats["Start Value"]    = account_size
    stats["End Value"]      = account_size + abs_pnl
    stats["Total Return [%]"] = return_on_margin

    trades_df = pf.trades.records_readable.copy()
    n_trades  = pf.trades.count()
    win_rate  = pf.trades.win_rate() * 100  if n_trades > 0 else 0.0
    pf_factor = pf.trades.profit_factor()   if n_trades > 0 else 0.0

    print("\n" + "=" * 60)
    print(f" RANGE BREAKOUT — {symbol} 1H BACKTEST RESULTS")
    print(f" Period    : {start_date}  to  {end_date}")
    print(f" Account   : ${account_size:,}  |  Leverage: {leverage}:1  |  "
          f"Notional: ${account_size * leverage:,}")
    print(f" Risk/trade: {risk_per_trade * 100:.1f}%  "
          f"(${account_size * risk_per_trade:,.0f})  |  "
          f"Fees: {FX_FEES:.1f} pip/side")
    print("=" * 60)
    print(f" Abs P&L          : ${abs_pnl:+,.2f}")
    print(f" Return on Margin : {return_on_margin:+.2f}%  (on ${account_size:,})")
    print(f" Final Value      : ${account_size + abs_pnl:,.2f}")
    print(f" Sharpe Ratio     : {pf.sharpe_ratio():.3f}")
    print(f" Sortino Ratio    : {pf.sortino_ratio():.3f}")
    print(f" Max Drawdown     : {pf.max_drawdown() * 100:.2f}%")
    print(f" Total Trades     : {n_trades}")
    print(f" Win Rate         : {win_rate:.1f}%")
    print(f" Profit Factor    : {pf_factor:.3f}")
    trades_pnl = trades_df["PnL"]
    print(f" Avg Win          : ${trades_pnl[trades_pnl > 0].mean():.2f}")
    print(f" Avg Loss         : ${trades_pnl[trades_pnl < 0].mean():.2f}")
    print("=" * 60)

    # Direction breakdown
    if "Direction" in trades_df.columns:
        for direction in trades_df["Direction"].unique():
            sub  = trades_df[trades_df["Direction"] == direction]
            wins = (sub["PnL"] > 0).sum()
            tot  = len(sub)
            print(f" {direction:<10}: {tot} trades | "
                  f"win rate {wins / tot * 100:.1f}% | "
                  f"total PnL ${sub['PnL'].sum():.2f}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 6. Build equity + drawdown chart
    # ------------------------------------------------------------------
    equity = pf.value()
    dd     = (equity - equity.cummax()) / equity.cummax() * 100

    fig_eq = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.05, row_heights=[0.65, 0.35],
        subplot_titles=(
            f"Range Breakout — {symbol} 1H  |  Equity Curve",
            "Drawdown %",
        ),
    )

    fig_eq.add_trace(go.Scatter(
        x=equity.index, y=equity + account_size - account_size * leverage,
        name="Equity",
        line=dict(color="#26a69a", width=2),
        fill="tozeroy", fillcolor="rgba(38,166,154,0.08)",
    ), row=1, col=1)

    # Win / loss dots on the equity curve
    if len(trades_df) > 0:
        entry_times  = pd.to_datetime(trades_df["Entry Timestamp"].values, utc=True)
        # Re-anchor equity values to real margin scale for the dots
        eq_offset    = account_size - account_size * leverage
        entry_equity = (equity.reindex(entry_times, method="nearest") + eq_offset).values
        win_mask     = (trades_df["PnL"] > 0).values
        loss_mask    = (trades_df["PnL"] <= 0).values

        if win_mask.any():
            fig_eq.add_trace(go.Scatter(
                x=entry_times[win_mask], y=entry_equity[win_mask],
                mode="markers", name="Win",
                marker=dict(symbol="circle", size=7, color="#00e676",
                            line=dict(color="#1b5e20", width=1)),
            ), row=1, col=1)
        if loss_mask.any():
            fig_eq.add_trace(go.Scatter(
                x=entry_times[loss_mask], y=entry_equity[loss_mask],
                mode="markers", name="Loss",
                marker=dict(symbol="circle", size=7, color="#f44336",
                            line=dict(color="#b71c1c", width=1)),
            ), row=1, col=1)

    fig_eq.add_trace(go.Scatter(
        x=dd.index, y=dd, name="Drawdown",
        line=dict(color="#ef5350", width=1),
        fill="tozeroy", fillcolor="rgba(239,83,80,0.15)",
    ), row=2, col=1)

    fig_eq.update_layout(
        height=700, template="plotly_dark", hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(
            text=(
                f"<b>Range Breakout — {symbol} 1H</b><br>"
                f"<sup>"
                f"Return: {return_on_margin:+.2f}%  |  "
                f"Sharpe: {pf.sharpe_ratio():.2f}  |  "
                f"Win Rate: {win_rate:.1f}%  |  "
                f"Trades: {n_trades}  |  "
                f"Max DD: {pf.max_drawdown() * 100:.2f}%"
                f"</sup>"
            ),
            x=0.5, xanchor="center",
        ),
    )
    fig_eq.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig_eq.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig_eq.update_xaxes(title_text="Date", row=2, col=1)

    # ------------------------------------------------------------------
    # 7. P&L distribution chart
    # ------------------------------------------------------------------
    trades_df["pnl_pips"] = trades_df["PnL"].div(
        vbt_in["size_series"][vbt_in["size_series"] > 0].mean()
    ) / PIP * close.mean()  # approximate pip P&L

    # Simpler and more accurate: derive pip P&L from price difference
    if "Entry Price" in trades_df.columns and "Exit Price" in trades_df.columns:
        direction_sign = trades_df["Direction"].map({"Long": 1, "Short": -1}).fillna(1)
        trades_df["pnl_pips"] = (
            (trades_df["Exit Price"] - trades_df["Entry Price"]) * direction_sign / PIP
        )

    fig_dist = go.Figure()
    wins_pips  = trades_df.loc[trades_df["pnl_pips"] > 0, "pnl_pips"]
    loss_pips  = trades_df.loc[trades_df["pnl_pips"] <= 0, "pnl_pips"]

    if len(wins_pips) > 0:
        fig_dist.add_trace(go.Histogram(
            x=wins_pips, nbinsx=30, name="Wins",
            marker_color="rgba(38,166,154,0.7)",
        ))
    if len(loss_pips) > 0:
        fig_dist.add_trace(go.Histogram(
            x=loss_pips, nbinsx=30, name="Losses",
            marker_color="rgba(239,83,80,0.7)",
        ))

    fig_dist.update_layout(
        barmode="overlay",
        title=dict(
            text=(
                f"<b>Trade P&L Distribution (pips) — {symbol} 1H</b><br>"
                f"<sup>Avg win: {wins_pips.mean():.1f} pips  |  "
                f"Avg loss: {loss_pips.mean():.1f} pips</sup>"
            ),
            x=0.5, xanchor="center",
        ),
        xaxis_title="Pips",
        yaxis_title="Trade Count",
        template="plotly_dark",
        height=450,
    )

    # ------------------------------------------------------------------
    # 8. Save outputs + MLflow
    # ------------------------------------------------------------------
    run_ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(__file__).resolve().parent / "reports" / f"{symbol}_{run_ts}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    mlflow.start_run(run_name=f"{symbol}_1H_{start_date}_{end_date}")
    mlflow.log_params({
        "symbol":          symbol,
        "asset_type":      ASSET_TYPE,
        "start_date":      start_date,
        "end_date":        end_date,
        "account_size":    account_size,
        "leverage":        leverage,
        "risk_per_trade":  risk_per_trade,
        "fx_fees_pips":    FX_FEES,
        "rr":              RR,
        "freq":            FREQ,
        "timezone":        TIMEZONE,
        "trade_close_hour": TRADE_CLOSE_HOUR,
    })
    mlflow.log_metrics({
        "abs_pnl":              float(abs_pnl),
        "return_on_margin_pct": float(return_on_margin),
        "sharpe_ratio":         float(pf.sharpe_ratio()),
        "sortino_ratio":        float(pf.sortino_ratio()),
        "max_drawdown_pct":     float(pf.max_drawdown() * 100),
        "total_trades":         float(n_trades),
        "win_rate_pct":         float(win_rate),
        "profit_factor":        float(pf_factor),
        "n_long_entries":       float(n_long_vbt),
        "n_short_entries":      float(n_short_vbt),
    })

    if save_outputs:
        # Text report
        report_path = reports_dir / "backtest_report.txt"
        with open(report_path, "w") as f:
            f.write(f"Range Breakout — {symbol} 1H Backtest\n")
            f.write(f"Period     : {start_date} to {end_date}\n")
            f.write(f"Account    : ${account_size:,}  |  Leverage: {leverage}:1\n")
            f.write(f"Risk/trade : {risk_per_trade * 100:.1f}%  "
                    f"(${account_size * risk_per_trade:,.0f})\n")
            f.write(f"R:R        : {RR}  |  Fees: {FX_FEES:.1f} pip/side\n\n")
            f.write(str(stats))
            f.write(f"\n\nAbs P&L           : ${abs_pnl:+,.2f}\n")
            f.write(f"Return on Margin  : {return_on_margin:+.2f}%\n")
            f.write(f"Sharpe Ratio      : {pf.sharpe_ratio():.3f}\n")
            f.write(f"Sortino Ratio     : {pf.sortino_ratio():.3f}\n")
            f.write(f"Max Drawdown      : {pf.max_drawdown() * 100:.2f}%\n")
            f.write(f"Win Rate          : {win_rate:.1f}%\n")
            f.write(f"Profit Factor     : {pf_factor:.3f}\n")
        print(f"  Report saved   -> {report_path}")

        # Trades CSV
        trades_path = reports_dir / "trades.csv"
        trades_df.to_csv(trades_path)
        print(f"  Trades saved   -> {trades_path}")

        # Equity chart
        equity_path = reports_dir / "equity.html"
        fig_eq.write_html(str(equity_path))
        print(f"  Equity chart   -> {equity_path}")

        # P&L distribution
        dist_path = reports_dir / "pnl_distribution.html"
        fig_dist.write_html(str(dist_path))
        print(f"  P&L dist chart -> {dist_path}")

        # Native vectorbt 7-panel report
        print("  Building VBT report chart...")
        vbt_fig = pf.plot(
            subplots=[
                "value",
                "cum_returns",
                "underwater",
                "drawdowns",
                "orders",
                "trades",
                "net_exposure",
            ],
            make_subplots_kwargs=dict(
                rows=7, cols=1, shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.22, 0.12, 0.12, 0.14, 0.14, 0.14, 0.12],
            ),
            template="plotly_dark",
            title=f"Range Breakout — {symbol} 1H  |  VectorBT Performance Report",
        )
        vbt_path = reports_dir / "vbt_report.html"
        vbt_fig.write_html(str(vbt_path))
        print(f"  VBT report     -> {vbt_path}")

        # Portfolio pickle
        pkl_path = reports_dir / "portfolio.pkl"
        pf.save(str(pkl_path))
        print(f"  Portfolio pkl  -> {pkl_path}")

        # Upload everything to MLflow
        mlflow.log_artifacts(str(reports_dir), artifact_path="outputs")
        print(f"\n  All outputs in -> {reports_dir}")

    mlflow.end_run()
    print(f"  MLflow run logged -> {MLFLOW_TRACKING_URI}")

    if show_chart:
        fig_eq.show()

    return pf, df


if __name__ == "__main__":
    run_backtest()
