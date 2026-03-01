"""
ICT Unicorn Model — VectorBT Backtest

Risk management per trade:
  - Entry  : close of the signal bar (FVG retrace confirmed)
  - BUY  SL: below the sweep low (liquidity level that was taken out)
  - BUY  TP: at the prior swing high that was broken (CHoCH level = BSL target)
  - SELL SL: above the sweep high
  - SELL TP: at the prior swing low that was broken (CHoCH level = SSL target)

Target risk-reward >= 1:2.  Trades where the setup geometry produces a R:R below
a configurable minimum are filtered out before simulation.

Outputs saved to experiments/Unicorn-ICT/reports/{SYMBOL}_{YYYYMMDD_HHMMSS}/:
  - backtest_report.txt   full stats
  - trades.csv            trade-by-trade log
  - equity.html           interactive equity + drawdown chart
  - vbt_report.html       native vectorbt 7-panel performance chart
  - portfolio.pkl         raw Portfolio object for reloading
"""

import sys
import io
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Fix Windows cp1252 terminal encoding so MLflow's emoji output doesn't crash
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

# ---------------------------------------------------------------------------
# MLflow — point at the local server
# ---------------------------------------------------------------------------
import os

os.environ["MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS"] = "true"

MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("unicorn-ict")

# Import signal logic from strategy.py in the same folder
import importlib.util

_strategy_path = Path(__file__).resolve().parent / "strategy.py"
_spec = importlib.util.spec_from_file_location("unicorn_strategy", _strategy_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
calculate_signals = _mod.calculate_signals


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "EURUSD"
ASSET_TYPE = "fx"
START_DATE = "2025-10-01"
END_DATE = "2026-02-27"

# Account
ACCOUNT_SIZE = 10_000  # real margin capital in USD
LEVERAGE = 100  # 1:100 retail FX leverage
RISK_PER_TRADE = 0.01  # 1% of account risked per trade

# FX instrument constants (EURUSD)
PIP = 0.0001  # pip size for 4-decimal pairs
PIP_VALUE_PER_LOT = 10.0  # USD value of 1 pip on 1 standard lot (EURUSD, USD account)
STD_LOT = 100_000  # units per standard lot

FX_FEES = 0.00007  # ~0.7 pip spread per side (round trip ~1.4 pip)
MIN_RR = 1.5  # minimum R:R ratio — trades below this are skipped
SL_BUFFER = 0.0002  # 2 pip buffer added beyond sweep high/low for SL
FREQ = "1h"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def build_per_bar_stops(
    df: pd.DataFrame,
    signal_col: str,
    sl_price_col: str,
    tp_price_col: str,
    entry_price_col: str = "close",
    sl_buffer: float = SL_BUFFER,
    min_rr: float = MIN_RR,
    direction: str = "long",  # "long" or "short"
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    For each signal bar compute fractional SL and TP distances from entry.

    vectorbt expects sl_stop / tp_stop as a fraction of entry price:
        sl_stop = abs(entry - sl_price) / entry
        tp_stop = abs(tp_price - entry) / entry

    Returns
    -------
    filtered_signal : boolean Series (trades with R:R below min_rr removed)
    sl_series       : float Series, NaN on non-signal bars
    tp_series       : float Series, NaN on non-signal bars
    """
    filtered_signal = df[signal_col].copy()
    sl_series = pd.Series(np.nan, index=df.index)
    tp_series = pd.Series(np.nan, index=df.index)

    sig_idx = df.index[df[signal_col]]

    for ts in sig_idx:
        entry = df.loc[ts, entry_price_col]
        sl_raw = df.loc[ts, sl_price_col]
        tp_raw = df.loc[ts, tp_price_col]

        if np.isnan(sl_raw) or np.isnan(tp_raw):
            filtered_signal.loc[ts] = False
            continue

        # Apply SL buffer beyond the sweep level
        if direction == "long":
            sl_price = sl_raw - sl_buffer  # below the sweep low
            tp_price = tp_raw
        else:
            sl_price = sl_raw + sl_buffer  # above the sweep high
            tp_price = tp_raw

        sl_dist = abs(entry - sl_price)
        tp_dist = abs(tp_price - entry)

        if sl_dist <= 0:
            filtered_signal.loc[ts] = False
            continue

        rr = tp_dist / sl_dist
        if rr < min_rr:
            filtered_signal.loc[ts] = False
            continue

        sl_series.loc[ts] = sl_dist / entry
        tp_series.loc[ts] = tp_dist / entry

    return filtered_signal, sl_series, tp_series


def compute_lot_units(
    account_equity: float,
    sl_frac: float,
    entry_price: float,
    risk_pct: float = RISK_PER_TRADE,
    pip_size: float = PIP,
    pip_value_per_lot: float = PIP_VALUE_PER_LOT,
    std_lot: float = STD_LOT,
) -> float:
    """
    Compute position size in base currency units so that hitting SL loses
    exactly risk_pct of account equity.

    Derivation:
        risk_$      = account_equity * risk_pct
        sl_pips     = sl_frac * entry_price / pip_size
        lots        = risk_$ / (sl_pips * pip_value_per_lot)
        units       = lots * std_lot

    Returns units (float) — pass as size= with size_type='amount'.
    Returns 0.0 if sl_frac is zero or negative.
    """
    if sl_frac <= 0 or entry_price <= 0:
        return 0.0
    risk_dollars = account_equity * risk_pct
    sl_pips = (sl_frac * entry_price) / pip_size
    lots = risk_dollars / (sl_pips * pip_value_per_lot)
    return lots * std_lot


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------


def run_backtest(
    symbol: str = SYMBOL,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    account_size: float = ACCOUNT_SIZE,
    leverage: int = LEVERAGE,
    risk_per_trade: float = RISK_PER_TRADE,
    min_rr: float = MIN_RR,
    sl_buffer: float = SL_BUFFER,
    save_outputs: bool = True,
    show_chart: bool = True,
):
    # ------------------------------------------------------------------
    # 1. Load & resample data
    # ------------------------------------------------------------------
    print(f"Loading {symbol} 1-minute data ({start_date} to {end_date})...")
    loader = DataLoader()
    df_1m = loader.load_fx(symbol, start_date=start_date, end_date=end_date)
    print(f"  Loaded {len(df_1m):,} 1-minute bars.")

    print("Resampling to 1H and detecting Unicorn setups...")
    df = loader.resample_ohlcv(df_1m, "1h")
    df = calculate_signals(df)
    close = df["close"]
    print(
        f"  {len(df):,} hourly bars  |  "
        f"Buy signals: {df['buy_signal'].sum()}  |  "
        f"Sell signals: {df['sell_signal'].sum()}"
    )

    # ------------------------------------------------------------------
    # 2. Build per-bar SL / TP series (long and short separately)
    # ------------------------------------------------------------------
    long_entries, long_sl, long_tp = build_per_bar_stops(
        df,
        signal_col="buy_signal",
        sl_price_col="buy_sl_price",
        tp_price_col="buy_tp_price",
        direction="long",
        sl_buffer=sl_buffer,
        min_rr=min_rr,
    )

    short_entries, short_sl, short_tp = build_per_bar_stops(
        df,
        signal_col="sell_signal",
        sl_price_col="sell_sl_price",
        tp_price_col="sell_tp_price",
        direction="short",
        sl_buffer=sl_buffer,
        min_rr=min_rr,
    )

    n_long = int(long_entries.sum())
    n_short = int(short_entries.sum())
    print(
        f"  After R:R >= {min_rr} filter: {n_long} long entries, {n_short} short entries"
    )

    if n_long + n_short == 0:
        print("No valid trades after R:R filter. Exiting.")
        return None

    # ------------------------------------------------------------------
    # 3. Position sizing — risk-based lot sizing with leverage
    #
    #    With 1:100 leverage a $10,000 account controls $1,000,000 notional.
    #    Margin required per standard lot = 100,000 / 100 = $1,000.
    #
    #    We size each trade so that hitting the SL costs exactly
    #    risk_per_trade * account_size in USD:
    #
    #        risk_$   = account_size * risk_per_trade
    #        sl_pips  = sl_frac * entry_price / PIP
    #        lots     = risk_$ / (sl_pips * PIP_VALUE_PER_LOT)
    #        units    = lots * STD_LOT
    #
    #    vectorbt is passed init_cash = account_size * leverage (notional
    #    buying power) so it has sufficient cash to fill the lot sizes.
    #    Real P&L and returns are re-anchored to account_size afterwards.
    # ------------------------------------------------------------------
    entry_price_approx = close.mean()

    # Build a per-signal-bar Series of lot units
    all_signals = long_entries | short_entries
    combined_sl = long_sl.fillna(short_sl)

    lot_units_series = pd.Series(0.0, index=df.index)
    for ts in df.index[all_signals]:
        sl_frac = combined_sl.loc[ts]
        if np.isnan(sl_frac) or sl_frac <= 0:
            continue
        units = compute_lot_units(
            account_equity=account_size,
            sl_frac=sl_frac,
            entry_price=close.loc[ts],
            risk_pct=risk_per_trade,
        )
        lot_units_series.loc[ts] = units

    mean_units = lot_units_series[all_signals].replace(0, np.nan).mean()
    mean_sl_frac = combined_sl[all_signals].dropna().mean()
    mean_sl_pips = (mean_sl_frac * entry_price_approx) / PIP if mean_sl_frac > 0 else 0

    print(
        f"  Mean SL distance   : {mean_sl_pips:.1f} pips  ({mean_sl_frac * 100:.3f}% of price)"
    )
    print(
        f"  Mean lot size      : {mean_units / STD_LOT:.3f} standard lots  ({mean_units:,.0f} units)"
    )
    print(
        f"  Risk per trade     : ${account_size * risk_per_trade:,.0f}  ({risk_per_trade * 100:.1f}% of ${account_size:,})"
    )
    print(
        f"  Notional per trade : ${mean_units * entry_price_approx:,.0f}  (margin: ${mean_units * entry_price_approx / leverage:,.0f})"
    )

    # ------------------------------------------------------------------
    # 4. Run vectorbt simulation (long + short simultaneously)
    # ------------------------------------------------------------------
    print("Running vectorbt backtest...")

    # init_cash = notional buying power (account_size * leverage) so vectorbt
    # can fill the full lot sizes. Real return is computed against account_size.
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=long_entries,
        exits=pd.Series(False, index=df.index),  # exits handled by SL/TP
        short_entries=short_entries,
        short_exits=pd.Series(False, index=df.index),
        sl_stop=long_sl.fillna(short_sl).fillna(0),
        tp_stop=long_tp.fillna(short_tp).fillna(0),
        init_cash=account_size * leverage,  # notional buying power
        size=lot_units_series,  # per-bar lot units
        size_type="amount",
        fees=FX_FEES,
        freq=FREQ,
        accumulate=False,
    )

    # Re-anchor equity to real margin capital
    abs_pnl = pf.value().iloc[-1] - (account_size * leverage)
    return_on_margin = (abs_pnl / account_size) * 100

    # ------------------------------------------------------------------
    # 5. Print stats
    # ------------------------------------------------------------------
    stats = pf.stats()

    # Patch stats so Start Value / End Value / Total Return reflect the
    # real margin account ($10,000) rather than the notional ($1,000,000)
    stats["Start Value"] = account_size
    stats["End Value"] = account_size + abs_pnl
    stats["Total Return [%]"] = return_on_margin

    print("\n" + "=" * 60)
    print(f" ICT UNICORN — {symbol} 1H BACKTEST RESULTS")
    print(f" Period   : {start_date} to {end_date}")
    print(
        f" Account  : ${account_size:,}  |  Leverage: {leverage}:1  |  Notional: ${account_size * leverage:,}"
    )
    print(
        f" Risk/trade: {risk_per_trade * 100:.1f}%  (${account_size * risk_per_trade:,.0f})  |  Fees: {FX_FEES * 10000:.1f} pips/side"
    )
    print(f" R:R min  : {min_rr}  |  SL buffer: {sl_buffer * 10000:.1f} pips")
    print("=" * 60)
    print(f" Abs P&L          : ${abs_pnl:+,.2f}")
    print(
        f" Return on Margin : {return_on_margin:+.2f}%  (on ${account_size:,} margin)"
    )
    print(f" Final Value      : ${account_size + abs_pnl:,.2f}")
    print(f" Sharpe Ratio     : {pf.sharpe_ratio():.2f}")
    print(f" Sortino Ratio    : {pf.sortino_ratio():.2f}")
    print(f" Max Drawdown     : {pf.max_drawdown() * 100:.2f}%")
    print(f" Total Trades     : {pf.trades.count()}")
    print(f" Win Rate         : {pf.trades.win_rate() * 100:.1f}%")
    print(f" Profit Factor    : {pf.trades.profit_factor():.2f}")
    trades_pnl = pf.trades.records_readable["PnL"]
    print(f" Avg Win          : ${trades_pnl.where(trades_pnl > 0).mean():.2f}")
    print(f" Avg Loss         : ${trades_pnl.where(trades_pnl < 0).mean():.2f}")
    print("=" * 60)

    # Separate long vs short breakdown
    trades_df = pf.trades.records_readable.copy()
    if "Direction" in trades_df.columns:
        for direction in trades_df["Direction"].unique():
            sub = trades_df[trades_df["Direction"] == direction]
            wins = (sub["PnL"] > 0).sum()
            total = len(sub)
            print(
                f" {direction:<10}: {total} trades | win rate {wins / total * 100:.1f}% | "
                f"total PnL ${sub['PnL'].sum():.2f}"
            )
    print("=" * 60)

    # ------------------------------------------------------------------
    # 6. Save outputs + MLflow logging
    # ------------------------------------------------------------------
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(__file__).resolve().parent / "reports" / f"{symbol}_{run_ts}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Start MLflow run (ended after all artifacts are saved below)
    mlflow.start_run(run_name=f"{symbol}_1H_{start_date}_{end_date}")
    mlflow.log_params(
        {
            "symbol": symbol,
            "asset_type": ASSET_TYPE,
            "start_date": start_date,
            "end_date": end_date,
            "account_size": account_size,
            "leverage": leverage,
            "risk_per_trade": risk_per_trade,
            "min_rr": min_rr,
            "sl_buffer": sl_buffer,
            "fx_fees": FX_FEES,
            "freq": FREQ,
        }
    )
    mlflow.log_metrics(
        {
            "abs_pnl": float(abs_pnl),
            "return_on_margin_pct": float(return_on_margin),
            "sharpe_ratio": float(pf.sharpe_ratio()),
            "sortino_ratio": float(pf.sortino_ratio()),
            "max_drawdown_pct": float(pf.max_drawdown() * 100),
            "total_trades": float(pf.trades.count()),
            "win_rate_pct": float(pf.trades.win_rate() * 100)
            if pf.trades.count() > 0
            else 0.0,
            "profit_factor": float(pf.trades.profit_factor())
            if pf.trades.count() > 0
            else 0.0,
            "n_long_entries": float(n_long),
            "n_short_entries": float(n_short),
        }
    )

    if save_outputs:
        # Text report
        report_path = reports_dir / "backtest_report.txt"
        with open(report_path, "w") as f:
            f.write(f"ICT Unicorn Model — {symbol} 1H Backtest\n")
            f.write(f"Period    : {start_date} to {end_date}\n")
            f.write(f"Account   : ${account_size:,}  |  Leverage: {leverage}:1\n")
            f.write(
                f"Risk/trade: {risk_per_trade * 100:.1f}%  (${account_size * risk_per_trade:,.0f})\n"
            )
            f.write(
                f"R:R min   : {min_rr}   SL buffer: {sl_buffer * 10000:.1f} pips\n\n"
            )
            f.write(str(stats))
            f.write(f"\n\nAbs P&L           : ${abs_pnl:+,.2f}\n")
            f.write(
                f"Return on Margin  : {return_on_margin:+.2f}%  (on ${account_size:,})\n"
            )
            f.write(f"Sharpe Ratio      : {pf.sharpe_ratio():.2f}\n")
            f.write(f"Max Drawdown      : {pf.max_drawdown() * 100:.2f}%\n")
            f.write(f"Win Rate          : {pf.trades.win_rate() * 100:.1f}%\n")
            f.write(f"Profit Factor     : {pf.trades.profit_factor():.2f}\n")
        print(f"\n  Report saved   -> {report_path}")

        # Trades CSV
        trades_path = reports_dir / "trades.csv"
        trades_df.to_csv(trades_path)
        print(f"  Trades saved   -> {trades_path}")

        # Native vectorbt 7-panel performance report
        # Disable FigureWidget (requires anywidget) — use plain Plotly Figure instead
        vbt.settings.plotting["use_widgets"] = False
        print("  Building native VBT report chart...")
        vbt_fig = pf.plot(
            subplots=[
                "value",  # equity curve
                "cum_returns",  # cumulative return %
                "underwater",  # drawdown filled area
                "drawdowns",  # top drawdown period bands
                "orders",  # buy/sell markers on price
                "trades",  # entry/exit lines per trade
                "net_exposure",  # net exposure over time
            ],
            make_subplots_kwargs=dict(
                rows=7,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.04,
                row_heights=[0.22, 0.12, 0.12, 0.14, 0.14, 0.14, 0.12],
            ),
            template="plotly_dark",
            title=f"ICT Unicorn — {symbol} 1H  |  VectorBT Performance Report",
        )
        vbt_chart_path = reports_dir / "vbt_report.html"
        vbt_fig.write_html(str(vbt_chart_path))
        print(f"  VBT report     -> {vbt_chart_path}")

        # Raw Portfolio object — reload later without re-running backtest
        pkl_path = reports_dir / "portfolio.pkl"
        pf.save(str(pkl_path))
        print(f"  Portfolio pkl  -> {pkl_path}")

    # ------------------------------------------------------------------
    # 7. Equity + drawdown chart
    # ------------------------------------------------------------------
    equity = pf.value()
    dd = (equity - equity.cummax()) / equity.cummax() * 100

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=(
            f"ICT Unicorn — {symbol} 1H  |  Equity Curve",
            "Drawdown %",
        ),
    )

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

    # Mark trade entries on equity curve
    if len(trades_df) > 0:
        entry_times = pd.to_datetime(trades_df["Entry Timestamp"].values)
        entry_equity = equity.reindex(entry_times, method="nearest").values
        win_mask = (trades_df["PnL"] > 0).values
        loss_mask = (trades_df["PnL"] <= 0).values

        if win_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=entry_times[win_mask],
                    y=entry_equity[win_mask],
                    mode="markers",
                    name="Winning Trade",
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
                    name="Losing Trade",
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

    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd,
            name="Drawdown",
            line=dict(color="#ef5350", width=1),
            fill="tozeroy",
            fillcolor="rgba(239,83,80,0.15)",
        ),
        row=2,
        col=1,
    )

    total_ret = pf.total_return() * 100
    win_rate = pf.trades.win_rate() * 100 if pf.trades.count() > 0 else 0.0

    fig.update_layout(
        height=700,
        template="plotly_dark",
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(
            text=(
                f"<b>ICT Unicorn — {symbol} 1H</b><br>"
                f"<sup>Return: {total_ret:.2f}%  |  "
                f"Sharpe: {pf.sharpe_ratio():.2f}  |  "
                f"Win Rate: {win_rate:.1f}%  |  "
                f"Trades: {pf.trades.count()}  |  "
                f"Max DD: {pf.max_drawdown() * 100:.2f}%</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
    )
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1)

    if save_outputs:
        chart_path = reports_dir / "equity.html"
        fig.write_html(str(chart_path))
        print(f"  Equity chart   -> {chart_path}")
        print(f"\n  All outputs in -> {reports_dir}")

        # Upload the entire run folder to MLflow as a single artifact directory
        mlflow.log_artifacts(str(reports_dir), artifact_path="outputs")

    mlflow.end_run()
    print(f"  MLflow run logged -> {MLFLOW_TRACKING_URI}")

    if show_chart:
        fig.show()

    return pf, df


if __name__ == "__main__":
    run_backtest()
