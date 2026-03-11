"""
Failed OB + FVG — Backtest
==========================

Pulls trade setups from strategy.py:detect_setups() and runs them through
vectorbt. Each setup provides a precise entry bar, SL price, and TP price
(2R). These are converted to per-bar fractional SL/TP arrays that vectorbt
uses to exit positions automatically.

Position sizing: risk-based — each trade risks exactly RISK_PER_TRADE % of
the account. SL distance in pips determines the lot size.

Outputs saved locally then uploaded to MLflow as artifacts.

Run:
    uv run python experiments/Unicorn-ICT/backtest.py
"""

import os
import sys
import io
from pathlib import Path
from datetime import datetime
import importlib.util

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Windows cp1252 fix
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ["SMC_CREDIT"] = "0"
os.environ["MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS"] = "true"

import numpy as np
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import mlflow

from utils import DataLoader

# ── load strategy module ───────────────────────────────────────────────────────
_strategy_path = Path(__file__).resolve().parent / "strategy.py"
_spec = importlib.util.spec_from_file_location("strategy", _strategy_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
detect_setups = _mod.detect_setups
load_ohlcv = _mod.load_ohlcv

# ── MLflow ─────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("unicorn-ict")

# ── config ─────────────────────────────────────────────────────────────────────
SYMBOL = "EURUSD"
START_DATE = "2025-11-18"
END_DATE = "2026-02-27"
TIMEFRAME = "1h"
SWING_LENGTH = 10

ACCOUNT_SIZE = 10_000  # USD — real margin capital
LEVERAGE = 100  # 1:100
RISK_PER_TRADE = 0.01  # 1% per trade

PIP = 0.0001
PIP_VALUE_PER_LOT = 10.0  # USD per pip per standard lot (EURUSD, USD account)
STD_LOT = 100_000
FX_FEES = 0.00007  # ~0.7 pip spread per side


# ── helpers ────────────────────────────────────────────────────────────────────
def setups_to_signal_arrays(
    df: pd.DataFrame,
    sell_setups: list,
    buy_setups: list,
) -> tuple:
    """
    Convert setup dicts to per-bar arrays for vectorbt.

    Returns
    -------
    short_entries, long_entries : boolean Series
    short_sl, short_tp         : fractional Series (NaN on non-signal bars)
    long_sl,  long_tp          : fractional Series
    lot_units                  : float Series (position size in base currency units)
    """
    n = len(df)
    idx = df.index
    closes = df["close"].to_numpy()

    short_entries = pd.Series(False, index=idx)
    long_entries = pd.Series(False, index=idx)
    short_sl_arr = pd.Series(np.nan, index=idx)
    short_tp_arr = pd.Series(np.nan, index=idx)
    long_sl_arr = pd.Series(np.nan, index=idx)
    long_tp_arr = pd.Series(np.nan, index=idx)
    lot_units = pd.Series(0.0, index=idx)

    def _frac(price_dist, entry):
        return abs(price_dist) / entry if entry > 0 else np.nan

    def _lot_size(sl_frac, entry):
        if sl_frac <= 0 or entry <= 0:
            return 0.0
        risk_usd = ACCOUNT_SIZE * RISK_PER_TRADE
        sl_pips = (sl_frac * entry) / PIP
        lots = risk_usd / (sl_pips * PIP_VALUE_PER_LOT)
        return lots * STD_LOT

    # SELL setups
    for s in sell_setups:
        ts = s["entry_ts"]
        entry = s["entry"]
        sl = s["sl"]
        tp = s["tp"]

        if ts not in idx:
            continue
        if sl <= entry:  # SL must be above entry for a short
            continue

        sl_frac = _frac(sl - entry, entry)
        tp_frac = _frac(entry - tp, entry)

        if np.isnan(sl_frac) or np.isnan(tp_frac) or sl_frac <= 0 or tp_frac <= 0:
            continue

        short_entries.loc[ts] = True
        short_sl_arr.loc[ts] = sl_frac
        short_tp_arr.loc[ts] = tp_frac
        lot_units.loc[ts] = _lot_size(sl_frac, entry)

    # BUY setups
    for s in buy_setups:
        ts = s["entry_ts"]
        entry = s["entry"]
        sl = s["sl"]
        tp = s["tp"]

        if ts not in idx:
            continue
        if sl >= entry:  # SL must be below entry for a long
            continue

        sl_frac = _frac(entry - sl, entry)
        tp_frac = _frac(tp - entry, entry)

        if np.isnan(sl_frac) or np.isnan(tp_frac) or sl_frac <= 0 or tp_frac <= 0:
            continue

        long_entries.loc[ts] = True
        long_sl_arr.loc[ts] = sl_frac
        long_tp_arr.loc[ts] = tp_frac
        lot_units.loc[ts] = _lot_size(sl_frac, entry)

    return (
        short_entries,
        long_entries,
        short_sl_arr,
        short_tp_arr,
        long_sl_arr,
        long_tp_arr,
        lot_units,
    )


def build_equity_chart(pf, symbol, account_size):
    # equity is on notional scale (account_size * LEVERAGE); re-anchor to real margin
    equity_notional = pf.value()
    init_notional = account_size * LEVERAGE
    abs_pnl = equity_notional.iloc[-1] - init_notional

    # shift equity to real dollar terms: start at account_size
    equity = equity_notional - init_notional + account_size
    dd = (equity - equity.cummax()) / equity.cummax() * 100
    trades_df = pf.trades.records_readable.copy()

    # y-axis padding: 10% of the total equity range, minimum $200
    eq_range = equity.max() - equity.min()
    pad = max(eq_range * 0.10, 200)
    y_min = equity.min() - pad
    y_max = equity.max() + pad

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.65, 0.35],
        subplot_titles=(f"{symbol} — Equity Curve", "Drawdown %"),
    )

    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity,
            name="Equity",
            line=dict(color="#26a69a", width=2),
            fill="tonexty",
            fillcolor="rgba(38,166,154,0.08)",
        ),
        row=1,
        col=1,
    )

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

    rom = (abs_pnl / account_size) * 100
    fig.update_layout(
        height=700,
        template="plotly_dark",
        hovermode="x unified",
        title=dict(
            text=(
                f"<b>{symbol} 1H — Failed OB + FVG</b><br>"
                f"<sup>Return on Margin: {rom:.2f}%  |  "
                f"Sharpe: {pf.sharpe_ratio():.2f}  |  "
                f"Win Rate: {pf.trades.win_rate() * 100:.1f}%  |  "
                f"Trades: {pf.trades.count()}  |  "
                f"Max DD: {pf.max_drawdown() * 100:.2f}%</sup>"
            ),
            x=0.5,
            xanchor="center",
        ),
    )
    fig.update_yaxes(
        title_text="Portfolio Value ($)", row=1, col=1, range=[y_min, y_max]
    )
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)
    return fig


# ── main ───────────────────────────────────────────────────────────────────────
def run_backtest():
    # 1. Load data + detect setups
    df = load_ohlcv(SYMBOL, START_DATE, END_DATE, TIMEFRAME)
    sell_setups, buy_setups = detect_setups(df, swing_length=SWING_LENGTH)

    n_sell = len(sell_setups)
    n_buy = len(buy_setups)
    print(f"Total setups — SELL: {n_sell}  BUY: {n_buy}")

    if n_sell + n_buy == 0:
        print("No setups found. Exiting.")
        return

    # 2. Convert to vectorbt arrays
    (
        short_entries,
        long_entries,
        short_sl,
        short_tp,
        long_sl,
        long_tp,
        lot_units,
    ) = setups_to_signal_arrays(df, sell_setups, buy_setups)

    n_short = int(short_entries.sum())
    n_long = int(long_entries.sum())
    print(f"Valid vectorbt signals — short: {n_short}  long: {n_long}")

    if n_short + n_long == 0:
        print("No valid signals after SL/TP validation. Exiting.")
        return

    # 3. Run vectorbt
    # init_cash = notional buying power; real P&L re-anchored to ACCOUNT_SIZE
    print("Running vectorbt backtest...")
    vbt.settings.plotting["use_widgets"] = False

    combined_sl = long_sl.fillna(short_sl).fillna(0)
    combined_tp = long_tp.fillna(short_tp).fillna(0)

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=long_entries,
        exits=pd.Series(False, index=df.index),
        short_entries=short_entries,
        short_exits=pd.Series(False, index=df.index),
        sl_stop=combined_sl,
        tp_stop=combined_tp,
        init_cash=ACCOUNT_SIZE * LEVERAGE,
        size=lot_units,
        size_type="amount",
        fees=FX_FEES,
        freq=TIMEFRAME,
        accumulate=False,
    )

    # 4. Compute metrics
    abs_pnl = pf.value().iloc[-1] - (ACCOUNT_SIZE * LEVERAGE)
    return_on_margin = (abs_pnl / ACCOUNT_SIZE) * 100
    sharpe = pf.sharpe_ratio()
    sortino = pf.sortino_ratio()
    max_dd = pf.max_drawdown() * 100
    total_trades = pf.trades.count()
    win_rate = pf.trades.win_rate() * 100 if total_trades > 0 else 0.0
    profit_factor = pf.trades.profit_factor() if total_trades > 0 else 0.0

    print()
    print("=" * 58)
    print(f"  FAILED OB + FVG — {SYMBOL} 1H")
    print(f"  Period : {START_DATE} to {END_DATE}")
    print(f"  Account: ${ACCOUNT_SIZE:,}  |  Leverage: {LEVERAGE}:1")
    print(f"  Risk   : {RISK_PER_TRADE * 100:.1f}% per trade")
    print("=" * 58)
    print(f"  Abs P&L           : ${abs_pnl:+,.2f}")
    print(f"  Return on Margin  : {return_on_margin:+.2f}%")
    print(f"  Sharpe Ratio      : {sharpe:.2f}")
    print(f"  Sortino Ratio     : {sortino:.2f}")
    print(f"  Max Drawdown      : {max_dd:.2f}%")
    print(f"  Total Trades      : {total_trades}")
    print(f"  Win Rate          : {win_rate:.1f}%")
    print(f"  Profit Factor     : {profit_factor:.2f}")
    print("=" * 58)

    # 5. Save artifacts locally
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(__file__).resolve().parent / "reports" / f"{SYMBOL}_{run_ts}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # text report
    report_path = reports_dir / "backtest_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Failed OB + FVG — {SYMBOL} 1H\n")
        f.write(f"Period    : {START_DATE} to {END_DATE}\n")
        f.write(f"Account   : ${ACCOUNT_SIZE:,}  Leverage: {LEVERAGE}:1\n")
        f.write(f"Risk/trade: {RISK_PER_TRADE * 100:.1f}%\n")
        f.write(f"Swing len : {SWING_LENGTH}\n\n")
        f.write(str(pf.stats()))
        f.write(f"\n\nAbs P&L          : ${abs_pnl:+,.2f}\n")
        f.write(f"Return on Margin : {return_on_margin:+.2f}%\n")
        f.write(f"Sharpe           : {sharpe:.2f}\n")
        f.write(f"Max Drawdown     : {max_dd:.2f}%\n")
        f.write(f"Win Rate         : {win_rate:.1f}%\n")
        f.write(f"Profit Factor    : {profit_factor:.2f}\n")
    print(f"  Report  -> {report_path}")

    # trades CSV
    trades_df = pf.trades.records_readable.copy()
    trades_path = reports_dir / "trades.csv"
    trades_df.to_csv(trades_path)
    print(f"  Trades  -> {trades_path}")

    # equity chart
    eq_fig = build_equity_chart(pf, SYMBOL, ACCOUNT_SIZE)
    equity_path = reports_dir / "equity.html"
    eq_fig.write_html(str(equity_path))
    print(f"  Equity  -> {equity_path}")

    # native vbt report
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
            rows=7,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.22, 0.12, 0.12, 0.14, 0.14, 0.14, 0.12],
        ),
        template="plotly_dark",
        title=f"Failed OB + FVG — {SYMBOL} 1H",
    )
    vbt_path = reports_dir / "vbt_report.html"
    vbt_fig.write_html(str(vbt_path))
    print(f"  VBT     -> {vbt_path}")

    # portfolio pkl
    pkl_path = reports_dir / "portfolio.pkl"
    pf.save(str(pkl_path))
    print(f"  PKL     -> {pkl_path}")

    # 6. Log to MLflow
    with mlflow.start_run(run_name=f"{SYMBOL}_1H_{START_DATE}_{END_DATE}"):
        mlflow.log_params(
            {
                "symbol": SYMBOL,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "timeframe": TIMEFRAME,
                "swing_length": SWING_LENGTH,
                "account_size": ACCOUNT_SIZE,
                "leverage": LEVERAGE,
                "risk_per_trade": RISK_PER_TRADE,
                "fx_fees": FX_FEES,
                "n_sell_setups": n_sell,
                "n_buy_setups": n_buy,
            }
        )
        mlflow.log_metrics(
            {
                "abs_pnl": float(abs_pnl),
                "return_on_margin_pct": float(return_on_margin),
                "sharpe_ratio": float(sharpe),
                "sortino_ratio": float(sortino),
                "max_drawdown_pct": float(max_dd),
                "total_trades": float(total_trades),
                "win_rate_pct": float(win_rate),
                "profit_factor": float(profit_factor),
                "n_short_signals": float(n_short),
                "n_long_signals": float(n_long),
            }
        )
        mlflow.log_artifacts(str(reports_dir), artifact_path="outputs")

    print(f"  MLflow  -> {MLFLOW_TRACKING_URI}")
    print(f"\n  All outputs in -> {reports_dir}")


if __name__ == "__main__":
    run_backtest()
