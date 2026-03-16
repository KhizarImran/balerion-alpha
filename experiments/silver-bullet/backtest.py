"""
ICT Silver Bullet Strategy — VectorBT Backtest + MLflow Logging

Loads strategy.py via importlib, converts setup dicts to per-bar vectorbt
signal arrays, runs the simulation, and logs params, metrics, and artifacts
to the MLflow server at http://localhost:5000.

Run:
    uv run python experiments/silver-bullet/backtest.py

MLflow must be running first:
    cd ../mlflow-server && docker compose up -d
"""

import os
import sys
import io
import importlib.util
import pickle
from datetime import datetime
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

os.environ["MLFLOW_ENABLE_PROXY_MLFLOW_ARTIFACTS"] = "true"
os.environ["SMC_CREDIT"] = "0"

import numpy as np
import pandas as pd
import vectorbt as vbt
import mlflow

from utils import build_equity_chart, build_analytics_chart

# ---------------------------------------------------------------------------
# Load strategy module via importlib (picks up any strategy.py edits)
# ---------------------------------------------------------------------------

_strategy_path = Path(__file__).resolve().parent / "strategy.py"
_spec = importlib.util.spec_from_file_location("strategy", _strategy_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

detect_setups = _mod.detect_setups
load_ohlcv = _mod.load_ohlcv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "US30"
START_DATE = "2023-11-11"
END_DATE = "2026-02-25"
TIMEFRAME = "1min"

# Risk management (optimised parameters)
RR_RATIO = 2.0
SL_BUFFER_PTS = 0.0
SESSION_CAP_USD = -5_500.0

# Signal parameters (optimised)
LIQUIDITY_WINDOW = 150
SWEEP_LOOKBACK = 50
PULLBACK_WINDOW = 10

# Account & sizing
ACCOUNT_SIZE = 100_000  # USD real margin capital
LEVERAGE = 100  # 100:1 — standard CFD leverage for indices
LOT_SIZE = 1  # lots per trade ($1 P&L per point per lot)

# Fee model — fixed dollar per round-trip, deducted post-simulation.
# Using a percentage fee in vbt (fees=FEES) applies it to the full notional
# (LOT_SIZE × price ≈ $440,000 per trade at US30 ~44,000), which overstates
# the cost by ~4× vs a realistic 1–3 pt spread.
# Instead: fees=0.0 in vbt, subtract FIXED_FEE_PER_TRADE × total_trades.
#   2-pt spread × 10 lots × $1/pt/lot = $20 round-trip
FIXED_FEE_PER_TRADE = 2.0  # USD per round-trip (2-pt spread × 10 lots)

# vectorbt frequency for Sharpe/Sortino annualisation
VBT_FREQ = "1min"

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "silver-bullet"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def setups_to_signal_arrays(
    df: pd.DataFrame,
    sell_setups: list,
    buy_setups: list,
) -> tuple:
    """
    Convert lists of setup dicts into per-bar pandas Series for vectorbt.

    Returns:
        long_entries   : boolean Series — True on long entry bars
        short_entries  : boolean Series — True on short entry bars
        long_sl        : float Series   — fractional SL (NaN elsewhere)
        long_tp        : float Series   — fractional TP (NaN elsewhere)
        short_sl       : float Series   — fractional SL (NaN elsewhere)
        short_tp       : float Series   — fractional TP (NaN elsewhere)
        lot_series     : float Series   — position size in units (LOT_SIZE on entry bars)
    """
    long_entries = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    long_sl = pd.Series(np.nan, index=df.index)
    long_tp = pd.Series(np.nan, index=df.index)
    short_sl = pd.Series(np.nan, index=df.index)
    short_tp = pd.Series(np.nan, index=df.index)
    lot_series = pd.Series(np.nan, index=df.index)

    for s in buy_setups:
        ts = s["entry_ts"]
        if ts not in df.index:
            continue
        entry = s["entry"]
        sl = s["sl"]
        tp = s["tp"]
        if sl >= entry:
            continue  # SL must be below entry for long
        sl_frac = (entry - sl) / entry
        tp_frac = (tp - entry) / entry
        if sl_frac <= 0 or tp_frac <= 0 or np.isnan(sl_frac) or np.isnan(tp_frac):
            continue
        long_entries[ts] = True
        long_sl[ts] = sl_frac
        long_tp[ts] = tp_frac
        lot_series[ts] = LOT_SIZE

    for s in sell_setups:
        ts = s["entry_ts"]
        if ts not in df.index:
            continue
        entry = s["entry"]
        sl = s["sl"]
        tp = s["tp"]
        if sl <= entry:
            continue  # SL must be above entry for short
        sl_frac = (sl - entry) / entry
        tp_frac = (entry - tp) / entry
        if sl_frac <= 0 or tp_frac <= 0 or np.isnan(sl_frac) or np.isnan(tp_frac):
            continue
        short_entries[ts] = True
        short_sl[ts] = sl_frac
        short_tp[ts] = tp_frac
        lot_series[ts] = LOT_SIZE

    return long_entries, short_entries, long_sl, long_tp, short_sl, short_tp, lot_series


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------


def run_backtest() -> None:
    print("=" * 70)
    print("ICT SILVER BULLET — VECTORBT BACKTEST")
    print("=" * 70)

    # -- Data --
    print(f"\nLoading {SYMBOL} data ({START_DATE} to {END_DATE})...")
    df, source_tz = load_ohlcv(SYMBOL, START_DATE, END_DATE, TIMEFRAME)
    print(
        f"  {len(df):,} rows  [{df.index.min()} -> {df.index.max()}]  (source_tz={source_tz})"
    )

    # -- Signals --
    print("\nDetecting setups...")
    sell_setups, buy_setups, _ = detect_setups(
        df,
        liquidity_window=LIQUIDITY_WINDOW,
        sweep_lookback=SWEEP_LOOKBACK,
        pullback_window=PULLBACK_WINDOW,
        rr_ratio=RR_RATIO,
        sl_buffer_pts=SL_BUFFER_PTS,
        session_cap_usd=SESSION_CAP_USD,
        lot_size=LOT_SIZE,
        source_tz=source_tz,
    )
    print(f"  Long  setups : {len(buy_setups)}")
    print(f"  Short setups : {len(sell_setups)}")

    # -- Convert to arrays --
    (
        long_entries,
        short_entries,
        long_sl,
        long_tp,
        short_sl,
        short_tp,
        lot_series,
    ) = setups_to_signal_arrays(df, sell_setups, buy_setups)

    # Combined SL/TP: merge long and short (non-overlapping by construction)
    combined_sl = long_sl.fillna(short_sl).fillna(0)
    combined_tp = long_tp.fillna(short_tp).fillna(0)
    combined_size = lot_series.fillna(LOT_SIZE)

    # -- Run vectorbt --
    print("\nRunning vectorbt simulation...")
    vbt.settings.plotting["use_widgets"] = False

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=long_entries,
        exits=pd.Series(False, index=df.index),  # exits via SL/TP only
        short_entries=short_entries,
        short_exits=pd.Series(False, index=df.index),
        sl_stop=combined_sl,
        tp_stop=combined_tp,
        stop_exit_price="StopMarket",  # fills at exact SL/TP level, not bar close
        init_cash=ACCOUNT_SIZE * LEVERAGE,
        size=combined_size,
        size_type="amount",
        fees=0.0,  # fees applied post-simulation as fixed dollar per trade
        freq=VBT_FREQ,
        accumulate=False,
    )

    # -- Metrics --
    stats = pf.stats()
    trades = pf.trades.records_readable

    total_trades = int(stats.get("Total Trades", 0))

    # Fixed-dollar fee deducted post-simulation (vbt ran with fees=0.0)
    total_fees_paid = FIXED_FEE_PER_TRADE * total_trades

    gross_pnl = pf.value().iloc[-1] - (ACCOUNT_SIZE * LEVERAGE)
    abs_pnl = gross_pnl - total_fees_paid
    return_on_margin = (abs_pnl / ACCOUNT_SIZE) * 100

    # Re-anchored drawdown: shift equity to real-margin scale then compute peak-to-trough
    # Also subtract accrued fees proportionally so the equity curve reflects real costs.
    # Simple approach: shift the terminal value; drawdown uses the gross curve (conservative).
    equity_real = pf.value() - (ACCOUNT_SIZE * LEVERAGE) + ACCOUNT_SIZE
    max_dd = float(
        (equity_real - equity_real.cummax()).div(equity_real.cummax()).min() * 100
    )

    sharpe = float(stats.get("Sharpe Ratio", float("nan")))
    sortino = float(stats.get("Sortino Ratio", float("nan")))
    win_rate = float(stats.get("Win Rate [%]", float("nan")))
    profit_factor = float(stats.get("Profit Factor", float("nan")))

    print(f"\n  Trades           : {total_trades}")
    print(
        f"  Total Fees Paid  : ${total_fees_paid:,.2f}  (${FIXED_FEE_PER_TRADE:.2f} x {total_trades} trades)"
    )
    print(f"  Gross P&L        : ${gross_pnl:,.2f}")
    print(f"  Net P&L          : ${abs_pnl:,.2f}")
    print(
        f"  Return on Margin : {return_on_margin:+.2f}%  (${ACCOUNT_SIZE:,} at {LEVERAGE}:1)"
    )
    print(f"  Win Rate         : {win_rate:.1f}%")
    print(f"  Sharpe           : {sharpe:.2f}")
    print(f"  Max Drawdown     : {max_dd:.2f}%")

    # -- Artifacts --
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(__file__).resolve().parent / "reports" / f"{SYMBOL}_{run_ts}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot — written as a standalone artifact so any MLflow run
    # can be reproduced exactly by reading this file.
    config_txt = reports_dir / "config.txt"
    with open(config_txt, "w", encoding="utf-8") as f:
        f.write("ICT Silver Bullet — Config Snapshot\n")
        f.write("=" * 60 + "\n\n")
        f.write("# --- Data ---\n")
        f.write(f"SYMBOL              = {SYMBOL!r}\n")
        f.write(f"START_DATE          = {START_DATE!r}\n")
        f.write(f"END_DATE            = {END_DATE!r}\n")
        f.write(f"TIMEFRAME           = {TIMEFRAME!r}\n")
        f.write("\n# --- Risk management ---\n")
        f.write(f"RR_RATIO            = {RR_RATIO}\n")
        f.write(f"SL_BUFFER_PTS       = {SL_BUFFER_PTS}\n")
        f.write(f"SESSION_CAP_USD     = {SESSION_CAP_USD}\n")
        f.write("\n# --- Signal parameters ---\n")
        f.write(f"LIQUIDITY_WINDOW    = {LIQUIDITY_WINDOW}\n")
        f.write(f"SWEEP_LOOKBACK      = {SWEEP_LOOKBACK}\n")
        f.write(f"PULLBACK_WINDOW     = {PULLBACK_WINDOW}\n")
        f.write("\n# --- Account & sizing ---\n")
        f.write(f"ACCOUNT_SIZE        = {ACCOUNT_SIZE}\n")
        f.write(f"LEVERAGE            = {LEVERAGE}\n")
        f.write(f"LOT_SIZE            = {LOT_SIZE}\n")
        f.write(f"FIXED_FEE_PER_TRADE = {FIXED_FEE_PER_TRADE}  # USD per round-trip\n")
        f.write("\n# --- vectorbt ---\n")
        f.write(f"VBT_FREQ            = {VBT_FREQ!r}\n")
        f.write(f"fees in vbt         = 0.0  (post-sim deduction used instead)\n")
        f.write("\n# --- MLflow ---\n")
        f.write(f"MLFLOW_TRACKING_URI = {MLFLOW_TRACKING_URI!r}\n")
        f.write(f"MLFLOW_EXPERIMENT   = {MLFLOW_EXPERIMENT!r}\n")

    # Text summary
    report_txt = reports_dir / "backtest_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("ICT Silver Bullet — Backtest Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Symbol          : {SYMBOL}\n")
        f.write(f"Period          : {START_DATE} to {END_DATE}\n")
        f.write(f"Account         : ${ACCOUNT_SIZE:,} at {LEVERAGE}:1 leverage\n")
        f.write(f"Lot Size        : {LOT_SIZE} lots\n")
        f.write(f"Fee/Trade       : ${FIXED_FEE_PER_TRADE:.2f} fixed (post-sim)\n")
        f.write(f"RR Ratio        : {RR_RATIO}\n")
        f.write(f"SL Buffer       : {SL_BUFFER_PTS} pts\n")
        f.write(f"Session Cap     : ${SESSION_CAP_USD:,.0f}\n")
        f.write(f"Liq Window      : {LIQUIDITY_WINDOW}\n")
        f.write(f"Sweep Lookback  : {SWEEP_LOOKBACK}\n")
        f.write(f"Pullback Window : {PULLBACK_WINDOW}\n")
        f.write("\n--- Results ---\n")
        f.write(f"Total Trades    : {total_trades}\n")
        f.write(f"Total Fees Paid : ${total_fees_paid:,.2f}\n")
        f.write(f"Gross P&L       : ${gross_pnl:,.2f}\n")
        f.write(f"Net P&L         : ${abs_pnl:,.2f}\n")
        f.write(f"Return on Margin: {return_on_margin:+.2f}%\n")
        f.write(f"Win Rate        : {win_rate:.1f}%\n")
        f.write(f"Sharpe Ratio    : {sharpe:.4f}\n")
        f.write(f"Sortino Ratio   : {sortino:.4f}\n")
        f.write(f"Max Drawdown    : {max_dd:.2f}%\n")
        f.write(f"Profit Factor   : {profit_factor:.4f}\n")

    # Trades CSV
    trades_csv = reports_dir / "trades.csv"
    trades.to_csv(trades_csv, index=False)

    # Equity + analytics charts
    equity_html = reports_dir / "equity.html"
    analytics_html = reports_dir / "analytics.html"

    eq_fig = build_equity_chart(pf, SYMBOL, ACCOUNT_SIZE, LEVERAGE)
    eq_fig.write_html(str(equity_html))

    an_fig = build_analytics_chart(pf, SYMBOL, ACCOUNT_SIZE, LEVERAGE)
    an_fig.write_html(str(analytics_html))

    # Native vectorbt report
    vbt_html = reports_dir / "vbt_report.html"
    try:
        pf.plot().write_html(str(vbt_html))
    except Exception as e:
        print(f"  (vbt native plot skipped: {e})")

    # Serialised portfolio (vectorbt portfolios can't always be pickled — skip silently)
    pkl_path = reports_dir / "portfolio.pkl"
    try:
        with open(pkl_path, "wb") as f:
            pickle.dump(pf, f)
    except Exception as e:
        print(f"  (portfolio.pkl skipped: {e})")

    print(f"\n  Artifacts saved -> {reports_dir}")

    # -- MLflow --
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"{SYMBOL}_{START_DATE}_{END_DATE}"):
        mlflow.log_params(
            {
                "symbol": SYMBOL,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "timeframe": TIMEFRAME,
                "account_size": ACCOUNT_SIZE,
                "leverage": LEVERAGE,
                "lot_size": LOT_SIZE,
                "fixed_fee_per_trade": FIXED_FEE_PER_TRADE,
                "rr_ratio": RR_RATIO,
                "sl_buffer_pts": SL_BUFFER_PTS,
                "session_cap_usd": SESSION_CAP_USD,
                "liquidity_window": LIQUIDITY_WINDOW,
                "sweep_lookback": SWEEP_LOOKBACK,
                "pullback_window": PULLBACK_WINDOW,
                "long_setups": len(buy_setups),
                "short_setups": len(sell_setups),
            }
        )
        mlflow.log_metrics(
            {
                "gross_pnl": float(gross_pnl),
                "total_fees_paid": float(total_fees_paid),
                "net_pnl": float(abs_pnl),
                "return_on_margin_pct": float(return_on_margin),
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "win_rate_pct": win_rate,
                "max_drawdown_pct": max_dd,
                "total_trades": float(total_trades),
                "profit_factor": profit_factor,
                "best_trade_pct": float(stats.get("Best Trade [%]", float("nan"))),
                "worst_trade_pct": float(stats.get("Worst Trade [%]", float("nan"))),
                "avg_winning_trade_pct": float(
                    stats.get("Avg Winning Trade [%]", float("nan"))
                ),
                "avg_losing_trade_pct": float(
                    stats.get("Avg Losing Trade [%]", float("nan"))
                ),
                "expectancy": float(stats.get("Expectancy", float("nan"))),
            }
        )
        mlflow.log_artifacts(str(reports_dir), artifact_path="outputs")

    print(f"  MLflow run logged -> {MLFLOW_TRACKING_URI}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_backtest()
