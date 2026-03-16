"""
TrendBO — VectorBT Backtest + MLflow Logging

Intraday trend-following breakout strategy on USDJPY 1H:
  - EMA 50 / 200 determines trend direction (long-only in uptrend, short-only in downtrend)
  - Consolidation detected via ADX < 25 + compressed rolling range / ATR
  - Entry on first bar that closes beyond the consolidation range in the trend direction
  - SL at the opposite side of the consolidation zone
  - TP at RISK_REWARD * SL distance
  - No forced time exit — SL/TP handles all exits

Risk management:
  - 1% of account risked per trade
  - Lot size computed from SL distance so each SL hit = exactly 1% loss
  - Fees: fixed $8 per round-trip (conservative USDJPY retail spread estimate)

Outputs saved to experiments/TrendBO/reports/{SYMBOL}_{YYYYMMDD_HHMMSS}/:
  - backtest_report.txt
  - trades.csv
  - equity.html            (from utils.build_equity_chart)
  - analytics.html         (from utils.build_analytics_chart)
  - vbt_report.html        (native vectorbt plot)
  - portfolio.pkl

All outputs pushed to MLflow experiment "trendbo".

Conforms to STRATEGY_BACKTEST_PATTERN.md.

Usage:
    uv run python experiments/TrendBO/backtest.py
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
# Load strategy module via importlib
# ---------------------------------------------------------------------------

_strategy_path = Path(__file__).resolve().parent / "strategy.py"
_spec = importlib.util.spec_from_file_location("trendbo_strategy", _strategy_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

detect_setups = _mod.detect_setups
load_ohlcv = _mod.load_ohlcv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "USDJPY"
START_DATE = "2020-01-01"
END_DATE = "2026-03-05"
TIMEFRAME = "1h"

# Strategy parameters (passed to detect_setups to override strategy.py defaults)
RISK_REWARD = 2.0
EMA_FAST = 50
EMA_SLOW = 200
ADX_THRESHOLD = 25
RANGE_ATR_MULT = 4.0
CONSOL_MIN_BARS = 5
MIN_CONSOLIDATION_PIPS = 10
MIN_SL_PIPS = 10
SESSION_START_HOUR = 7
SESSION_END_HOUR = 20

# Account & sizing
ACCOUNT_SIZE = 10_000  # real margin capital (USD)
LEVERAGE = 100  # 1:100
RISK_PER_TRADE = 0.01  # 1% of account per trade

# USDJPY instrument constants
PIP = 0.01  # pip size for JPY pairs
PIP_VALUE_PER_LOT = 9.5  # approx USD per pip on 1 standard lot at ~150 USD/JPY
STD_LOT = 100_000  # units per standard lot

# For JPY-quoted pairs vectorbt computes P&L as:
#   units * (exit_price - entry_price)
# which yields a result in JPY, not USD, because the quote currency is JPY.
# Dividing units by the entry price (USDJPY rate) corrects this so that
# vectorbt's internal P&L is expressed in USD.
_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"}
_IS_JPY_PAIR = SYMBOL in _JPY_PAIRS

# Fee model: fixed cost per round-trip trade.
# 1 pip spread at USDJPY ~150, 0.08 avg lots -> ~$0.76; using $8 as conservative retail estimate.
FX_FEES = 0.0  # disabled in vectorbt — subtract fixed costs manually post-simulation
FIXED_FEE_PER_TRADE = 8.0  # USD per round-trip (entry + exit)

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "trendbo"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_lot_units(sl_pips: float) -> float:
    """
    Size a trade so hitting SL loses exactly RISK_PER_TRADE * ACCOUNT_SIZE.

        risk_$ = ACCOUNT_SIZE * RISK_PER_TRADE
        lots   = risk_$ / (sl_pips * PIP_VALUE_PER_LOT)
        units  = lots * STD_LOT
    """
    if sl_pips <= 0:
        return 0.0
    risk_dollars = ACCOUNT_SIZE * RISK_PER_TRADE
    lots = risk_dollars / (sl_pips * PIP_VALUE_PER_LOT)
    return lots * STD_LOT


def setups_to_signal_arrays(
    df: pd.DataFrame,
    sell_setups: list,
    buy_setups: list,
) -> tuple:
    """
    Convert setup dicts into per-bar pandas Series for vectorbt.

    Returns
    -------
    long_entries   : bool Series
    short_entries  : bool Series
    long_sl        : float Series — fractional SL distance (NaN on non-entry bars)
    long_tp        : float Series — fractional TP distance (NaN on non-entry bars)
    short_sl       : float Series
    short_tp       : float Series
    size_series    : float Series — lot units on entry bars (0.0 elsewhere)
    """
    long_entries = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    long_sl = pd.Series(np.nan, index=df.index)
    long_tp = pd.Series(np.nan, index=df.index)
    short_sl = pd.Series(np.nan, index=df.index)
    short_tp = pd.Series(np.nan, index=df.index)
    size_series = pd.Series(np.nan, index=df.index)

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
        sl_pips = (entry - sl) / PIP
        sl_pips = max(sl_pips, MIN_SL_PIPS)
        units = _compute_lot_units(sl_pips)
        # Quote currency adjustment for JPY pairs.
        # vectorbt computes P&L as: units * (exit_price - entry_price)
        # For USDJPY this gives JPY, not USD. Dividing units by entry_price
        # converts the notional so that the P&L vectorbt produces is in USD.
        if _IS_JPY_PAIR:
            units = units / entry
        # Cap notional: units * entry must not exceed ACCOUNT_SIZE * LEVERAGE
        # so vectorbt never rejects the order for insufficient cash.
        max_units = (ACCOUNT_SIZE * LEVERAGE) / entry
        units = min(units, max_units)
        if units <= 0:
            continue
        long_entries[ts] = True
        long_sl[ts] = sl_frac
        long_tp[ts] = tp_frac
        size_series[ts] = units

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
        sl_pips = (sl - entry) / PIP
        sl_pips = max(sl_pips, MIN_SL_PIPS)
        units = _compute_lot_units(sl_pips)
        # Quote currency adjustment for JPY pairs.
        if _IS_JPY_PAIR:
            units = units / entry
        # Cap notional: units * entry must not exceed ACCOUNT_SIZE * LEVERAGE.
        max_units = (ACCOUNT_SIZE * LEVERAGE) / entry
        units = min(units, max_units)
        if units <= 0:
            continue
        short_entries[ts] = True
        short_sl[ts] = sl_frac
        short_tp[ts] = tp_frac
        size_series[ts] = units

    return (
        long_entries,
        short_entries,
        long_sl,
        long_tp,
        short_sl,
        short_tp,
        size_series,
    )


# ---------------------------------------------------------------------------
# Main backtest
# ---------------------------------------------------------------------------


def run_backtest() -> None:
    print("=" * 70)
    print("TRENDBO — VECTORBT BACKTEST")
    print("=" * 70)

    # -- Data --
    print(f"\nLoading {SYMBOL} {TIMEFRAME} data ({START_DATE} to {END_DATE})...")
    df = load_ohlcv(SYMBOL, START_DATE, END_DATE, TIMEFRAME)
    print(f"  {len(df):,} bars  [{df.index.min()} -> {df.index.max()}]")

    # -- Signals --
    print("\nDetecting setups...")
    sell_setups, buy_setups, extra = detect_setups(
        df,
        risk_reward=RISK_REWARD,
        ema_fast=EMA_FAST,
        ema_slow=EMA_SLOW,
        adx_threshold=ADX_THRESHOLD,
        range_atr_mult=RANGE_ATR_MULT,
        consol_min_bars=CONSOL_MIN_BARS,
        min_consolidation_pips=MIN_CONSOLIDATION_PIPS,
        min_sl_pips=MIN_SL_PIPS,
        session_start=SESSION_START_HOUR,
        session_end=SESSION_END_HOUR,
    )
    print(f"  Long  setups : {len(buy_setups)}")
    print(f"  Short setups : {len(sell_setups)}")

    if len(buy_setups) + len(sell_setups) == 0:
        print("No valid setups detected. Exiting.")
        return

    # -- Convert to arrays --
    (
        long_entries,
        short_entries,
        long_sl,
        long_tp,
        short_sl,
        short_tp,
        size_series,
    ) = setups_to_signal_arrays(df, sell_setups, buy_setups)

    n_long_vbt = int(long_entries.sum())
    n_short_vbt = int(short_entries.sum())
    print(f"  After validation: {n_long_vbt} long, {n_short_vbt} short")

    # Merged SL/TP arrays.
    # Leave NaN on non-entry bars — vectorbt treats NaN as "no stop on this bar".
    # fillna(0) would set sl_stop=0 on every bar, triggering phantom stop-outs.
    combined_sl = long_sl.fillna(short_sl)
    combined_tp = long_tp.fillna(short_tp)
    combined_size = size_series.fillna(0)

    mean_sl_pips = combined_sl[combined_sl > 0].mul(df["close"]).div(PIP).mean()
    mean_units = combined_size[combined_size > 0].mean()
    print(f"  Mean SL distance : {mean_sl_pips:.1f} pips")
    print(
        f"  Mean lot size    : {mean_units / STD_LOT:.4f} lots  ({mean_units:,.0f} units)"
    )
    print(
        f"  Risk per trade   : ${ACCOUNT_SIZE * RISK_PER_TRADE:,.0f}  "
        f"({RISK_PER_TRADE * 100:.1f}% of ${ACCOUNT_SIZE:,})"
    )

    # -- Run vectorbt --
    print("\nRunning vectorbt simulation...")
    vbt.settings.plotting["use_widgets"] = False

    pf = vbt.Portfolio.from_signals(
        close=df["close"],
        entries=long_entries,
        exits=pd.Series(False, index=df.index),  # SL/TP handles all exits
        short_entries=short_entries,
        short_exits=pd.Series(False, index=df.index),
        sl_stop=combined_sl,
        tp_stop=combined_tp,
        stop_exit_price="StopMarket",  # fills at exact SL/TP level
        init_cash=ACCOUNT_SIZE,  # real margin capital — lot sizes are capped
        # so notional never exceeds ACCOUNT_SIZE*LEVERAGE,
        # so vbt never rejects an order for insufficient cash.
        size=combined_size,
        size_type="amount",
        fees=FX_FEES,  # 0.0 — fees subtracted manually post-simulation
        freq=TIMEFRAME,
        accumulate=False,
    )

    # -- Metrics --
    # init_cash=ACCOUNT_SIZE so all vbt internal % stats are computed against $10,000.
    # No re-anchoring needed — pf.value() already starts at ACCOUNT_SIZE.
    stats = pf.stats()
    trades = pf.trades.records_readable

    gross_pnl = pf.value().iloc[-1] - ACCOUNT_SIZE
    total_trades = int(stats.get("Total Trades", 0))
    total_fees = total_trades * FIXED_FEE_PER_TRADE
    abs_pnl = gross_pnl - total_fees
    return_on_margin = (abs_pnl / ACCOUNT_SIZE) * 100

    # Max drawdown directly from vbt — computed against ACCOUNT_SIZE base.
    max_dd = float(pf.max_drawdown() * 100)

    sharpe = float(stats.get("Sharpe Ratio", float("nan")))
    sortino = float(stats.get("Sortino Ratio", float("nan")))
    win_rate = float(stats.get("Win Rate [%]", float("nan")))
    profit_factor = float(stats.get("Profit Factor", float("nan")))

    print(f"\n  Trades           : {total_trades}")
    print(f"  Gross P&L        : ${gross_pnl:,.2f}")
    print(f"  Total fees       : ${total_fees:,.2f} (${FIXED_FEE_PER_TRADE}/trade)")
    print(f"  Net P&L          : ${abs_pnl:,.2f}")
    print(
        f"  Return on Margin : {return_on_margin:+.2f}%  (${ACCOUNT_SIZE:,} at {LEVERAGE}:1)"
    )
    print(f"  Win Rate         : {win_rate:.1f}%")
    print(f"  Sharpe           : {sharpe:.2f}")
    print(f"  Sortino          : {sortino:.2f}")
    print(f"  Max Drawdown     : {max_dd:.2f}%  (real margin)")
    print(f"  Profit Factor    : {profit_factor:.3f}")

    # Direction breakdown
    if "Direction" in trades.columns:
        for direction in trades["Direction"].unique():
            sub = trades[trades["Direction"] == direction]
            wins = (sub["PnL"] > 0).sum()
            tot = len(sub)
            wr = wins / tot * 100 if tot > 0 else 0.0
            dir_fees = tot * FIXED_FEE_PER_TRADE
            dir_net_pnl = sub["PnL"].sum() - dir_fees
            print(
                f"  {direction:<8} : {tot} trades  |  win rate {wr:.1f}%  |  "
                f"net PnL ${dir_net_pnl:.2f}"
            )

    # -- Artifacts --
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(__file__).resolve().parent / "reports" / f"{SYMBOL}_{run_ts}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot
    config_txt = reports_dir / "config.txt"
    with open(config_txt, "w", encoding="utf-8") as f:
        f.write("TrendBO — Config Snapshot\n")
        f.write("=" * 60 + "\n\n")
        f.write("# --- Data ---\n")
        f.write(f"SYMBOL                  = {SYMBOL!r}\n")
        f.write(f"START_DATE              = {START_DATE!r}\n")
        f.write(f"END_DATE                = {END_DATE!r}\n")
        f.write(f"TIMEFRAME               = {TIMEFRAME!r}\n")
        f.write("\n# --- Strategy parameters ---\n")
        f.write(f"RISK_REWARD             = {RISK_REWARD}\n")
        f.write(f"EMA_FAST                = {EMA_FAST}\n")
        f.write(f"EMA_SLOW                = {EMA_SLOW}\n")
        f.write(f"ADX_THRESHOLD           = {ADX_THRESHOLD}\n")
        f.write(f"RANGE_ATR_MULT          = {RANGE_ATR_MULT}\n")
        f.write(f"CONSOL_MIN_BARS         = {CONSOL_MIN_BARS}\n")
        f.write(f"MIN_CONSOLIDATION_PIPS  = {MIN_CONSOLIDATION_PIPS}\n")
        f.write(f"MIN_SL_PIPS             = {MIN_SL_PIPS}\n")
        f.write(f"SESSION_START_HOUR      = {SESSION_START_HOUR}\n")
        f.write(f"SESSION_END_HOUR        = {SESSION_END_HOUR}\n")
        f.write("\n# --- Account & sizing ---\n")
        f.write(f"ACCOUNT_SIZE            = {ACCOUNT_SIZE}\n")
        f.write(f"LEVERAGE                = {LEVERAGE}\n")
        f.write(f"RISK_PER_TRADE          = {RISK_PER_TRADE}\n")
        f.write(
            f"FIXED_FEE_PER_TRADE     = {FIXED_FEE_PER_TRADE}  # USD per round-trip\n"
        )
        f.write("\n# --- Instrument ---\n")
        f.write(f"PIP                     = {PIP}\n")
        f.write(f"PIP_VALUE_PER_LOT       = {PIP_VALUE_PER_LOT}\n")
        f.write(f"STD_LOT                 = {STD_LOT}\n")
        f.write(f"_IS_JPY_PAIR            = {_IS_JPY_PAIR}\n")
        f.write("\n# --- vectorbt ---\n")
        f.write(f"fees in vbt             = 0.0  (post-sim deduction used instead)\n")
        f.write("\n# --- MLflow ---\n")
        f.write(f"MLFLOW_TRACKING_URI     = {MLFLOW_TRACKING_URI!r}\n")
        f.write(f"MLFLOW_EXPERIMENT       = {MLFLOW_EXPERIMENT!r}\n")

    # Text report
    report_txt = reports_dir / "backtest_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("TrendBO — Backtest Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Symbol              : {SYMBOL}\n")
        f.write(f"Period              : {START_DATE} to {END_DATE}\n")
        f.write(f"Timeframe           : {TIMEFRAME}\n")
        f.write(f"Account             : ${ACCOUNT_SIZE:,} at {LEVERAGE}:1 leverage\n")
        f.write(
            f"Risk/trade          : {RISK_PER_TRADE * 100:.1f}%  (${ACCOUNT_SIZE * RISK_PER_TRADE:,.0f})\n"
        )
        f.write(
            f"Fixed fee           : ${FIXED_FEE_PER_TRADE} per trade (round-trip)\n"
        )
        f.write(f"RR Ratio            : {RISK_REWARD}\n")
        f.write(f"EMA fast / slow     : {EMA_FAST} / {EMA_SLOW}\n")
        f.write(f"ADX threshold       : {ADX_THRESHOLD}\n")
        f.write(f"Range/ATR mult      : {RANGE_ATR_MULT}\n")
        f.write(f"Consol min bars     : {CONSOL_MIN_BARS}\n")
        f.write(f"Min consol pips     : {MIN_CONSOLIDATION_PIPS}\n")
        f.write(f"Min SL pips         : {MIN_SL_PIPS}\n")
        f.write(
            f"Session (UTC)       : {SESSION_START_HOUR:02d}:00 - {SESSION_END_HOUR:02d}:00\n"
        )
        f.write("\n--- Results ---\n")
        f.write(f"Total Trades        : {total_trades}\n")
        f.write(f"Long setups         : {len(buy_setups)}\n")
        f.write(f"Short setups        : {len(sell_setups)}\n")
        f.write(f"Gross P&L           : ${gross_pnl:,.2f}\n")
        f.write(f"Total fees          : ${total_fees:,.2f}\n")
        f.write(f"Net P&L             : ${abs_pnl:,.2f}\n")
        f.write(f"Return on Margin    : {return_on_margin:+.2f}%\n")
        f.write(f"Win Rate            : {win_rate:.1f}%\n")
        f.write(f"Sharpe Ratio        : {sharpe:.4f}\n")
        f.write(f"Sortino Ratio       : {sortino:.4f}\n")
        f.write(f"Max Drawdown        : {max_dd:.2f}%\n")
        f.write(f"Profit Factor       : {profit_factor:.4f}\n")
        f.write("\n--- Full VBT Stats ---\n")
        f.write(str(stats))

    # Trades CSV
    trades_csv = reports_dir / "trades.csv"
    trades.to_csv(trades_csv, index=False)

    # Equity + analytics charts
    equity_html = reports_dir / "equity.html"
    analytics_html = reports_dir / "analytics.html"

    # leverage=1 because init_cash=ACCOUNT_SIZE — no re-anchoring needed in chart utils.
    eq_fig = build_equity_chart(pf, SYMBOL, ACCOUNT_SIZE, 1)
    eq_fig.write_html(str(equity_html))

    an_fig = build_analytics_chart(pf, SYMBOL, ACCOUNT_SIZE, 1)
    an_fig.write_html(str(analytics_html))

    # Native vectorbt report
    vbt_html = reports_dir / "vbt_report.html"
    try:
        pf.plot().write_html(str(vbt_html))
    except Exception as e:
        print(f"  (vbt native plot skipped: {e})")

    # Serialised portfolio
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

    with mlflow.start_run(run_name=f"{SYMBOL}_{TIMEFRAME}_{START_DATE}_{END_DATE}"):
        mlflow.log_params(
            {
                "symbol": SYMBOL,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "timeframe": TIMEFRAME,
                "account_size": ACCOUNT_SIZE,
                "leverage": LEVERAGE,
                "risk_per_trade": RISK_PER_TRADE,
                "fixed_fee_per_trade": FIXED_FEE_PER_TRADE,
                "rr_ratio": RISK_REWARD,
                "ema_fast": EMA_FAST,
                "ema_slow": EMA_SLOW,
                "adx_threshold": ADX_THRESHOLD,
                "range_atr_mult": RANGE_ATR_MULT,
                "consol_min_bars": CONSOL_MIN_BARS,
                "min_consolidation_pips": MIN_CONSOLIDATION_PIPS,
                "min_sl_pips": MIN_SL_PIPS,
                "session_start_utc": SESSION_START_HOUR,
                "session_end_utc": SESSION_END_HOUR,
                "long_setups": len(buy_setups),
                "short_setups": len(sell_setups),
            }
        )
        mlflow.log_metrics(
            {
                "gross_pnl": float(gross_pnl),
                "total_fees": float(total_fees),
                "abs_pnl": float(abs_pnl),
                "return_on_margin_pct": float(return_on_margin),
                "sharpe_ratio": sharpe,
                "sortino_ratio": sortino,
                "win_rate_pct": win_rate,
                "max_drawdown_pct": max_dd,
                "total_trades": float(total_trades),
                "profit_factor": profit_factor,
                "n_long_entries": float(n_long_vbt),
                "n_short_entries": float(n_short_vbt),
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
