"""
BbandsMR — VectorBT Backtest + MLflow Logging

Bollinger Bands mean reversion strategy on EURGBP 1H:
  - Enter long when close < lower BB and ADX < threshold (ranging market)
  - Enter short when close > upper BB and ADX < threshold
  - TP at BB midline (SMA 20) at signal bar
  - SL at entry +/- SL_ATR_MULT * ATR
  - No time exit — SL/TP handles all exits
  - One trade at a time (detect_setups deduplicates via forward simulation)

Risk management :
  - 1% of account risked per trade
  - Lot size computed from SL distance so each SL hit = exactly 1% loss
  - Fees: fixed $5 per round-trip (conservative EURGBP retail spread estimate)

EURGBP sizing note:
  EURGBP is GBP-quoted. vectorbt computes P&L as:
      units * (exit_price - entry_price)
  which yields GBP, not USD. The correction is to divide units by entry_price
  before passing to vectorbt, so the resulting P&L is in USD.
  See VECTORBT_FX_SIZING.md for full derivation.

  PIP_VALUE_PER_LOT for EURGBP (GBP-quoted, 1 pip = 0.0001):
      pip_value = 0.0001 * 100,000 * (1 / EURGBP_rate)
      At EURGBP ~0.838 -> GBP/USD ~1.193 -> pip_value_per_lot ~ $11.93
      We use $12.00 as a round conservative approximation.

All outputs pushed to MLflow experiment "bbands-mr".

Usage:
    uv run python experiments/BbandsMR/backtest.py
"""

import os
import sys
import io
import importlib.util
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

from utils import build_analytics_chart

# ---------------------------------------------------------------------------
# Load strategy module via importlib (no package install needed)
# ---------------------------------------------------------------------------

_strategy_path = Path(__file__).resolve().parent / "strategy.py"
_spec = importlib.util.spec_from_file_location("bbands_mr_strategy", _strategy_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

detect_setups = _mod.detect_setups
load_ohlcv = _mod.load_ohlcv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "EURGBP"
START_DATE = "2024-03-17"  # 2 years of Dukascopy 1H data
END_DATE = None  # today
TIMEFRAME = "1H"

# Strategy parameters (passed to detect_setups — override strategy.py defaults here)
BB_PERIOD = 20
BB_STD = 2.0
ADX_THRESHOLD = 25
SL_ATR_MULT = 1.5
MIN_RR = 0.5

# EMA200 directional filter
EMA_PERIOD = 200  # slow EMA period
EMA_DIST_THRESHOLD = 0.002  # 0.2% distance from EMA200 required to trade

# Account & sizing
ACCOUNT_SIZE = 10_000  # real margin capital (USD)
LEVERAGE = 100  # 1:100
RISK_PER_TRADE = 0.01  # 1% of account per trade

# EURGBP instrument constants
# EURGBP is GBP-quoted: price = GBP per 1 EUR
# 1 pip = 0.0001 price units
# Pip value per standard lot (100,000 units):
#   = 0.0001 * 100,000 * (USD/GBP rate)
#   At GBP/USD ~1.26 -> pip_value_per_lot = 0.0001 * 100,000 * 1.26 = $12.60
#   We use $12.00 as a conservative round approximation.
PIP = 0.0001
PIP_VALUE_PER_LOT = 12.00  # approx USD per pip on 1 standard lot (EURGBP)
STD_LOT = 100_000  # units per standard lot

# EURGBP is GBP-quoted — quote currency is not USD.
# vectorbt's raw P&L would be in GBP. Dividing units by entry_price corrects
# this so that vbt's internal P&L is expressed in USD.
# See VECTORBT_FX_SIZING.md — same correction as JPY/CAD pairs.
_IS_NON_USD_QUOTED = True  # EURGBP: GBP-quoted

# Fee model: fixed cost per round-trip trade.
# EURGBP typical spread 0.5–1.0 pip at ~$6–12/lot.
# Using $5 as a conservative mid estimate for retail.
FX_FEES = 0.0  # disabled in vectorbt — subtract manually post-simulation
FIXED_FEE_PER_TRADE = 5.0  # USD per round-trip (entry + exit)

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "bbands-mr"

# ---------------------------------------------------------------------------
# Position sizing helper
# ---------------------------------------------------------------------------


def _compute_lot_units(sl_pips: float) -> float:
    """
    Size a trade so that hitting the SL loses exactly RISK_PER_TRADE * ACCOUNT_SIZE.

    Formula:
        risk_$  = ACCOUNT_SIZE * RISK_PER_TRADE      e.g. $100
        lots    = risk_$ / (sl_pips * PIP_VALUE_PER_LOT)
        units   = lots * STD_LOT

    Returns base-currency units (not corrected for quote currency yet).
    The caller must apply the /entry_price correction for EURGBP before
    passing to vectorbt.
    """
    if sl_pips <= 0:
        return 0.0
    risk_dollars = ACCOUNT_SIZE * RISK_PER_TRADE
    lots = risk_dollars / (sl_pips * PIP_VALUE_PER_LOT)
    return lots * STD_LOT


# ---------------------------------------------------------------------------
# Convert setups -> vectorbt signal arrays
# ---------------------------------------------------------------------------


def setups_to_signal_arrays(
    df: pd.DataFrame,
    sell_setups: list,
    buy_setups: list,
) -> tuple:
    """
    Convert setup dicts into per-bar pandas Series for vectorbt.

    Sizing logic (EURGBP GBP-quoted):
      1. sl_pips  = abs(entry - sl) / PIP
      2. units    = _compute_lot_units(sl_pips)   [risk-based, in EUR units]
      3. units   /= entry_price                   [GBP-quoted correction for vectorbt]
      4. units    = min(units, max_units)          [cap so notional <= ACCOUNT_SIZE * LEVERAGE]

    SL / TP are expressed as fractional distances from entry (required by vectorbt):
      long  sl_frac = (entry - sl) / entry
      long  tp_frac = (tp - entry) / entry
      short sl_frac = (sl - entry) / entry
      short tp_frac = (entry - tp) / entry

    Returns
    -------
    long_entries   : bool Series
    short_entries  : bool Series
    long_sl        : float Series — fractional SL (NaN on non-entry bars)
    long_tp        : float Series — fractional TP (NaN on non-entry bars)
    short_sl       : float Series
    short_tp       : float Series
    size_series    : float Series — corrected units on entry bars (NaN elsewhere)
    """
    long_entries = pd.Series(False, index=df.index)
    short_entries = pd.Series(False, index=df.index)
    long_sl = pd.Series(np.nan, index=df.index)
    long_tp = pd.Series(np.nan, index=df.index)
    short_sl = pd.Series(np.nan, index=df.index)
    short_tp = pd.Series(np.nan, index=df.index)
    size_series = pd.Series(np.nan, index=df.index)

    # Max notional cap: vectorbt holds init_cash=ACCOUNT_SIZE, so any order
    # with (units * entry_price) > ACCOUNT_SIZE would be silently rejected.
    # After the /entry correction, notional = (units / entry) * entry = units (raw),
    # which must be <= ACCOUNT_SIZE * LEVERAGE.
    # Cap raw units before correction: max_raw_units = ACCOUNT_SIZE * LEVERAGE.
    # After correction cap: max_corrected_units = (ACCOUNT_SIZE * LEVERAGE) / entry.

    for s in buy_setups:
        ts = s["entry_ts"]
        if ts not in df.index:
            continue
        entry = s["entry"]
        sl = s["sl"]
        tp = s["tp"]

        # Validate direction
        if sl >= entry:
            continue  # SL must be below entry for long
        if tp <= entry:
            continue  # TP must be above entry for long

        # Fractional SL/TP for vectorbt
        sl_frac = (entry - sl) / entry
        tp_frac = (tp - entry) / entry
        if sl_frac <= 0 or tp_frac <= 0 or np.isnan(sl_frac) or np.isnan(tp_frac):
            continue

        # Pip-based sizing
        sl_pips = (entry - sl) / PIP
        if sl_pips <= 0:
            continue

        units = _compute_lot_units(sl_pips)

        # GBP-quoted correction: vectorbt P&L = units * (exit - entry) [in GBP].
        # Dividing by entry converts the notional so vbt computes USD P&L.
        if _IS_NON_USD_QUOTED:
            units = units / entry

        # Notional cap: after correction, vbt sees notional = units * entry.
        # units_corrected * entry = raw_units -> so cap corrected units at
        # (ACCOUNT_SIZE * LEVERAGE) / entry to ensure notional <= ACCOUNT_SIZE * LEVERAGE.
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

        # Validate direction
        if sl <= entry:
            continue  # SL must be above entry for short
        if tp >= entry:
            continue  # TP must be below entry for short

        # Fractional SL/TP for vectorbt
        sl_frac = (sl - entry) / entry
        tp_frac = (entry - tp) / entry
        if sl_frac <= 0 or tp_frac <= 0 or np.isnan(sl_frac) or np.isnan(tp_frac):
            continue

        # Pip-based sizing
        sl_pips = (sl - entry) / PIP
        if sl_pips <= 0:
            continue

        units = _compute_lot_units(sl_pips)

        # GBP-quoted correction
        if _IS_NON_USD_QUOTED:
            units = units / entry

        # Notional cap
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
    print("BBANDSMR — VECTORBT BACKTEST")
    print("=" * 70)

    # -- Data --
    print(f"\nLoading {SYMBOL} {TIMEFRAME} data ({START_DATE} to {END_DATE})...")
    df = load_ohlcv(SYMBOL, START_DATE, END_DATE, TIMEFRAME)
    print(f"  {len(df):,} bars  [{df.index.min()} -> {df.index.max()}]")

    # -- Signals --
    print("\nDetecting setups...")
    sell_setups, buy_setups, extra = detect_setups(
        df,
        bb_period=BB_PERIOD,
        bb_std=BB_STD,
        adx_threshold=ADX_THRESHOLD,
        sl_atr_mult=SL_ATR_MULT,
        min_rr=MIN_RR,
        ema_period=EMA_PERIOD,
        ema_dist_threshold=EMA_DIST_THRESHOLD,
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
    # IMPORTANT: leave NaN on non-entry bars. fillna(0) would set sl_stop=0
    # on every bar — vectorbt treats that as a stop at price=0, triggering
    # phantom stop-outs on every single bar.
    combined_sl = long_sl.fillna(short_sl)
    combined_tp = long_tp.fillna(short_tp)
    combined_size = size_series.fillna(0)

    # Diagnostics: convert fractional SL back to pips for sanity checking
    # sl_frac = (entry - sl) / entry  ->  sl_pips = sl_frac * entry / PIP
    # We approximate using close price as a proxy for entry (close of signal bar)
    active_sl = combined_sl[combined_sl > 0]
    active_close = df["close"][combined_sl > 0]
    mean_sl_pips = (active_sl * active_close / PIP).mean()
    active_size = combined_size[combined_size > 0]
    mean_units_corrected = active_size.mean()
    # Approximate lots: corrected_units * entry ≈ raw_units = lots * STD_LOT
    mean_lots_approx = mean_units_corrected * active_close.mean() / STD_LOT

    print(f"\n  Mean SL distance       : {mean_sl_pips:.1f} pips")
    print(f"  Mean lot size (approx) : {mean_lots_approx:.4f} lots")
    print(f"  Mean corrected units   : {mean_units_corrected:,.2f}")
    print(
        f"  Risk per trade         : ${ACCOUNT_SIZE * RISK_PER_TRADE:,.0f}"
        f"  ({RISK_PER_TRADE * 100:.1f}% of ${ACCOUNT_SIZE:,})"
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
        stop_exit_price="StopMarket",  # fill at exact SL/TP price, not bar close
        init_cash=ACCOUNT_SIZE,  # real margin capital ($10,000)
        # lot sizes are capped so notional never
        # exceeds ACCOUNT_SIZE*LEVERAGE, so vbt
        # never rejects an order for insufficient cash
        size=combined_size,
        size_type="amount",
        fees=FX_FEES,  # 0.0 — deducted manually post-simulation
        freq=TIMEFRAME,
        accumulate=False,
    )

    # -- Metrics --
    stats = pf.stats()
    trades = pf.trades.records_readable

    gross_pnl = pf.value().iloc[-1] - ACCOUNT_SIZE
    total_trades = int(stats.get("Total Trades", 0))
    total_fees = total_trades * FIXED_FEE_PER_TRADE
    abs_pnl = gross_pnl - total_fees
    return_on_margin = (abs_pnl / ACCOUNT_SIZE) * 100

    max_dd = float(pf.max_drawdown() * 100)
    sharpe = float(stats.get("Sharpe Ratio", float("nan")))
    sortino = float(stats.get("Sortino Ratio", float("nan")))
    win_rate = float(stats.get("Win Rate [%]", float("nan")))
    profit_factor = float(stats.get("Profit Factor", float("nan")))

    # Sanity check: average losing trade should be ~-$100 (= 1% of $10k)
    # If it's ~-$12,000 the GBP correction is missing.
    # If it's ~-$0.67 the correction was double-applied.
    closed_trades = trades[trades["PnL"] != 0] if len(trades) > 0 else trades
    avg_win = (
        float(closed_trades.loc[closed_trades["PnL"] > 0, "PnL"].mean())
        if len(closed_trades) > 0
        else float("nan")
    )
    avg_loss = (
        float(closed_trades.loc[closed_trades["PnL"] < 0, "PnL"].mean())
        if len(closed_trades) > 0
        else float("nan")
    )
    expected_loss = ACCOUNT_SIZE * RISK_PER_TRADE  # $100

    print(f"\n  Trades               : {total_trades}")
    print(f"  Gross P&L            : ${gross_pnl:,.2f}")
    print(
        f"  Total fees           : ${total_fees:,.2f}  (${FIXED_FEE_PER_TRADE}/trade)"
    )
    print(f"  Net P&L              : ${abs_pnl:,.2f}")
    print(
        f"  Return on Margin     : {return_on_margin:+.2f}%"
        f"  (${ACCOUNT_SIZE:,} at {LEVERAGE}:1)"
    )
    print(f"  Win Rate             : {win_rate:.1f}%")
    print(f"  Sharpe               : {sharpe:.2f}")
    print(f"  Sortino              : {sortino:.2f}")
    print(f"  Max Drawdown         : {max_dd:.2f}%")
    print(f"  Profit Factor        : {profit_factor:.3f}")
    print(f"\n  --- Sizing sanity check ---")
    print(f"  Expected avg loss    : ~-${expected_loss:.0f}")
    print(f"  Actual avg loss      : ${avg_loss:.2f}")
    print(f"  Actual avg win       : ${avg_win:.2f}")

    # Warn if sizing looks wrong
    if not np.isnan(avg_loss) and abs(avg_loss) > expected_loss * 5:
        print(
            f"  WARNING: avg loss is {abs(avg_loss) / expected_loss:.0f}x expected — "
            f"check GBP quote currency correction."
        )
    elif not np.isnan(avg_loss) and abs(avg_loss) < expected_loss * 0.1:
        print(
            f"  WARNING: avg loss is very small — correction may have been double-applied."
        )

    # Direction breakdown
    if len(trades) > 0 and "Direction" in trades.columns:
        print()
        for direction in sorted(trades["Direction"].unique()):
            sub = trades[trades["Direction"] == direction]
            wins = (sub["PnL"] > 0).sum()
            tot = len(sub)
            wr = wins / tot * 100 if tot > 0 else 0.0
            dir_fees = tot * FIXED_FEE_PER_TRADE
            dir_net_pnl = sub["PnL"].sum() - dir_fees
            print(
                f"  {direction:<8} : {tot} trades  |  win rate {wr:.1f}%  |"
                f"  net PnL ${dir_net_pnl:.2f}"
            )

    # -- Artifacts --
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_dir = Path(__file__).resolve().parent / "reports" / f"{SYMBOL}_{run_ts}"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Config snapshot
    config_txt = reports_dir / "config.txt"
    with open(config_txt, "w", encoding="utf-8") as f:
        f.write("BbandsMR — Config Snapshot\n")
        f.write("=" * 60 + "\n\n")
        f.write("# --- Data ---\n")
        f.write(f"SYMBOL                  = {SYMBOL!r}\n")
        f.write(f"START_DATE              = {START_DATE!r}\n")
        f.write(f"END_DATE                = {END_DATE!r}\n")
        f.write(f"TIMEFRAME               = {TIMEFRAME!r}\n")
        f.write("\n# --- Strategy parameters ---\n")
        f.write(f"BB_PERIOD               = {BB_PERIOD}\n")
        f.write(f"BB_STD                  = {BB_STD}\n")
        f.write(f"ADX_THRESHOLD           = {ADX_THRESHOLD}\n")
        f.write(f"SL_ATR_MULT             = {SL_ATR_MULT}\n")
        f.write(f"MIN_RR                  = {MIN_RR}\n")
        f.write(f"EMA_PERIOD              = {EMA_PERIOD}\n")
        f.write(f"EMA_DIST_THRESHOLD      = {EMA_DIST_THRESHOLD}  # frac of price\n")
        f.write("\n# --- Account & sizing ---\n")
        f.write(f"ACCOUNT_SIZE            = {ACCOUNT_SIZE}\n")
        f.write(f"LEVERAGE                = {LEVERAGE}\n")
        f.write(f"RISK_PER_TRADE          = {RISK_PER_TRADE}\n")
        f.write(
            f"FIXED_FEE_PER_TRADE     = {FIXED_FEE_PER_TRADE}  # USD per round-trip\n"
        )
        f.write("\n# --- Instrument (EURGBP) ---\n")
        f.write(f"PIP                     = {PIP}\n")
        f.write(
            f"PIP_VALUE_PER_LOT       = {PIP_VALUE_PER_LOT}  # USD (at ~1.26 GBP/USD)\n"
        )
        f.write(f"STD_LOT                 = {STD_LOT}\n")
        f.write(
            f"_IS_NON_USD_QUOTED      = {_IS_NON_USD_QUOTED}  # GBP-quoted -> /entry correction\n"
        )
        f.write("\n# --- vectorbt ---\n")
        f.write("fees in vbt             = 0.0  (post-sim deduction used instead)\n")
        f.write("stop_exit_price         = StopMarket\n")
        f.write("init_cash               = ACCOUNT_SIZE (not * LEVERAGE)\n")
        f.write("\n# --- MLflow ---\n")
        f.write(f"MLFLOW_TRACKING_URI     = {MLFLOW_TRACKING_URI!r}\n")
        f.write(f"MLFLOW_EXPERIMENT       = {MLFLOW_EXPERIMENT!r}\n")

    # Text report
    report_txt = reports_dir / "backtest_report.txt"
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("BbandsMR — Backtest Report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Symbol              : {SYMBOL}\n")
        f.write(f"Period              : {START_DATE} to {END_DATE}\n")
        f.write(f"Timeframe           : {TIMEFRAME}\n")
        f.write(f"Account             : ${ACCOUNT_SIZE:,} at {LEVERAGE}:1 leverage\n")
        f.write(
            f"Risk/trade          : {RISK_PER_TRADE * 100:.1f}%"
            f"  (${ACCOUNT_SIZE * RISK_PER_TRADE:,.0f})\n"
        )
        f.write(
            f"Fixed fee           : ${FIXED_FEE_PER_TRADE} per trade (round-trip)\n"
        )
        f.write(f"BB Period / Std Dev : {BB_PERIOD} / {BB_STD}\n")
        f.write(f"ADX threshold       : {ADX_THRESHOLD}\n")
        f.write(f"SL ATR multiplier   : {SL_ATR_MULT}\n")
        f.write(f"Min RR              : {MIN_RR}\n")
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
        f.write(f"\n--- Sizing Sanity Check ---\n")
        f.write(f"Expected avg loss   : ~-${expected_loss:.0f}\n")
        f.write(f"Actual avg loss     : ${avg_loss:.2f}\n")
        f.write(f"Actual avg win      : ${avg_win:.2f}\n")
        f.write("\n--- Full VBT Stats ---\n")
        f.write(str(stats))

    # Analytics chart
    analytics_html = reports_dir / "analytics.html"
    an_fig = build_analytics_chart(pf, SYMBOL, ACCOUNT_SIZE, 1)
    an_fig.write_html(str(analytics_html))

    # Native vectorbt report
    vbt_html = reports_dir / "vbt_report.html"
    try:
        pf.plot().write_html(str(vbt_html))
    except Exception as e:
        print(f"  (vbt native plot skipped: {e})")

    print(f"\n  Artifacts saved -> {reports_dir}")

    # -- MLflow --
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    run_name = f"{SYMBOL}_{TIMEFRAME}_{START_DATE}_to_{END_DATE or 'today'}"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(
            {
                "symbol": SYMBOL,
                "start_date": START_DATE,
                "end_date": str(END_DATE),
                "timeframe": TIMEFRAME,
                "account_size": ACCOUNT_SIZE,
                "leverage": LEVERAGE,
                "risk_per_trade": RISK_PER_TRADE,
                "fixed_fee_per_trade": FIXED_FEE_PER_TRADE,
                "bb_period": BB_PERIOD,
                "bb_std": BB_STD,
                "adx_threshold": ADX_THRESHOLD,
                "sl_atr_mult": SL_ATR_MULT,
                "min_rr": MIN_RR,
                "ema_period": EMA_PERIOD,
                "ema_dist_threshold": EMA_DIST_THRESHOLD,
                "pip_value_per_lot": PIP_VALUE_PER_LOT,
                "is_non_usd_quoted": _IS_NON_USD_QUOTED,
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
                "avg_win": avg_win,
                "avg_loss": avg_loss,
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
