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

Outputs saved to experiments/Unicorn-ICT/reports/:
  - unicorn_ict_EURUSD_backtest_report.txt   full stats
  - unicorn_ict_EURUSD_trades.csv            trade-by-trade log
  - unicorn_ict_EURUSD_equity.html           interactive equity + drawdown chart
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import vectorbt as vbt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import DataLoader

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
INIT_CASH = 10_000  # USD account size
RISK_PER_TRADE = 0.01  # 1% of equity risked per trade
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


def size_from_risk(
    equity: float, sl_frac: float, risk_pct: float = RISK_PER_TRADE
) -> float:
    """
    Compute position size so that hitting SL loses exactly risk_pct of equity.
    Returns fraction of equity to allocate (size_type='percent').

    size * equity * sl_frac = risk_pct * equity
    => size = risk_pct / sl_frac
    capped at 1.0 (100% of equity).
    """
    if sl_frac <= 0:
        return 0.0
    return min(risk_pct / sl_frac, 1.0)


# ---------------------------------------------------------------------------
# Main backtest runner
# ---------------------------------------------------------------------------


def run_backtest(
    symbol: str = SYMBOL,
    start_date: str = START_DATE,
    end_date: str = END_DATE,
    init_cash: float = INIT_CASH,
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
    # 3. Position sizing: risk-based, computed per signal bar
    #    We pass a single size_type='percent' value; because each trade has
    #    a different SL distance we compute an average fractional size.
    #    For per-trade exact sizing use from_order_func — this approach uses
    #    a conservative fixed 1% allocation which approximates 1% risk given
    #    typical SL distances on EURUSD 1H.
    # ------------------------------------------------------------------
    # Compute mean SL fraction across all valid signals for sizing reference
    all_sl = pd.concat(
        [
            long_sl[long_entries],
            short_sl[short_entries],
        ]
    ).dropna()

    if len(all_sl) > 0:
        mean_sl_frac = all_sl.mean()
        position_size = min(risk_per_trade / mean_sl_frac, 0.95)
    else:
        mean_sl_frac = 0.0
        position_size = risk_per_trade

    print(f"  Mean SL distance: {mean_sl_frac * 100:.3f}% of price")
    print(f"  Position size   : {position_size * 100:.1f}% of equity per trade")

    # ------------------------------------------------------------------
    # 4. Run vectorbt simulation (long + short simultaneously)
    # ------------------------------------------------------------------
    print("Running vectorbt backtest...")

    # sl_stop / tp_stop as per-bar Series — vectorbt reads the value at
    # the entry bar and applies it as a fixed fraction from entry price.
    # Forward-fill so every bar has a value (vectorbt needs aligned series).
    # Non-signal bars carry NaN which vectorbt ignores for non-open positions.

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=long_entries,
        exits=pd.Series(False, index=df.index),  # exits handled by SL/TP
        short_entries=short_entries,
        short_exits=pd.Series(False, index=df.index),
        sl_stop=long_sl.fillna(short_sl).fillna(0),
        tp_stop=long_tp.fillna(short_tp).fillna(0),
        init_cash=init_cash,
        size=position_size,
        size_type="percent",
        fees=FX_FEES,
        freq=FREQ,
        accumulate=False,
    )

    # ------------------------------------------------------------------
    # 5. Print stats
    # ------------------------------------------------------------------
    stats = pf.stats()

    print("\n" + "=" * 60)
    print(f" ICT UNICORN — EURUSD 1H BACKTEST RESULTS")
    print(f" Period  : {start_date} to {end_date}")
    print(f" Capital : ${init_cash:,.0f}   Fees: {FX_FEES * 100:.4f}% per side")
    print(f" R:R min : {min_rr}   SL buffer: {sl_buffer * 10000:.1f} pips")
    print("=" * 60)
    print(f" Total Return     : {pf.total_return() * 100:.2f}%")
    print(f" Final Value      : ${pf.final_value():.2f}")
    print(f" Sharpe Ratio     : {pf.sharpe_ratio():.2f}")
    print(f" Sortino Ratio    : {pf.sortino_ratio():.2f}")
    print(f" Max Drawdown     : {pf.max_drawdown() * 100:.2f}%")
    print(f" Total Trades     : {pf.trades.count()}")
    print(f" Win Rate         : {pf.trades.win_rate() * 100:.1f}%")
    print(f" Profit Factor    : {pf.trades.profit_factor():.2f}")
    print(
        f" Avg Win          : {pf.trades.records_readable['PnL'].where(pf.trades.records_readable['PnL'] > 0).mean():.2f}"
    )
    print(
        f" Avg Loss         : {pf.trades.records_readable['PnL'].where(pf.trades.records_readable['PnL'] < 0).mean():.2f}"
    )
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
    # 6. Save outputs
    # ------------------------------------------------------------------
    reports_dir = Path(__file__).resolve().parent / "reports"
    reports_dir.mkdir(exist_ok=True)

    if save_outputs:
        # Text report
        report_path = reports_dir / f"unicorn_ict_{symbol}_backtest_report.txt"
        with open(report_path, "w") as f:
            f.write(f"ICT Unicorn Model — {symbol} 1H Backtest\n")
            f.write(f"Period  : {start_date} to {end_date}\n")
            f.write(f"Capital : ${init_cash:,.0f}\n")
            f.write(f"R:R min : {min_rr}   SL buffer: {sl_buffer * 10000:.1f} pips\n\n")
            f.write(str(stats))
            f.write(f"\n\nTotal Return  : {pf.total_return() * 100:.2f}%\n")
            f.write(f"Sharpe Ratio  : {pf.sharpe_ratio():.2f}\n")
            f.write(f"Max Drawdown  : {pf.max_drawdown() * 100:.2f}%\n")
            f.write(f"Win Rate      : {pf.trades.win_rate() * 100:.1f}%\n")
            f.write(f"Profit Factor : {pf.trades.profit_factor():.2f}\n")
        print(f"\n  Report saved   -> {report_path}")

        # Trades CSV
        trades_path = reports_dir / f"unicorn_ict_{symbol}_trades.csv"
        trades_df.to_csv(trades_path)
        print(f"  Trades saved   -> {trades_path}")

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
        chart_path = reports_dir / f"unicorn_ict_{symbol}_equity.html"
        fig.write_html(str(chart_path))
        print(f"  Equity chart   -> {chart_path}")

    if show_chart:
        fig.show()

    return pf, df


if __name__ == "__main__":
    run_backtest()
