"""
ICT Silver Bullet Strategy — Two-Phase Parameter Optimisation

Phase 1 (coarse): Wide sparse grid across all 6 parameters.
Phase 2 (fine):   Neighbourhood search around the top Phase 1 seeds.

Optimisation target: Sharpe Ratio (secondary: Return on Margin).
A minimum trade count filter (MIN_TRADES) is applied to avoid curve-fitting
on sparse results.

Usage:
    # Run both phases from scratch (~3-4 hours):
    uv run python experiments/silver-bullet/optimize.py

    # Skip backtesting, rebuild HTML report from saved CSVs:
    # Set REPORT_ONLY = True at the top of this file, then re-run.

Outputs (to experiments/silver-bullet/reports/):
    silver_bullet_optimization_phase1.csv
    silver_bullet_optimization_phase2.csv
    silver_bullet_optimization_report.html

MLflow experiment: silver-bullet-optimization
"""

import os
import sys
import io
import importlib.util
import itertools
import time
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
import plotly.graph_objects as go
import mlflow

# ---------------------------------------------------------------------------
# Load strategy module via importlib
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

# Set REPORT_ONLY = True to skip all backtesting and rebuild the HTML report
# from previously saved CSVs without re-running any simulations.
REPORT_ONLY = False

SYMBOL = "US30"
START_DATE = "2025-11-11"
END_DATE = "2026-02-25"
TIMEFRAME = "1min"

ACCOUNT_SIZE = 100_000  # USD real margin capital
LEVERAGE = 100  # 100:1 leverage
LOT_SIZE = 10  # lots per trade
FEES = 0.0001  # 0.01% per side
VBT_FREQ = "1min"

MIN_TRADES = 10  # discard runs with fewer trades (curve-fitting guard)
TOP_N_PHASE1 = 10  # number of Phase 1 seeds to drill into in Phase 2

MLFLOW_TRACKING_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "silver-bullet-optimization"

# ---------------------------------------------------------------------------
# Parameter grids
# ---------------------------------------------------------------------------

# Phase 1 — coarse (wide ranges, larger steps)
PHASE1_GRID = {
    "rr_ratio": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
    "liquidity_window": [100, 150, 200, 300, 400, 500],
    "sl_buffer_pts": [0.0, 1.0, 3.0, 5.0],
    "session_cap_usd": [-500.0, -1000.0, -2000.0, -5000.0],
    "sweep_lookback": [60, 120, 180, 240],
    "pullback_window": [5, 10, 20, 30],
}

# Phase 2 — perturbation deltas applied around each Phase 1 seed
PHASE2_DELTAS = {
    "rr_ratio": [-0.5, 0.0, +0.5],
    "liquidity_window": [-50, 0, +50],
    "sl_buffer_pts": [-0.5, 0.0, +0.5],
    "session_cap_usd": [-500, 0.0, +500],
    "sweep_lookback": [-30, 0, +30],
    "pullback_window": [-5, 0, +5],
}

PHASE2_BOUNDS = {
    "rr_ratio": (0.5, 8.0),
    "liquidity_window": (50, 600),
    "sl_buffer_pts": (0.0, 10.0),
    "session_cap_usd": (-10_000.0, -100.0),
    "sweep_lookback": (30, 300),
    "pullback_window": (3, 60),
}

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------

reports_dir = Path(__file__).resolve().parent / "reports"
reports_dir.mkdir(exist_ok=True)

CSV_PHASE1 = reports_dir / "silver_bullet_optimization_phase1.csv"
CSV_PHASE2 = reports_dir / "silver_bullet_optimization_phase2.csv"
HTML_REPORT = reports_dir / "silver_bullet_optimization_report.html"

# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------


def run_one(
    df_raw: pd.DataFrame,
    source_tz: str,
    rr_ratio: float,
    liquidity_window: int,
    sl_buffer_pts: float,
    session_cap_usd: float,
    sweep_lookback: int,
    pullback_window: int,
) -> dict:
    """
    Run a single backtest for the given parameter set.

    Uses detect_setups() from strategy.py and builds a minimal vectorbt
    Portfolio. Returns a flat dict of metrics; NaN metrics are returned
    when the run fails or produces fewer than MIN_TRADES.
    """
    params = dict(
        rr_ratio=float(rr_ratio),
        liquidity_window=int(liquidity_window),
        sl_buffer_pts=float(sl_buffer_pts),
        session_cap_usd=float(session_cap_usd),
        sweep_lookback=int(sweep_lookback),
        pullback_window=int(pullback_window),
        lot_size=LOT_SIZE,
    )
    nan_row = {
        **params,
        "trades": 0,
        "sharpe": np.nan,
        "total_return_pct": np.nan,
        "win_rate": np.nan,
        "max_drawdown_pct": np.nan,
        "profit_factor": np.nan,
        "abs_pnl": np.nan,
    }

    try:
        sell_setups, buy_setups, _ = detect_setups(
            df_raw, source_tz=source_tz, **params
        )
        n_setups = len(buy_setups) + len(sell_setups)

        if n_setups < MIN_TRADES:
            nan_row["trades"] = n_setups
            return nan_row

        # Build per-bar signal arrays (long only for optimisation speed;
        # short side is symmetric and would double the evaluation cost)
        close = df_raw["close"]
        long_entries = pd.Series(False, index=df_raw.index)
        long_sl = pd.Series(np.nan, index=df_raw.index)
        long_tp = pd.Series(np.nan, index=df_raw.index)

        for s in buy_setups:
            ts = s["entry_ts"]
            if ts not in df_raw.index:
                continue
            entry = s["entry"]
            sl = s["sl"]
            tp = s["tp"]
            if sl >= entry:
                continue
            sl_frac = (entry - sl) / entry
            tp_frac = (tp - entry) / entry
            if sl_frac <= 0 or tp_frac <= 0:
                continue
            long_entries[ts] = True
            long_sl[ts] = sl_frac
            long_tp[ts] = tp_frac

        short_entries = pd.Series(False, index=df_raw.index)
        short_sl = pd.Series(np.nan, index=df_raw.index)
        short_tp = pd.Series(np.nan, index=df_raw.index)

        for s in sell_setups:
            ts = s["entry_ts"]
            if ts not in df_raw.index:
                continue
            entry = s["entry"]
            sl = s["sl"]
            tp = s["tp"]
            if sl <= entry:
                continue
            sl_frac = (sl - entry) / entry
            tp_frac = (entry - tp) / entry
            if sl_frac <= 0 or tp_frac <= 0:
                continue
            short_entries[ts] = True
            short_sl[ts] = sl_frac
            short_tp[ts] = tp_frac

        combined_sl = long_sl.fillna(short_sl).fillna(0)
        combined_tp = long_tp.fillna(short_tp).fillna(0)

        n_signals = int(long_entries.sum()) + int(short_entries.sum())
        if n_signals < MIN_TRADES:
            nan_row["trades"] = n_signals
            return nan_row

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=long_entries,
            exits=pd.Series(False, index=df_raw.index),
            short_entries=short_entries,
            short_exits=pd.Series(False, index=df_raw.index),
            sl_stop=combined_sl,
            tp_stop=combined_tp,
            stop_exit_price="StopMarket",
            init_cash=ACCOUNT_SIZE * LEVERAGE,
            size=LOT_SIZE,
            size_type="amount",
            fees=FEES,
            freq=VBT_FREQ,
            accumulate=False,
        )

        stats = pf.stats()
        abs_pnl = pf.value().iloc[-1] - (ACCOUNT_SIZE * LEVERAGE)
        rom = (abs_pnl / ACCOUNT_SIZE) * 100

        return {
            **params,
            "trades": int(stats.get("Total Trades", n_signals)),
            "sharpe": float(stats.get("Sharpe Ratio", np.nan)),
            "total_return_pct": float(rom),
            "win_rate": float(stats.get("Win Rate [%]", np.nan)),
            "max_drawdown_pct": float(stats.get("Max Drawdown [%]", np.nan)),
            "profit_factor": float(stats.get("Profit Factor", np.nan)),
            "abs_pnl": float(abs_pnl),
        }

    except Exception as exc:
        nan_row["_error"] = str(exc)
        return nan_row


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------


def run_phase(
    label: str, grid: dict, df_raw: pd.DataFrame, source_tz: str
) -> pd.DataFrame:
    """Iterate over the Cartesian product of `grid` and collect results."""
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)

    print(f"\n--- {label} ---")
    print(f"  {total:,} combinations")

    results = []
    t_start = time.time()

    for i, combo in enumerate(combos, 1):
        kwargs = dict(zip(keys, combo))
        row = run_one(df_raw, source_tz, **kwargs)
        results.append(row)

        if i % 100 == 0 or i == total:
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 1
            eta = (total - i) / rate
            print(
                f"  {i:>6}/{total}  elapsed {elapsed:>6.0f}s  ~{eta:>5.0f}s remaining"
            )

    df_out = pd.DataFrame(results)
    print(f"  Done — {total} runs in {time.time() - t_start:.0f}s")
    return df_out


# ---------------------------------------------------------------------------
# HTML report helpers
# ---------------------------------------------------------------------------


def _scatter_chart(df_valid: pd.DataFrame, title: str) -> str:
    """Sharpe vs Return scatter, bubble size = trade count."""
    if len(df_valid) == 0:
        return "<p>No valid results.</p>"
    max_trades = df_valid["trades"].max() or 1
    fig = go.Figure(
        go.Scatter(
            x=df_valid["sharpe"],
            y=df_valid["total_return_pct"],
            mode="markers",
            marker=dict(
                size=np.clip(df_valid["trades"] / max_trades * 30, 6, 30),
                color=df_valid["sharpe"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Sharpe"),
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            text=[
                f"RR={r['rr_ratio']}  LiqWin={r['liquidity_window']}  SLbuf={r['sl_buffer_pts']}<br>"
                f"Cap={r['session_cap_usd']}  Sweep={r['sweep_lookback']}  PBwin={r['pullback_window']}<br>"
                f"Trades={r['trades']}  WR={r['win_rate']:.1f}%  DD={r['max_drawdown_pct']:.2f}%"
                for _, r in df_valid.iterrows()
            ],
            hoverinfo="text+x+y",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Sharpe Ratio",
        yaxis_title="Return on Margin (%)",
        template="plotly_white",
        height=600,
        margin=dict(l=60, r=40, t=60, b=60),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _box_chart(df_valid: pd.DataFrame, param: str, title: str) -> str:
    """Box plot of Sharpe distribution per unique value of `param`."""
    if len(df_valid) == 0 or param not in df_valid.columns:
        return ""
    fig = go.Figure()
    for v in sorted(df_valid[param].unique()):
        sub = df_valid[df_valid[param] == v]["sharpe"].dropna()
        fig.add_trace(
            go.Box(y=sub.values, name=str(v), boxpoints="all", jitter=0.3, pointpos=0)
        )
    fig.update_layout(
        title=title,
        yaxis_title="Sharpe Ratio",
        xaxis_title=param,
        template="plotly_white",
        height=400,
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def _table_html(df: pd.DataFrame, n: int = 20) -> str:
    sub = df.head(n).copy()
    for col in (
        "sharpe",
        "total_return_pct",
        "win_rate",
        "max_drawdown_pct",
        "profit_factor",
    ):
        if col in sub.columns:
            sub[col] = sub[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    if "abs_pnl" in sub.columns:
        sub["abs_pnl"] = sub["abs_pnl"].map(
            lambda x: f"${x:,.0f}" if pd.notna(x) else ""
        )
    return sub.to_html(index=False, border=0, classes="results-table", escape=False)


def _best_kpis(row: pd.Series) -> str:
    fields = [
        ("RR Ratio", f"{row['rr_ratio']}"),
        ("Liquidity Window", f"{int(row['liquidity_window'])}"),
        ("SL Buffer (pts)", f"{row['sl_buffer_pts']}"),
        ("Session Cap ($)", f"{row['session_cap_usd']:,.0f}"),
        ("Sweep Lookback", f"{int(row['sweep_lookback'])}"),
        ("FVG Pullback Win", f"{int(row['pullback_window'])}"),
        ("Sharpe", f"{row['sharpe']:.3f}"),
        ("Return on Margin", f"{row['total_return_pct']:+.2f}%"),
        ("Win Rate", f"{row['win_rate']:.1f}%"),
        ("Max Drawdown", f"{row['max_drawdown_pct']:.2f}%"),
    ]
    cards = "".join(
        f'<div style="background:rgba(255,255,255,.22);padding:16px;border-radius:8px;text-align:center;">'
        f'<div style="font-size:10px;opacity:.85;text-transform:uppercase;">{label}</div>'
        f'<div style="font-size:24px;font-weight:bold;">{value}</div>'
        f"</div>"
        for label, value in fields
    )
    return f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:12px;">{cards}</div>'


def build_report(
    df_p1_valid: pd.DataFrame,
    df_p2_valid: pd.DataFrame,
    n_p1_total: int,
    n_p2_total: int,
    data_start: str,
    data_end: str,
    n_candles: int,
) -> str:
    """Assemble and return the full HTML optimisation report."""
    final = df_p2_valid if len(df_p2_valid) > 0 else df_p1_valid
    final_sorted = final.sort_values(["sharpe", "total_return_pct"], ascending=False)
    p1_sorted = df_p1_valid.sort_values(["sharpe", "total_return_pct"], ascending=False)

    best_kpis_html = (
        _best_kpis(final_sorted.iloc[0])
        if len(final_sorted) > 0
        else "<p>No valid results.</p>"
    )

    p1_scatter = _scatter_chart(
        p1_sorted, "Phase 1 — Coarse: Sharpe vs Return (bubble = trade count)"
    )
    p2_scatter = (
        _scatter_chart(
            df_p2_valid.sort_values("sharpe", ascending=False),
            "Phase 2 — Fine: Sharpe vs Return (bubble = trade count)",
        )
        if len(df_p2_valid) > 0
        else ""
    )

    box_charts = "".join(
        _box_chart(p1_sorted, param, f"Phase 1 — Sharpe by {param}")
        for param in PHASE2_DELTAS
    )

    best_table = _table_html(final_sorted, 20)
    p1_table = _table_html(p1_sorted, 20)

    p2_section = (
        f'<div class="section"><h2>Phase 2 — Fine Search: Sharpe vs Return</h2>{p2_scatter}</div>'
        if p2_scatter
        else ""
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Silver Bullet — Optimisation Report</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #2c3e50; }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white; padding: 40px 20px; text-align: center;
        }}
        .header h1 {{ font-size: 28px; margin-bottom: 8px; }}
        .header p  {{ font-size: 13px; opacity: 0.8; }}
        .banner {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border-radius: 12px; padding: 28px;
            max-width: 1600px; margin: 24px auto;
        }}
        .banner h2 {{ font-size: 18px; margin-bottom: 16px; opacity: .9; }}
        .section {{
            max-width: 1600px; margin: 24px auto; background: white;
            border-radius: 12px; padding: 28px;
            box-shadow: 0 2px 12px rgba(0,0,0,.08);
        }}
        .section h2 {{ font-size: 18px; margin-bottom: 16px; border-bottom: 2px solid #667eea; padding-bottom: 8px; }}
        .charts-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
        .results-table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
        .results-table th {{ background: #667eea; color: white; padding: 8px 10px; text-align: left; }}
        .results-table td {{ padding: 6px 10px; border-bottom: 1px solid #eee; }}
        .results-table tr:hover td {{ background: #f8f9ff; }}
        .footer {{ text-align: center; padding: 20px; color: #888; font-size: 12px; }}
    </style>
</head>
<body>

<div class="header">
    <h1>Silver Bullet Strategy (ICT) — Parameter Optimisation</h1>
    <p>
        Asset: {SYMBOL} &nbsp;|&nbsp;
        Period: {START_DATE} to {END_DATE} &nbsp;|&nbsp;
        Phase 1: {n_p1_total:,} runs &nbsp;|&nbsp;
        Phase 2: {n_p2_total:,} runs &nbsp;|&nbsp;
        Min trades filter: {MIN_TRADES}
    </p>
</div>

<div class="banner">
    <h2>Best Parameter Set (optimised for Sharpe Ratio)</h2>
    {best_kpis_html}
</div>

<div class="section">
    <h2>Phase 1 — Coarse Search: Sharpe vs Return</h2>
    {p1_scatter}
</div>

<div class="section">
    <h2>Phase 1 — Sharpe Distribution by Parameter Value</h2>
    <div class="charts-grid">{box_charts}</div>
</div>

{p2_section}

<div class="section">
    <h2>Top 20 Results (Phase 2 if available, else Phase 1)</h2>
    {best_table}
</div>

<div class="section">
    <h2>Top 20 Phase 1 Results</h2>
    {p1_table}
</div>

<div class="footer">
    Generated with VectorBT &nbsp;|&nbsp;
    {n_candles:,} candles &nbsp;|&nbsp;
    {data_start} to {data_end}
</div>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print("=" * 70)
    print("ICT SILVER BULLET — TWO-PHASE PARAMETER OPTIMISATION")
    print("=" * 70)

    # Disable vbt FigureWidget before any portfolio call
    vbt.settings.plotting["use_widgets"] = False

    # -- Load data (once) --
    print(f"\nLoading {SYMBOL} data ({START_DATE} to {END_DATE})...")
    df_raw, source_tz = load_ohlcv(SYMBOL, START_DATE, END_DATE, TIMEFRAME)
    print(
        f"  {len(df_raw):,} rows  [{df_raw.index.min()} -> {df_raw.index.max()}]  (source_tz={source_tz})"
    )

    data_start = str(df_raw.index[0].date())
    data_end = str(df_raw.index[-1].date())
    n_candles = len(df_raw)

    # ------------------------------------------------------------------
    # REPORT_ONLY mode — rebuild HTML from saved CSVs, skip backtesting
    # ------------------------------------------------------------------
    if REPORT_ONLY:
        print("\nREPORT_ONLY mode — loading saved CSVs, skipping backtesting.")
        if not CSV_PHASE1.exists():
            print(f"ERROR: {CSV_PHASE1} not found. Run with REPORT_ONLY=False first.")
            return
        df_p1 = pd.read_csv(CSV_PHASE1)
        df_p2 = pd.read_csv(CSV_PHASE2) if CSV_PHASE2.exists() else pd.DataFrame()

        df_p1_valid = df_p1.dropna(subset=["sharpe"])
        df_p1_valid = df_p1_valid[df_p1_valid["trades"] >= MIN_TRADES]
        df_p2_valid = pd.DataFrame()
        if len(df_p2) > 0:
            df_p2_valid = df_p2.dropna(subset=["sharpe"])
            df_p2_valid = df_p2_valid[df_p2_valid["trades"] >= MIN_TRADES]

        html = build_report(
            df_p1_valid,
            df_p2_valid,
            len(df_p1),
            len(df_p2),
            data_start,
            data_end,
            n_candles,
        )
        with open(HTML_REPORT, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"\nReport saved -> {HTML_REPORT}")
        return

    # ------------------------------------------------------------------
    # MLflow run
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=f"optimize_{SYMBOL}_{START_DATE}_{END_DATE}"):
        mlflow.log_params(
            {
                "symbol": SYMBOL,
                "start_date": START_DATE,
                "end_date": END_DATE,
                "account_size": ACCOUNT_SIZE,
                "leverage": LEVERAGE,
                "lot_size": LOT_SIZE,
                "fees": FEES,
                "min_trades": MIN_TRADES,
                "top_n_phase1": TOP_N_PHASE1,
                "phase1_combinations": len(
                    list(itertools.product(*[PHASE1_GRID[k] for k in PHASE1_GRID]))
                ),
            }
        )

        # ------------------------------------------------------------------
        # Phase 1 — coarse
        # ------------------------------------------------------------------
        df_p1 = run_phase("PHASE 1 — COARSE", PHASE1_GRID, df_raw, source_tz)
        df_p1.to_csv(CSV_PHASE1, index=False)

        df_p1_valid = df_p1.dropna(subset=["sharpe"])
        df_p1_valid = df_p1_valid[df_p1_valid["trades"] >= MIN_TRADES]
        df_p1_sorted = df_p1_valid.sort_values(
            ["sharpe", "total_return_pct"], ascending=False
        )

        print(f"\nPhase 1 summary:")
        print(f"  Total runs         : {len(df_p1)}")
        print(f"  Valid runs (>={MIN_TRADES} trades): {len(df_p1_valid)}")
        if len(df_p1_sorted) > 0:
            b1 = df_p1_sorted.iloc[0]
            print(f"  Best Sharpe        : {b1['sharpe']:.3f}")
            print(f"  Best Return        : {b1['total_return_pct']:+.2f}%")
            print(
                f"  Best params        : RR={b1['rr_ratio']}, LiqWin={int(b1['liquidity_window'])}, "
                f"SLbuf={b1['sl_buffer_pts']}, Cap={b1['session_cap_usd']}, "
                f"Sweep={int(b1['sweep_lookback'])}, PBwin={int(b1['pullback_window'])}"
            )

        # ------------------------------------------------------------------
        # Phase 2 — fine neighbourhood search
        # ------------------------------------------------------------------
        df_p2_valid = pd.DataFrame()
        n_p2_total = 0

        if len(df_p1_sorted) == 0:
            print("\nNo valid Phase 1 results — skipping Phase 2.")
        else:
            top_seeds = df_p1_sorted.head(TOP_N_PHASE1)
            phase2_sets: dict = {k: set() for k in PHASE2_DELTAS}

            for _, seed in top_seeds.iterrows():
                for param, deltas in PHASE2_DELTAS.items():
                    lo, hi = PHASE2_BOUNDS[param]
                    for d in deltas:
                        val = seed[param] + d
                        val = max(lo, min(hi, val))
                        if param in (
                            "liquidity_window",
                            "sweep_lookback",
                            "pullback_window",
                        ):
                            val = int(round(val))
                        phase2_sets[param].add(val)

            phase2_grid = {k: sorted(v) for k, v in phase2_sets.items()}
            print(
                f"\nPhase 2 grid sizes: { {k: len(v) for k, v in phase2_grid.items()} }"
            )

            df_p2 = run_phase("PHASE 2 — FINE", phase2_grid, df_raw, source_tz)
            n_p2_total = len(df_p2)
            df_p2.to_csv(CSV_PHASE2, index=False)

            df_p2_valid = df_p2.dropna(subset=["sharpe"])
            df_p2_valid = df_p2_valid[df_p2_valid["trades"] >= MIN_TRADES]
            df_p2_sorted = df_p2_valid.sort_values(
                ["sharpe", "total_return_pct"], ascending=False
            )

            print(f"\nPhase 2 summary:")
            print(f"  Total runs  : {n_p2_total}")
            print(f"  Valid runs  : {len(df_p2_valid)}")
            if len(df_p2_sorted) > 0:
                b2 = df_p2_sorted.iloc[0]
                print(f"  Best Sharpe : {b2['sharpe']:.3f}")
                print(f"  Best Return : {b2['total_return_pct']:+.2f}%")
                print(
                    f"  Best params : RR={b2['rr_ratio']}, LiqWin={int(b2['liquidity_window'])}, "
                    f"SLbuf={b2['sl_buffer_pts']}, Cap={b2['session_cap_usd']}, "
                    f"Sweep={int(b2['sweep_lookback'])}, PBwin={int(b2['pullback_window'])}"
                )

        # ------------------------------------------------------------------
        # Build HTML report
        # ------------------------------------------------------------------
        print("\nBuilding HTML optimisation report...")
        html = build_report(
            df_p1_valid,
            df_p2_valid,
            len(df_p1),
            n_p2_total,
            data_start,
            data_end,
            n_candles,
        )
        with open(HTML_REPORT, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"  Report saved -> {HTML_REPORT}")

        # ------------------------------------------------------------------
        # MLflow logging
        # ------------------------------------------------------------------
        final = df_p2_valid if len(df_p2_valid) > 0 else df_p1_valid
        final_sorted = final.sort_values(
            ["sharpe", "total_return_pct"], ascending=False
        )

        if len(final_sorted) > 0:
            b = final_sorted.iloc[0]
            mlflow.log_metrics(
                {
                    "best_sharpe": float(b["sharpe"]),
                    "best_return_on_margin_pct": float(b["total_return_pct"]),
                    "best_win_rate_pct": float(b["win_rate"]),
                    "best_max_drawdown_pct": float(b["max_drawdown_pct"]),
                    "best_profit_factor": float(b["profit_factor"])
                    if pd.notna(b["profit_factor"])
                    else 0.0,
                    "best_trades": float(b["trades"]),
                    "phase1_valid_runs": float(len(df_p1_valid)),
                    "phase1_total_runs": float(len(df_p1)),
                    "phase2_valid_runs": float(len(df_p2_valid)),
                    "phase2_total_runs": float(n_p2_total),
                }
            )
            mlflow.log_params(
                {
                    "best_rr_ratio": float(b["rr_ratio"]),
                    "best_liquidity_window": int(b["liquidity_window"]),
                    "best_sl_buffer_pts": float(b["sl_buffer_pts"]),
                    "best_session_cap_usd": float(b["session_cap_usd"]),
                    "best_sweep_lookback": int(b["sweep_lookback"]),
                    "best_pullback_window": int(b["pullback_window"]),
                }
            )

        # Upload the entire reports folder as artifacts
        mlflow.log_artifacts(str(reports_dir), artifact_path="outputs")
        print(f"  MLflow run logged -> {MLFLOW_TRACKING_URI}")

    # Final summary
    print("\n" + "=" * 70)
    print("OPTIMISATION COMPLETE")
    print("=" * 70)
    if len(final_sorted) > 0:
        b = final_sorted.iloc[0]
        print(f"\nRecommended parameters:")
        print(f"  rr_ratio        = {b['rr_ratio']}")
        print(f"  liquidity_window= {int(b['liquidity_window'])}")
        print(f"  sl_buffer_pts   = {b['sl_buffer_pts']}")
        print(f"  session_cap_usd = {b['session_cap_usd']}")
        print(f"  sweep_lookback  = {int(b['sweep_lookback'])}")
        print(f"  pullback_window = {int(b['pullback_window'])}")
        print(f"\nExpected performance (in-sample, treat as optimistic):")
        print(f"  Sharpe          = {b['sharpe']:.3f}")
        print(f"  Return on Margin= {b['total_return_pct']:+.2f}%")
        print(f"  Win Rate        = {b['win_rate']:.1f}%")
        print(f"  Max Drawdown    = {b['max_drawdown_pct']:.2f}%")
        print(f"  Trades          = {int(b['trades'])}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
