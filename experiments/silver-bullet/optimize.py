"""
ICT Silver Bullet Strategy — Two-Phase Parameter Optimisation

Phase 1 (coarse): Sweeps a wide, sparse grid across all 6 parameters.
Phase 2 (fine):   Drills into the neighbourhood of the top Phase-1 results.

Optimisation target: Sharpe Ratio
(secondary sort on Total Return, then Min Trades filter to avoid curve-fitting)

Run:
    uv run python experiments/silver-bullet/optimize.py

Outputs (to reports/):
    silver_bullet_optimization_phase1.csv
    silver_bullet_optimization_phase2.csv
    silver_bullet_optimization_report.html
"""

import sys
import time
import itertools
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import vectorbt as vbt

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
LEVERAGE = 100
LOT_SIZE = 10
SIZE_TYPE = "amount"
FEES = 0.0001
FREQ = "1min"
MIN_TRADES = 10  # discard runs with fewer trades (curve-fitting guard)

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

# Phase 2 — fine (narrow ranges around phase-1 winners, finer steps)
# These are derived dynamically from phase-1 results (see run_phase2).
PHASE2_DELTAS = {
    "rr_ratio": [-0.5, 0.0, +0.5],
    "liquidity_window": [-50, 0, +50],
    "sl_buffer_pts": [-0.5, 0.0, +0.5],
    "session_cap_usd": [-500, 0.0, +500],
    "sweep_lookback": [-30, 0, +30],
    "pullback_window": [-5, 0, +5],
}

# Bounds to clip phase-2 values into valid ranges
PHASE2_BOUNDS = {
    "rr_ratio": (0.5, 8.0),
    "liquidity_window": (50, 600),
    "sl_buffer_pts": (0.0, 10.0),
    "session_cap_usd": (-10_000.0, -100.0),
    "sweep_lookback": (30, 300),
    "pullback_window": (3, 60),
}

TOP_N_PHASE1 = 10  # number of best phase-1 results to seed phase 2

# ---------------------------------------------------------------------------
# Data loading (once, shared across all runs)
# ---------------------------------------------------------------------------

print("=" * 70)
print("ICT SILVER BULLET — TWO-PHASE PARAMETER OPTIMISATION")
print("=" * 70)

print(f"\nLoading {SYMBOL} data ({START_DATE} to {END_DATE})...")
df_raw = load_data(
    SYMBOL, asset_type=ASSET_TYPE, start_date=START_DATE, end_date=END_DATE
)
print(f"  {len(df_raw):,} rows  [{df_raw.index.min()} -> {df_raw.index.max()}]")


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------


def run_one(
    rr_ratio: float,
    liquidity_window: int,
    sl_buffer_pts: float,
    session_cap_usd: float,
    sweep_lookback: int,
    pullback_window: int,
) -> dict:
    """
    Run a single backtest for the given parameter set.
    Returns a dict of metrics (or NaN metrics if the run fails / too few trades).
    """
    params = dict(
        rr_ratio=rr_ratio,
        liquidity_window=int(liquidity_window),
        sl_buffer_pts=sl_buffer_pts,
        session_cap_usd=session_cap_usd,
        sweep_lookback=int(sweep_lookback),
        pullback_window=int(pullback_window),
        lot_size=LOT_SIZE,
    )

    try:
        df = calculate_signals(df_raw.copy(), **params)

        entries = df["buy_signal_daily"]
        exits = df["sell_signal"]
        close = df["close"]

        n_entries = int(entries.sum())
        if n_entries < MIN_TRADES:
            return {
                **params,
                "trades": n_entries,
                "sharpe": np.nan,
                "total_return_pct": np.nan,
                "win_rate": np.nan,
                "max_drawdown_pct": np.nan,
                "profit_factor": np.nan,
                "abs_pnl": np.nan,
            }

        pf = vbt.Portfolio.from_signals(
            close=close,
            entries=entries,
            exits=exits,
            init_cash=INIT_CASH * LEVERAGE,
            size=LOT_SIZE,
            size_type=SIZE_TYPE,
            sl_stop=df["sl_stop_frac"],
            tp_stop=df["tp_stop_frac"],
            fees=FEES,
            freq=FREQ,
        )

        stats = pf.stats()
        abs_pnl = pf.value().iloc[-1] - (INIT_CASH * LEVERAGE)
        rom = abs_pnl / INIT_CASH * 100

        return {
            **params,
            "trades": int(stats.get("Total Trades", n_entries)),
            "sharpe": float(stats.get("Sharpe Ratio", np.nan)),
            "total_return_pct": float(rom),
            "win_rate": float(stats.get("Win Rate [%]", np.nan)),
            "max_drawdown_pct": float(stats.get("Max Drawdown [%]", np.nan)),
            "profit_factor": float(stats.get("Profit Factor", np.nan)),
            "abs_pnl": float(abs_pnl),
        }

    except Exception as exc:
        return {
            **params,
            "trades": 0,
            "sharpe": np.nan,
            "total_return_pct": np.nan,
            "win_rate": np.nan,
            "max_drawdown_pct": np.nan,
            "profit_factor": np.nan,
            "abs_pnl": np.nan,
            "_error": str(exc),
        }


# ---------------------------------------------------------------------------
# Phase runner
# ---------------------------------------------------------------------------


def run_phase(label: str, grid: dict[str, list]) -> pd.DataFrame:
    keys = list(grid.keys())
    combos = list(itertools.product(*[grid[k] for k in keys]))
    total = len(combos)

    print(f"\n--- {label} ---")
    print(f"  {total:,} combinations to evaluate")

    results = []
    t_start = time.time()

    for i, combo in enumerate(combos, 1):
        kwargs = dict(zip(keys, combo))
        row = run_one(**kwargs)
        results.append(row)

        # Progress every 50 runs
        if i % 50 == 0 or i == total:
            elapsed = time.time() - t_start
            rate = i / elapsed
            remaining = (total - i) / rate if rate > 0 else 0
            print(
                f"  {i:>5}/{total}  "
                f"elapsed {elapsed:>6.0f}s  "
                f"~{remaining:>5.0f}s remaining"
            )

    df_results = pd.DataFrame(results)
    print(f"  Done — {total} runs in {time.time() - t_start:.0f}s")
    return df_results


# ---------------------------------------------------------------------------
# Phase 1 — coarse
# ---------------------------------------------------------------------------

df_phase1 = run_phase("PHASE 1 — COARSE", PHASE1_GRID)

# Save raw results
reports_dir = project_root / "reports"
reports_dir.mkdir(exist_ok=True)
df_phase1.to_csv(reports_dir / "silver_bullet_optimization_phase1.csv", index=False)

# Summary of phase 1
df_p1_valid = df_phase1.dropna(subset=["sharpe"]).copy()
df_p1_valid = df_p1_valid[df_p1_valid["trades"] >= MIN_TRADES]
df_p1_sorted = df_p1_valid.sort_values(["sharpe", "total_return_pct"], ascending=False)

print(f"\nPhase 1 summary:")
print(f"  Total runs         : {len(df_phase1)}")
print(f"  Valid runs (>= {MIN_TRADES} trades): {len(df_p1_valid)}")
if len(df_p1_sorted) > 0:
    best = df_p1_sorted.iloc[0]
    print(f"  Best Sharpe        : {best['sharpe']:.3f}")
    print(f"  Best Return        : {best['total_return_pct']:+.2f}%")
    print(
        f"  Best params        : RR={best['rr_ratio']}, LiqWin={best['liquidity_window']}, "
        f"SLbuf={best['sl_buffer_pts']}, Cap={best['session_cap_usd']}, "
        f"Sweep={best['sweep_lookback']}, PBwin={best['pullback_window']}"
    )

# ---------------------------------------------------------------------------
# Phase 2 — fine (seeds from top N phase-1 results)
# ---------------------------------------------------------------------------

if len(df_p1_sorted) == 0:
    print("\nNo valid phase-1 results — skipping phase 2.")
    df_phase2 = pd.DataFrame()
else:
    top_seeds = df_p1_sorted.head(TOP_N_PHASE1)

    # Build phase-2 grid: union of neighbourhoods around each seed
    phase2_sets: dict[str, set] = {k: set() for k in PHASE2_DELTAS}

    for _, seed in top_seeds.iterrows():
        for param, deltas in PHASE2_DELTAS.items():
            lo, hi = PHASE2_BOUNDS[param]
            for d in deltas:
                val = seed[param] + d
                val = max(lo, min(hi, val))
                # Snap integers
                if param in ("liquidity_window", "sweep_lookback", "pullback_window"):
                    val = int(round(val))
                phase2_sets[param].add(val)

    phase2_grid = {k: sorted(v) for k, v in phase2_sets.items()}

    print(f"\nPhase 2 grid sizes: { {k: len(v) for k, v in phase2_grid.items()} }")
    df_phase2 = run_phase("PHASE 2 — FINE", phase2_grid)

    df_phase2.to_csv(reports_dir / "silver_bullet_optimization_phase2.csv", index=False)

    df_p2_valid = df_phase2.dropna(subset=["sharpe"]).copy()
    df_p2_valid = df_p2_valid[df_p2_valid["trades"] >= MIN_TRADES]
    df_p2_sorted = df_p2_valid.sort_values(
        ["sharpe", "total_return_pct"], ascending=False
    )

    print(f"\nPhase 2 summary:")
    print(f"  Total runs         : {len(df_phase2)}")
    print(f"  Valid runs         : {len(df_p2_valid)}")
    if len(df_p2_sorted) > 0:
        best2 = df_p2_sorted.iloc[0]
        print(f"  Best Sharpe        : {best2['sharpe']:.3f}")
        print(f"  Best Return        : {best2['total_return_pct']:+.2f}%")
        print(
            f"  Best params        : RR={best2['rr_ratio']}, LiqWin={best2['liquidity_window']}, "
            f"SLbuf={best2['sl_buffer_pts']}, Cap={best2['session_cap_usd']}, "
            f"Sweep={best2['sweep_lookback']}, PBwin={best2['pullback_window']}"
        )

# ---------------------------------------------------------------------------
# Build HTML report
# ---------------------------------------------------------------------------

print("\nBuilding HTML optimisation report...")

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def make_scatter_matrix(df_valid: pd.DataFrame, title: str) -> str:
    """Return HTML string for a Plotly scatter chart: Return vs Sharpe, bubble=trades."""
    if len(df_valid) == 0:
        return "<p>No valid results.</p>"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df_valid["sharpe"],
            y=df_valid["total_return_pct"],
            mode="markers",
            marker=dict(
                size=np.clip(df_valid["trades"] / df_valid["trades"].max() * 30, 6, 30),
                color=df_valid["sharpe"],
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="Sharpe"),
                opacity=0.7,
                line=dict(width=0.5, color="white"),
            ),
            text=[
                f"RR={r['rr_ratio']} LW={r['liquidity_window']} SLbuf={r['sl_buffer_pts']}<br>"
                f"Cap={r['session_cap_usd']} Sweep={r['sweep_lookback']} PBwin={r['pullback_window']}<br>"
                f"Trades={r['trades']}  WR={r['win_rate']:.1f}%  DD={r['max_drawdown_pct']:.2f}%"
                for _, r in df_valid.iterrows()
            ],
            hoverinfo="text+x+y",
            name="Parameter sets",
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


def make_param_violin(df_valid: pd.DataFrame, param: str, title: str) -> str:
    """Strip plot of Sharpe by parameter value."""
    if len(df_valid) == 0 or param not in df_valid.columns:
        return ""
    vals = sorted(df_valid[param].unique())
    fig = go.Figure()
    for v in vals:
        sub = df_valid[df_valid[param] == v]["sharpe"].dropna()
        fig.add_trace(
            go.Box(
                y=sub.values,
                name=str(v),
                boxpoints="all",
                jitter=0.3,
                pointpos=0,
            )
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


def df_to_html_table(df: pd.DataFrame, n: int = 20) -> str:
    sub = df.head(n).copy()
    # Format numbers
    for col in [
        "sharpe",
        "total_return_pct",
        "win_rate",
        "max_drawdown_pct",
        "profit_factor",
    ]:
        if col in sub.columns:
            sub[col] = sub[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "—")
    if "abs_pnl" in sub.columns:
        sub["abs_pnl"] = sub["abs_pnl"].map(
            lambda x: f"${x:,.0f}" if pd.notna(x) else "—"
        )
    html = sub.to_html(index=False, border=0, classes="results-table", escape=False)
    return html


# Build charts
p1_scatter_html = make_scatter_matrix(
    df_p1_valid if len(df_phase1) > 0 else pd.DataFrame(),
    "Phase 1 — Coarse: Sharpe vs Return (bubble = trade count)",
)

param_charts_html = ""
for param in PHASE2_DELTAS:
    if len(df_p1_valid) > 0:
        param_charts_html += make_param_violin(
            df_p1_valid, param, f"Phase 1 — Sharpe distribution by {param}"
        )

p2_scatter_html = ""
best_table_html = ""
if "df_p2_sorted" in dir() and len(df_p2_sorted) > 0:
    p2_scatter_html = make_scatter_matrix(
        df_p2_sorted, "Phase 2 — Fine: Sharpe vs Return (bubble = trade count)"
    )
    best_table_html = df_to_html_table(df_p2_sorted, n=20)
elif len(df_p1_sorted) > 0:
    best_table_html = df_to_html_table(df_p1_sorted, n=20)

p1_table_html = (
    df_to_html_table(df_p1_sorted, n=20)
    if len(df_p1_sorted) > 0
    else "<p>No valid results.</p>"
)

best_row_summary = ""
final_sorted = (
    df_p2_sorted
    if ("df_p2_sorted" in dir() and len(df_p2_sorted) > 0)
    else (df_p1_sorted if len(df_p1_sorted) > 0 else pd.DataFrame())
)
if len(final_sorted) > 0:
    b = final_sorted.iloc[0]
    best_row_summary = f"""
    <div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; margin-bottom:24px;">
        <div style="background:rgba(255,255,255,.22); padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:10px; opacity:.85; text-transform:uppercase;">RR Ratio</div>
            <div style="font-size:26px; font-weight:bold;">{b["rr_ratio"]}</div>
        </div>
        <div style="background:rgba(255,255,255,.22); padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:10px; opacity:.85; text-transform:uppercase;">Liquidity Window</div>
            <div style="font-size:26px; font-weight:bold;">{int(b["liquidity_window"])}</div>
        </div>
        <div style="background:rgba(255,255,255,.22); padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:10px; opacity:.85; text-transform:uppercase;">SL Buffer (pts)</div>
            <div style="font-size:26px; font-weight:bold;">{b["sl_buffer_pts"]}</div>
        </div>
        <div style="background:rgba(255,255,255,.22); padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:10px; opacity:.85; text-transform:uppercase;">Session Cap ($)</div>
            <div style="font-size:26px; font-weight:bold;">{b["session_cap_usd"]:,.0f}</div>
        </div>
        <div style="background:rgba(255,255,255,.22); padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:10px; opacity:.85; text-transform:uppercase;">Sweep Lookback</div>
            <div style="font-size:26px; font-weight:bold;">{int(b["sweep_lookback"])}</div>
        </div>
        <div style="background:rgba(255,255,255,.22); padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:10px; opacity:.85; text-transform:uppercase;">FVG Pullback Win</div>
            <div style="font-size:26px; font-weight:bold;">{int(b["pullback_window"])}</div>
        </div>
        <div style="background:rgba(255,255,255,.22); padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:10px; opacity:.85; text-transform:uppercase;">Sharpe</div>
            <div style="font-size:26px; font-weight:bold;">{b["sharpe"]:.3f}</div>
        </div>
        <div style="background:rgba(255,255,255,.22); padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:10px; opacity:.85; text-transform:uppercase;">Return on Margin</div>
            <div style="font-size:26px; font-weight:bold;">{b["total_return_pct"]:+.2f}%</div>
        </div>
        <div style="background:rgba(255,255,255,.22); padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:10px; opacity:.85; text-transform:uppercase;">Win Rate</div>
            <div style="font-size:26px; font-weight:bold;">{b["win_rate"]:.1f}%</div>
        </div>
        <div style="background:rgba(255,255,255,.22); padding:16px; border-radius:8px; text-align:center;">
            <div style="font-size:10px; opacity:.85; text-transform:uppercase;">Max Drawdown</div>
            <div style="font-size:26px; font-weight:bold;">{b["max_drawdown_pct"]:.2f}%</div>
        </div>
    </div>
    """

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Silver Bullet — Optimisation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', sans-serif; background: #f0f2f5; color: #2c3e50; }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            color: white; padding: 40px 20px; text-align: center;
        }}
        .header h1 {{ font-size: 30px; margin-bottom: 8px; }}
        .header p  {{ font-size: 13px; opacity: 0.8; }}
        .section {{
            max-width: 1600px; margin: 24px auto; background: white;
            border-radius: 12px; padding: 28px;
            box-shadow: 0 2px 12px rgba(0,0,0,.08);
        }}
        .section h2 {{ font-size: 18px; margin-bottom: 16px; border-bottom: 2px solid #667eea; padding-bottom: 8px; }}
        .best-banner {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; border-radius: 12px; padding: 28px;
            max-width: 1600px; margin: 24px auto;
        }}
        .best-banner h2 {{ font-size: 18px; margin-bottom: 16px; opacity: .9; }}
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
        Phase 1: {len(df_phase1):,} runs &nbsp;|&nbsp;
        Min trades filter: {MIN_TRADES}
    </p>
</div>

<div class="best-banner">
    <h2>Best Parameter Set (optimised for Sharpe Ratio)</h2>
    {best_row_summary if best_row_summary else "<p>No valid results found.</p>"}
</div>

<div class="section">
    <h2>Phase 1 — Coarse Search: Sharpe vs Return</h2>
    {p1_scatter_html}
</div>

<div class="section">
    <h2>Phase 1 — Sharpe Distribution by Parameter Value</h2>
    <div class="charts-grid">
        {param_charts_html}
    </div>
</div>

{"<div class='section'><h2>Phase 2 — Fine Search: Sharpe vs Return</h2>" + p2_scatter_html + "</div>" if p2_scatter_html else ""}

<div class="section">
    <h2>Top 20 Results (Phase 2 if available, else Phase 1) — sorted by Sharpe</h2>
    {best_table_html}
</div>

<div class="section">
    <h2>Top 20 Phase 1 Results — sorted by Sharpe</h2>
    {p1_table_html}
</div>

<div class="footer">
    Generated with VectorBT &nbsp;|&nbsp;
    {len(df_raw):,} candles &nbsp;|&nbsp;
    {df_raw.index[0].date()} to {df_raw.index[-1].date()}
</div>

</body>
</html>
"""

output_path = reports_dir / "silver_bullet_optimization_report.html"
with open(output_path, "w", encoding="utf-8") as f:
    f.write(full_html)

print(f"\nReport saved -> {output_path}")
print("\nOptimisation complete.")
if len(final_sorted) > 0:
    b = final_sorted.iloc[0]
    print(f"\nRecommended parameters:")
    print(f"  rr_ratio        = {b['rr_ratio']}")
    print(f"  liquidity_window= {int(b['liquidity_window'])}")
    print(f"  sl_buffer_pts   = {b['sl_buffer_pts']}")
    print(f"  session_cap_usd = {b['session_cap_usd']}")
    print(f"  sweep_lookback  = {int(b['sweep_lookback'])}")
    print(f"  pullback_window = {int(b['pullback_window'])}")
    print(f"\nExpected performance:")
    print(f"  Sharpe          = {b['sharpe']:.3f}")
    print(f"  Return on Margin= {b['total_return_pct']:+.2f}%")
    print(f"  Win Rate        = {b['win_rate']:.1f}%")
    print(f"  Max Drawdown    = {b['max_drawdown_pct']:.2f}%")
    print(f"  Trades          = {int(b['trades'])}")
