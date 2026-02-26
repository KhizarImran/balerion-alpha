"""
ICT Silver Bullet — Phase 2 (fine search) + HTML report generation.

Reads the already-completed phase1 CSV from reports/ and runs the fine search
around the top results.  Run this after optimize.py has completed phase 1
(or after it was interrupted once the phase1 CSV is saved).

Run:
    uv run python experiments/silver-bullet/optimize_phase2.py
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
# Config (must match optimize.py)
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
MIN_TRADES = 10
TOP_N_PHASE1 = 10

# Set True to skip all backtesting and rebuild the HTML report from saved CSVs only.
REPORT_ONLY = True

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

reports_dir = project_root / "reports"

# ---------------------------------------------------------------------------
# Load phase 1 results (always needed)
# ---------------------------------------------------------------------------

print("=" * 70)
print("ICT SILVER BULLET — PHASE 2 FINE SEARCH + REPORT")
print("=" * 70)

p1_path = reports_dir / "silver_bullet_optimization_phase1.csv"
if not p1_path.exists():
    print(f"ERROR: {p1_path} not found. Run optimize.py first.")
    sys.exit(1)

df_phase1 = pd.read_csv(p1_path)
df_p1_valid = df_phase1.dropna(subset=["sharpe"]).copy()
df_p1_valid = df_p1_valid[df_p1_valid["trades"] >= MIN_TRADES]
df_p1_sorted = df_p1_valid.sort_values(["sharpe", "total_return_pct"], ascending=False)

print(f"\nPhase 1 loaded: {len(df_phase1):,} rows  ({len(df_p1_valid):,} valid)")
if len(df_p1_sorted) > 0:
    b1 = df_p1_sorted.iloc[0]
    print(
        f"  Phase 1 best Sharpe = {b1['sharpe']:.3f}  Return = {b1['total_return_pct']:+.2f}%"
    )

if REPORT_ONLY:
    # Load previously saved phase 2 CSV — skip all backtesting
    p2_path = reports_dir / "silver_bullet_optimization_phase2.csv"
    if not p2_path.exists():
        print(f"ERROR: {p2_path} not found. Set REPORT_ONLY=False to run phase 2.")
        sys.exit(1)
    df_phase2 = pd.read_csv(p2_path)
    df_p2_valid = df_phase2.dropna(subset=["sharpe"]).copy()
    df_p2_valid = df_p2_valid[df_p2_valid["trades"] >= MIN_TRADES]
    df_p2_sorted = df_p2_valid.sort_values(
        ["sharpe", "total_return_pct"], ascending=False
    )
    print(
        f"Phase 2 loaded from CSV: {len(df_phase2):,} rows  ({len(df_p2_valid):,} valid)"
    )
    if len(df_p2_sorted) > 0:
        b2 = df_p2_sorted.iloc[0]
        print(
            f"  Phase 2 best Sharpe = {b2['sharpe']:.3f}  Return = {b2['total_return_pct']:+.2f}%"
        )
    # Still need df_raw for the footer candle count
    print(f"\nLoading {SYMBOL} data for footer info...")
    df_raw = load_data(
        SYMBOL, asset_type=ASSET_TYPE, start_date=START_DATE, end_date=END_DATE
    )
    print(f"  {len(df_raw):,} rows loaded.")


# ---------------------------------------------------------------------------
# Single-run helper
# ---------------------------------------------------------------------------


def run_one(
    rr_ratio,
    liquidity_window,
    sl_buffer_pts,
    session_cap_usd,
    sweep_lookback,
    pullback_window,
):
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


if not REPORT_ONLY:
    # ---------------------------------------------------------------------------
    # Load market data (only needed for backtesting)
    # ---------------------------------------------------------------------------
    print(f"\nLoading {SYMBOL} data ({START_DATE} to {END_DATE})...")
    df_raw = load_data(
        SYMBOL, asset_type=ASSET_TYPE, start_date=START_DATE, end_date=END_DATE
    )
    print(f"  {len(df_raw):,} rows  [{df_raw.index.min()} -> {df_raw.index.max()}]")

    # ---------------------------------------------------------------------------
    # Build phase 2 grid from top phase-1 seeds
    # ---------------------------------------------------------------------------

    top_seeds = df_p1_sorted.head(TOP_N_PHASE1)

    phase2_sets = {k: set() for k in PHASE2_DELTAS}
    for _, seed in top_seeds.iterrows():
        for param, deltas in PHASE2_DELTAS.items():
            lo, hi = PHASE2_BOUNDS[param]
            for d in deltas:
                val = seed[param] + d
                val = max(lo, min(hi, val))
                if param in ("liquidity_window", "sweep_lookback", "pullback_window"):
                    val = int(round(val))
                phase2_sets[param].add(val)

    phase2_grid = {k: sorted(v) for k, v in phase2_sets.items()}
    print(f"\nPhase 2 grid sizes: { {k: len(v) for k, v in phase2_grid.items()} }")

    keys = list(phase2_grid.keys())
    combos = list(itertools.product(*[phase2_grid[k] for k in keys]))
    total = len(combos)
    print(f"  {total:,} combinations to evaluate")

    results = []
    t_start = time.time()
    for i, combo in enumerate(combos, 1):
        kwargs = dict(zip(keys, combo))
        results.append(run_one(**kwargs))
        if i % 50 == 0 or i == total:
            elapsed = time.time() - t_start
            rate = i / elapsed
            remaining = (total - i) / rate if rate > 0 else 0
            print(
                f"  {i:>5}/{total}  elapsed {elapsed:>5.0f}s  ~{remaining:>5.0f}s remaining"
            )

    df_phase2 = pd.DataFrame(results)
    df_phase2.to_csv(reports_dir / "silver_bullet_optimization_phase2.csv", index=False)

    df_p2_valid = df_phase2.dropna(subset=["sharpe"]).copy()
    df_p2_valid = df_p2_valid[df_p2_valid["trades"] >= MIN_TRADES]
    df_p2_sorted = df_p2_valid.sort_values(
        ["sharpe", "total_return_pct"], ascending=False
    )

    print(
        f"\nPhase 2 done: {len(df_phase2)} runs  ({len(df_p2_valid)} valid)  "
        f"in {time.time() - t_start:.0f}s"
    )
    if len(df_p2_sorted) > 0:
        b = df_p2_sorted.iloc[0]
        print(
            f"  Best Sharpe = {b['sharpe']:.3f}  Return = {b['total_return_pct']:+.2f}%  "
            f"Trades = {int(b['trades'])}"
        )
        print(
            f"  rr={b['rr_ratio']}  lw={int(b['liquidity_window'])}  "
            f"sl_buf={b['sl_buffer_pts']}  cap={b['session_cap_usd']}  "
            f"sweep={int(b['sweep_lookback'])}  pb={int(b['pullback_window'])}"
        )

# ---------------------------------------------------------------------------
# HTML Report
# ---------------------------------------------------------------------------

print("\nBuilding HTML report...")
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# Force standard JSON serialisation — avoids the binary bdata format introduced
# in Plotly.py 5.x which requires a matching Plotly.js 2.x CDN to render.
pio.json.config.default_engine = "json"


def scatter_html(df_valid, title):
    if len(df_valid) == 0:
        return "<p>No valid results.</p>"
    # Convert to plain Python lists so Plotly 6.x does not binary-encode the arrays
    x = df_valid["sharpe"].tolist()
    y = df_valid["total_return_pct"].tolist()
    sizes = np.clip(df_valid["trades"] / df_valid["trades"].max() * 30, 6, 30).tolist()
    colors = df_valid["sharpe"].tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            marker=dict(
                size=sizes,
                color=colors,
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


def violin_html(df_valid, param, title):
    if len(df_valid) == 0 or param not in df_valid.columns:
        return ""
    vals = sorted(df_valid[param].unique())
    fig = go.Figure()
    for v in vals:
        sub = df_valid[df_valid[param] == v]["sharpe"].dropna()
        # Use .tolist() to prevent binary encoding of numpy arrays
        fig.add_trace(
            go.Box(y=sub.tolist(), name=str(v), boxpoints="all", jitter=0.3, pointpos=0)
        )
    fig.update_layout(
        title=title,
        yaxis_title="Sharpe Ratio",
        template="plotly_white",
        height=400,
        showlegend=False,
        margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def table_html(df, n=20):
    sub = df.head(n).copy()
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
    return sub.to_html(index=False, border=0, classes="results-table", escape=False)


p1_scatter = scatter_html(df_p1_valid, "Phase 1 Coarse — Sharpe vs Return")
p2_scatter = scatter_html(df_p2_sorted, "Phase 2 Fine — Sharpe vs Return")

violin_charts = ""
for param in PHASE2_DELTAS:
    violin_charts += violin_html(
        df_p1_valid, param, f"Phase 1 — Sharpe distribution by {param}"
    )

final_sorted = df_p2_sorted if len(df_p2_sorted) > 0 else df_p1_sorted
best_table = table_html(final_sorted, 20)
p1_table = table_html(df_p1_sorted, 20)

best_kpis = ""
if len(final_sorted) > 0:
    b = final_sorted.iloc[0]

    def kpi(label, value):
        return (
            f'<div style="background:rgba(255,255,255,.22); padding:16px; '
            f'border-radius:8px; text-align:center;">'
            f'<div style="font-size:10px; opacity:.85; text-transform:uppercase;">{label}</div>'
            f'<div style="font-size:26px; font-weight:bold;">{value}</div></div>'
        )

    best_kpis = (
        '<div style="display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); '
        'gap:12px; margin-bottom:24px;">'
        + kpi("RR Ratio", b["rr_ratio"])
        + kpi("Liquidity Window", int(b["liquidity_window"]))
        + kpi("SL Buffer (pts)", b["sl_buffer_pts"])
        + kpi("Session Cap ($)", f"{b['session_cap_usd']:,.0f}")
        + kpi("Sweep Lookback", int(b["sweep_lookback"]))
        + kpi("FVG Pullback Win", int(b["pullback_window"]))
        + kpi("Sharpe", f"{b['sharpe']:.3f}")
        + kpi("Return on Margin", f"{b['total_return_pct']:+.2f}%")
        + kpi("Win Rate", f"{b['win_rate']:.1f}%")
        + kpi("Max Drawdown", f"{b['max_drawdown_pct']:.2f}%")
        + "</div>"
    )

full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Silver Bullet — Optimisation Report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ box-sizing:border-box; margin:0; padding:0; }}
  body {{ font-family:'Segoe UI',sans-serif; background:#f0f2f5; color:#2c3e50; }}
  .header {{ background:linear-gradient(135deg,#1a1a2e 0%,#16213e 50%,#0f3460 100%);
             color:white; padding:40px 20px; text-align:center; }}
  .header h1 {{ font-size:30px; margin-bottom:8px; }}
  .header p  {{ font-size:13px; opacity:.8; }}
  .section {{ max-width:1600px; margin:24px auto; background:white;
              border-radius:12px; padding:28px;
              box-shadow:0 2px 12px rgba(0,0,0,.08); }}
  .section h2 {{ font-size:18px; margin-bottom:16px;
                 border-bottom:2px solid #667eea; padding-bottom:8px; }}
  .best-banner {{ background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);
                  color:white; border-radius:12px; padding:28px;
                  max-width:1600px; margin:24px auto; }}
  .best-banner h2 {{ font-size:18px; margin-bottom:16px; opacity:.9; }}
  .charts-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:20px; }}
  .results-table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  .results-table th {{ background:#667eea; color:white; padding:8px 10px; text-align:left; }}
  .results-table td {{ padding:6px 10px; border-bottom:1px solid #eee; }}
  .results-table tr:hover td {{ background:#f8f9ff; }}
  .footer {{ text-align:center; padding:20px; color:#888; font-size:12px; }}
</style>
</head>
<body>

<div class="header">
  <h1>Silver Bullet Strategy (ICT) — Parameter Optimisation</h1>
  <p>Asset: {SYMBOL} &nbsp;|&nbsp; Period: {START_DATE} to {END_DATE} &nbsp;|&nbsp;
     Phase 1: {len(df_phase1):,} runs &nbsp;|&nbsp;
     Phase 2: {len(df_phase2):,} runs &nbsp;|&nbsp;
     Min trades filter: {MIN_TRADES}</p>
</div>

<div class="best-banner">
  <h2>Best Parameter Set — optimised for Sharpe Ratio</h2>
  {best_kpis}
</div>

<div class="section">
  <h2>Phase 1 — Coarse Search: Sharpe vs Return on Margin</h2>
  {p1_scatter}
</div>

<div class="section">
  <h2>Phase 1 — Sharpe Distribution by Parameter Value</h2>
  <div class="charts-grid">{violin_charts}</div>
</div>

<div class="section">
  <h2>Phase 2 — Fine Search: Sharpe vs Return on Margin</h2>
  {p2_scatter}
</div>

<div class="section">
  <h2>Top 20 Results (Phase 2) — sorted by Sharpe</h2>
  {best_table}
</div>

<div class="section">
  <h2>Top 20 Phase 1 Results — sorted by Sharpe</h2>
  {p1_table}
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
print("\nDone.")
if len(final_sorted) > 0:
    b = final_sorted.iloc[0]
    print(f"\nRecommended parameters:")
    print(f"  rr_ratio         = {b['rr_ratio']}")
    print(f"  liquidity_window = {int(b['liquidity_window'])}")
    print(f"  sl_buffer_pts    = {b['sl_buffer_pts']}")
    print(f"  session_cap_usd  = {b['session_cap_usd']}")
    print(f"  sweep_lookback   = {int(b['sweep_lookback'])}")
    print(f"  pullback_window  = {int(b['pullback_window'])}")
    print(f"\nExpected performance (in-sample):")
    print(f"  Sharpe           = {b['sharpe']:.3f}")
    print(f"  Return on Margin = {b['total_return_pct']:+.2f}%")
    print(f"  Win Rate         = {b['win_rate']:.1f}%")
    print(f"  Max Drawdown     = {b['max_drawdown_pct']:.2f}%")
    print(f"  Trades           = {int(b['trades'])}")
