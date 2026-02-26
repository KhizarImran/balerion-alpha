import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

p = project_root / "reports" / "silver_bullet_optimization_phase1.csv"
if not p.exists():
    print("Phase 1 CSV not found.")
    sys.exit(1)

df = pd.read_csv(p)
print(f"Rows saved   : {len(df)}")

valid = df.dropna(subset=["sharpe"])
valid = valid[valid["trades"] >= 10].copy()
sorted_df = valid.sort_values(["sharpe", "total_return_pct"], ascending=False)

print(f"Valid runs   : {len(valid)}")

if len(sorted_df) == 0:
    print("No valid results yet.")
    sys.exit(0)

b = sorted_df.iloc[0]
print(f"\nBest so far:")
print(f"  Sharpe       = {b['sharpe']:.3f}")
print(f"  Return       = {b['total_return_pct']:+.2f}%")
print(f"  Win Rate     = {b['win_rate']:.1f}%")
print(f"  Trades       = {int(b['trades'])}")
print(f"  rr_ratio     = {b['rr_ratio']}")
print(f"  liq_window   = {int(b['liquidity_window'])}")
print(f"  sl_buffer    = {b['sl_buffer_pts']}")
print(f"  session_cap  = {b['session_cap_usd']}")
print(f"  sweep_look   = {int(b['sweep_lookback'])}")
print(f"  pullback_win = {int(b['pullback_window'])}")

print("\nTop 10:")
cols = [
    "rr_ratio",
    "liquidity_window",
    "sl_buffer_pts",
    "session_cap_usd",
    "sweep_lookback",
    "pullback_window",
    "sharpe",
    "total_return_pct",
    "win_rate",
    "trades",
]
print(sorted_df[cols].head(10).to_string(index=False))
