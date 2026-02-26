import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from utils import load_data
from strategy import calculate_signals
import vectorbt as vbt
import numpy as np

df_raw = load_data(
    "US30", asset_type="indices", start_date="2025-11-11", end_date="2026-02-25"
)
print(f"Data loaded: {len(df_raw)} rows")

test_cases = [
    dict(
        rr_ratio=2.0,
        liquidity_window=240,
        sl_buffer_pts=1.0,
        session_cap_usd=-2000,
        sweep_lookback=120,
        pullback_window=15,
    ),
    dict(
        rr_ratio=1.5,
        liquidity_window=100,
        sl_buffer_pts=0.0,
        session_cap_usd=-500,
        sweep_lookback=60,
        pullback_window=5,
    ),
    dict(
        rr_ratio=4.0,
        liquidity_window=500,
        sl_buffer_pts=5.0,
        session_cap_usd=-5000,
        sweep_lookback=240,
        pullback_window=30,
    ),
]

for i, p in enumerate(test_cases, 1):
    df = calculate_signals(df_raw.copy(), lot_size=10, **p)
    n = int(df["buy_signal_daily"].sum())
    print(f"Case {i}: {n} entries  rr={p['rr_ratio']} lw={p['liquidity_window']}")
    if n >= 3:
        pf = vbt.Portfolio.from_signals(
            close=df["close"],
            entries=df["buy_signal_daily"],
            exits=df["sell_signal"],
            init_cash=10_000_000,
            size=10,
            size_type="amount",
            sl_stop=df["sl_stop_frac"],
            tp_stop=df["tp_stop_frac"],
            fees=0.0001,
            freq="1min",
        )
        stats = pf.stats()
        print(
            f"  Sharpe={stats['Sharpe Ratio']:.2f}  Trades={int(stats['Total Trades'])}"
        )

print("Smoke test passed.")
