"""
ICT Silver Bullet Strategy — Signal Generation & Visualization

Strategy Logic:
  - Asset:      US30 (Dow Jones), 1-minute data
  - Sessions:   Three ICT killzone windows (all times EST):
                  London       03:00 – 04:00
                  New York AM  10:00 – 11:00
                  New York PM  14:00 – 15:00
  - Structure:
      1. Liquidity sweep: a 240-bar low has been swept (low_swept),
         but a 240-bar high has NOT yet been swept (bullish bias).
      2. Bullish FVG (Fair Value Gap): candle N-2 high < candle N low,
         creating an imbalance that price pulls back into.
      3. Entry: price retraces into the FVG zone during any session window.
      4. One entry per session per trading day (first qualifying bar wins).
  - Stop Loss:  Below the swept liquidity low minus a 1-point buffer.
  - Take Profit: 2:1 Risk/Reward ratio (TP distance = 2 × SL distance).
  - Time Exit:  52-bar fallback if neither SL nor TP is hit.
  - Session Cap: Stop trading a session once cumulative loss exceeds -$2,000.
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from utils import load_data, plot_candlestick_with_signals


# ---------------------------------------------------------------------------
# Signal Calculation
# ---------------------------------------------------------------------------


def calculate_signals(
    df: pd.DataFrame,
    liquidity_window: int = 240,
    sweep_lookback: int = 120,
    pullback_window: int = 15,
    rr_ratio: float = 2.0,
    sl_buffer_pts: float = 1.0,
    session_cap_usd: float = -2_000.0,
    lot_size: int = 10,
) -> pd.DataFrame:
    """
    Calculate ICT Silver Bullet entry signals across three sessions.

    Sessions (all New York / Eastern time):
      - London      : 03:00 – 03:59 ET
      - New York AM : 10:00 – 10:59 ET
      - New York PM : 14:00 – 14:59 ET

    One entry is allowed per session per calendar day, subject to the
    session P&L cap.  A qualifying bar must:
      1. Have lows swept (bullish liquidity bias).
      2. Have highs NOT swept (bullish bias confirmed).
      3. Close inside an active bullish FVG zone.
      4. Fall within one of the three session windows.
      5. The session's cumulative estimated P&L has not hit session_cap_usd.

    Timezone note — broker server time:
      The MT5 data timestamps are in broker server time, which is UTC+2 in
      winter (EET) and UTC+3 in summer (EEST).  Eastern Time is UTC-5 in
      winter (EST) and UTC-4 in summer (EDT).

      To convert broker server time -> ET we use pytz so that DST transitions
      in both the EU and the US are handled automatically:

        server (naive) -> assume UTC+2/+3 via 'Europe/Athens'
        then convert   -> 'America/New_York'

      Flat-offset summary for reference:
        Period                     EU tz    ET tz   Server -> ET offset
        Nov–Mar (both on winter)   UTC+2    UTC-5   subtract 7 h
        Mar–Oct (both on summer)   UTC+3    UTC-4   subtract 7 h
        Late Mar gap (EU summer,   UTC+3    UTC-5   subtract 8 h
          US still winter)
        Early Nov gap (EU winter,  UTC+2    UTC-4   subtract 6 h
          US still summer)

    Args:
        df:                  OHLCV DataFrame with naive datetime index in
                             broker server time (UTC+2 winter / UTC+3 summer).
        liquidity_window:    Rolling window (bars) for defining significant
                             highs/lows (default 240 = 4 hours on 1-min data).
        sweep_lookback:      Bars to look back when detecting whether a
                             liquidity level has been swept (default 120 = 2 h).
        pullback_window:     How many bars the FVG zone is kept alive for a
                             pullback entry (default 15 minutes).
        rr_ratio:            Take-profit = rr_ratio × SL distance (default 2.0).
        sl_buffer_pts:       Extra points below the swept liquidity low for the
                             stop loss (default 1.0 point).
        session_cap_usd:     Maximum cumulative loss per session per day in $.
                             Once hit, no further entries are taken in that
                             session for the rest of that day (default -2000).
        lot_size:            Lot size used for session cap estimation — must
                             match the lot size used in the backtest (default 10).

    Returns:
        DataFrame with extra columns:
          - liquidity_high / liquidity_low    : rolling extremes
          - bullish_fvg                       : True on the bar the gap forms
          - bullish_fvg_top/_bottom           : price levels of the gap
          - low_swept / high_swept            : liquidity sweep flags
          - in_bullish_fvg                    : price inside the gap zone
          - in_london / in_ny_am / in_ny_pm   : per-session window flags
          - in_session                        : True in any of the three windows
          - session_label                     : 'london' | 'ny_am' | 'ny_pm' | ''
          - buy_signal                        : raw entry signals (all qualifying bars)
          - buy_signal_daily                  : one entry per session per day
                                                (after session cap applied)
          - sl_price                          : stop loss price for each entry bar
          - tp_price                          : take profit price for each entry bar
          - sl_stop_frac                      : SL as fraction of entry price
                                                (for vectorbt sl_stop parameter)
          - tp_stop_frac                      : TP as fraction of entry price
                                                (for vectorbt tp_stop parameter)
          - sell_signal                       : 52-bar fallback time exit
    """
    df = df.copy()

    high = df["high"]
    low = df["low"]
    close = df["close"]

    # ------------------------------------------------------------------
    # 1. Liquidity levels — rolling high/low over liquidity_window bars
    # ------------------------------------------------------------------
    df["liquidity_high"] = high.rolling(window=liquidity_window, min_periods=1).max()
    df["liquidity_low"] = low.rolling(window=liquidity_window, min_periods=1).min()

    # ------------------------------------------------------------------
    # 2. Liquidity sweep detection
    #    A level is "swept" if price has touched it at any point in the
    #    last sweep_lookback bars (rolling max of the boolean touch flag).
    # ------------------------------------------------------------------
    high_touched = high >= df["liquidity_high"]
    df["high_swept"] = (
        high_touched.rolling(window=sweep_lookback, min_periods=1).max().astype(bool)
    )

    low_touched = low <= df["liquidity_low"]
    df["low_swept"] = (
        low_touched.rolling(window=sweep_lookback, min_periods=1).max().astype(bool)
    )

    # ------------------------------------------------------------------
    # 3. Bullish Fair Value Gap (FVG)
    #    On bar N: gap exists when candle[N-2].high < candle[N].low
    #    (a two-bar imbalance to the upside).
    # ------------------------------------------------------------------
    df["bullish_fvg"] = high.shift(2) < low

    # FVG price zone — only defined on bars where the gap forms
    df["bullish_fvg_top"] = low.where(df["bullish_fvg"])
    df["bullish_fvg_bottom"] = high.shift(2).where(df["bullish_fvg"])

    # Forward-fill the zone for pullback_window bars so price can
    # re-enter the gap after it forms
    df["bullish_fvg_top_ff"] = df["bullish_fvg_top"].ffill(limit=pullback_window)
    df["bullish_fvg_bottom_ff"] = df["bullish_fvg_bottom"].ffill(limit=pullback_window)

    # Price is "in the FVG" when close is between the two levels
    df["in_bullish_fvg"] = (close <= df["bullish_fvg_top_ff"]) & (
        close >= df["bullish_fvg_bottom_ff"]
    )

    # ------------------------------------------------------------------
    # 4. Session windows — DST-aware conversion to Eastern Time
    #
    #    Data timestamps are in broker server time (Europe/Athens):
    #      UTC+2 in winter (EET), UTC+3 in summer (EEST)
    #    Target timezone: America/New_York (EST/EDT auto-handled by pytz)
    #
    #    Sessions (ET):
    #      London       03:00 – 03:59
    #      New York AM  10:00 – 10:59
    #      New York PM  14:00 – 14:59
    # ------------------------------------------------------------------
    import pytz

    server_tz = pytz.timezone("Europe/Athens")  # broker server: UTC+2/+3
    eastern_tz = pytz.timezone("America/New_York")

    # Localise the naive index as broker server time, then convert to ET
    server_index = df.index.tz_localize(
        server_tz, ambiguous="infer", nonexistent="shift_forward"
    )
    eastern_index = server_index.tz_convert(eastern_tz)
    et_hour = eastern_index.hour

    df["in_london"] = et_hour == 3
    df["in_ny_am"] = et_hour == 10
    df["in_ny_pm"] = et_hour == 14
    df["in_session"] = df["in_london"] | df["in_ny_am"] | df["in_ny_pm"]

    # Label each bar with its session name (empty string outside sessions)
    df["session_label"] = ""
    df.loc[df["in_london"], "session_label"] = "london"
    df.loc[df["in_ny_am"], "session_label"] = "ny_am"
    df.loc[df["in_ny_pm"], "session_label"] = "ny_pm"

    # ------------------------------------------------------------------
    # 5. Raw entry condition (all qualifying bars)
    #    Bullish bias  : lows swept, highs NOT swept
    #    FVG pullback  : close is inside the active bullish FVG zone
    #    Session filter: inside any of the three session windows
    # ------------------------------------------------------------------
    df["buy_signal"] = (
        df["low_swept"] & ~df["high_swept"] & df["in_bullish_fvg"] & df["in_session"]
    )

    # ------------------------------------------------------------------
    # 6. One entry per SESSION per calendar day + session P&L cap
    #
    #    Process each (date, session) group chronologically.
    #    Track a running estimated P&L for the session — when it drops
    #    below session_cap_usd, no further entries are allowed that day.
    #
    #    P&L estimate per trade:
    #      - SL hit   -> loss  = -sl_distance * lot_size
    #      - TP hit   -> gain  =  sl_distance * rr_ratio * lot_size
    #    We assume the trade resolves at SL or TP (50/50 prior) but only
    #    use the SL loss for a conservative cap estimate — i.e. if the
    #    previous trade in the session was a loser we count its full SL loss.
    #    Winners don't offset the cap (conservative approach).
    # ------------------------------------------------------------------
    df["buy_signal_daily"] = False
    df["sl_price"] = np.nan
    df["tp_price"] = np.nan
    df["_date"] = df.index.date

    for (date, session), group in df.groupby(["_date", "session_label"]):
        if session == "":
            continue

        session_pnl = 0.0  # running estimated loss for this session
        prev_sl_dist = None  # SL distance (points) of last trade taken

        day_session_mask = group["buy_signal"]

        for bar_time, is_signal in day_session_mask.items():
            if not is_signal:
                continue

            # Apply session cap: if running loss already at/below cap, skip
            if session_pnl <= session_cap_usd:
                continue

            entry_price = close.loc[bar_time]
            liq_low = df.loc[bar_time, "liquidity_low"]

            sl_price = liq_low - sl_buffer_pts
            sl_dist = entry_price - sl_price

            # Guard: skip if SL distance is zero or negative (malformed bar)
            if sl_dist <= 0:
                continue

            tp_price = entry_price + sl_dist * rr_ratio

            df.loc[bar_time, "buy_signal_daily"] = True
            df.loc[bar_time, "sl_price"] = sl_price
            df.loc[bar_time, "tp_price"] = tp_price

            # Estimate the P&L impact of this trade for the cap
            # Conservatively assume a loss (SL hit) to be cautious
            estimated_loss = -sl_dist * lot_size
            session_pnl += estimated_loss
            prev_sl_dist = sl_dist

            # Only allow one entry per session per day
            break

    df.drop(columns=["_date"], inplace=True)

    # ------------------------------------------------------------------
    # 7. Compute SL/TP as fractions of entry price for vectorbt
    #    sl_stop_frac = sl_distance / entry_price  (fraction below entry)
    #    tp_stop_frac = tp_distance / entry_price  (fraction above entry)
    # ------------------------------------------------------------------
    entry_close = close.where(df["buy_signal_daily"])
    sl_dist_col = entry_close - df["sl_price"]
    tp_dist_col = df["tp_price"] - entry_close

    df["sl_stop_frac"] = sl_dist_col / entry_close
    df["tp_stop_frac"] = tp_dist_col / entry_close

    # ------------------------------------------------------------------
    # 8. Time-based fallback exit — 52 bars (minutes) after each entry
    #    vectorbt will use SL/TP first; this only triggers if neither fires.
    # ------------------------------------------------------------------
    exit_bars = 52
    df["sell_signal"] = False

    for entry_time in df.index[df["buy_signal_daily"]]:
        entry_pos = df.index.get_loc(entry_time)
        exit_pos = min(entry_pos + exit_bars, len(df) - 1)
        exit_time = df.index[exit_pos]
        df.loc[exit_time, "sell_signal"] = True

    return df


# ---------------------------------------------------------------------------
# Experiment Runner
# ---------------------------------------------------------------------------


def run_experiment(
    symbol: str = "US30",
    asset_type: str = "indices",
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    liquidity_window: int = 240,
    sweep_lookback: int = 120,
    pullback_window: int = 15,
    show_chart: bool = True,
    save_chart: bool = True,
) -> tuple[pd.DataFrame, object]:
    """
    Run the Silver Bullet signal experiment and visualise results.

    Args:
        symbol:            Trading symbol (default 'US30').
        asset_type:        'fx' or 'indices' (default 'indices').
        start_date:        Date range start (YYYY-MM-DD).
        end_date:          Date range end (YYYY-MM-DD).
        liquidity_window:  Rolling window for significant highs/lows.
        sweep_lookback:    Lookback for sweep detection.
        pullback_window:   Bars to keep FVG zone alive.
        show_chart:        Open chart in browser.
        save_chart:        Save chart HTML to reports/.

    Returns:
        (df, fig) tuple.
    """
    print(f"Loading {symbol} data ({start_date} to {end_date})...")
    df = load_data(
        symbol, asset_type=asset_type, start_date=start_date, end_date=end_date
    )
    print(f"  Loaded {len(df):,} rows  [{df.index.min()} -> {df.index.max()}]")

    print("Calculating Silver Bullet signals (London / NY AM / NY PM)...")
    df = calculate_signals(
        df,
        liquidity_window=liquidity_window,
        sweep_lookback=sweep_lookback,
        pullback_window=pullback_window,
    )

    n_entries = int(df["buy_signal_daily"].sum())
    n_exits = int(df["sell_signal"].sum())
    n_london = int((df["buy_signal_daily"] & df["in_london"]).sum())
    n_ny_am = int((df["buy_signal_daily"] & df["in_ny_am"]).sum())
    n_ny_pm = int((df["buy_signal_daily"] & df["in_ny_pm"]).sum())
    print(
        f"  Entry signals : {n_entries}  (London={n_london}, NY AM={n_ny_am}, NY PM={n_ny_pm})"
    )
    print(f"  Exit signals  : {n_exits}")

    # Build a display-friendly sample — last 5 trading days so the chart
    # is not too dense on 1-minute data for a long date range
    print("Generating visualization (last 5 trading days of data)...")
    unique_dates = sorted(set(df.index.date))
    if len(unique_dates) == 0:
        raise ValueError(
            f"No data found for {symbol} between {start_date} and {end_date}.\n"
            "Check the date range — available data may differ."
        )
    display_cutoff = unique_dates[max(0, len(unique_dates) - 5)]
    display_start = pd.Timestamp(display_cutoff)
    df_display = df[df.index >= display_start]

    fig = plot_candlestick_with_signals(
        df_display,
        title=f"{symbol} - ICT Silver Bullet  (liq={liquidity_window}, sweep={sweep_lookback}, pfbk={pullback_window})",
        buy_signals=df_display["buy_signal_daily"],
        sell_signals=df_display["sell_signal"],
        indicators={
            "Liquidity High": df_display["liquidity_high"],
            "Liquidity Low": df_display["liquidity_low"],
            "FVG Top": df_display["bullish_fvg_top_ff"],
            "FVG Bottom": df_display["bullish_fvg_bottom_ff"],
        },
        show_volume=True,
    )

    if save_chart:
        reports_dir = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        output_file = (
            reports_dir / f"silver_bullet_{symbol}_{start_date}_{end_date}.html"
        )
        fig.write_html(str(output_file))
        print(f"  Chart saved -> {output_file}")

    if show_chart:
        fig.show()

    return df, fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df, fig = run_experiment(
        symbol="US30",
        asset_type="indices",
        start_date="2025-11-11",
        end_date="2026-02-25",
        show_chart=True,
        save_chart=True,
    )
