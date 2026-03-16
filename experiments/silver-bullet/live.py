"""
ICT Silver Bullet — Live Execution Bot

Connects to MetaTrader 5, fetches US30 1-minute bars on every candle close,
runs the ICT Silver Bullet strategy logic, and executes trades automatically
via the MT5 Python API.

=============================================================================
SETUP — copy .env.example to .env and fill in your values
=============================================================================

    cp .env.example .env   (or copy manually on Windows)

Required variables in .env:
    MT5_LOGIN        your MT5 account number (integer)
    MT5_PASSWORD     your MT5 password
    MT5_SERVER       broker server name shown on the MT5 login screen

Optional variables (Telegram alerts):
    TELEGRAM_TOKEN   bot token from @BotFather
    TELEGRAM_CHAT_ID your personal chat ID

=============================================================================
TELEGRAM SETUP
=============================================================================
1. Open Telegram and message @BotFather
2. Send /newbot — follow prompts to name your bot
3. BotFather gives you a TOKEN like: 7123456789:AAHdqTcvCHL1K2b...
4. Message your new bot once (send any text) so it can receive messages
5. Open a browser and visit:
       https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates
   Find "chat":{"id": XXXXXXX} — that number is your CHAT_ID
6. Add both to your .env file

=============================================================================
MT5 SETUP
=============================================================================
- MetaTrader 5 terminal must be installed and running on this machine
- "Allow automated trading" must be enabled:
      MT5 -> Tools -> Options -> Expert Advisors -> Allow automated trading
- Your account must be logged in before running this script
- US30 must be visible in Market Watch

=============================================================================
STRATEGY SUMMARY
=============================================================================
Asset:    US30 (Dow Jones), 1-minute bars
Sessions: Three ICT killzone windows in Eastern Time:
            London      03:00 ET
            New York AM 10:00 ET
            New York PM 14:00 ET

Long entry (all must be true):
  1. Rolling liquidity LOW swept within last SWEEP_LOOKBACK bars (bullish bias)
  2. Corresponding HIGH has NOT been swept (bias intact)
  3. Bullish FVG exists: high[N-2] < low[N]
  4. Price pulls back inside FVG zone within PULLBACK_WINDOW bars
  5. Bar falls within a killzone window

Short entry: mirror of long with inverted conditions.

One trade per session per calendar day (first qualifying bar wins).
Session cap: stops new entries once cumulative estimated loss exceeds
SESSION_CAP_USD for that session on that day. Resets on bot restart.

Stop Loss:   Long  — below swept liquidity low - SL_BUFFER_PTS
             Short — above swept liquidity high + SL_BUFFER_PTS
Take Profit: RR_RATIO x SL distance
Time exit:   52 bars (minutes) if neither SL nor TP is hit

=============================================================================
USAGE
=============================================================================
    uv run python experiments/silver-bullet/live.py

The script runs forever. Press Ctrl+C to stop.
It wakes up every ~60 seconds (after each 1-minute candle close), checks
for signals, and sleeps again until the next minute. Telegram alerts are
sent on all key events.
"""

import os
import sys
import io
import time
import math
import logging
import zoneinfo
from pathlib import Path
from datetime import datetime, timezone, timedelta, date

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import requests
import MetaTrader5 as mt5
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYMBOL = "US30"

# MT5 terminal path — set to the terminal64.exe of the specific installation
# you want this strategy to use. None = auto-detect (only works if you have
# exactly one MT5 installed).
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Timezone used for killzone session windows
_ET_TZ = zoneinfo.ZoneInfo("America/New_York")
_UTC_TZ = timezone.utc

# ICT killzone hours in Eastern Time (start of each 1-hour window)
_KILLZONE_HOURS_ET = {3, 10, 14}
_SESSION_NAMES = {3: "london", 10: "ny_am", 14: "ny_pm"}

# Strategy parameters — match backtest.py optimised values
LIQUIDITY_WINDOW = 150  # rolling window (bars) for significant high/low levels
SWEEP_LOOKBACK = 50  # bars to look back for sweep detection
# NOTE: strategy.py default is 50; 210 is the value
# found optimal in optimize.py and used in backtest.py
PULLBACK_WINDOW = 10  # bars the FVG zone stays alive for pullback entry
RR_RATIO = 2.0  # take profit = RR_RATIO x SL distance
SL_BUFFER_PTS = 0.0  # extra points beyond liquidity level for SL
SESSION_CAP_USD = -5_500.0  # max cumulative estimated loss per session per day
TIME_EXIT_BARS = 52  # close trade at market after this many 1-minute bars

# Position sizing — fixed lot (US30 index, mirrors backtest.py)
LOT_SIZE = 1  # lots per trade
POINT = 1.0  # US30 moves in whole points (not pips)

# How many 1-minute bars to fetch from MT5 on each tick
BARS_TO_FETCH = 500  # must exceed LIQUIDITY_WINDOW (150) + SWEEP_LOOKBACK (210)

# Seconds before :00 to wake up so we don't miss the candle close
WAKEUP_EARLY_SECS = 3

# Magic number — uniquely identifies trades placed by this bot
MAGIC = 20261001


# ---------------------------------------------------------------------------
# Session cap — in-memory (resets on restart, mirrors backtest logic)
# ---------------------------------------------------------------------------

# Key: "YYYY-MM-DD_session_name" -> cumulative estimated loss ($) this session
_session_pnl: dict[str, float] = {}

# Active trade tracker for time exit
# Key: ticket (int) -> bars elapsed since entry (int)
_active_trades: dict[int, int] = {}


# ---------------------------------------------------------------------------
# Signal calculation (self-contained — no dependency on strategy.py)
# ---------------------------------------------------------------------------


def _get_et_hours(index: pd.DatetimeIndex) -> np.ndarray:
    """
    Convert a UTC-aware DatetimeIndex to Eastern Time hours.

    MT5 copy_rates_from_pos returns bars with UTC Unix timestamps.
    After our conversion they carry UTC tzinfo, so we just tz_convert.
    """
    et_idx = index.tz_convert(_ET_TZ)
    return et_idx.hour.to_numpy()


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the ICT Silver Bullet signal logic on a UTC-aware 1-minute OHLCV
    DataFrame and return it with signal columns appended.

    New columns added
    -----------------
    et_hour      : int    — bar open hour in Eastern Time
    session      : str    — "london" | "ny_am" | "ny_pm" | ""
    buy_signal   : bool   — first qualifying long bar in this session today
    sell_signal  : bool   — first qualifying short bar in this session today
    entry_price  : float  — close of the signal bar
    sl_price     : float  — stop-loss price
    tp_price     : float  — take-profit price
    sl_dist      : float  — absolute SL distance in points

    One signal per session per calendar day is emitted (first qualifying bar
    wins). The session cap is applied using the module-level _session_pnl
    dict so state persists across calls within the same bot session.
    """
    df = df.copy()
    n = len(df)

    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    idx = df.index

    # -- Eastern Time hours --
    et_hours = _get_et_hours(idx)
    df["et_hour"] = et_hours
    df["session"] = [_SESSION_NAMES.get(h, "") for h in et_hours]

    # -- 1. Rolling liquidity levels --
    liq_high = (
        pd.Series(high, index=idx)
        .rolling(LIQUIDITY_WINDOW, min_periods=1)
        .max()
        .to_numpy()
    )
    liq_low = (
        pd.Series(low, index=idx)
        .rolling(LIQUIDITY_WINDOW, min_periods=1)
        .min()
        .to_numpy()
    )

    # -- 2. Sweep detection --
    high_touched = high >= liq_high
    low_touched = low <= liq_low

    high_swept = (
        pd.Series(high_touched.astype(float), index=idx)
        .rolling(SWEEP_LOOKBACK, min_periods=1)
        .max()
        .astype(bool)
        .to_numpy()
    )
    low_swept = (
        pd.Series(low_touched.astype(float), index=idx)
        .rolling(SWEEP_LOOKBACK, min_periods=1)
        .max()
        .astype(bool)
        .to_numpy()
    )

    # -- 3. Bullish FVG: high[N-2] < low[N] --
    bull_fvg_top = np.full(n, np.nan)
    bull_fvg_bot = np.full(n, np.nan)
    for i in range(2, n):
        if high[i - 2] < low[i]:
            bull_fvg_top[i] = low[i]
            bull_fvg_bot[i] = high[i - 2]

    # Forward-fill each bullish FVG zone for PULLBACK_WINDOW bars
    bull_top_ff = np.full(n, np.nan)
    bull_bot_ff = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(bull_fvg_top[i]):
            end = min(n, i + PULLBACK_WINDOW + 1)
            for j in range(i, end):
                if np.isnan(bull_top_ff[j]):
                    bull_top_ff[j] = bull_fvg_top[i]
                    bull_bot_ff[j] = bull_fvg_bot[i]

    in_bull_fvg = (close <= bull_top_ff) & (close >= bull_bot_ff)

    # -- 4. Bearish FVG: low[N-2] > high[N] --
    bear_fvg_top = np.full(n, np.nan)
    bear_fvg_bot = np.full(n, np.nan)
    for i in range(2, n):
        if low[i - 2] > high[i]:
            bear_fvg_top[i] = low[i - 2]
            bear_fvg_bot[i] = high[i]

    bear_top_ff = np.full(n, np.nan)
    bear_bot_ff = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(bear_fvg_top[i]):
            end = min(n, i + PULLBACK_WINDOW + 1)
            for j in range(i, end):
                if np.isnan(bear_top_ff[j]):
                    bear_top_ff[j] = bear_fvg_top[i]
                    bear_bot_ff[j] = bear_fvg_bot[i]

    in_bear_fvg = (close >= bear_bot_ff) & (close <= bear_top_ff)

    # -- 5. Raw setup flags --
    in_killzone = np.isin(et_hours, list(_KILLZONE_HOURS_ET))
    raw_long = low_swept & ~high_swept & in_bull_fvg & in_killzone
    raw_short = high_swept & ~low_swept & in_bear_fvg & in_killzone

    # -- 6. Initialise output columns --
    df["buy_signal"] = False
    df["sell_signal"] = False
    df["entry_price"] = np.nan
    df["sl_price"] = np.nan
    df["tp_price"] = np.nan
    df["sl_dist"] = np.nan

    # ET dates for deduplication (date in Eastern Time, not UTC)
    et_dates = np.array([idx[i].tz_convert(_ET_TZ).date() for i in range(n)])

    # -- 7. Deduplicate: one entry per (et_date, session) --
    seen: set[tuple] = set()

    def _emit(raw_mask: np.ndarray, direction: str) -> None:
        for i in range(n):
            if not raw_mask[i]:
                continue
            sess = _SESSION_NAMES.get(int(et_hours[i]), "")
            if not sess:
                continue

            key = (et_dates[i], sess)
            if key in seen:
                continue

            # Session cap check
            cap_key = f"{et_dates[i]}_{sess}"
            current_pnl = _session_pnl.get(cap_key, 0.0)
            if current_pnl <= SESSION_CAP_USD:
                seen.add(key)
                continue

            entry = close[i]
            if direction == "long":
                sl = liq_low[i] - SL_BUFFER_PTS
                sl_dist = entry - sl
            else:
                sl = liq_high[i] + SL_BUFFER_PTS
                sl_dist = sl - entry

            if sl_dist <= 0:
                continue

            tp = (
                entry + RR_RATIO * sl_dist
                if direction == "long"
                else entry - RR_RATIO * sl_dist
            )

            if direction == "long":
                df.at[idx[i], "buy_signal"] = True
            else:
                df.at[idx[i], "sell_signal"] = True

            df.at[idx[i], "entry_price"] = entry
            df.at[idx[i], "sl_price"] = sl
            df.at[idx[i], "tp_price"] = tp
            df.at[idx[i], "sl_dist"] = sl_dist

            seen.add(key)

    _emit(raw_long, "long")
    _emit(raw_short, "short")

    return df


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            Path(__file__).resolve().parent / "live.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("SilverBullet")


# ---------------------------------------------------------------------------
# Config — loaded from .env (copy .env.example -> .env and fill in values)
# ---------------------------------------------------------------------------

_env_path = project_root / ".env"
load_dotenv(dotenv_path=_env_path)


def _require(key: str) -> str:
    val = os.environ.get(key, "").strip()
    if not val:
        print(f"ERROR: required variable '{key}' is missing from .env")
        print(f"       Copy .env.example to .env and fill in your values.")
        sys.exit(1)
    return val


def _optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


MT5_LOGIN = int(_require("MT5_LOGIN"))
MT5_PASSWORD = _require("MT5_PASSWORD")
MT5_SERVER = _require("MT5_SERVER")

TELEGRAM_TOKEN = _optional("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = _optional("TELEGRAM_CHAT_ID")


# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def send_telegram(message: str) -> None:
    """Send a Telegram message. Silently ignores errors."""
    if not TELEGRAM_TOKEN or TELEGRAM_TOKEN == "your_telegram_bot_token":
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        resp = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"},
            timeout=10,
        )
        if not resp.ok:
            log.warning(f"Telegram send failed: {resp.status_code} {resp.text[:100]}")
    except Exception as e:
        log.warning(f"Telegram error: {e}")


# ---------------------------------------------------------------------------
# MT5 connection
# ---------------------------------------------------------------------------


def connect_mt5() -> bool:
    """Initialise and log into MT5. Returns True on success."""
    init_kwargs: dict = {}
    if MT5_PATH is not None:
        init_kwargs["path"] = MT5_PATH

    if not mt5.initialize(**init_kwargs):
        log.error(f"MT5 initialize() failed: {mt5.last_error()}")
        return False

    if not mt5.login(MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
        log.error(f"MT5 login failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    info = mt5.account_info()
    if info is None:
        log.error("MT5 account_info() returned None after login")
        mt5.shutdown()
        return False

    if not mt5.symbol_select(SYMBOL, True):
        log.error(f"symbol_select({SYMBOL}) failed: {mt5.last_error()}")
        mt5.shutdown()
        return False

    # Brief pause so the terminal is ready to serve historical data
    time.sleep(2)

    log.info(
        f"MT5 connected | Account: {info.login} | "
        f"Balance: ${info.balance:,.2f} | "
        f"Server: {info.server}"
    )
    return True


def ensure_connected() -> None:
    """Block until MT5 is connected, retrying every 60 seconds."""
    if mt5.account_info() is not None:
        return

    log.warning("MT5 not connected — retrying every 60 seconds...")
    send_telegram(
        "<b>SilverBullet Alert</b>\nMT5 connection lost. Retrying every 60 seconds..."
    )

    attempt = 0
    while True:
        attempt += 1
        log.info(f"Connection attempt {attempt}...")
        if connect_mt5():
            info = mt5.account_info()
            msg = (
                f"<b>SilverBullet</b> — MT5 reconnected\n"
                f"Account: {info.login} | Balance: ${info.balance:,.2f}"
            )
            log.info("MT5 reconnected successfully")
            send_telegram(msg)
            return
        time.sleep(60)


# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------


def fetch_bars(n: int = BARS_TO_FETCH) -> pd.DataFrame:
    """
    Fetch the last n closed 1-minute bars from MT5 for SYMBOL.

    pos=1 skips the currently forming bar; returns n fully closed bars.
    Returns a UTC-aware DataFrame with columns: open, high, low, close, volume.
    Index is a DatetimeIndex in UTC.
    """
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_M1, 1, n)

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"copy_rates_from_pos returned no data for {SYMBOL}: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.index.name = "timestamp"

    df = df.rename(columns={"tick_volume": "volume"})[
        ["open", "high", "low", "close", "volume"]
    ]
    return df


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------


def get_balance() -> float:
    info = mt5.account_info()
    if info is None:
        raise RuntimeError("Cannot read account info from MT5")
    return float(info.balance)


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------


def get_open_position() -> "mt5.TradePosition | None":
    """Return the first open position for SYMBOL placed by this bot, or None."""
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None or len(positions) == 0:
        return None
    # Filter by magic number so we only manage our own trades
    bot_positions = [p for p in positions if p.magic == MAGIC]
    return bot_positions[0] if bot_positions else None


def open_position(
    direction: str,
    lots: float,
    sl_price: float,
    tp_price: float,
) -> "mt5.OrderSendResult | None":
    """
    Send a market order. Returns the OrderSendResult on success, None on failure.
    US30 prices are in whole points; round SL/TP to 1 decimal place.
    """
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        log.error(f"Cannot get tick for {SYMBOL}: {mt5.last_error()}")
        return None

    order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
    price = tick.ask if direction == "long" else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": float(lots),
        "type": order_type,
        "price": price,
        "sl": round(sl_price, 1),
        "tp": round(tp_price, 1),
        "deviation": 50,  # US30 can gap; allow wider deviation
        "magic": MAGIC,
        "comment": "SilverBullet",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        log.error(f"order_send failed | retcode: {retcode} | {mt5.last_error()}")
        return None

    log.info(
        f"Order executed | {direction.upper()} {lots} lots {SYMBOL} "
        f"@ {result.price:.1f} | SL {sl_price:.1f} | TP {tp_price:.1f} | "
        f"ticket #{result.order}"
    )
    return result


def close_position(position: "mt5.TradePosition") -> tuple[bool, float]:
    """
    Close the given open position at market.
    Returns (success, pnl_points).
    """
    direction = position.type  # 0 = BUY, 1 = SELL
    close_type = mt5.ORDER_TYPE_SELL if direction == 0 else mt5.ORDER_TYPE_BUY
    tick = mt5.symbol_info_tick(SYMBOL)

    if tick is None:
        log.error(f"Cannot get tick for {SYMBOL}: {mt5.last_error()}")
        return False, 0.0

    close_price = tick.bid if direction == 0 else tick.ask

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": position.volume,
        "type": close_type,
        "position": position.ticket,
        "price": close_price,
        "deviation": 50,
        "magic": MAGIC,
        "comment": "SilverBullet-TimeExit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        log.error(f"close order_send failed | retcode: {retcode} | {mt5.last_error()}")
        return False, 0.0

    if direction == 0:  # was long
        pnl_points = (result.price - position.price_open) / POINT
    else:  # was short
        pnl_points = (position.price_open - result.price) / POINT

    log.info(
        f"Position closed | ticket #{position.ticket} | "
        f"close @ {result.price:.1f} | P&L: {pnl_points:+.1f} pts"
    )
    return True, pnl_points


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------


def evaluate_bar(df: pd.DataFrame) -> dict:
    """
    Run calculate_signals on the fetched bars and return a summary dict
    describing what action (if any) should be taken on the most recently
    closed bar.

    Returns
    -------
    {
        "et_hour"     : int,
        "session"     : str | None,          # "london" | "ny_am" | "ny_pm" | None
        "signal"      : "long" | "short" | None,
        "entry_price" : float | None,
        "sl_price"    : float | None,
        "tp_price"    : float | None,
        "sl_dist"     : float | None,
        "latest_close": float,
        "latest_ts"   : pd.Timestamp,
    }
    """
    df = calculate_signals(df)
    last = df.iloc[-1]

    signal = None
    entry_price = None
    sl_price = None
    tp_price = None
    sl_dist = None

    if bool(last["buy_signal"]):
        signal = "long"
        entry_price = float(last["entry_price"])
        sl_price = float(last["sl_price"])
        tp_price = float(last["tp_price"])
        sl_dist = float(last["sl_dist"])
    elif bool(last["sell_signal"]):
        signal = "short"
        entry_price = float(last["entry_price"])
        sl_price = float(last["sl_price"])
        tp_price = float(last["tp_price"])
        sl_dist = float(last["sl_dist"])

    session_val = str(last["session"])
    session = session_val if session_val else None

    return {
        "et_hour": int(last["et_hour"]),
        "session": session,
        "signal": signal,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_dist": sl_dist,
        "latest_close": float(last["close"]),
        "latest_ts": df.index[-1],
    }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def seconds_until_next_minute() -> float:
    """
    Returns seconds until the next top-of-minute UTC,
    minus WAKEUP_EARLY_SECS so we wake up slightly before
    the candle is fully formed and available in MT5.
    """
    now = datetime.now(_UTC_TZ)
    next_min = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)
    delta = (next_min - now).total_seconds() - WAKEUP_EARLY_SECS
    return max(delta, 0.0)


def now_et_str() -> str:
    """Current time as a readable Eastern Time string."""
    return datetime.now(_ET_TZ).strftime("%Y-%m-%d %H:%M %Z")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run() -> None:
    """Main execution loop. Runs indefinitely until Ctrl+C."""
    log.info("=" * 60)
    log.info("  SilverBullet Live Bot — starting up")
    log.info(f"  Symbol    : {SYMBOL}")
    log.info(f"  Timeframe : 1-minute")
    log.info(f"  Sizing    : fixed lot {LOT_SIZE}  |  R:R: {RR_RATIO}")
    log.info(f"  Killzones : 03:00, 10:00, 14:00 ET (each 60 min)")
    log.info(f"  Time exit : {TIME_EXIT_BARS} bars")
    log.info(f"  Sweep LB  : {SWEEP_LOOKBACK} (optimised)")
    log.info("=" * 60)

    # --- Connect to MT5 (blocks until successful) ---
    log.info("Connecting to MT5...")
    while not connect_mt5():
        log.warning("MT5 connection failed — retrying in 60s...")
        time.sleep(60)

    balance = get_balance()
    startup_msg = (
        f"<b>SilverBullet Bot Online</b>\n"
        f"Symbol: {SYMBOL} | Fixed lot: {LOT_SIZE}\n"
        f"Account balance: <b>${balance:,.2f}</b>\n"
        f"Killzones: 03:00 / 10:00 / 14:00 ET\n"
        f"Time exit: {TIME_EXIT_BARS} bars | R:R {RR_RATIO}"
    )
    log.info(startup_msg.replace("<b>", "").replace("</b>", ""))
    send_telegram(startup_msg)

    # --- Main loop ---
    while True:
        # Sleep until the top of the next minute
        wait = seconds_until_next_minute()
        log.debug(f"Sleeping {wait:.1f}s until next candle close...")
        time.sleep(wait)

        # Small extra buffer to ensure MT5 has the completed bar
        time.sleep(WAKEUP_EARLY_SECS + 1)

        # --- Ensure connection is alive ---
        ensure_connected()

        # --- Fetch bars ---
        try:
            df = fetch_bars(BARS_TO_FETCH)
        except RuntimeError as e:
            log.error(f"fetch_bars error: {e}")
            send_telegram(f"<b>SilverBullet Error</b>\nfetch_bars failed: {e}")
            continue

        # --- Evaluate ---
        try:
            result = evaluate_bar(df)
        except Exception as e:
            log.error(f"evaluate_bar error: {e}", exc_info=True)
            send_telegram(f"<b>SilverBullet Error</b>\nevaluate_bar failed: {e}")
            continue

        et_hour = result["et_hour"]
        session = result["session"]
        signal = result["signal"]
        entry_price = result["entry_price"]
        sl_price = result["sl_price"]
        tp_price = result["tp_price"]
        sl_dist = result["sl_dist"]
        latest_ts = result["latest_ts"]
        latest_close = result["latest_close"]

        log.info(
            f"[{now_et_str()}] "
            f"Bar: {latest_ts}  Close: {latest_close:.1f}  "
            f"ET hour: {et_hour:02d}  Session: {session or 'none'}"
        )

        # ----------------------------------------------------------------
        # Check active trade — increment bar counter, time exit if due
        # ----------------------------------------------------------------
        position = get_open_position()

        if position is not None:
            ticket = position.ticket
            _active_trades[ticket] = _active_trades.get(ticket, 0) + 1
            bars_elapsed = _active_trades[ticket]

            if bars_elapsed >= TIME_EXIT_BARS:
                direction_str = "LONG" if position.type == 0 else "SHORT"
                log.info(
                    f"TIME EXIT — {bars_elapsed} bars elapsed | "
                    f"ticket #{ticket} | {direction_str} | "
                    f"open @ {position.price_open:.1f}"
                )
                success, pnl_points = close_position(position)
                if success:
                    del _active_trades[ticket]
                    balance = get_balance()
                    direction_label = "BUY" if position.type == 0 else "SELL"
                    msg = (
                        f"<b>SilverBullet — Time Exit ({TIME_EXIT_BARS} bars)</b>\n"
                        f"Closed {direction_label} {SYMBOL} @ {latest_close:.1f}\n"
                        f"P&L: <b>{pnl_points:+.1f} pts</b>\n"
                        f"Account balance: ${balance:,.2f}"
                    )
                    log.info(msg.replace("<b>", "").replace("</b>", ""))
                    send_telegram(msg)
                else:
                    send_telegram(
                        f"<b>SilverBullet ERROR</b>\n"
                        f"Failed to close time exit!\n"
                        f"Ticket: #{ticket} — please close manually."
                    )
            else:
                direction_str = "LONG" if position.type == 0 else "SHORT"
                log.info(
                    f"Position open: {direction_str} {position.volume} lots | "
                    f"Open @ {position.price_open:.1f} | "
                    f"Bars elapsed: {bars_elapsed}/{TIME_EXIT_BARS} | "
                    f"Current P&L: ${position.profit:.2f}"
                )

        else:
            # No position open — clean up any stale tracker entries
            # (position was closed by SL or TP, not by us)
            closed_tickets = list(_active_trades.keys())
            for ticket in closed_tickets:
                log.info(
                    f"Position #{ticket} no longer open — closed by SL/TP or externally"
                )
                del _active_trades[ticket]

            # ----------------------------------------------------------------
            # Enter new trade if signal fired
            # ----------------------------------------------------------------
            if signal is not None and session is not None:
                # Session cap check
                et_date_str = latest_ts.tz_convert(_ET_TZ).strftime("%Y-%m-%d")
                cap_key = f"{et_date_str}_{session}"
                current_session_pnl = _session_pnl.get(cap_key, 0.0)

                if current_session_pnl <= SESSION_CAP_USD:
                    log.info(
                        f"SIGNAL skipped — session cap reached | "
                        f"session: {session} | "
                        f"cap_key: {cap_key} | "
                        f"pnl so far: ${current_session_pnl:.2f}"
                    )
                else:
                    log.info(
                        f"SIGNAL: {signal.upper()} | "
                        f"Session: {session} | "
                        f"Entry: {entry_price:.1f} | "
                        f"SL: {sl_price:.1f} ({sl_dist:.1f} pts) | "
                        f"TP: {tp_price:.1f} | "
                        f"Lots: {LOT_SIZE}"
                    )

                    order_result = open_position(signal, LOT_SIZE, sl_price, tp_price)

                    if order_result is not None:
                        # Register in active trade tracker
                        _active_trades[order_result.order] = 0

                        # Update session cap with conservative estimated loss
                        estimated_loss = -sl_dist * LOT_SIZE
                        _session_pnl[cap_key] = current_session_pnl + estimated_loss

                        balance = get_balance()
                        direction_label = "BUY" if signal == "long" else "SELL"
                        msg = (
                            f"<b>SilverBullet — {direction_label} {SYMBOL}</b>\n"
                            f"Session: {session}\n"
                            f"Entry:   {entry_price:.1f}\n"
                            f"SL:      {sl_price:.1f}  ({sl_dist:.1f} pts)\n"
                            f"TP:      {tp_price:.1f}  ({sl_dist * RR_RATIO:.1f} pts)\n"
                            f"Lots:    {LOT_SIZE}\n"
                            f"Balance: ${balance:,.2f}"
                        )
                        log.info(msg.replace("<b>", "").replace("</b>", ""))
                        send_telegram(msg)
                    else:
                        send_telegram(
                            f"<b>SilverBullet ERROR</b>\n"
                            f"order_send failed for {signal.upper()} {SYMBOL}.\n"
                            f"Check MT5 terminal — trade may not have executed."
                        )
            else:
                log.debug(
                    f"No signal | ET hour: {et_hour:02d} | Close: {latest_close:.1f}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log.info("Shutdown requested by user (Ctrl+C)")
        send_telegram("<b>SilverBullet Bot</b> — stopped by user (Ctrl+C)")
        mt5.shutdown()
        log.info("MT5 disconnected. Goodbye.")
    except Exception as e:
        log.critical(f"Unhandled exception: {e}", exc_info=True)
        send_telegram(f"<b>SilverBullet CRITICAL ERROR</b>\n{e}\nBot has stopped.")
        mt5.shutdown()
        raise
