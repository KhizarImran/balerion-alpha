"""
TrendBO — Live Execution Bot

Connects to MetaTrader 5, fetches 1-hour bars for the specified symbol on
every candle close, runs the TrendBO strategy logic, and executes trades
automatically via the MT5 Python API.

Pass the symbol via the --symbol argument at runtime (see USAGE below).
One process per symbol — run multiple instances for multiple pairs.

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
- The symbol must be visible in Market Watch

=============================================================================
STRATEGY SUMMARY
=============================================================================
Asset:     Any FX pair supported by the broker, 1-hour bars
Sessions:  UTC 07:00 - 20:00 (London open to NY close)
           No trades outside this window.

Long entry (all must be true):
  1. EMA 50 > EMA 200  (bullish macro trend)
  2. Prior bar was in an active consolidation episode:
       - ADX < 25 for >= CONSOL_MIN_BARS consecutive bars
       - 20-bar rolling range < RANGE_ATR_MULT * ATR_14
       - rolling range >= MIN_CONSOLIDATION_PIPS pips wide
  3. Current bar closes above the consolidation rolling high (breakout)
  4. Current ATR >= ATR_EXPAND_MULT * prior ATR  (volatility expanding)
  5. One setup per consolidation episode (first qualifying bar wins)

Short entry: mirror of long with inverted conditions.

Stop Loss:   Long  — rolling low at breakout bar
             Short — rolling high at breakout bar
Take Profit: RISK_REWARD * SL distance
Sizing:      Risk-based — 1% of ACCOUNT_SIZE per trade
             lots = (ACCOUNT_SIZE * RISK_PER_TRADE) / (sl_pips * pip_value_per_lot)
             pip_value_per_lot computed from live symbol rate at startup

Magic:       MAGIC = 4444 for all TrendBO instances regardless of symbol.
             Isolation between symbols is achieved by positions_get(symbol=SYMBOL).

No time-based exit — SL/TP handles all exits.

=============================================================================
USAGE
=============================================================================
    # Single pair:
    uv run python experiments/TrendBO/live.py --symbol USDJPY

    # Multiple pairs — one process each:
    uv run python experiments/TrendBO/live.py --symbol USDJPY
    uv run python experiments/TrendBO/live.py --symbol EURUSD
    uv run python experiments/TrendBO/live.py --symbol GBPUSD

Each process writes its own log file: live_USDJPY.log, live_EURUSD.log, etc.
The script runs forever. Press Ctrl+C to stop.
It wakes up every ~3600 seconds (after each 1-hour candle close), checks
for signals, and sleeps again until the next hour. Telegram alerts are
sent on all key events.
"""

import os
import sys
import io
import time
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta

project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import talib
import requests
import MetaTrader5 as mt5
from dotenv import load_dotenv


# ---------------------------------------------------------------------------
# Symbol — parsed from --symbol command-line argument
# ---------------------------------------------------------------------------

_parser = argparse.ArgumentParser(description="TrendBO Live Bot")
_parser.add_argument(
    "--symbol",
    required=True,
    help="MT5 symbol to trade, e.g. USDJPY, EURUSD, GBPUSD",
)
_args = _parser.parse_args()

SYMBOL = _args.symbol.upper()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# MT5 terminal path — set to the terminal64.exe of the specific installation
# you want this strategy to use. None = auto-detect (only works if you have
# exactly one MT5 installed).
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"

# Magic number — same for all TrendBO instances regardless of symbol.
# Each process is isolated by symbol via positions_get(symbol=SYMBOL).
MAGIC = 4444

_UTC_TZ = timezone.utc

# Pip size — derived from symbol (JPY pairs use 0.01, all others 0.0001)
_JPY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "CADJPY", "CHFJPY", "NZDJPY"}
PIP = 0.01 if SYMBOL in _JPY_PAIRS else 0.0001

# Strategy parameters — match backtest.py values exactly
EMA_FAST = 50
EMA_SLOW = 200
ATR_PERIOD = 14
ADX_PERIOD = 14
RANGE_PERIOD = 20
CONSOL_MIN_BARS = 5
ADX_THRESHOLD = 25
RANGE_ATR_MULT = 4.0
MIN_CONSOLIDATION_PIPS = 10
ATR_EXPAND_MULT = 1.0
MIN_SL_PIPS = 10
RISK_REWARD = 2.0
SESSION_START_HOUR = 7  # UTC — London open
SESSION_END_HOUR = 20  # UTC — NY close

# Account & sizing
ACCOUNT_SIZE = 10_000  # real margin capital (USD)
LEVERAGE = 100  # 1:100
RISK_PER_TRADE = 0.01  # 1% of account per trade

# Instrument constants
STD_LOT = 100_000  # units per standard lot

# _pip_value_per_lot is NOT hardcoded here — it is computed from the live
# symbol rate once at startup via _compute_pip_value() and stored as a
# module-level variable. See connect_mt5().

# How many 1-hour bars to fetch on each tick
BARS_TO_FETCH = 500  # must exceed EMA_SLOW (200) + ADX_PERIOD + RANGE_PERIOD warmup

# Seconds before :00 to wake up so we don't miss the candle close
WAKEUP_EARLY_SECS = 5


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Computed from live symbol rate in connect_mt5() — used by _compute_lot_units()
_pip_value_per_lot: float = 0.0

# Number of decimal places for this symbol's prices — set from mt5.symbol_info().digits
# in connect_mt5(). Used for SL/TP rounding and log formatting.
_price_digits: int = 5

# Deduplication: stores the timestamp of the last bar we entered on.
# Prevents re-firing the same consolidation breakout signal on every hourly tick
# while the bar remains the "latest" bar in the 500-bar window.
_last_entry_ts: "pd.Timestamp | None" = None


# ---------------------------------------------------------------------------
# Signal calculation (self-contained — no dependency on strategy.py)
# ---------------------------------------------------------------------------


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attach all indicator columns to a copy of the DataFrame using ta-lib.

    Indicators added:
      ema_fast   — EMA(close, EMA_FAST)
      ema_slow   — EMA(close, EMA_SLOW)
      atr        — ATR(high, low, close, ATR_PERIOD)
      adx        — ADX(high, low, close, ADX_PERIOD)
      roll_high  — rolling max of high over RANGE_PERIOD bars
      roll_low   — rolling min of low  over RANGE_PERIOD bars
      roll_range — roll_high - roll_low
    """
    df = df.copy()

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    df["ema_fast"] = talib.EMA(close, timeperiod=EMA_FAST)
    df["ema_slow"] = talib.EMA(close, timeperiod=EMA_SLOW)
    df["atr"] = talib.ATR(high, low, close, timeperiod=ATR_PERIOD)
    df["adx"] = talib.ADX(high, low, close, timeperiod=ADX_PERIOD)

    df["roll_high"] = df["high"].rolling(RANGE_PERIOD).max()
    df["roll_low"] = df["low"].rolling(RANGE_PERIOD).min()
    df["roll_range"] = df["roll_high"] - df["roll_low"]

    return df


def _find_consolidation_episodes(df: pd.DataFrame) -> pd.Series:
    """
    Returns a Series of integer episode IDs (0 = not in consolidation).

    A new episode starts after CONSOL_MIN_BARS consecutive bars with:
      - ADX < ADX_THRESHOLD
      - roll_range < RANGE_ATR_MULT * ATR
      - roll_range >= MIN_CONSOLIDATION_PIPS * PIP

    Episode ends when any condition breaks.
    """
    adx = df["adx"].values
    roll_range = df["roll_range"].values
    atr = df["atr"].values

    min_range_abs = MIN_CONSOLIDATION_PIPS * PIP

    episodes = np.zeros(len(df), dtype=int)
    episode_id = 0
    consec = 0
    in_episode = False

    for i in range(len(df)):
        ranging = (
            not np.isnan(adx[i])
            and adx[i] < ADX_THRESHOLD
            and not np.isnan(roll_range[i])
            and not np.isnan(atr[i])
            and atr[i] > 0
            and roll_range[i] < RANGE_ATR_MULT * atr[i]
            and roll_range[i] >= min_range_abs
        )

        if ranging:
            consec += 1
            if consec >= CONSOL_MIN_BARS:
                if not in_episode:
                    episode_id += 1
                    in_episode = True
                episodes[i] = episode_id
        else:
            consec = 0
            if in_episode:
                in_episode = False

    return pd.Series(episodes, index=df.index, name="consol_episode")


def calculate_signals(df: pd.DataFrame) -> dict:
    """
    Run the TrendBO signal logic on a UTC-aware 1-hour OHLCV DataFrame
    and return a summary dict describing the most recently closed bar.

    Parameters
    ----------
    df : pd.DataFrame
        UTC-aware 1H OHLCV with columns: open, high, low, close, volume.
        Must have at least BARS_TO_FETCH rows for indicator warmup.

    Returns
    -------
    dict with keys:
        signal        : "long" | "short" | None
        entry_price   : float | None
        sl_price      : float | None
        tp_price      : float | None
        sl_dist_pips  : float | None
        lots          : float | None
        latest_close  : float
        latest_ts     : pd.Timestamp
        session_active: bool   — True if latest bar falls in session window
    """
    df = _add_indicators(df)
    df["consol_episode"] = _find_consolidation_episodes(df)

    used_episodes: set = set()

    signal_result = {
        "signal": None,
        "entry_price": None,
        "sl_price": None,
        "tp_price": None,
        "sl_dist_pips": None,
        "lots": None,
        "latest_close": float(df["close"].iloc[-1]),
        "latest_ts": df.index[-1],
        "session_active": False,
    }

    # Session check for the latest bar
    latest_hour = df.index[-1].hour
    signal_result["session_active"] = (
        SESSION_START_HOUR <= latest_hour < SESSION_END_HOUR
    )

    # Scan all bars to build used_episodes and detect the signal on the latest bar.
    # We must scan the full window so that used_episodes is populated correctly —
    # if an earlier bar in this 500-bar window already fired a setup for a given
    # episode, the latest bar must not fire again for the same episode.
    warmup = EMA_SLOW + ADX_PERIOD + RANGE_PERIOD + 1

    for i in range(warmup, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]

        episode = int(row["consol_episode"])
        prev_episode = int(prev["consol_episode"])

        # Breakout bar: current bar not in consolidation, previous bar was
        if prev_episode == 0:
            continue
        if prev_episode in used_episodes:
            continue

        # Session filter (UTC hour of the bar timestamp)
        hour = df.index[i].hour
        if not (SESSION_START_HOUR <= hour < SESSION_END_HOUR):
            continue

        close = row["close"]
        prev_roll_high = prev["roll_high"]
        prev_roll_low = prev["roll_low"]

        if pd.isna(prev_roll_high) or pd.isna(prev_roll_low):
            continue

        # ATR expansion check
        if pd.isna(row["atr"]) or pd.isna(prev["atr"]) or prev["atr"] <= 0:
            continue
        if row["atr"] < ATR_EXPAND_MULT * prev["atr"]:
            continue

        # EMA trend direction
        ema_fast_val = row["ema_fast"]
        ema_slow_val = row["ema_slow"]
        if pd.isna(ema_fast_val) or pd.isna(ema_slow_val):
            continue

        bullish_trend = ema_fast_val > ema_slow_val
        bearish_trend = ema_fast_val < ema_slow_val

        fired = False

        # Long breakout
        if bullish_trend and close > prev_roll_high:
            entry = close
            sl_raw = prev_roll_low
            sl_dist = max(abs(entry - sl_raw), MIN_SL_PIPS * PIP)
            sl = entry - sl_dist
            tp = entry + RISK_REWARD * sl_dist

            if sl < entry:
                sl_pips = sl_dist / PIP
                lots = _compute_lot_units(sl_pips)

                if i == len(df) - 1:
                    # This is the latest bar — populate the signal result
                    signal_result["signal"] = "long"
                    signal_result["entry_price"] = entry
                    signal_result["sl_price"] = sl
                    signal_result["tp_price"] = tp
                    signal_result["sl_dist_pips"] = sl_pips
                    signal_result["lots"] = lots

                fired = True

        # Short breakout
        elif bearish_trend and close < prev_roll_low:
            entry = close
            sl_raw = prev_roll_high
            sl_dist = max(abs(sl_raw - entry), MIN_SL_PIPS * PIP)
            sl = entry + sl_dist
            tp = entry - RISK_REWARD * sl_dist

            if sl > entry:
                sl_pips = sl_dist / PIP
                lots = _compute_lot_units(sl_pips)

                if i == len(df) - 1:
                    signal_result["signal"] = "short"
                    signal_result["entry_price"] = entry
                    signal_result["sl_price"] = sl
                    signal_result["tp_price"] = tp
                    signal_result["sl_dist_pips"] = sl_pips
                    signal_result["lots"] = lots

                fired = True

        if fired:
            used_episodes.add(prev_episode)

    return signal_result


# ---------------------------------------------------------------------------
# Position sizing
# ---------------------------------------------------------------------------


def _compute_lot_units(sl_pips: float) -> float:
    """
    Size a trade so hitting SL loses exactly RISK_PER_TRADE * ACCOUNT_SIZE.

        risk_$  = ACCOUNT_SIZE * RISK_PER_TRADE
        lots    = risk_$ / (sl_pips * _pip_value_per_lot)

    _pip_value_per_lot is set from the live USDJPY rate at startup.
    Returns lots rounded to 2 decimal places (standard MT5 precision).
    Clamped to a minimum of 0.01 lots.
    """
    if sl_pips <= 0 or _pip_value_per_lot <= 0:
        return 0.01
    risk_dollars = ACCOUNT_SIZE * RISK_PER_TRADE
    lots = risk_dollars / (sl_pips * _pip_value_per_lot)
    lots = max(round(lots, 2), 0.01)
    return lots


def _compute_pip_value(symbol: str, rate: float) -> float:
    """
    Compute USD pip value per standard lot for the given symbol and live rate.

    USD-quoted pairs (EURUSD, GBPUSD, AUDUSD, NZDUSD):
        pip_value = PIP * STD_LOT  — $10.00 flat, quote currency is already USD.

    All other pairs (JPY-quoted, CAD-quoted, etc.):
        pip_value = (PIP * STD_LOT) / rate
        For JPY pairs this is exact (e.g. USDJPY=150 -> $6.67/pip/lot).
        For other non-USD-quoted pairs it is an acceptable approximation.
    """
    _USD_QUOTED = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD"}
    if symbol in _USD_QUOTED:
        return PIP * STD_LOT
    return (PIP * STD_LOT) / rate


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
            Path(__file__).resolve().parent / f"live_{SYMBOL}.log",
            encoding="utf-8",
        ),
    ],
)
log = logging.getLogger("TrendBO")


# ---------------------------------------------------------------------------
# Credentials — loaded from .env
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
    """
    Initialise and log into MT5. On success:
      - fetches symbol_info to set _price_digits
      - fetches the live rate to compute _pip_value_per_lot via _compute_pip_value()
    Returns True on success.
    """
    global _pip_value_per_lot, _price_digits

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

    # Fetch symbol metadata — digits used for SL/TP rounding and log formatting
    sym_info = mt5.symbol_info(SYMBOL)
    if sym_info is None:
        log.error(f"symbol_info({SYMBOL}) failed: {mt5.last_error()}")
        mt5.shutdown()
        return False
    _price_digits = sym_info.digits

    # Brief pause so the terminal is ready to serve data
    time.sleep(2)

    # Compute pip value per lot from live symbol rate
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        log.error(f"Cannot get tick for {SYMBOL} after login: {mt5.last_error()}")
        mt5.shutdown()
        return False

    live_rate = tick.ask
    _pip_value_per_lot = _compute_pip_value(SYMBOL, live_rate)
    log.info(
        f"MT5 connected | Account: {info.login} | "
        f"Balance: ${info.balance:,.2f} | "
        f"Server: {info.server}"
    )
    log.info(
        f"{SYMBOL} rate at startup: {live_rate:.{_price_digits}f} | "
        f"Pip value per lot: ${_pip_value_per_lot:.4f} | "
        f"Price digits: {_price_digits}"
    )
    return True


def ensure_connected() -> None:
    """Block until MT5 is connected, retrying every 60 seconds."""
    if mt5.account_info() is not None:
        return

    log.warning("MT5 not connected — retrying every 60 seconds...")
    send_telegram(
        "<b>TrendBO Alert</b>\nMT5 connection lost. Retrying every 60 seconds..."
    )

    attempt = 0
    while True:
        attempt += 1
        log.info(f"Connection attempt {attempt}...")
        if connect_mt5():
            info = mt5.account_info()
            msg = (
                f"<b>TrendBO</b> — MT5 reconnected\n"
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
    Fetch the last n closed 1-hour bars from MT5 for SYMBOL.

    pos=1 skips the currently forming bar; returns n fully closed bars.
    Returns a UTC-aware DataFrame with columns: open, high, low, close, volume.
    Index is a DatetimeIndex in UTC.
    """
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 1, n)

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
    USDJPY prices have 3 decimal places; SL/TP rounded to 3dp.
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
        "sl": round(sl_price, _price_digits),
        "tp": round(tp_price, _price_digits),
        "deviation": 20,  # FX standard
        "magic": MAGIC,
        "comment": "TrendBO",
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
        f"@ {result.price:.{_price_digits}f} | "
        f"SL {sl_price:.{_price_digits}f} | TP {tp_price:.{_price_digits}f} | "
        f"ticket #{result.order}"
    )
    return result


def close_position(position: "mt5.TradePosition") -> tuple:
    """
    Close the given open position at market.
    Returns (success: bool, pnl_pips: float).
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
        "deviation": 20,
        "magic": MAGIC,
        "comment": "TrendBO-Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        log.error(f"close order_send failed | retcode: {retcode} | {mt5.last_error()}")
        return False, 0.0

    if direction == 0:  # was long
        pnl_pips = (result.price - position.price_open) / PIP
    else:  # was short
        pnl_pips = (position.price_open - result.price) / PIP

    log.info(
        f"Position closed | ticket #{position.ticket} | "
        f"close @ {result.price:.{_price_digits}f} | P&L: {pnl_pips:+.1f} pips"
    )
    return True, pnl_pips


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def seconds_until_next_hour() -> float:
    """
    Returns seconds until the next top-of-hour UTC,
    minus WAKEUP_EARLY_SECS so we wake up slightly before the candle
    is fully formed and available in MT5.
    """
    now = datetime.now(_UTC_TZ)
    next_hr = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    delta = (next_hr - now).total_seconds() - WAKEUP_EARLY_SECS
    return max(delta, 0.0)


def now_utc_str() -> str:
    """Current time as a readable UTC string."""
    return datetime.now(_UTC_TZ).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run() -> None:
    """Main execution loop. Runs indefinitely until Ctrl+C."""
    global _last_entry_ts

    log.info("=" * 60)
    log.info("  TrendBO Live Bot — starting up")
    log.info(f"  Symbol    : {SYMBOL}")
    log.info(f"  Timeframe : 1-hour")
    log.info(f"  Magic     : {MAGIC}")
    log.info(
        f"  Session   : {SESSION_START_HOUR:02d}:00 - {SESSION_END_HOUR:02d}:00 UTC"
    )
    log.info(f"  R:R       : {RISK_REWARD}  |  Risk/trade: {RISK_PER_TRADE * 100:.1f}%")
    log.info(f"  EMA       : {EMA_FAST} / {EMA_SLOW}")
    log.info(f"  ADX thr   : {ADX_THRESHOLD}  |  Range/ATR: {RANGE_ATR_MULT}")
    log.info("=" * 60)

    # --- Connect to MT5 (blocks until successful) ---
    log.info("Connecting to MT5...")
    while not connect_mt5():
        log.warning("MT5 connection failed — retrying in 60s...")
        time.sleep(60)

    balance = get_balance()
    startup_msg = (
        f"<b>TrendBO Bot Online</b>\n"
        f"Symbol: {SYMBOL} | Magic: {MAGIC}\n"
        f"Session: {SESSION_START_HOUR:02d}:00 - {SESSION_END_HOUR:02d}:00 UTC\n"
        f"R:R: {RISK_REWARD} | Risk/trade: {RISK_PER_TRADE * 100:.1f}%\n"
        f"Pip value/lot: ${_pip_value_per_lot:.4f}\n"
        f"Account balance: <b>${balance:,.2f}</b>"
    )
    log.info(startup_msg.replace("<b>", "").replace("</b>", ""))
    send_telegram(startup_msg)

    # --- Main loop ---
    while True:
        # Sleep until the top of the next hour
        wait = seconds_until_next_hour()
        log.debug(f"Sleeping {wait:.1f}s until next candle close...")
        time.sleep(wait)

        # Small extra buffer to ensure MT5 has the completed H1 bar
        time.sleep(WAKEUP_EARLY_SECS + 1)

        # --- Ensure connection is alive ---
        ensure_connected()

        # --- Fetch bars ---
        try:
            df = fetch_bars(BARS_TO_FETCH)
        except RuntimeError as e:
            log.error(f"fetch_bars error: {e}")
            send_telegram(f"<b>TrendBO Error</b>\nfetch_bars failed: {e}")
            continue

        # --- Evaluate ---
        try:
            result = calculate_signals(df)
        except Exception as e:
            log.error(f"calculate_signals error: {e}", exc_info=True)
            send_telegram(f"<b>TrendBO Error</b>\ncalculate_signals failed: {e}")
            continue

        signal = result["signal"]
        entry_price = result["entry_price"]
        sl_price = result["sl_price"]
        tp_price = result["tp_price"]
        sl_dist_pips = result["sl_dist_pips"]
        lots = result["lots"]
        latest_ts = result["latest_ts"]
        latest_close = result["latest_close"]
        session_active = result["session_active"]

        log.info(
            f"[{now_utc_str()}] "
            f"Bar: {latest_ts}  Close: {latest_close:.3f}  "
            f"Session active: {session_active}"
        )

        # ----------------------------------------------------------------
        # Check if a position is already open (this bot's own trade)
        # ----------------------------------------------------------------
        position = get_open_position()

        if position is not None:
            direction_str = "LONG" if position.type == 0 else "SHORT"
            unrealised_pips = (
                (latest_close - position.price_open) / PIP
                if position.type == 0
                else (position.price_open - latest_close) / PIP
            )
            log.info(
                f"Position open: {direction_str} {position.volume} lots | "
                f"Open @ {position.price_open:.3f} | "
                f"Unrealised: {unrealised_pips:+.1f} pips | "
                f"Current P&L: ${position.profit:.2f}"
            )

        else:
            # ----------------------------------------------------------------
            # No position — check for new entry signal
            # ----------------------------------------------------------------
            if signal is not None:
                # Deduplication: skip if we already entered on this exact bar
                if _last_entry_ts is not None and latest_ts == _last_entry_ts:
                    log.info(
                        f"Signal on {latest_ts} already acted on — skipping "
                        f"(dedup: _last_entry_ts={_last_entry_ts})"
                    )
                else:
                    sl_dist_pts = sl_dist_pips if sl_dist_pips is not None else 0.0
                    tp_dist_pips = sl_dist_pts * RISK_REWARD

                    log.info(
                        f"SIGNAL: {signal.upper()} | "
                        f"Entry: {entry_price:.3f} | "
                        f"SL: {sl_price:.3f} ({sl_dist_pts:.1f} pips) | "
                        f"TP: {tp_price:.3f} ({tp_dist_pips:.1f} pips) | "
                        f"Lots: {lots:.2f}"
                    )

                    order_result = open_position(signal, lots, sl_price, tp_price)

                    if order_result is not None:
                        _last_entry_ts = latest_ts
                        balance = get_balance()
                        direction_label = "BUY" if signal == "long" else "SELL"
                        msg = (
                            f"<b>TrendBO — {direction_label} {SYMBOL}</b>\n"
                            f"Entry:   {entry_price:.3f}\n"
                            f"SL:      {sl_price:.3f}  ({sl_dist_pts:.1f} pips)\n"
                            f"TP:      {tp_price:.3f}  ({tp_dist_pips:.1f} pips)\n"
                            f"Lots:    {lots:.2f}\n"
                            f"Balance: ${balance:,.2f}"
                        )
                        log.info(msg.replace("<b>", "").replace("</b>", ""))
                        send_telegram(msg)
                    else:
                        send_telegram(
                            f"<b>TrendBO ERROR</b>\n"
                            f"order_send failed for {signal.upper()} {SYMBOL}.\n"
                            f"Check MT5 terminal — trade may not have executed."
                        )
            else:
                log.debug(
                    f"No signal | Session active: {session_active} | "
                    f"Close: {latest_close:.3f}"
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log.info("Shutdown requested by user (Ctrl+C)")
        send_telegram("<b>TrendBO Bot</b> — stopped by user (Ctrl+C)")
        mt5.shutdown()
        log.info("MT5 disconnected. Goodbye.")
    except Exception as e:
        log.critical(f"Unhandled exception: {e}", exc_info=True)
        send_telegram(f"<b>TrendBO CRITICAL ERROR</b>\n{e}\nBot has stopped.")
        mt5.shutdown()
        raise
