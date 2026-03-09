"""
Range Breakout — Live Execution Bot

Connects to MetaTrader 5, fetches USDJPY 1H bars on every hourly candle
close, runs the Asian range breakout strategy logic, and executes trades
automatically via the MT5 Python API.

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
      MT5 → Tools → Options → Expert Advisors → Allow automated trading
- Your account must be logged in before running this script

=============================================================================
USAGE
=============================================================================
    uv run python experiments/RangeBO/live.py

The script runs forever. Press Ctrl+C to stop.
It wakes up at every :00 (top of the hour), checks for signals, and
sleeps again until the next hour. Telegram alerts are sent on all key events.
"""

import os
import sys
import io
import time
import math
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta

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
# Signal calculation (self-contained — no dependency on strategy.py)
# ---------------------------------------------------------------------------


def calculate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a 1H OHLCV DataFrame (UTC-aware index), compute the Asian range
    per day and detect breakout entries for both long and short directions.

    New columns added
    -----------------
    uk_hour     : int   — bar open hour in UK local time
    range_high  : float — daily range high
    range_low   : float — daily range low
    buy_signal  : bool  — bar that first closes above range_high after 07:00
    sell_signal : bool  — bar that first closes below range_low  after 07:00
    entry_price : float — close of the entry bar
    sl_price    : float — stop-loss price
    tp_price    : float — take-profit price (1:2 R:R)
    """
    df = df.copy()

    df_uk = df.copy()
    df_uk.index = df_uk.index.tz_convert(TIMEZONE)
    df["uk_hour"] = df_uk.index.hour
    df["uk_date"] = df_uk.index.normalize()

    df["range_high"] = np.nan
    df["range_low"] = np.nan
    df["buy_signal"] = False
    df["sell_signal"] = False
    df["entry_price"] = np.nan
    df["sl_price"] = np.nan
    df["tp_price"] = np.nan

    uk_dates = df["uk_date"].unique()

    for uk_day in uk_dates:
        day_mask = df["uk_date"] == uk_day

        range_mask = (
            day_mask
            & (df["uk_hour"] >= RANGE_START_HOUR)
            & (df["uk_hour"] < RANGE_END_HOUR)
        )
        if range_mask.sum() == 0:
            continue

        r_high = df.loc[range_mask, "high"].max()
        r_low = df.loc[range_mask, "low"].min()

        df.loc[day_mask, "range_high"] = r_high
        df.loc[day_mask, "range_low"] = r_low

        trade_mask = (
            day_mask
            & (df["uk_hour"] >= RANGE_END_HOUR)
            & (df["uk_hour"] < TRADE_CLOSE_HOUR)
        )
        trade_bars = df.index[trade_mask]
        trade_taken = False

        for ts in trade_bars:
            if trade_taken:
                break

            bar = df.loc[ts]
            c = bar["close"]

            is_long = c > r_high
            is_short = c < r_low

            if not is_long and not is_short:
                continue

            direction = "long" if is_long else "short"
            entry = c
            sl = r_low if direction == "long" else r_high
            sl_dist = abs(entry - sl)

            if sl_dist <= 0:
                continue

            tp = entry + RR * sl_dist if direction == "long" else entry - RR * sl_dist

            df.loc[ts, "buy_signal"] = direction == "long"
            df.loc[ts, "sell_signal"] = direction == "short"
            df.loc[ts, "entry_price"] = entry
            df.loc[ts, "sl_price"] = sl
            df.loc[ts, "tp_price"] = tp
            trade_taken = True

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
log = logging.getLogger("RangeBO")

# ---------------------------------------------------------------------------
# CONFIG — loaded from .env (copy .env.example → .env and fill in values)
# ---------------------------------------------------------------------------

# Load .env from the project root (balerion-alpha/)
_env_path = project_root / ".env"
load_dotenv(dotenv_path=_env_path)


def _require(key: str) -> str:
    """Read a required env var, exit with a clear message if missing."""
    val = os.environ.get(key, "").strip()
    if not val:
        print(f"ERROR: required variable '{key}' is missing from .env")
        print(f"       Copy .env.example to .env and fill in your values.")
        sys.exit(1)
    return val


def _optional(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# MetaTrader 5 — required
MT5_LOGIN = int(_require("MT5_LOGIN"))
MT5_PASSWORD = _require("MT5_PASSWORD")
MT5_SERVER = _require("MT5_SERVER")

# Telegram — optional (alerts silently disabled if not set)
TELEGRAM_TOKEN = _optional("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = _optional("TELEGRAM_CHAT_ID")

# ---------------------------------------------------------------------------
# MT5 terminal path — set this to the terminal64.exe of the specific MT5
# installation you want this strategy to use.
# Leave as None to let the MT5 Python library find the default installation.
#
# Examples:
#   MT5_PATH = r"C:\Program Files\MetaTrader 5 ICMarkets\terminal64.exe"
#   MT5_PATH = r"C:\Program Files\MetaTrader 5 Pepperstone\terminal64.exe"
#   MT5_PATH = None   # auto-detect (only works if you have one MT5 installed)
# ---------------------------------------------------------------------------
MT5_PATH = r"C:\Program Files\MetaTrader 5\terminal64.exe"  # e.g. r"C:\Program Files\MetaTrader 5 ICMarkets\terminal64.exe"

# Strategy — fixed constants (not user-configurable via .env)
SYMBOL = "USDJPY.pro"
TIMEZONE = "Europe/London"
RANGE_START_HOUR = 0  # 00:00 UK inclusive
RANGE_END_HOUR = 7  # 07:00 UK exclusive
TRADE_CLOSE_HOUR = 16  # 16:00 UK hard exit
RR = 2.0  # reward-to-risk

# ---------------------------------------------------------------------------
# Sizing — edit these two lines to switch mode
#
#   SIZING_MODE = "risk_pct"
#       Size each trade so hitting SL loses exactly RISK_PCT of account balance.
#       RISK_PCT = 0.01 means 1% risk per trade.
#
#   SIZING_MODE = "fixed_lot"
#       Use FIXED_LOT on every trade regardless of SL distance or balance.
# ---------------------------------------------------------------------------
SIZING_MODE = "risk_pct"  # "risk_pct" | "fixed_lot"
RISK_PCT = 0.005  # used when SIZING_MODE = "risk_pct"  (1% = 0.01)
FIXED_LOT = 0.10  # used when SIZING_MODE = "fixed_lot"
PIP = 0.01  # USDJPY pip size (0.01 for JPY pairs)
STD_LOT = 100_000  # units per standard lot
MIN_LOT = 0.01  # minimum lot size accepted by broker
LOT_STEP = 0.01  # lot granularity — MT5 rejects more than 2 decimal places

# How many 1H bars to fetch from MT5 each tick
BARS_TO_FETCH = 100

# Seconds before :00 to wake up early so we don't miss the candle close
WAKEUP_EARLY_SECS = 5

# ---------------------------------------------------------------------------
# Telegram
# ---------------------------------------------------------------------------


def send_telegram(message: str) -> None:
    """
    Send a Telegram message. Silently ignores errors so a Telegram outage
    never kills the trading loop.
    """
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
    Initialise and log into MT5. Returns True on success.
    Uses MT5_PATH to target a specific terminal installation if set.
    """
    init_kwargs = {}
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

    # Ensure the symbol is visible in Market Watch so data calls work
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
    """
    Block until MT5 is connected, retrying every 60 seconds.
    Sends a Telegram alert on first failure and on recovery.
    """
    if mt5.account_info() is not None:
        return  # already connected

    log.warning("MT5 not connected — retrying every 60 seconds...")
    send_telegram(
        "<b>RangeBO Alert</b>\nMT5 connection lost. Retrying every 60 seconds..."
    )

    attempt = 0
    while True:
        attempt += 1
        log.info(f"Connection attempt {attempt}...")
        if connect_mt5():
            info = mt5.account_info()
            msg = (
                f"<b>RangeBO</b> — MT5 reconnected\n"
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
    Fetch the last n closed 1H bars from MT5 for SYMBOL.
    Bar at index 0 in copy_rates_from_pos is the most recently CLOSED bar
    (the current forming bar is excluded by using pos=1).

    Returns a UTC-aware DataFrame with columns: open, high, low, close, volume.
    Index is a DatetimeIndex in UTC.
    """
    # pos=1 means skip the currently forming bar; fetch n closed bars
    rates = mt5.copy_rates_from_pos(SYMBOL, mt5.TIMEFRAME_H1, 1, n)

    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"copy_rates_from_pos returned no data for {SYMBOL}: {mt5.last_error()}"
        )

    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.index.name = "timestamp"

    # Keep only columns used by calculate_signals
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


def compute_lots(sl_pips: float, balance: float) -> float:
    """
    Return the lot size to use for the next trade.

    SIZING_MODE = "risk_pct":
        Size so that hitting SL loses exactly RISK_PCT * balance.

        Pip value per lot is derived from the live USDJPY bid price:
            pip_value_per_lot ($) = (PIP / usdjpy_rate) * STD_LOT
        e.g. at 150.00:  (0.01 / 150.00) * 100,000 = $6.67 per pip per lot

        Then:
            risk_$  = balance * RISK_PCT
            lots    = risk_$ / (sl_pips * pip_value_per_lot)

    SIZING_MODE = "fixed_lot":
        Always return FIXED_LOT regardless of SL distance or balance.

    Result is always rounded DOWN to 2 decimal places (LOT_STEP = 0.01).
    MT5 rejects lot sizes with more than 2 decimal places.
    Floored at MIN_LOT (0.01).
    """
    if SIZING_MODE == "fixed_lot":
        return FIXED_LOT

    if sl_pips <= 0:
        return MIN_LOT

    # Fetch live bid price from MT5 to compute exact pip value
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        log.warning(
            f"Cannot get live tick for pip value calculation — "
            f"falling back to MIN_LOT: {mt5.last_error()}"
        )
        return MIN_LOT

    usdjpy_rate = tick.bid
    pip_value_per_lot = (PIP / usdjpy_rate) * STD_LOT  # USD per pip per standard lot

    risk_dollars = balance * RISK_PCT
    raw_lots = risk_dollars / (sl_pips * pip_value_per_lot)

    # Round DOWN to 2 decimal places — never round up (avoids exceeding risk)
    lots = math.floor(raw_lots * 100) / 100
    lots = max(lots, MIN_LOT)

    log.debug(
        f"Lot sizing | rate: {usdjpy_rate:.3f} | "
        f"pip_val/lot: ${pip_value_per_lot:.4f} | "
        f"risk: ${risk_dollars:.2f} | "
        f"raw lots: {raw_lots:.4f} | "
        f"final lots: {lots}"
    )
    return lots


# ---------------------------------------------------------------------------
# Order execution
# ---------------------------------------------------------------------------


def get_open_position() -> "mt5.TradePosition | None":
    """
    Return the first open position for SYMBOL, or None if flat.
    """
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None or len(positions) == 0:
        return None
    return positions[0]


def open_position(
    direction: str,  # "long" or "short"
    lots: float,
    sl_price: float,
    tp_price: float,
) -> bool:
    """
    Send a market order. Returns True on success.
    """
    tick = mt5.symbol_info_tick(SYMBOL)
    if tick is None:
        log.error(f"Cannot get tick for {SYMBOL}: {mt5.last_error()}")
        return False

    order_type = mt5.ORDER_TYPE_BUY if direction == "long" else mt5.ORDER_TYPE_SELL
    price = tick.ask if direction == "long" else tick.bid

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": SYMBOL,
        "volume": lots,
        "type": order_type,
        "price": price,
        "sl": round(sl_price, 3),
        "tp": round(tp_price, 3),
        "deviation": 20,  # max price deviation in points
        "magic": 20260306,  # magic number to identify bot trades
        "comment": "RangeBO",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        log.error(f"order_send failed | retcode: {retcode} | {mt5.last_error()}")
        return False

    log.info(
        f"Order executed | {direction.upper()} {lots} lots {SYMBOL} "
        f"@ {result.price:.3f} | SL {sl_price:.3f} | TP {tp_price:.3f} | "
        f"ticket #{result.order}"
    )
    return True


def close_position(position) -> tuple[bool, float]:
    """
    Close the given open position at market.
    Returns (success, pnl_pips).
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
        "magic": 20260306,
        "comment": "RangeBO-TimeExit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        retcode = result.retcode if result else "None"
        log.error(f"close order_send failed | retcode: {retcode} | {mt5.last_error()}")
        return False, 0.0

    # P&L in pips
    if direction == 0:  # was long
        pnl_pips = (result.price - position.price_open) / PIP
    else:  # was short
        pnl_pips = (position.price_open - result.price) / PIP

    log.info(
        f"Position closed | ticket #{position.ticket} | "
        f"close @ {result.price:.3f} | P&L: {pnl_pips:+.1f} pips"
    )
    return True, pnl_pips


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------


def evaluate_bar(df: pd.DataFrame) -> dict:
    """
    Run calculate_signals on the fetched bars and return a summary dict
    describing what action (if any) should be taken based on the most
    recently closed bar.

    Returns
    -------
    {
        "uk_hour"     : int,
        "range_high"  : float | None,
        "range_low"   : float | None,
        "signal"      : "long" | "short" | None,
        "entry_price" : float | None,
        "sl_price"    : float | None,
        "tp_price"    : float | None,
        "sl_pips"     : float | None,
        "latest_close": float,
        "latest_ts"   : pd.Timestamp,
    }
    """
    df = calculate_signals(df)
    last = df.iloc[-1]

    range_high = float(last["range_high"]) if pd.notna(last["range_high"]) else None
    range_low = float(last["range_low"]) if pd.notna(last["range_low"]) else None

    signal = None
    entry_price = None
    sl_price = None
    tp_price = None
    sl_pips = None

    if bool(last["buy_signal"]):
        signal = "long"
        entry_price = float(last["entry_price"])
        sl_price = float(last["sl_price"])
        tp_price = float(last["tp_price"])
        sl_pips = abs(entry_price - sl_price) / PIP

    elif bool(last["sell_signal"]):
        signal = "short"
        entry_price = float(last["entry_price"])
        sl_price = float(last["sl_price"])
        tp_price = float(last["tp_price"])
        sl_pips = abs(entry_price - sl_price) / PIP

    return {
        "uk_hour": int(last["uk_hour"]),
        "range_high": range_high,
        "range_low": range_low,
        "signal": signal,
        "entry_price": entry_price,
        "sl_price": sl_price,
        "tp_price": tp_price,
        "sl_pips": sl_pips,
        "latest_close": float(last["close"]),
        "latest_ts": df.index[-1],
    }


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def seconds_until_next_hour() -> float:
    """
    Returns the number of seconds until the next top-of-hour UTC,
    minus WAKEUP_EARLY_SECS so we wake up slightly before the candle
    is fully formed and available.
    """
    now = datetime.now(timezone.utc)
    next_hour = (now + timedelta(hours=1)).replace(minute=0, second=0, microsecond=0)
    delta = (next_hour - now).total_seconds() - WAKEUP_EARLY_SECS
    return max(delta, 0.0)


def now_uk_str() -> str:
    """Current time as a readable UK-local string."""
    import zoneinfo

    uk = zoneinfo.ZoneInfo(TIMEZONE)
    return datetime.now(uk).strftime("%Y-%m-%d %H:%M %Z")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run() -> None:
    """
    Main execution loop. Runs indefinitely until Ctrl+C.
    """
    log.info("=" * 60)
    log.info("  RangeBO Live Bot — starting up")
    log.info(f"  Symbol    : {SYMBOL}")
    if SIZING_MODE == "risk_pct":
        log.info(
            f"  Sizing    : risk-based {RISK_PCT * 100:.1f}% per trade  |  R:R: {RR}"
        )
    else:
        log.info(f"  Sizing    : fixed lot {FIXED_LOT}  |  R:R: {RR}")
    log.info(f"  Range     : {RANGE_START_HOUR:02d}:00–{RANGE_END_HOUR:02d}:00 UK")
    log.info(f"  Time exit : {TRADE_CLOSE_HOUR:02d}:00 UK")
    log.info("=" * 60)

    # --- Connect to MT5 (blocks until successful) ---
    log.info("Connecting to MT5...")
    while not connect_mt5():
        log.warning("MT5 connection failed — retrying in 60s...")
        time.sleep(60)

    balance = get_balance()
    sizing_desc = (
        f"Risk: {RISK_PCT * 100:.1f}%/trade"
        if SIZING_MODE == "risk_pct"
        else f"Fixed lot: {FIXED_LOT}"
    )
    startup_msg = (
        f"<b>RangeBO Bot Online</b>\n"
        f"Symbol: {SYMBOL} | {sizing_desc}\n"
        f"Account balance: <b>${balance:,.2f}</b>\n"
        f"Range window: {RANGE_START_HOUR:02d}:00 – {RANGE_END_HOUR:02d}:00 UK\n"
        f"Auto-close: {TRADE_CLOSE_HOUR:02d}:00 UK"
    )
    log.info(startup_msg.replace("<b>", "").replace("</b>", ""))
    send_telegram(startup_msg)

    # --- Main loop ---
    while True:
        # Sleep until the top of the next hour
        wait = seconds_until_next_hour()
        log.info(f"Sleeping {wait:.0f}s until next candle close...")
        time.sleep(wait)

        # Small extra buffer to ensure MT5 has the completed bar available
        time.sleep(WAKEUP_EARLY_SECS + 2)

        # --- Ensure connection is alive ---
        ensure_connected()

        # --- Fetch bars and evaluate ---
        try:
            df = fetch_bars(BARS_TO_FETCH)
        except RuntimeError as e:
            log.error(f"fetch_bars error: {e}")
            send_telegram(f"<b>RangeBO Error</b>\nfetch_bars failed: {e}")
            continue

        try:
            result = evaluate_bar(df)
        except Exception as e:
            log.error(f"evaluate_bar error: {e}", exc_info=True)
            send_telegram(f"<b>RangeBO Error</b>\nevaluate_bar failed: {e}")
            continue

        uk_hour = result["uk_hour"]
        range_high = result["range_high"]
        range_low = result["range_low"]
        signal = result["signal"]
        entry_price = result["entry_price"]
        sl_price = result["sl_price"]
        tp_price = result["tp_price"]
        sl_pips = result["sl_pips"]
        latest_ts = result["latest_ts"]
        latest_close = result["latest_close"]

        log.info(
            f"[{now_uk_str()}] "
            f"Bar: {latest_ts}  Close: {latest_close:.3f}  UK hour: {uk_hour:02d}  "
            f"Range: {range_low or '?'} – {range_high or '?'}"
        )

        # ----------------------------------------------------------------
        # 16:00 UK — TIME EXIT
        # ----------------------------------------------------------------
        if uk_hour == TRADE_CLOSE_HOUR:
            position = get_open_position()
            if position is not None:
                log.info(
                    f"16:00 UK time exit triggered | "
                    f"ticket #{position.ticket} | "
                    f"open @ {position.price_open:.3f}"
                )
                success, pnl_pips = close_position(position)
                if success:
                    balance = get_balance()
                    direction_str = "LONG" if position.type == 0 else "SHORT"
                    msg = (
                        f"<b>RangeBO — Time Exit (16:00 UK)</b>\n"
                        f"Closed {direction_str} {SYMBOL} @ {latest_close:.3f}\n"
                        f"P&L: <b>{pnl_pips:+.1f} pips</b>\n"
                        f"Account balance: ${balance:,.2f}"
                    )
                    log.info(msg.replace("<b>", "").replace("</b>", ""))
                    send_telegram(msg)
                else:
                    send_telegram(
                        f"<b>RangeBO ERROR</b>\n"
                        f"Failed to close position at 16:00 UK time exit!\n"
                        f"Ticket: #{position.ticket} — please close manually."
                    )
            else:
                log.info("16:00 UK — no open position (already closed by SL/TP)")

        # ----------------------------------------------------------------
        # 07:00–15:59 UK — TRADING WINDOW
        # ----------------------------------------------------------------
        elif RANGE_END_HOUR <= uk_hour < TRADE_CLOSE_HOUR:
            position = get_open_position()

            if position is not None:
                # Position already open — MT5 is managing SL/TP
                direction_str = "LONG" if position.type == 0 else "SHORT"
                log.info(
                    f"Position open: {direction_str} {position.volume} lots | "
                    f"Open @ {position.price_open:.3f} | "
                    f"Current P&L: ${position.profit:.2f}"
                )

            elif signal is not None:
                # Fresh signal on a bar with no existing position — enter
                balance = get_balance()
                lots = compute_lots(sl_pips, balance)

                log.info(
                    f"SIGNAL: {signal.upper()} | "
                    f"Entry: {entry_price:.3f} | "
                    f"SL: {sl_price:.3f} ({sl_pips:.1f} pips) | "
                    f"TP: {tp_price:.3f} | "
                    f"Lots: {lots}"
                )

                success = open_position(signal, lots, sl_price, tp_price)

                if success:
                    balance = get_balance()
                    direction_label = "BUY" if signal == "long" else "SELL"
                    risk_line = (
                        f"Risk:    ${balance * RISK_PCT:,.2f}  ({RISK_PCT * 100:.1f}%)"
                        if SIZING_MODE == "risk_pct"
                        else f"Sizing:  fixed lot {FIXED_LOT}"
                    )
                    msg = (
                        f"<b>RangeBO — {direction_label} {SYMBOL}</b>\n"
                        f"Entry:   {entry_price:.3f}\n"
                        f"SL:      {sl_price:.3f}  ({sl_pips:.1f} pips)\n"
                        f"TP:      {tp_price:.3f}  ({sl_pips * RR:.1f} pips)\n"
                        f"Lots:    {lots}\n"
                        f"{risk_line}\n"
                        f"Balance: ${balance:,.2f}"
                    )
                    log.info(msg.replace("<b>", "").replace("</b>", ""))
                    send_telegram(msg)
                else:
                    send_telegram(
                        f"<b>RangeBO ERROR</b>\n"
                        f"order_send failed for {signal.upper()} {SYMBOL}.\n"
                        f"Check MT5 terminal — trade may not have executed."
                    )

            else:
                log.info(
                    f"Trading window — no signal on this bar | "
                    f"Close: {latest_close:.3f} | "
                    f"Range: {range_low:.3f} – {range_high:.3f}"
                    if range_high and range_low
                    else f"Trading window — no signal | Close: {latest_close:.3f}"
                )

        # ----------------------------------------------------------------
        # 00:00–06:59 UK — ACCUMULATION PHASE
        # ----------------------------------------------------------------
        else:
            log.info(
                f"Accumulation phase ({uk_hour:02d}:00 UK) — "
                f"building range | Close: {latest_close:.3f}"
            )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        log.info("Shutdown requested by user (Ctrl+C)")
        send_telegram("<b>RangeBO Bot</b> — stopped by user (Ctrl+C)")
        mt5.shutdown()
        log.info("MT5 disconnected. Goodbye.")
    except Exception as e:
        log.critical(f"Unhandled exception: {e}", exc_info=True)
        send_telegram(f"<b>RangeBO CRITICAL ERROR</b>\n{e}\nBot has stopped.")
        mt5.shutdown()
        raise
