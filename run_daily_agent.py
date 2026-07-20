#!/usr/bin/env python3
"""
Daily MA Cross live agent.

Uses the exact MACrossStrategy (10/30 daily) that backtested at +3.71%
over 2020-2026.  Checks for new completed daily bars once per hour and
places OANDA market orders with 2% SL / 4% TP when a crossover fires.

Usage
-----
  python3 run_daily_agent.py --token YOUR_TOKEN --account YOUR_ACCOUNT_ID
  python3 run_daily_agent.py --token YOUR_TOKEN --account YOUR_ACCOUNT_ID --env live
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import datetime, timezone
from typing import Dict, Optional

import oandapyV20
import oandapyV20.endpoints.accounts as ep_accounts
import oandapyV20.endpoints.instruments as ep_instruments
import oandapyV20.endpoints.orders as ep_orders
import oandapyV20.endpoints.trades as ep_trades

from backtester.data import DailyBar, OANDA_INSTRUMENT
from backtester.strategies.ma_cross import MACrossStrategy

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PAIRS = [
    "EUR/USD", "GBP/USD", "AUD/USD", "NZD/USD",
    "USD/CHF", "USD/JPY", "USD/CAD",
]

SL_PCT  = 2.0
TP_PCT  = 4.0
MAX_POSITIONS = 3
POSITION_SIZE_USD = 5_000.0
CHECK_INTERVAL_SECONDS = 3600  # once per hour


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Daily MA Cross live agent")
    p.add_argument("--token",   default=os.environ.get("OANDA_API_TOKEN", ""))
    p.add_argument("--account", default=os.environ.get("OANDA_ACCOUNT_ID", ""))
    p.add_argument("--env",     default="practice", choices=["practice", "live"])
    p.add_argument("--sl",      type=float, default=SL_PCT,           metavar="PCT")
    p.add_argument("--tp",      type=float, default=TP_PCT,           metavar="PCT")
    p.add_argument("--size",    type=float, default=POSITION_SIZE_USD, metavar="USD")
    p.add_argument("--max-pos", type=int,   default=MAX_POSITIONS)
    p.add_argument("--interval",type=int,   default=CHECK_INTERVAL_SECONDS, metavar="SECS")
    p.add_argument("--once",    action="store_true",
                   help="Run a single check then exit (useful for testing)")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# OANDA helpers
# ---------------------------------------------------------------------------

def fetch_daily_bars(client, pair: str, count: int = 50):
    inst = OANDA_INSTRUMENT[pair]
    params = {"count": count, "granularity": "D", "price": "M"}
    r = ep_instruments.InstrumentsCandles(inst, params=params)
    client.request(r)
    bars = []
    for c in r.response.get("candles", []):
        if not c.get("complete", True):
            continue
        mid = c["mid"]
        bars.append(DailyBar(
            date=c["time"][:10],
            open=float(mid["o"]),
            high=float(mid["h"]),
            low=float(mid["l"]),
            close=float(mid["c"]),
            volume=float(c.get("volume", 0)),
        ))
    return bars


def open_trade_pairs(client, account_id: str) -> Dict[str, str]:
    """Return {pair: trade_id} for currently open OANDA trades."""
    r = ep_trades.TradesList(account_id, params={"state": "OPEN"})
    client.request(r)
    result = {}
    for t in r.response.get("trades", []):
        pair = t["instrument"].replace("_", "/")
        result[pair] = t["id"]
    return result


def place_order(client, account_id, pair, action, price, sl_pct, tp_pct, size_usd):
    inst  = OANDA_INSTRUMENT[pair]
    is_inverted = pair in ("USD/CHF", "USD/JPY", "USD/CAD")

    # action "buy"  = long the pair
    # action "sell" = short the pair
    if action == "buy":
        raw_units = int(size_usd) if is_inverted else int(size_usd / price)
        units = raw_units
        sl    = round(price * (1 - sl_pct / 100), 5 if "JPY" not in pair else 3)
        tp    = round(price * (1 + tp_pct / 100), 5 if "JPY" not in pair else 3)
    else:
        raw_units = int(size_usd) if is_inverted else int(size_usd / price)
        units = -raw_units
        sl    = round(price * (1 + sl_pct / 100), 5 if "JPY" not in pair else 3)
        tp    = round(price * (1 - tp_pct / 100), 5 if "JPY" not in pair else 3)

    dec   = 3 if "JPY" in pair else 5
    body  = {
        "order": {
            "type": "MARKET",
            "instrument": inst,
            "units": str(units),
            "timeInForce": "FOK",
            "stopLossOnFill":   {"price": f"{sl:.{dec}f}"},
            "takeProfitOnFill": {"price": f"{tp:.{dec}f}"},
        }
    }
    r = ep_orders.OrderCreate(account_id, data=body)
    client.request(r)

    cancel = r.response.get("orderCancelTransaction", {})
    if cancel:
        logger.warning("Order cancelled: %s", cancel.get("reason"))
        return None

    fill     = r.response.get("orderFillTransaction", {})
    trade_id = fill.get("tradeOpened", {}).get("tradeID")
    if trade_id:
        logger.info("OPENED  | %-10s | %s | units=%-6s | SL=%.5f | TP=%.5f | id=%s",
                    pair, action.upper(), units, sl, tp, trade_id)
    return trade_id


def account_summary(client, account_id):
    r = ep_accounts.AccountSummary(account_id)
    client.request(r)
    return r.response["account"]


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

def run_check(client, account_id, strategy, args):
    """Single check: fetch bars, generate signals, open/skip positions."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    logger.info("=== Check at %s ===", now)

    acct = account_summary(client, account_id)
    logger.info("Balance=$%s  NAV=$%s  Open=%s",
                acct["balance"], acct["NAV"], acct["openTradeCount"])

    open_pairs = open_trade_pairs(client, account_id)
    n_open = len(open_pairs)

    for pair in PAIRS:
        try:
            bars = fetch_daily_bars(client, pair, count=50)
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", pair, e)
            continue

        signal = strategy.generate_signal(pair, bars)
        current_price = bars[-1].close if bars else 0

        if pair in open_pairs:
            logger.info("  %-10s OPEN  | signal=%-5s | price=%.5f",
                        pair, signal.action.upper(), current_price)
            continue

        if signal.action == "hold":
            logger.info("  %-10s hold  | signal=HOLD  | price=%.5f | %s",
                        pair, current_price, signal.reason)
            continue

        if n_open >= args.max_pos:
            logger.info("  %-10s skip  | max positions reached (%d)", pair, args.max_pos)
            continue

        logger.info("  %-10s SIGNAL| action=%-5s | price=%.5f | %s",
                    pair, signal.action.upper(), current_price, signal.reason)

        tid = place_order(
            client, account_id, pair,
            action=signal.action,
            price=current_price,
            sl_pct=args.sl,
            tp_pct=args.tp,
            size_usd=args.size,
        )
        if tid:
            n_open += 1


def main(argv=None):
    args = parse_args(argv)

    if not args.token:
        print("ERROR: --token required (or set OANDA_API_TOKEN)")
        return 1
    if not args.account:
        print("ERROR: --account required (or set OANDA_ACCOUNT_ID)")
        return 1

    client   = oandapyV20.API(access_token=args.token, environment=args.env)
    strategy = MACrossStrategy(fast=10, slow=30)

    logger.info("Daily MA Cross Agent started")
    logger.info("Strategy: %s | SL=%.0f%% | TP=%.0f%% | Size=$%.0f | MaxPos=%d",
                strategy.name, args.sl, args.tp, args.size, args.max_pos)
    logger.info("Account: %s (%s)", args.account, args.env)

    if args.once:
        run_check(client, args.account, strategy, args)
        return 0

    while True:
        try:
            run_check(client, args.account, strategy, args)
        except KeyboardInterrupt:
            logger.info("Stopped by user.")
            break
        except Exception as e:
            logger.error("Unexpected error: %s", e)

        logger.info("Next check in %d minutes.", args.interval // 60)
        time.sleep(args.interval)

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
