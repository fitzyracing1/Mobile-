"""
OANDA v20 integration — real price feed and order execution.

Replaces the synthetic DataFeed with live M1 candles from OANDA and
places real market orders (with native stop-loss and take-profit) via
the v20 REST API.

Usage
-----
    from dutch_agent.oanda import OandaAgent
    from dutch_agent.config import AgentConfig

    agent = OandaAgent(
        token="YOUR_OANDA_TOKEN",
        account_id="101-001-XXXXXXX-001",
        environment="practice",   # or "live"
        config=AgentConfig(...),
    )
    agent.run(max_ticks=50)
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import oandapyV20
import oandapyV20.endpoints.accounts as ep_accounts
import oandapyV20.endpoints.instruments as ep_instruments
import oandapyV20.endpoints.orders as ep_orders
import oandapyV20.endpoints.trades as ep_trades

from .agent import DutchAgent, _fmt_price
from .config import AgentConfig, INVERTED_PAIRS
from .data import Bar, DataFeed, PriceHistory
from .position import Portfolio, Side

logger = logging.getLogger(__name__)

# Map agent pair names → OANDA instrument names
OANDA_INSTRUMENT: Dict[str, str] = {
    "EUR/USD": "EUR_USD",
    "GBP/USD": "GBP_USD",
    "AUD/USD": "AUD_USD",
    "NZD/USD": "NZD_USD",
    "USD/CHF": "USD_CHF",
    "USD/JPY": "USD_JPY",
    "USD/CAD": "USD_CAD",
}

GRANULARITY = "M1"   # 1-minute candles


# ---------------------------------------------------------------------------
# Real data feed
# ---------------------------------------------------------------------------

class OandaDataFeed(DataFeed):
    """
    DataFeed backed by OANDA v20 candle API.

    warmup()  — fetches the last N M1 candles from OANDA history.
    tick()    — fetches the latest completed M1 candle for each pair.
    """

    def __init__(self, client: oandapyV20.API, pairs: List[str], max_bars: int = 200) -> None:
        super().__init__(pairs, max_bars)
        self.client = client

    def _fetch_candles(self, pair: str, count: int) -> List[Bar]:
        instrument = OANDA_INSTRUMENT[pair]
        params = {"count": count, "granularity": GRANULARITY, "price": "M"}
        r = ep_instruments.InstrumentsCandles(instrument, params=params)
        self.client.request(r)
        bars = []
        for c in r.response.get("candles", []):
            if not c.get("complete", True):
                continue
            mid = c["mid"]
            bars.append(Bar(
                timestamp=_parse_oanda_time(c["time"]),
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
                volume=float(c.get("volume", 0)),
            ))
        return bars

    def warmup(self, n_bars: int) -> None:
        logger.info("Fetching %d bars of real OANDA history …", n_bars)
        for pair in self.pairs:
            hist = self._histories[pair]
            bars = self._fetch_candles(pair, n_bars + 5)  # +5 buffer for incomplete bar
            for bar in bars[-n_bars:]:
                hist.add_bar(bar)
                hist._price = bar.close
            logger.info("  %-10s  last close: %s", pair, _fmt_price(pair, hist.current_price()))
        logger.info("Warmup complete.")

    def tick(self) -> Dict[str, Bar]:
        result = {}
        for pair in self.pairs:
            hist = self._histories[pair]
            try:
                bars = self._fetch_candles(pair, 3)
                if bars:
                    bar = bars[-1]
                    hist.add_bar(bar)
                    hist._price = bar.close
                    result[pair] = bar
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", pair, e)
                # Fall back to last known price
                result[pair] = hist.bars[-1] if hist.bars else Bar(
                    timestamp=time.time(), open=hist._price,
                    high=hist._price, low=hist._price, close=hist._price,
                )
        return result


# ---------------------------------------------------------------------------
# Order executor
# ---------------------------------------------------------------------------

class OandaOrderExecutor:
    """Places and closes orders on OANDA, returns OANDA trade IDs."""

    def __init__(self, client: oandapyV20.API, account_id: str) -> None:
        self.client = client
        self.account_id = account_id

    def place_order(
        self,
        pair: str,
        side: Side,
        price: float,
        size_usd: float,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> Optional[str]:
        """
        Place a market order.  Returns the OANDA trade ID or None on failure.
        Units are sized so that notional value ≈ size_usd USD.
        """
        instrument = OANDA_INSTRUMENT[pair]
        units = _calc_units(pair, side, price, size_usd)

        sl_price = _stop_price(pair, side, price, stop_loss_pct)
        tp_price = _target_price(pair, side, price, take_profit_pct)

        body = {
            "order": {
                "type": "MARKET",
                "instrument": instrument,
                "units": str(units),
                "stopLossOnFill": {"price": _fmt_oanda_price(pair, sl_price)},
                "takeProfitOnFill": {"price": _fmt_oanda_price(pair, tp_price)},
                "timeInForce": "FOK",
            }
        }

        try:
            r = ep_orders.OrderCreate(self.account_id, data=body)
            self.client.request(r)
            resp = r.response

            # Check for cancellation (e.g. MARKET_HALTED on weekends)
            cancel = resp.get("orderCancelTransaction", {})
            if cancel:
                reason = cancel.get("reason", "UNKNOWN")
                logger.warning("Order cancelled for %s: %s", pair, reason)
                return None

            fill = resp.get("orderFillTransaction", {})
            trade_id = fill.get("tradeOpened", {}).get("tradeID")

            if not trade_id:
                logger.warning("No trade opened for %s — response: %s", pair, resp)
                return None

            logger.info(
                "OANDA ORDER  | %-10s | %s | units=%-8s | SL=%s | TP=%s | tradeID=%s",
                pair, side.value, units,
                _fmt_oanda_price(pair, sl_price),
                _fmt_oanda_price(pair, tp_price),
                trade_id,
            )
            return str(trade_id)
        except Exception as e:
            logger.error("Order failed for %s: %s", pair, e)
            return None

    def close_trade(self, trade_id: str) -> bool:
        """Close an open OANDA trade by ID."""
        try:
            r = ep_trades.TradeClose(self.account_id, trade_id)
            self.client.request(r)
            return True
        except Exception as e:
            logger.error("Failed to close trade %s: %s", trade_id, e)
            return False

    def open_trade_ids(self) -> List[str]:
        """Return IDs of currently open OANDA trades."""
        try:
            r = ep_trades.TradesList(self.account_id, params={"state": "OPEN"})
            self.client.request(r)
            return [t["id"] for t in r.response.get("trades", [])]
        except Exception as e:
            logger.warning("Could not fetch open trades: %s", e)
            return []

    def account_summary(self) -> Dict:
        r = ep_accounts.AccountSummary(self.account_id)
        self.client.request(r)
        return r.response["account"]


# ---------------------------------------------------------------------------
# Agent subclass
# ---------------------------------------------------------------------------

class OandaAgent(DutchAgent):
    """
    DutchAgent that uses real OANDA prices and places real orders.
    SL/TP are set natively on OANDA so exits are handled broker-side.
    """

    def __init__(
        self,
        token: str,
        account_id: str,
        environment: str = "practice",
        config: Optional[AgentConfig] = None,
    ) -> None:
        super().__init__(config=config)

        self.oanda_account_id = account_id
        self.client = oandapyV20.API(access_token=token, environment=environment)

        # Replace synthetic feed with real OANDA feed
        self.feed = OandaDataFeed(self.client, self.config.pairs)

        # Order executor
        self.executor = OandaOrderExecutor(self.client, account_id)

        # Map pair → OANDA trade ID for open positions
        self._oanda_trade_ids: Dict[str, str] = {}

    def warmup(self) -> None:
        self.feed.warmup(self.config.warmup_bars)

    def _check_exits(self, prices: Dict[str, float]) -> None:
        """
        Sync local position tracking with OANDA.
        OANDA handles SL/TP exits broker-side; we just detect when a
        trade has been closed and remove it from our local record.
        """
        if not self._oanda_trade_ids:
            return

        live_ids = set(self.executor.open_trade_ids())

        for pair in list(self._oanda_trade_ids.keys()):
            tid = self._oanda_trade_ids[pair]
            if tid not in live_ids:
                # Trade was closed by OANDA (SL or TP hit)
                trade = self.portfolio.close_position(pair, prices.get(pair, 0), "oanda_exit")
                del self._oanda_trade_ids[pair]
                if trade:
                    logger.info(
                        "CLOSED       | %-10s | tradeID=%s | PnL $%.2f",
                        pair, tid, trade.pnl,
                    )

    def _open_positions(self, prices: Dict[str, float], signals) -> None:
        cfg = self.config
        n_open = len(self._oanda_trade_ids)

        for pair, sig in signals.items():
            if n_open >= cfg.max_positions:
                break
            if pair in self._oanda_trade_ids:
                continue
            if sig.action != "sell_usd":
                continue

            price = prices[pair]
            side = Side.SHORT if pair in INVERTED_PAIRS else Side.LONG

            trade_id = self.executor.place_order(
                pair=pair,
                side=side,
                price=price,
                size_usd=cfg.position_size_usd,
                stop_loss_pct=cfg.stop_loss_pct,
                take_profit_pct=cfg.take_profit_pct,
            )

            if trade_id:
                self._oanda_trade_ids[pair] = trade_id
                self.portfolio.open_position(
                    pair=pair,
                    side=side,
                    price=price,
                    size_usd=cfg.position_size_usd,
                    stop_loss_pct=cfg.stop_loss_pct,
                    take_profit_pct=cfg.take_profit_pct,
                )
                n_open += 1

    def _log_status(self, prices: Dict[str, float]) -> None:
        try:
            acct = self.executor.account_summary()
            logger.info(
                "Tick #%d | OANDA balance=$%s | NAV=$%s | open=%s | unrealised=$%s",
                self._tick_count,
                acct["balance"],
                acct["NAV"],
                acct["openTradeCount"],
                acct["unrealizedPL"],
            )
        except Exception:
            super()._log_status(prices)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_oanda_time(s: str) -> float:
    """Parse OANDA RFC3339 timestamp to Unix epoch."""
    from datetime import datetime, timezone
    s = s[:26].rstrip("Z") + "+00:00"
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return time.time()


def _calc_units(pair: str, side: Side, price: float, size_usd: float) -> int:
    """
    Convert a USD notional size into OANDA units.

    For pairs where USD is the quote (EUR/USD, GBP/USD …):
        1 unit = 1 base-currency unit.  units = size_usd / price
    For pairs where USD is the base (USD/JPY, USD/CHF …):
        1 unit = 1 USD.  units = size_usd

    Sign convention: positive = buy (LONG), negative = sell (SHORT).
    """
    if pair in INVERTED_PAIRS:
        raw = int(size_usd)
    else:
        raw = int(size_usd / price)

    return -raw if side == Side.SHORT else raw


def _stop_price(pair: str, side: Side, entry: float, pct: float) -> float:
    multiplier = (1 - pct / 100) if side == Side.LONG else (1 + pct / 100)
    return entry * multiplier


def _target_price(pair: str, side: Side, entry: float, pct: float) -> float:
    multiplier = (1 + pct / 100) if side == Side.LONG else (1 - pct / 100)
    return entry * multiplier


def _fmt_oanda_price(pair: str, price: float) -> str:
    decimals = 3 if "JPY" in pair else 5
    return f"{price:.{decimals}f}"
