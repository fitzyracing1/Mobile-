"""
OANDA v20 integration — real price feed and order execution (v2).

Changes from v1
---------------
- Fetches H4 candles alongside M1 candles and populates agent._htf_closes
  so the H4 trend filter in agent.py has real higher-timeframe data.
- H4 data is refreshed every tick (cheap — only 2 candles needed to update).
- place_order now correctly detects MARKET_HALTED and other cancellations.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

import numpy as np
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

OANDA_INSTRUMENT: Dict[str, str] = {
    "EUR/USD": "EUR_USD",
    "GBP/USD": "GBP_USD",
    "AUD/USD": "AUD_USD",
    "NZD/USD": "NZD_USD",
    "USD/CHF": "USD_CHF",
    "USD/JPY": "USD_JPY",
    "USD/CAD": "USD_CAD",
}


# ---------------------------------------------------------------------------
# Real data feed (M1)
# ---------------------------------------------------------------------------

class OandaDataFeed(DataFeed):
    """DataFeed backed by OANDA v20 M1 candles."""

    def __init__(self, client: oandapyV20.API, pairs: List[str], max_bars: int = 200) -> None:
        super().__init__(pairs, max_bars)
        self.client = client

    def _fetch_candles(self, pair: str, count: int, granularity: str = "M1") -> List[Bar]:
        instrument = OANDA_INSTRUMENT[pair]
        params = {"count": count, "granularity": granularity, "price": "M"}
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
        logger.info("Fetching %d bars of real OANDA M1 history …", n_bars)
        for pair in self.pairs:
            hist = self._histories[pair]
            bars = self._fetch_candles(pair, n_bars + 5, "M1")
            for bar in bars[-n_bars:]:
                hist.add_bar(bar)
                hist._price = bar.close
            logger.info("  %-10s  M1 close: %s", pair, _fmt_price(pair, hist.current_price()))
        logger.info("M1 warmup complete.")

    def fetch_htf_candles(self, pair: str, count: int, granularity: str) -> List[Bar]:
        return self._fetch_candles(pair, count, granularity)

    def tick(self) -> Dict[str, Bar]:
        result = {}
        for pair in self.pairs:
            hist = self._histories[pair]
            try:
                bars = self._fetch_candles(pair, 3, "M1")
                if bars:
                    bar = bars[-1]
                    hist.add_bar(bar)
                    hist._price = bar.close
                    result[pair] = bar
            except Exception as e:
                logger.warning("Failed to fetch %s: %s", pair, e)
                result[pair] = hist.bars[-1] if hist.bars else Bar(
                    timestamp=time.time(), open=hist._price,
                    high=hist._price, low=hist._price, close=hist._price,
                )
        return result


# ---------------------------------------------------------------------------
# Order executor
# ---------------------------------------------------------------------------

class OandaOrderExecutor:
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
        instrument = OANDA_INSTRUMENT[pair]
        units = _calc_units(pair, side, price, size_usd)
        sl_price = _stop_price(side, price, stop_loss_pct)
        tp_price = _target_price(side, price, take_profit_pct)

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

            cancel = resp.get("orderCancelTransaction", {})
            if cancel:
                logger.warning("Order cancelled for %s: %s", pair, cancel.get("reason", "UNKNOWN"))
                return None

            fill = resp.get("orderFillTransaction", {})
            trade_id = fill.get("tradeOpened", {}).get("tradeID")
            if not trade_id:
                logger.warning("No trade opened for %s", pair)
                return None

            logger.info(
                "OANDA ORDER  | %-10s | %s | units=%-8s | SL=%s | TP=%s | id=%s",
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
        try:
            r = ep_trades.TradeClose(self.account_id, trade_id)
            self.client.request(r)
            return True
        except Exception as e:
            logger.error("Failed to close trade %s: %s", trade_id, e)
            return False

    def open_trade_ids(self) -> List[str]:
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
# OandaAgent — DutchAgent subclass wired to OANDA
# ---------------------------------------------------------------------------

class OandaAgent(DutchAgent):
    """
    DutchAgent that uses real OANDA prices and places real orders.

    Additions vs v1
    ---------------
    - Warms up H4 candles at startup and refreshes every tick.
    - H4 closes are stored in self._htf_closes (inherited from DutchAgent)
      and consumed by the H4 trend filter in _open_positions.
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
        self.feed = OandaDataFeed(self.client, self.config.pairs)
        self.executor = OandaOrderExecutor(self.client, account_id)
        self._oanda_trade_ids: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Warmup — M1 + H4
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        self.feed.warmup(self.config.warmup_bars)
        self._refresh_htf(full=True)

    def _refresh_htf(self, full: bool = False) -> None:
        """Fetch H4 candles for every pair and update self._htf_closes."""
        cfg = self.config
        count = cfg.htf_bars + 5 if full else 3
        for pair in cfg.pairs:
            try:
                bars = self.feed.fetch_htf_candles(pair, count, cfg.htf_granularity)
                closes = np.array([b.close for b in bars])
                if full or pair not in self._htf_closes:
                    self._htf_closes[pair] = closes
                else:
                    # Append only the new candle(s)
                    existing = self._htf_closes[pair]
                    if len(closes) > 0 and (len(existing) == 0 or closes[-1] != existing[-1]):
                        combined = np.append(existing, closes[-1])
                        self._htf_closes[pair] = combined[-cfg.htf_bars:]
            except Exception as e:
                logger.warning("H4 fetch failed for %s: %s", pair, e)
        if full:
            logger.info("H4 warmup complete (%s bars per pair).", cfg.htf_bars)

    # ------------------------------------------------------------------
    # Exit checks — sync with OANDA
    # ------------------------------------------------------------------

    def _check_exits(self, prices: Dict[str, float]) -> None:
        if not self._oanda_trade_ids:
            return
        live_ids = set(self.executor.open_trade_ids())
        for pair in list(self._oanda_trade_ids.keys()):
            tid = self._oanda_trade_ids[pair]
            if tid not in live_ids:
                trade = self.portfolio.close_position(pair, prices.get(pair, 0), "oanda_exit")
                del self._oanda_trade_ids[pair]
                if trade:
                    logger.info(
                        "CLOSED       | %-10s | id=%s | PnL $%.2f",
                        pair, tid, trade.pnl,
                    )

    # ------------------------------------------------------------------
    # Entry — delegates to DutchAgent._open_positions via override hook
    # ------------------------------------------------------------------

    def _open_positions(self, prices: Dict[str, float], signals) -> None:
        cfg = self.config
        from .filters import EntryCooldown, htf_trend_confirms, select_candidates
        from .config import INVERTED_PAIRS

        if len(self._oanda_trade_ids) >= cfg.max_positions:
            return

        if not self._cooldown.ready(self._tick_count):
            return

        open_pairs = set(self._oanda_trade_ids.keys())
        candidates = select_candidates(signals, open_pairs, cfg.min_score_threshold)

        for pair in candidates:
            if len(self._oanda_trade_ids) >= cfg.max_positions:
                break

            htf = self._htf_closes.get(pair, np.array([]))
            if not htf_trend_confirms(pair, htf, cfg.htf_ma_fast, cfg.htf_ma_slow):
                logger.info("H4 FILTER   | %-10s | H4 trend does not confirm — skip", pair)
                continue

            price = prices[pair]
            side = Side.SHORT if pair in INVERTED_PAIRS else Side.LONG
            sig = signals[pair]

            trade_id = self.executor.place_order(
                pair=pair, side=side, price=price,
                size_usd=cfg.position_size_usd,
                stop_loss_pct=cfg.stop_loss_pct,
                take_profit_pct=cfg.take_profit_pct,
            )
            if trade_id:
                self._oanda_trade_ids[pair] = trade_id
                self.portfolio.open_position(
                    pair=pair, side=side, price=price,
                    size_usd=cfg.position_size_usd,
                    stop_loss_pct=cfg.stop_loss_pct,
                    take_profit_pct=cfg.take_profit_pct,
                )
                self._cooldown.record_entry(self._tick_count)
                logger.info(
                    "OPEN %-5s  | %-10s | price=%-10s | score=%+.2f | RSI=%.1f",
                    side.value, pair, _fmt_price(pair, price),
                    sig.score, sig.rsi.value,
                )
                break  # one entry per tick

    # ------------------------------------------------------------------
    # Tick — also refreshes H4 data
    # ------------------------------------------------------------------

    def tick(self) -> Dict:
        self._tick_count += 1
        new_bars = self.feed.tick()
        prices = {pair: bar.close for pair, bar in new_bars.items()}

        # Refresh H4 every 10 ticks (new H4 candle every 240 M1 ticks, so
        # checking every 10 keeps the trend filter current without over-fetching)
        if self._tick_count % 10 == 0:
            self._refresh_htf(full=False)

        signals = self._get_signals()
        self._check_exits(prices)
        self._open_positions(prices, signals)

        if self.config.verbose:
            self._log_status(prices)

        return self.portfolio.summary(prices)

    # ------------------------------------------------------------------
    # Status log — OANDA balance + per-pair signal view
    # ------------------------------------------------------------------

    def _log_status(self, prices: Dict[str, float]) -> None:
        try:
            acct = self.executor.account_summary()
            logger.info(
                "Tick #%d | balance=$%s | NAV=$%s | open=%s | unrealised=$%s",
                self._tick_count,
                acct["balance"], acct["NAV"],
                acct["openTradeCount"], acct["unrealizedPL"],
            )
        except Exception:
            from .agent import DutchAgent
            DutchAgent._log_status(self, prices)

        from .signals import composite_signal
        cfg = self.config
        for pair in cfg.pairs:
            hist = self.feed.history(pair)
            closes = hist.closes()
            if len(closes) < cfg.slow_ma:
                continue
            sig = self._get_signals().get(pair)
            if sig is None:
                continue
            status = "OPEN" if pair in self._oanda_trade_ids else "    "
            htf = self._htf_closes.get(pair, np.array([]))
            h4_ok = htf_trend_confirms(pair, htf, cfg.htf_ma_fast, cfg.htf_ma_slow)
            logger.info(
                "  %s %-10s price=%-10s score=%+.2f RSI=%5.1f H4=%s → %s",
                status, pair, _fmt_price(pair, prices.get(pair, 0)),
                sig.score, sig.rsi.value,
                "✓" if h4_ok else "✗",
                sig.action,
            )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_oanda_time(s: str) -> float:
    from datetime import datetime, timezone
    s = s[:26].rstrip("Z") + "+00:00"
    try:
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return time.time()


def _calc_units(pair: str, side: Side, price: float, size_usd: float) -> int:
    raw = int(size_usd) if pair in INVERTED_PAIRS else int(size_usd / price)
    return -raw if side == Side.SHORT else raw


def _stop_price(side: Side, entry: float, pct: float) -> float:
    return entry * ((1 - pct / 100) if side == Side.LONG else (1 + pct / 100))


def _target_price(side: Side, entry: float, pct: float) -> float:
    return entry * ((1 + pct / 100) if side == Side.LONG else (1 - pct / 100))


def _fmt_oanda_price(pair: str, price: float) -> str:
    decimals = 3 if "JPY" in pair else 5
    return f"{price:.{decimals}f}"


def htf_trend_confirms(pair, htf_closes, fast_period, slow_period):
    from .filters import htf_trend_confirms as _htf
    return _htf(pair, htf_closes, fast_period, slow_period)
