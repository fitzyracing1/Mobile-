"""
DutchAgent — the main trading agent loop.

The agent's sole thesis is to be net short the US Dollar ("sell the Dutch
on the USD").  On non-inverted pairs like EUR/USD it goes *long* the pair
(long EUR, short USD).  On inverted pairs like USD/JPY it goes *short* the
pair (short USD, long JPY).

Decision logic
--------------
Every tick the agent:
  1. Fetches a new price bar for every tracked pair.
  2. Computes a composite signal (MA cross + RSI + Bollinger).
  3. Checks existing positions for stop-loss / take-profit exits.
  4. Opens new positions when the signal says "sell_usd" and room is available.
  5. Logs a status table to stdout.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional

from .config import AgentConfig, INVERTED_PAIRS
from .data import DataFeed
from .position import Portfolio, Side
from .signals import composite_signal, CompositeSignal

logger = logging.getLogger(__name__)


def _fmt_price(pair: str, price: float) -> str:
    decimals = 2 if "JPY" in pair else 5
    return f"{price:.{decimals}f}"


class DutchAgent:
    """Paper-trading agent that sells the US Dollar."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self.feed = DataFeed(self.config.pairs)
        self.portfolio = Portfolio(
            starting_balance=self.config.position_size_usd * self.config.max_positions * 2
        )
        self._tick_count = 0
        self._running = False

        if self.config.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s | %(levelname)-5s | %(message)s",
                datefmt="%H:%M:%S",
            )

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def warmup(self) -> None:
        """Pre-fill price histories so indicators are ready from tick 1."""
        logger.info("Warming up %d bars of history …", self.config.warmup_bars)
        self.feed.warmup(self.config.warmup_bars)
        logger.info("Warmup complete.")

    # ------------------------------------------------------------------
    # Core decision loop
    # ------------------------------------------------------------------

    def _get_signals(self) -> Dict[str, CompositeSignal]:
        cfg = self.config
        signals: Dict[str, CompositeSignal] = {}
        for pair in cfg.pairs:
            hist = self.feed.history(pair)
            closes = hist.closes()
            inverted = pair in INVERTED_PAIRS
            signals[pair] = composite_signal(
                closes=closes,
                fast_ma=cfg.fast_ma,
                slow_ma=cfg.slow_ma,
                rsi_period=cfg.rsi_period,
                bb_period=cfg.bb_period,
                bb_std=cfg.bb_std,
                rsi_oversold=cfg.rsi_oversold,
                rsi_overbought=cfg.rsi_overbought,
                is_inverted=inverted,
            )
        return signals

    def _check_exits(self, prices: Dict[str, float]) -> None:
        for pair in list(self.portfolio.positions.keys()):
            pos = self.portfolio.positions[pair]
            price = prices.get(pair, pos.entry_price)

            if pos.is_stopped(price):
                trade = self.portfolio.close_position(pair, price, "stop_loss")
                if trade:
                    logger.info(
                        "STOP-LOSS   | %-10s | %s → %s | PnL $%.2f",
                        pair, _fmt_price(pair, trade.entry_price),
                        _fmt_price(pair, trade.exit_price), trade.pnl,
                    )

            elif pos.is_target_hit(price):
                trade = self.portfolio.close_position(pair, price, "take_profit")
                if trade:
                    logger.info(
                        "TAKE-PROFIT | %-10s | %s → %s | PnL $%.2f",
                        pair, _fmt_price(pair, trade.entry_price),
                        _fmt_price(pair, trade.exit_price), trade.pnl,
                    )

    def _open_positions(
        self,
        prices: Dict[str, float],
        signals: Dict[str, CompositeSignal],
    ) -> None:
        cfg = self.config
        n_open = len(self.portfolio.positions)

        for pair, sig in signals.items():
            if n_open >= cfg.max_positions:
                break
            if pair in self.portfolio.positions:
                continue
            if sig.action != "sell_usd":
                continue

            price = prices[pair]
            # On inverted pairs (USD is base) we SHORT the pair to be short USD.
            # On non-inverted pairs (USD is quote) we LONG the pair.
            side = Side.SHORT if pair in INVERTED_PAIRS else Side.LONG

            pos = self.portfolio.open_position(
                pair=pair,
                side=side,
                price=price,
                size_usd=cfg.position_size_usd,
                stop_loss_pct=cfg.stop_loss_pct,
                take_profit_pct=cfg.take_profit_pct,
            )
            if pos:
                logger.info(
                    "OPEN %-5s  | %-10s | price=%-10s | score=%.2f | RSI=%.1f",
                    side.value, pair, _fmt_price(pair, price),
                    sig.score, sig.rsi.value,
                )
                n_open += 1

    def _log_status(self, prices: Dict[str, float]) -> None:
        summary = self.portfolio.summary(prices)
        logger.info(
            "Tick #%d | equity=$%s | pnl=$%s | open=%d | trades=%d | win=%.0f%%",
            self._tick_count,
            f"{summary['equity']:,.2f}",
            f"{summary['total_pnl']:+,.2f}",
            summary["open_positions"],
            summary["closed_trades"],
            summary["win_rate_pct"],
        )

    def tick(self) -> Dict:
        """Execute one agent tick.  Returns current portfolio summary."""
        self._tick_count += 1
        new_bars = self.feed.tick()
        prices = {pair: bar.close for pair, bar in new_bars.items()}
        signals = self._get_signals()

        self._check_exits(prices)
        self._open_positions(prices, signals)

        if self.config.verbose:
            self._log_status(prices)

        return self.portfolio.summary(prices)

    # ------------------------------------------------------------------
    # Live loop
    # ------------------------------------------------------------------

    def run(self, max_ticks: Optional[int] = None) -> None:
        """
        Run the agent continuously.

        Parameters
        ----------
        max_ticks:
            Stop after this many ticks (useful for back-testing / unit tests).
            Pass None to run indefinitely (stop with Ctrl-C).
        """
        self.warmup()
        self._running = True
        logger.info("Dutch Agent started.  Strategy: sell USD on every signal.")
        logger.info("Pairs: %s", ", ".join(self.config.pairs))

        try:
            while self._running:
                self.tick()
                if max_ticks is not None and self._tick_count >= max_ticks:
                    break
                if max_ticks is None:  # live mode
                    time.sleep(self.config.tick_interval_seconds)
        except KeyboardInterrupt:
            logger.info("Agent stopped by user.")
        finally:
            self._running = False
            prices = {p: self.feed.price(p) for p in self.config.pairs}
            logger.info("=== FINAL PORTFOLIO ===")
            for k, v in self.portfolio.summary(prices).items():
                logger.info("  %-20s %s", k, v)
            logger.info("======================")
