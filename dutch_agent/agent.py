"""
DutchAgent — the main trading agent loop (v2).

Strategy improvements over v1
------------------------------
1. Correlation filter  — at most one position per correlated pair group
   (EUR, commodity, inverted-USD).  Prevents tripling up on the same move.
2. H4 trend filter     — only open when the 4-hour MA also points to USD
   weakness.  Stops trading against the macro trend.
3. Score threshold     — requires a composite score < -0.35 (vs -0.10 in v1).
4. Entry cooldown      — minimum N ticks between any two entries; no more
   opening all positions in the same minute.

Decision logic every tick
--------------------------
  1. Fetch a new M1 price bar for every tracked pair.
  2. Compute composite signal (MA cross + RSI + Bollinger) on M1 closes.
  3. Check existing positions for OANDA-side SL/TP exits (or paper exits).
  4. For each correlation group that has no open position:
       a. Find the pair with the strongest sell-USD score.
       b. Require score < min_score_threshold.
       c. Confirm the H4 trend agrees (USD weakening on higher timeframe).
       d. Check cooldown — skip if too soon after the last entry.
       e. Open position.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Set

import numpy as np

from .config import AgentConfig, INVERTED_PAIRS
from .data import DataFeed
from .filters import (
    EntryCooldown,
    htf_direction,
    position_side,
    select_candidates_bidirectional,
)
from .position import Portfolio, Side
from .signals import composite_signal, CompositeSignal

logger = logging.getLogger(__name__)


def _fmt_price(pair: str, price: float) -> str:
    decimals = 2 if "JPY" in pair else 5
    return f"{price:.{decimals}f}"


class DutchAgent:
    """Paper-trading agent that sells the US Dollar (v2)."""

    def __init__(self, config: Optional[AgentConfig] = None) -> None:
        self.config = config or AgentConfig()
        self.feed = DataFeed(self.config.pairs)
        self.portfolio = Portfolio(
            starting_balance=self.config.position_size_usd * self.config.max_positions * 2
        )
        self._tick_count = 0
        self._running = False
        self._cooldown = EntryCooldown(self.config.entry_cooldown_ticks)

        # HTF closes per pair — populated by subclasses or left empty for paper mode
        self._htf_closes: Dict[str, np.ndarray] = {}

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
        logger.info("Warming up %d bars of history …", self.config.warmup_bars)
        self.feed.warmup(self.config.warmup_bars)
        logger.info("Warmup complete.")

    # ------------------------------------------------------------------
    # Signal computation
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

    # ------------------------------------------------------------------
    # Exit checks
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Entry logic (v2 with all three filters)
    # ------------------------------------------------------------------

    def _open_positions(
        self,
        prices: Dict[str, float],
        signals: Dict[str, CompositeSignal],
    ) -> None:
        cfg = self.config

        if len(self.portfolio.positions) >= cfg.max_positions:
            return

        # Cooldown check — one gate for ALL new entries this tick
        if not self._cooldown.ready(self._tick_count):
            remaining = self._cooldown.ticks_remaining(self._tick_count)
            logger.debug("Cooldown: %d ticks remaining before next entry", remaining)
            return

        open_pairs: Set[str] = set(self.portfolio.positions.keys())

        # Bidirectional selection: correlation + H4 + score threshold in one pass
        candidates = select_candidates_bidirectional(
            signals=signals,
            open_pairs=open_pairs,
            min_score=cfg.min_score_threshold,
            htf_closes=self._htf_closes,
            htf_fast=cfg.htf_ma_fast,
            htf_slow=cfg.htf_ma_slow,
        )

        for pair, action in candidates:
            if len(self.portfolio.positions) >= cfg.max_positions:
                break

            price = prices[pair]
            side = position_side(pair, action)
            sig = signals[pair]

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
                    "OPEN %-5s  | %-10s | %s | price=%-10s | score=%+.2f | RSI=%.1f",
                    side.value, pair, action.upper(),
                    _fmt_price(pair, price), sig.score, sig.rsi.value,
                )
                self._cooldown.record_entry(self._tick_count)
                break  # one entry per tick

    # ------------------------------------------------------------------
    # Status logging
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def tick(self) -> Dict:
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
        self.warmup()
        self._running = True
        logger.info("Dutch Agent v3 started.  Strategy: bidirectional USD — H4 trend-following.")
        logger.info(
            "Filters: corr-groups | min_score=%.2f | cooldown=%d ticks | H4=bidirectional",
            self.config.min_score_threshold,
            self.config.entry_cooldown_ticks,
        )
        logger.info("Pairs: %s", ", ".join(self.config.pairs))

        try:
            while self._running:
                self.tick()
                if max_ticks is not None and self._tick_count >= max_ticks:
                    break
                if max_ticks is None:
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
