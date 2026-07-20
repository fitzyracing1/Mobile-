"""
Backtest simulation engine.

Realistic fill assumptions:
  - Entry on the NEXT bar's open after a signal.
  - SL/TP are checked against the bar's high/low.
  - If both SL and TP are hit in the same bar, SL is assumed to trigger
    first (worst-case conservative assumption).
  - One position per pair at a time.
  - No slippage or commission modeled (OANDA major-pair spreads are ~1 pip).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .data import DailyBar
from .strategy import Signal, Strategy


@dataclass
class BacktestTrade:
    pair: str
    side: str            # "long" | "short"
    entry_date: str
    entry_price: float
    exit_date: str
    exit_price: float
    size_usd: float
    exit_reason: str     # "take_profit" | "stop_loss" | "end_of_test"

    @property
    def pnl(self) -> float:
        direction = 1.0 if self.side == "long" else -1.0
        return direction * (self.exit_price - self.entry_price) / self.entry_price * self.size_usd

    @property
    def pnl_pct(self) -> float:
        direction = 1.0 if self.side == "long" else -1.0
        return direction * (self.exit_price - self.entry_price) / self.entry_price * 100


@dataclass
class OpenPosition:
    pair: str
    side: str
    entry_date: str
    entry_price: float
    size_usd: float
    stop_loss: float
    take_profit: float


class BacktestEngine:
    """Simulates a strategy across historical daily bars."""

    def __init__(
        self,
        strategy: Strategy,
        pairs: List[str],
        stop_loss_pct: float = 2.0,
        take_profit_pct: float = 4.0,
        position_size_usd: float = 10_000.0,
        max_positions: int = 3,
    ) -> None:
        self.strategy = strategy
        self.pairs = pairs
        self.sl_pct = stop_loss_pct
        self.tp_pct = take_profit_pct
        self.size_usd = position_size_usd
        self.max_positions = max_positions

    def run(self, history: Dict[str, List[DailyBar]]) -> List[BacktestTrade]:
        """
        Run the backtest.  history[pair] must be sorted oldest-first.
        Returns all completed trades.
        """
        trades: List[BacktestTrade] = []
        open_positions: Dict[str, OpenPosition] = {}

        # Determine the shared set of dates across all pairs
        all_dates = sorted(set(
            bar.date
            for bars in history.values()
            for bar in bars
        ))

        # Build lookup: pair → date → bar
        lookup: Dict[str, Dict[str, DailyBar]] = {
            pair: {bar.date: bar for bar in bars}
            for pair, bars in history.items()
        }

        for i, date in enumerate(all_dates):
            for pair in self.pairs:
                bar = lookup[pair].get(date)
                if bar is None:
                    continue

                # ---- Check exits on open positions ----
                if pair in open_positions:
                    pos = open_positions[pair]
                    closed = self._check_exit(pos, bar, date)
                    if closed:
                        trades.append(closed)
                        del open_positions[pair]

                # ---- Generate signal on today's close ----
                if pair in open_positions:
                    continue  # already in a trade on this pair
                if len(open_positions) >= self.max_positions:
                    continue

                bars_so_far = [
                    lookup[pair][d]
                    for d in all_dates[: i + 1]
                    if d in lookup[pair]
                ]
                sig = self.strategy.generate_signal(pair, bars_so_far)

                if sig.action == "hold":
                    continue

                # Entry on NEXT bar's open
                if i + 1 >= len(all_dates):
                    continue
                next_date = all_dates[i + 1]
                next_bar = lookup[pair].get(next_date)
                if next_bar is None:
                    continue

                entry_price = next_bar.open
                side = "long" if sig.action == "buy" else "short"
                sl, tp = self._sl_tp(side, entry_price)

                open_positions[pair] = OpenPosition(
                    pair=pair,
                    side=side,
                    entry_date=next_date,
                    entry_price=entry_price,
                    size_usd=self.size_usd,
                    stop_loss=sl,
                    take_profit=tp,
                )

        # Close anything still open at end of test
        last_date = all_dates[-1]
        for pair, pos in open_positions.items():
            last_bar = lookup[pair].get(last_date)
            if last_bar:
                trades.append(BacktestTrade(
                    pair=pair,
                    side=pos.side,
                    entry_date=pos.entry_date,
                    entry_price=pos.entry_price,
                    exit_date=last_date,
                    exit_price=last_bar.close,
                    size_usd=pos.size_usd,
                    exit_reason="end_of_test",
                ))

        return trades

    def _sl_tp(self, side: str, entry: float) -> Tuple[float, float]:
        if side == "long":
            return (
                entry * (1 - self.sl_pct / 100),
                entry * (1 + self.tp_pct / 100),
            )
        else:
            return (
                entry * (1 + self.sl_pct / 100),
                entry * (1 - self.tp_pct / 100),
            )

    def _check_exit(
        self, pos: OpenPosition, bar: DailyBar, date: str
    ) -> Optional[BacktestTrade]:
        sl_hit = tp_hit = False

        if pos.side == "long":
            sl_hit = bar.low <= pos.stop_loss
            tp_hit = bar.high >= pos.take_profit
        else:
            sl_hit = bar.high >= pos.stop_loss
            tp_hit = bar.low <= pos.take_profit

        if sl_hit and tp_hit:
            # Conservative: assume stop-loss hit first
            sl_hit, tp_hit = True, False

        if sl_hit:
            return BacktestTrade(
                pair=pos.pair, side=pos.side,
                entry_date=pos.entry_date, entry_price=pos.entry_price,
                exit_date=date, exit_price=pos.stop_loss,
                size_usd=pos.size_usd, exit_reason="stop_loss",
            )
        if tp_hit:
            return BacktestTrade(
                pair=pos.pair, side=pos.side,
                entry_date=pos.entry_date, entry_price=pos.entry_price,
                exit_date=date, exit_price=pos.take_profit,
                size_usd=pos.size_usd, exit_reason="take_profit",
            )
        return None
