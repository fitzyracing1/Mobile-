"""
Position and portfolio management for the Dutch agent.

All positions are paper-traded; no real orders are ever sent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class Side(str, Enum):
    LONG = "LONG"    # bought the non-USD currency (e.g. long EUR, short USD)
    SHORT = "SHORT"  # sold the non-USD currency (long USD)


@dataclass
class Trade:
    """A completed round-trip trade."""

    pair: str
    side: Side
    entry_price: float
    exit_price: float
    size_usd: float
    entry_time: float
    exit_time: float
    reason: str  # "stop_loss" | "take_profit" | "signal_exit" | "agent_close"

    @property
    def pnl(self) -> float:
        """Profit/loss in USD."""
        direction = 1.0 if self.side == Side.LONG else -1.0
        price_change_pct = (self.exit_price - self.entry_price) / self.entry_price
        return direction * price_change_pct * self.size_usd

    @property
    def duration_seconds(self) -> float:
        return self.exit_time - self.entry_time


@dataclass
class Position:
    """An open position."""

    pair: str
    side: Side
    entry_price: float
    size_usd: float
    entry_time: float = field(default_factory=time.time)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

    def unrealised_pnl(self, current_price: float) -> float:
        direction = 1.0 if self.side == Side.LONG else -1.0
        return direction * ((current_price - self.entry_price) / self.entry_price) * self.size_usd

    def is_stopped(self, current_price: float) -> bool:
        if self.stop_loss is None:
            return False
        if self.side == Side.LONG:
            return current_price <= self.stop_loss
        return current_price >= self.stop_loss

    def is_target_hit(self, current_price: float) -> bool:
        if self.take_profit is None:
            return False
        if self.side == Side.LONG:
            return current_price >= self.take_profit
        return current_price <= self.take_profit


class Portfolio:
    """Tracks open positions and closed trade history."""

    def __init__(self, starting_balance: float = 100_000.0) -> None:
        self.starting_balance = starting_balance
        self.cash = starting_balance
        self.positions: Dict[str, Position] = {}   # keyed by pair
        self.trades: List[Trade] = []

    # ------------------------------------------------------------------
    # Position lifecycle
    # ------------------------------------------------------------------

    def open_position(
        self,
        pair: str,
        side: Side,
        price: float,
        size_usd: float,
        stop_loss_pct: float,
        take_profit_pct: float,
    ) -> Optional[Position]:
        """Open a new position.  Returns None if pair already has an open position."""
        if pair in self.positions:
            return None

        sl_multiplier = (1 - stop_loss_pct / 100) if side == Side.LONG else (1 + stop_loss_pct / 100)
        tp_multiplier = (1 + take_profit_pct / 100) if side == Side.LONG else (1 - take_profit_pct / 100)

        pos = Position(
            pair=pair,
            side=side,
            entry_price=price,
            size_usd=size_usd,
            stop_loss=price * sl_multiplier,
            take_profit=price * tp_multiplier,
        )
        self.positions[pair] = pos
        self.cash -= size_usd
        return pos

    def close_position(self, pair: str, current_price: float, reason: str) -> Optional[Trade]:
        """Close an open position and record the trade."""
        pos = self.positions.pop(pair, None)
        if pos is None:
            return None

        trade = Trade(
            pair=pair,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=current_price,
            size_usd=pos.size_usd,
            entry_time=pos.entry_time,
            exit_time=time.time(),
            reason=reason,
        )
        self.cash += pos.size_usd + trade.pnl
        self.trades.append(trade)
        return trade

    # ------------------------------------------------------------------
    # Portfolio-level metrics
    # ------------------------------------------------------------------

    def equity(self, current_prices: Dict[str, float]) -> float:
        """Mark-to-market equity."""
        unrealised = sum(
            pos.unrealised_pnl(current_prices.get(pair, pos.entry_price))
            for pair, pos in self.positions.items()
        )
        return self.cash + sum(pos.size_usd for pos in self.positions.values()) + unrealised

    def total_pnl(self, current_prices: Dict[str, float]) -> float:
        return self.equity(current_prices) - self.starting_balance

    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        wins = sum(1 for t in self.trades if t.pnl > 0)
        return wins / len(self.trades)

    def total_realised_pnl(self) -> float:
        return sum(t.pnl for t in self.trades)

    def summary(self, current_prices: Dict[str, float]) -> Dict:
        return {
            "starting_balance": self.starting_balance,
            "cash": round(self.cash, 2),
            "equity": round(self.equity(current_prices), 2),
            "total_pnl": round(self.total_pnl(current_prices), 2),
            "realised_pnl": round(self.total_realised_pnl(), 2),
            "open_positions": len(self.positions),
            "closed_trades": len(self.trades),
            "win_rate_pct": round(self.win_rate() * 100, 1),
        }
