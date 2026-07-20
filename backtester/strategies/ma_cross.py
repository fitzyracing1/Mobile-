"""
Daily MA crossover — the original strategy on a daily timeframe.
Included as a baseline to confirm the backtest catches losing strategies.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..data import DailyBar
from ..strategy import Signal, Strategy


class MACrossStrategy(Strategy):
    """
    Buy when fast daily MA > slow daily MA (USD weakening on non-inverted pairs),
    sell when fast < slow.

    This is the same logic as the live agent but on daily bars.
    """

    INVERTED = {"USD/CHF", "USD/JPY", "USD/CAD"}

    def __init__(self, fast: int = 10, slow: int = 30) -> None:
        self.fast = fast
        self.slow = slow

    @property
    def name(self) -> str:
        return f"MA Cross ({self.fast}/{self.slow} daily)"

    def generate_signal(self, pair: str, bars: List[DailyBar]) -> Signal:
        if len(bars) < self.slow:
            return Signal("hold", pair)

        closes = np.array([b.close for b in bars])
        fast_val = float(np.mean(closes[-self.fast:]))
        slow_val = float(np.mean(closes[-self.slow:]))

        inverted = pair in self.INVERTED
        if fast_val > slow_val:
            action = "sell" if inverted else "buy"
        elif fast_val < slow_val:
            action = "buy" if inverted else "sell"
        else:
            return Signal("hold", pair)

        return Signal(action, pair, f"fast={fast_val:.5f} slow={slow_val:.5f}")
