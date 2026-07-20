"""
Daily RSI Extreme Mean Reversion strategy.

Only trades when the daily RSI reaches genuinely extreme levels —
below 25 (very oversold) or above 75 (very overbought).  At these
extremes the probability of mean reversion is much higher than at
normal RSI levels.

Signal logic:
  RSI < oversold_threshold  →  pair has been falling hard  →  BUY  (expect bounce)
  RSI > overbought_threshold →  pair has been rising hard  →  SELL (expect pullback)

For inverted pairs (USD is the base), the pair direction is reversed
relative to USD direction, but the mean-reversion logic is the same —
a very oversold USD/JPY means USD has been sold hard → expect a bounce
in USD/JPY → BUY USD/JPY.
"""

from __future__ import annotations

from typing import List

import numpy as np

from ..data import DailyBar
from ..strategy import Signal, Strategy


def _rsi(closes: np.ndarray, period: int) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0, deltas,  0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_g  = float(np.mean(gains))
    avg_l  = float(np.mean(losses))
    if avg_l == 0:
        return 100.0
    rs = avg_g / avg_l
    return 100.0 - 100.0 / (1.0 + rs)


class RSIExtremeStrategy(Strategy):
    """Mean reversion on daily RSI extremes."""

    def __init__(
        self,
        rsi_period: int = 14,
        oversold: float = 25.0,
        overbought: float = 75.0,
        cooldown_bars: int = 5,  # minimum bars between trades on same pair
    ) -> None:
        self.rsi_period  = rsi_period
        self.oversold    = oversold
        self.overbought  = overbought
        self.cooldown    = cooldown_bars
        self._last_signal_bar: dict = {}  # pair → bar index

    @property
    def name(self) -> str:
        return f"RSI Extreme MR (period={self.rsi_period}, OS={self.oversold}, OB={self.overbought})"

    def generate_signal(self, pair: str, bars: List[DailyBar]) -> Signal:
        if len(bars) < self.rsi_period + 2:
            return Signal("hold", pair)

        # Cooldown: don't fire again too soon after the last signal
        last = self._last_signal_bar.get(pair, -self.cooldown - 1)
        if len(bars) - 1 - last < self.cooldown:
            return Signal("hold", pair, "cooldown")

        closes = np.array([b.close for b in bars])
        rsi_val = _rsi(closes, self.rsi_period)

        if rsi_val < self.oversold:
            self._last_signal_bar[pair] = len(bars) - 1
            return Signal("buy", pair, f"RSI={rsi_val:.1f} < {self.oversold} (oversold → buy)")
        elif rsi_val > self.overbought:
            self._last_signal_bar[pair] = len(bars) - 1
            return Signal("sell", pair, f"RSI={rsi_val:.1f} > {self.overbought} (overbought → sell)")

        return Signal("hold", pair, f"RSI={rsi_val:.1f}")
