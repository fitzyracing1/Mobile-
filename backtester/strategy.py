"""Abstract base class for backtest strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from .data import DailyBar


@dataclass
class Signal:
    action: str   # "buy" | "sell" | "hold"
    pair: str
    reason: str = ""


class Strategy(ABC):
    """
    A strategy receives the full bar history up to (and including) today's
    close and returns a Signal.  Entry is simulated on the NEXT bar's open
    — you see the signal after the close, act at the next open.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def generate_signal(self, pair: str, bars: List[DailyBar]) -> Signal: ...
