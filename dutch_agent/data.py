"""
Market data layer.

In production you would plug in a real broker / exchange feed here
(e.g. OANDA, Interactive Brokers, or a crypto exchange via ccxt).
For demo / paper-trading the module generates realistic synthetic
OHLC bars using geometric Brownian motion seeded from approximate
real-world starting prices.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

# Approximate mid prices as of mid-2024 (used only to seed the simulation)
SEED_PRICES: Dict[str, float] = {
    "EUR/USD": 1.0850,
    "GBP/USD": 1.2700,
    "AUD/USD": 0.6600,
    "NZD/USD": 0.6100,
    "USD/CHF": 0.9050,
    "USD/JPY": 154.00,
    "USD/CAD": 1.3600,
}

# Approximate annualised volatility for each pair
PAIR_VOL: Dict[str, float] = {
    "EUR/USD": 0.070,
    "GBP/USD": 0.080,
    "AUD/USD": 0.090,
    "NZD/USD": 0.090,
    "USD/CHF": 0.065,
    "USD/JPY": 0.080,
    "USD/CAD": 0.065,
}

BARS_PER_YEAR = 252 * 24  # hourly bars


@dataclass
class Bar:
    """A single OHLC price bar."""

    timestamp: float  # Unix epoch seconds
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0


class PriceHistory:
    """Holds a rolling window of bars for one pair."""

    def __init__(self, pair: str, max_bars: int = 200) -> None:
        self.pair = pair
        self.max_bars = max_bars
        self._bars: List[Bar] = []
        self._price = SEED_PRICES.get(pair, 1.0)

    def add_bar(self, bar: Bar) -> None:
        self._bars.append(bar)
        if len(self._bars) > self.max_bars:
            self._bars.pop(0)

    def generate_next_bar(self, timestamp: Optional[float] = None) -> Bar:
        """Generate the next synthetic bar using GBM."""
        pair = self.pair
        sigma = PAIR_VOL.get(pair, 0.07)
        dt = 1.0 / BARS_PER_YEAR
        drift = -0.01 * dt  # mild bearish-USD drift embedded in the sim

        z = random.gauss(0, 1)
        ret = drift + sigma * math.sqrt(dt) * z
        open_px = self._price
        close_px = open_px * math.exp(ret)

        intra_noise = abs(random.gauss(0, sigma * math.sqrt(dt) * open_px))
        high_px = max(open_px, close_px) + intra_noise * random.uniform(0.3, 1.0)
        low_px = min(open_px, close_px) - intra_noise * random.uniform(0.3, 1.0)

        self._price = close_px

        bar = Bar(
            timestamp=timestamp or time.time(),
            open=round(open_px, 5),
            high=round(high_px, 5),
            low=round(low_px, 5),
            close=round(close_px, 5),
            volume=round(random.uniform(100, 1000), 0),
        )
        self.add_bar(bar)
        return bar

    @property
    def bars(self) -> List[Bar]:
        return list(self._bars)

    def closes(self) -> np.ndarray:
        return np.array([b.close for b in self._bars])

    def highs(self) -> np.ndarray:
        return np.array([b.high for b in self._bars])

    def lows(self) -> np.ndarray:
        return np.array([b.low for b in self._bars])

    def current_price(self) -> float:
        if self._bars:
            return self._bars[-1].close
        return self._price


class DataFeed:
    """Manages price history for all tracked pairs."""

    def __init__(self, pairs: List[str], max_bars: int = 200) -> None:
        self.pairs = pairs
        self._histories: Dict[str, PriceHistory] = {
            p: PriceHistory(p, max_bars) for p in pairs
        }

    def warmup(self, n_bars: int) -> None:
        """Pre-populate histories with synthetic bars."""
        t0 = time.time() - n_bars * 3600  # pretend bars are hourly
        for i in range(n_bars):
            ts = t0 + i * 3600
            for hist in self._histories.values():
                hist.generate_next_bar(timestamp=ts)

    def tick(self) -> Dict[str, Bar]:
        """Generate one new bar for every pair and return them."""
        ts = time.time()
        return {p: h.generate_next_bar(ts) for p, h in self._histories.items()}

    def history(self, pair: str) -> PriceHistory:
        return self._histories[pair]

    def price(self, pair: str) -> float:
        return self._histories[pair].current_price()
