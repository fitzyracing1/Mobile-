"""Agent configuration and defaults."""

from dataclasses import dataclass, field
from typing import List


# EUR/USD is the primary Dutch pair (the Netherlands uses the Euro).
# The agent also monitors a basket of G10 USD pairs so signals are
# diversified across multiple legs of the same USD-short thesis.
DEFAULT_PAIRS: List[str] = [
    "EUR/USD",  # Euro — primary Dutch pair
    "GBP/USD",  # Sterling
    "AUD/USD",  # Aussie
    "NZD/USD",  # Kiwi
    "USD/CHF",  # Swiss Franc (inverted — USD is the base)
    "USD/JPY",  # Yen       (inverted)
    "USD/CAD",  # Loonie    (inverted)
]

# USD is *base* for these pairs, so a falling price = weaker USD = our thesis.
INVERTED_PAIRS: List[str] = ["USD/CHF", "USD/JPY", "USD/CAD"]


@dataclass
class AgentConfig:
    """Tuneable parameters for the Dutch agent."""

    # Which currency pairs to trade
    pairs: List[str] = field(default_factory=lambda: list(DEFAULT_PAIRS))

    # Capital per position in notional USD
    position_size_usd: float = 10_000.0

    # Maximum simultaneous open positions
    max_positions: int = 4

    # Technical-indicator windows (bars)
    fast_ma: int = 10        # fast moving-average period
    slow_ma: int = 30        # slow moving-average period
    rsi_period: int = 14
    bb_period: int = 20      # Bollinger Band period
    bb_std: float = 2.0      # Bollinger Band standard-deviation multiplier

    # Signal thresholds
    rsi_oversold: float = 40.0   # RSI below this → USD weakening → buy signal
    rsi_overbought: float = 60.0

    # Stop-loss and take-profit in % of entry price
    stop_loss_pct: float = 1.0
    take_profit_pct: float = 2.0

    # How many bars of simulated history to pre-generate on startup
    warmup_bars: int = 60

    # Seconds between agent ticks in live-loop mode
    tick_interval_seconds: int = 5

    # Logging verbosity
    verbose: bool = True
