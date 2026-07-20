"""Agent configuration and defaults."""

from dataclasses import dataclass, field
from typing import Dict, List


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

# Pairs grouped by correlation.  The agent takes at most ONE position per
# group to avoid betting the same macro move three times simultaneously.
# EUR/USD and GBP/USD move together (~0.85 correlation); AUD and NZD are
# nearly identical commodity proxies; the inverted USD-base pairs form their
# own cluster.
CORRELATION_GROUPS: Dict[str, List[str]] = {
    "EUR": ["EUR/USD", "GBP/USD"],
    "COM": ["AUD/USD", "NZD/USD"],
    "INV": ["USD/JPY", "USD/CHF", "USD/CAD"],
}


@dataclass
class AgentConfig:
    """Tuneable parameters for the Dutch agent."""

    # Which currency pairs to trade
    pairs: List[str] = field(default_factory=lambda: list(DEFAULT_PAIRS))

    # Capital per position in notional USD
    position_size_usd: float = 10_000.0

    # Maximum simultaneous open positions (hard cap; correlation filter
    # further limits to 1 per group = 3 effective max)
    max_positions: int = 3

    # Technical-indicator windows (bars, M1 timeframe)
    fast_ma: int = 10
    slow_ma: int = 30
    rsi_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0

    # Signal thresholds
    rsi_oversold: float = 40.0
    rsi_overbought: float = 60.0

    # Minimum composite score to open a position (must be < this negative
    # value).  Higher magnitude = only trade on strong signals.
    min_score_threshold: float = -0.35

    # Stop-loss and take-profit in % of entry price.
    # Wider stops give H4-driven positions room to breathe through intraday noise.
    stop_loss_pct: float = 2.0
    take_profit_pct: float = 4.0

    # Minimum ticks to wait between opening any two positions.
    # Prevents all positions being entered in a single burst.
    entry_cooldown_ticks: int = 15

    # Higher-timeframe (H4) trend filter.
    # A new position is only opened when the H4 MA confirms USD weakness.
    htf_granularity: str = "H4"
    htf_bars: int = 50        # number of H4 bars to keep
    htf_ma_fast: int = 10     # fast H4 MA period
    htf_ma_slow: int = 25     # slow H4 MA period

    # How many M1 bars of history to pre-load on startup
    warmup_bars: int = 60

    # Seconds between agent ticks in live-loop mode
    tick_interval_seconds: int = 60

    # Logging verbosity
    verbose: bool = True
