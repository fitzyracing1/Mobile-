"""
Dutch Agent — a paper-trading agent that sells the US Dollar.

The "Dutch Book" strategy exploits momentum and mean-reversion signals
across multiple USD pairs to build net short-USD positions.
"""

from .agent import DutchAgent

__all__ = ["DutchAgent"]
