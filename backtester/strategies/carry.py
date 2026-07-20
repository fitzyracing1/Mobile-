"""
Interest Rate Carry Trade strategy.

The carry trade earns the interest rate differential between two currencies.
When currency A yields more than currency B, you borrow in B, invest in A,
and pocket the difference — regardless of price movement.

This strategy:
  1. Determines the interest rate differential for each pair based on
     approximate central bank policy rates (hardcoded by date range).
  2. Goes LONG the pair when the base currency yields more than the quote
     (positive carry), SHORT when the quote yields more.
  3. Only enters when the differential is meaningful (>= min_diff_pct).
  4. Holds the position — does NOT try to time entries with technical signals.
     It re-enters after any exit (stop or TP) after a short cooling-off period.

Approximate central bank rates used (annual %):
  Fed (USD):  0.25 (2020-2021) → 4.50 (2023) → 5.25-5.50 (2024-2025) → 4.25 (2026)
  RBA (AUD):  0.10 (2020-2021) → 4.35 (2023+)
  RBNZ(NZD):  0.25 (2020-2021) → 5.50 (2023) → 4.75 (2026)
  BOE (GBP):  0.10 (2020-2021) → 5.25 (2023) → 4.25 (2026)
  ECB (EUR): -0.50 (2020-2021) → 4.00 (2023) → 2.50 (2026)
  SNB (CHF): -0.75 (2020-2021) → 1.75 (2023) → 0.25 (2026)
  BOJ (JPY):  0.10 (2020-2024) → 0.50 (2025+)
  BOC (CAD):  0.25 (2020-2021) → 5.00 (2023) → 3.25 (2026)
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from ..data import DailyBar
from ..strategy import Signal, Strategy

# (start_date, end_date, rate_pct)
_RATE_SCHEDULE: Dict[str, List[Tuple[str, str, float]]] = {
    "USD": [
        ("2020-01-01", "2022-02-28", 0.25),
        ("2022-03-01", "2022-12-31", 2.50),
        ("2023-01-01", "2024-06-30", 5.25),
        ("2024-07-01", "2025-06-30", 4.75),
        ("2025-07-01", "2030-12-31", 4.25),
    ],
    "AUD": [
        ("2020-01-01", "2022-04-30", 0.10),
        ("2022-05-01", "2023-12-31", 4.10),
        ("2024-01-01", "2030-12-31", 4.35),
    ],
    "NZD": [
        ("2020-01-01", "2021-09-30", 0.25),
        ("2021-10-01", "2023-06-30", 5.25),
        ("2023-07-01", "2024-12-31", 5.50),
        ("2025-01-01", "2030-12-31", 4.75),
    ],
    "GBP": [
        ("2020-01-01", "2021-12-31", 0.10),
        ("2022-01-01", "2023-06-30", 4.00),
        ("2023-07-01", "2024-06-30", 5.25),
        ("2024-07-01", "2030-12-31", 4.25),
    ],
    "EUR": [
        ("2020-01-01", "2022-06-30", -0.50),
        ("2022-07-01", "2023-09-30", 4.00),
        ("2023-10-01", "2025-06-30", 3.00),
        ("2025-07-01", "2030-12-31", 2.50),
    ],
    "CHF": [
        ("2020-01-01", "2022-06-30", -0.75),
        ("2022-07-01", "2023-06-30", 1.75),
        ("2023-07-01", "2024-12-31", 1.25),
        ("2025-01-01", "2030-12-31", 0.25),
    ],
    "JPY": [
        ("2020-01-01", "2024-03-31", 0.10),
        ("2024-04-01", "2030-12-31", 0.50),
    ],
    "CAD": [
        ("2020-01-01", "2022-01-31", 0.25),
        ("2022-02-01", "2023-09-30", 5.00),
        ("2023-10-01", "2025-06-30", 4.25),
        ("2025-07-01", "2030-12-31", 3.25),
    ],
}

# Extract base and quote currency from pair string
_CURRENCIES: Dict[str, Tuple[str, str]] = {
    "EUR/USD": ("EUR", "USD"),
    "GBP/USD": ("GBP", "USD"),
    "AUD/USD": ("AUD", "USD"),
    "NZD/USD": ("NZD", "USD"),
    "USD/CHF": ("USD", "CHF"),
    "USD/JPY": ("USD", "JPY"),
    "USD/CAD": ("USD", "CAD"),
}


def _rate(currency: str, date: str) -> float:
    for start, end, rate in _RATE_SCHEDULE.get(currency, []):
        if start <= date <= end:
            return rate
    return 0.0


def _carry_direction(pair: str, date: str) -> Tuple[str, float]:
    """
    Returns ("buy"|"sell"|"hold", differential_pct).
    "buy"  = long the base currency (positive carry for base).
    "sell" = short the base currency (positive carry for quote).
    """
    base, quote = _CURRENCIES[pair]
    base_rate  = _rate(base,  date)
    quote_rate = _rate(quote, date)
    diff = base_rate - quote_rate

    if diff > 0:
        return "buy", diff
    elif diff < 0:
        return "sell", -diff
    return "hold", 0.0


class CarryTradeStrategy(Strategy):
    """
    Always be positioned in the direction of the interest rate carry.
    Re-enters after stop-loss exits once a brief cooling-off period passes.
    """

    def __init__(self, min_diff_pct: float = 0.5, cooldown_bars: int = 5) -> None:
        self.min_diff = min_diff_pct
        self.cooldown = cooldown_bars
        self._last_exit_bar: Dict[str, int] = {}

    @property
    def name(self) -> str:
        return f"Carry Trade (min_diff={self.min_diff}%)"

    def generate_signal(self, pair: str, bars: List[DailyBar]) -> Signal:
        if not bars:
            return Signal("hold", pair)

        date = bars[-1].date

        # Cooling-off after a stop-loss
        last_exit = self._last_exit_bar.get(pair, -self.cooldown - 1)
        if len(bars) - 1 - last_exit < self.cooldown:
            return Signal("hold", pair, "post-exit cooldown")

        action, diff = _carry_direction(pair, date)

        if action == "hold" or diff < self.min_diff:
            return Signal("hold", pair, f"carry diff={diff:.2f}% below threshold")

        return Signal(action, pair, f"carry diff={diff:.2f}% → {action}")

    def notify_exit(self, pair: str, bar_index: int) -> None:
        """Call this when a position is exited so the cooldown starts."""
        self._last_exit_bar[pair] = bar_index
