"""
Entry filters for the rebuilt Dutch agent.

Three problems killed v1:
  1. All positions entered simultaneously on the same signal → correlated losses
  2. No higher-timeframe check → traded against the macro trend
  3. Score threshold too loose → opened on weak signals

This module fixes all three.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np

from .config import CORRELATION_GROUPS, INVERTED_PAIRS
from .signals import CompositeSignal


# ---------------------------------------------------------------------------
# 1. Correlation filter
# ---------------------------------------------------------------------------

def select_candidates(
    signals: Dict[str, CompositeSignal],
    open_pairs: Set[str],
    min_score: float,
) -> List[str]:
    """
    Return at most one candidate pair per correlation group.

    Rules:
    - Skip any group that already has an open position in it.
    - Within each group, pick the pair with the most negative score
      (strongest sell-USD signal) that also clears min_score.
    - Returns pairs ordered by score strength (most negative first).

    Parameters
    ----------
    signals     : composite signal per pair from this tick
    open_pairs  : pairs that currently have an open position
    min_score   : score must be <= this value to qualify (e.g. -0.35)
    """
    candidates: List[tuple] = []  # (score, pair)

    for group, pairs in CORRELATION_GROUPS.items():
        # If any pair in the group is already open, skip the whole group.
        if any(p in open_pairs for p in pairs):
            continue

        best_pair: Optional[str] = None
        best_score = min_score  # must beat this to qualify

        for pair in pairs:
            sig = signals.get(pair)
            if sig is None:
                continue
            if sig.score <= best_score:
                best_score = sig.score
                best_pair = pair

        if best_pair is not None:
            candidates.append((best_score, best_pair))

    # Sort by score ascending (most negative = strongest signal first)
    candidates.sort(key=lambda x: x[0])
    return [pair for _, pair in candidates]


# ---------------------------------------------------------------------------
# 2. Higher-timeframe trend filter
# ---------------------------------------------------------------------------

def htf_trend_confirms(
    pair: str,
    htf_closes: np.ndarray,
    fast_period: int,
    slow_period: int,
) -> bool:
    """
    Return True only when the H4 moving-average trend confirms USD weakness.

    For non-inverted pairs (e.g. EUR/USD):
        H4 fast MA > H4 slow MA  →  pair trending up  →  USD weakening  →  confirm
    For inverted pairs (e.g. USD/JPY):
        H4 fast MA < H4 slow MA  →  pair trending down  →  USD weakening  →  confirm

    If there is insufficient H4 history the filter passes (don't block on
    missing data — the M1 signal is still used).
    """
    if len(htf_closes) < slow_period:
        return True  # not enough history — don't block

    fast_val = float(np.mean(htf_closes[-fast_period:]))
    slow_val = float(np.mean(htf_closes[-slow_period:]))

    is_inverted = pair in INVERTED_PAIRS
    if is_inverted:
        return fast_val < slow_val   # pair falling → USD weakening
    else:
        return fast_val > slow_val   # pair rising  → USD weakening


# ---------------------------------------------------------------------------
# 3. Entry cooldown
# ---------------------------------------------------------------------------

class EntryCooldown:
    """
    Enforces a minimum number of ticks between consecutive position entries.

    After any new position is opened, further entries are blocked for
    `cooldown_ticks` ticks.  This prevents the agent from entering all
    positions in a single burst when a broad USD-weakness signal fires.
    """

    def __init__(self, cooldown_ticks: int) -> None:
        self.cooldown_ticks = cooldown_ticks
        self._last_entry_tick: int = -cooldown_ticks  # allow entry immediately on startup

    def ready(self, current_tick: int) -> bool:
        return (current_tick - self._last_entry_tick) >= self.cooldown_ticks

    def record_entry(self, current_tick: int) -> None:
        self._last_entry_tick = current_tick

    def ticks_remaining(self, current_tick: int) -> int:
        remaining = self.cooldown_ticks - (current_tick - self._last_entry_tick)
        return max(0, remaining)
