"""
Entry filters for the Dutch agent (v3 — bidirectional).

Three filters are applied before any position is opened:

1. Correlation filter  — at most one position per correlated pair group
   (EUR, commodity, inverted-USD), picking the pair with the strongest signal.

2. H4 trend filter     — the 4-hour moving-average direction determines WHICH
   way to trade, not just whether to trade.  M1 and H4 must agree on direction.

3. Entry cooldown      — minimum N ticks between consecutive entries.

Bidirectional logic
-------------------
The agent now trades in both directions:
  sell_usd (short USD): when H4 trend shows USD weakening.
  buy_usd  (long USD):  when H4 trend shows USD strengthening.

Position side mapping:
  sell_usd + non-inverted pair (EUR/USD) → LONG  (buy EUR, sell USD)
  sell_usd + inverted pair    (USD/JPY)  → SHORT (sell USD/JPY)
  buy_usd  + non-inverted pair (EUR/USD) → SHORT (sell EUR, buy USD)
  buy_usd  + inverted pair    (USD/JPY)  → LONG  (buy USD/JPY)
"""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from .config import CORRELATION_GROUPS, INVERTED_PAIRS
from .position import Side
from .signals import CompositeSignal


# ---------------------------------------------------------------------------
# H4 direction
# ---------------------------------------------------------------------------

def htf_direction(
    pair: str,
    closes: np.ndarray,
    fast_period: int,
    slow_period: int,
) -> str:
    """
    Determine the H4 USD direction for a pair.

    Returns
    -------
    "sell_usd"  H4 trend shows USD weakening   → go short USD
    "buy_usd"   H4 trend shows USD strengthening → go long USD
    "neutral"   insufficient data
    """
    if len(closes) < slow_period:
        return "neutral"

    fast_val = float(np.mean(closes[-fast_period:]))
    slow_val = float(np.mean(closes[-slow_period:]))
    is_inverted = pair in INVERTED_PAIRS

    if is_inverted:
        # Rising USD/JPY → USD strengthening
        return "buy_usd" if fast_val > slow_val else "sell_usd"
    else:
        # Rising EUR/USD → USD weakening
        return "sell_usd" if fast_val > slow_val else "buy_usd"


# ---------------------------------------------------------------------------
# Backward-compatible alias (used by old unit tests)
# ---------------------------------------------------------------------------

def htf_trend_confirms(
    pair: str,
    htf_closes: np.ndarray,
    fast_period: int,
    slow_period: int,
) -> bool:
    """True when H4 direction is sell_usd (original unidirectional API)."""
    return htf_direction(pair, htf_closes, fast_period, slow_period) in ("sell_usd", "neutral")


# ---------------------------------------------------------------------------
# Bidirectional candidate selection
# ---------------------------------------------------------------------------

def select_candidates_bidirectional(
    signals: Dict[str, CompositeSignal],
    open_pairs: Set[str],
    min_score: float,
    htf_closes: Dict[str, np.ndarray],
    htf_fast: int,
    htf_slow: int,
) -> List[Tuple[str, str]]:
    """
    Return at most one (pair, action) per correlation group.

    Both the M1 composite signal and the H4 trend must agree on direction
    ("sell_usd" or "buy_usd").  The pair with the highest |score| that
    clears min_score wins the group.

    Parameters
    ----------
    signals     : M1 composite signals keyed by pair
    open_pairs  : pairs that currently have an open position
    min_score   : |score| must exceed abs(min_score) to qualify
    htf_closes  : H4 close arrays keyed by pair
    htf_fast    : fast H4 MA period
    htf_slow    : slow H4 MA period

    Returns
    -------
    List of (pair, action) ordered by |score| descending.
    """
    threshold = abs(min_score)
    candidates: List[Tuple[float, str, str]] = []  # (|score|, pair, action)

    for group, pairs in CORRELATION_GROUPS.items():
        if any(p in open_pairs for p in pairs):
            continue

        best_abs = threshold
        best_pair: Optional[str] = None
        best_action: Optional[str] = None

        for pair in pairs:
            sig = signals.get(pair)
            if sig is None or sig.action not in ("sell_usd", "buy_usd"):
                continue
            if abs(sig.score) < best_abs:
                continue

            # H4 must agree with the M1 signal direction
            h4 = htf_closes.get(pair, np.array([]))
            h4_dir = htf_direction(pair, h4, htf_fast, htf_slow)

            if h4_dir == "neutral":
                continue
            if sig.action != h4_dir:
                continue

            best_abs = abs(sig.score)
            best_pair = pair
            best_action = sig.action

        if best_pair is not None:
            candidates.append((best_abs, best_pair, best_action))

    candidates.sort(reverse=True)
    return [(pair, action) for _, pair, action in candidates]


# ---------------------------------------------------------------------------
# Backward-compatible unidirectional selection (used by old unit tests)
# ---------------------------------------------------------------------------

def select_candidates(
    signals: Dict[str, CompositeSignal],
    open_pairs: Set[str],
    min_score: float,
) -> List[str]:
    """Original API: sell_usd only, no H4 filter."""
    threshold = abs(min_score)
    candidates: List[Tuple[float, str]] = []

    for group, pairs in CORRELATION_GROUPS.items():
        if any(p in open_pairs for p in pairs):
            continue
        best_abs = threshold
        best_pair: Optional[str] = None
        for pair in pairs:
            sig = signals.get(pair)
            if sig is None:
                continue
            if sig.score >= -threshold:
                continue
            if abs(sig.score) > best_abs:
                best_abs = abs(sig.score)
                best_pair = pair
        if best_pair is not None:
            candidates.append((best_abs, best_pair))

    candidates.sort(reverse=True)
    return [p for _, p in candidates]


# ---------------------------------------------------------------------------
# Position side helper
# ---------------------------------------------------------------------------

def position_side(pair: str, action: str) -> Side:
    """
    Translate (pair, action) into the correct OANDA order side.

    sell_usd: we want to be long the non-USD currency.
        non-inverted (EUR/USD) → LONG
        inverted     (USD/JPY) → SHORT
    buy_usd: we want to be long the USD.
        non-inverted (EUR/USD) → SHORT
        inverted     (USD/JPY) → LONG
    """
    is_inverted = pair in INVERTED_PAIRS
    if action == "sell_usd":
        return Side.SHORT if is_inverted else Side.LONG
    else:  # buy_usd
        return Side.LONG if is_inverted else Side.SHORT


# ---------------------------------------------------------------------------
# Entry cooldown
# ---------------------------------------------------------------------------

class EntryCooldown:
    """
    Enforces a minimum number of ticks between consecutive position entries.
    """

    def __init__(self, cooldown_ticks: int) -> None:
        self.cooldown_ticks = cooldown_ticks
        self._last_entry_tick: int = -cooldown_ticks

    def ready(self, current_tick: int) -> bool:
        return (current_tick - self._last_entry_tick) >= self.cooldown_ticks

    def record_entry(self, current_tick: int) -> None:
        self._last_entry_tick = current_tick

    def ticks_remaining(self, current_tick: int) -> int:
        return max(0, self.cooldown_ticks - (current_tick - self._last_entry_tick))
