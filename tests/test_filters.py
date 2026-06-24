"""Tests for the v2/v3 entry filters."""

import numpy as np
import pytest

from dutch_agent.filters import (
    EntryCooldown,
    htf_direction,
    htf_trend_confirms,
    position_side,
    select_candidates,
    select_candidates_bidirectional,
)
from dutch_agent.position import Side
from dutch_agent.signals import CompositeSignal, MACrossSignal, RSISignal, BollingerSignal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sig(score: float, action: str = None) -> CompositeSignal:
    if action is None:
        action = "sell_usd" if score < -0.1 else ("buy_usd" if score > 0.1 else "hold")
    return CompositeSignal(
        ma_cross=MACrossSignal(1.0, 1.0, 0),
        rsi=RSISignal(50.0, 0),
        bollinger=BollingerSignal(1.1, 1.0, 0.9, 0.5, 0),
        score=score,
        action=action,
    )


def make_trend(n, slope):
    return np.array([1.0 + i * slope for i in range(n)])


# ---------------------------------------------------------------------------
# select_candidates
# ---------------------------------------------------------------------------

class TestSelectCandidates:
    def test_returns_best_per_group(self):
        signals = {
            "EUR/USD": make_sig(-0.75),  # EUR group — strongest
            "GBP/USD": make_sig(-0.40),  # EUR group — weaker
            "AUD/USD": make_sig(-0.50),  # COM group
            "NZD/USD": make_sig(-0.20),  # COM group — below threshold
            "USD/JPY": make_sig(-0.60),  # INV group
            "USD/CHF": make_sig(+0.30),  # INV group — wrong direction
            "USD/CAD": make_sig(-0.10),  # INV group — too weak
        }
        candidates = select_candidates(signals, open_pairs=set(), min_score=-0.35)
        # EUR group: EUR/USD (-0.75) wins over GBP/USD (-0.40)
        # COM group: AUD/USD (-0.50) wins (NZD/USD only -0.20, doesn't clear -0.35)
        # INV group: USD/JPY (-0.60) wins
        assert "EUR/USD" in candidates
        assert "GBP/USD" not in candidates
        assert "AUD/USD" in candidates
        assert "USD/JPY" in candidates
        assert len(candidates) == 3

    def test_skips_group_with_open_position(self):
        signals = {
            "EUR/USD": make_sig(-0.80),
            "GBP/USD": make_sig(-0.80),
            "AUD/USD": make_sig(-0.80),
            "NZD/USD": make_sig(-0.80),
            "USD/JPY": make_sig(-0.80),
            "USD/CHF": make_sig(-0.80),
            "USD/CAD": make_sig(-0.80),
        }
        # EUR/USD already open → skip entire EUR group
        candidates = select_candidates(signals, open_pairs={"EUR/USD"}, min_score=-0.35)
        assert "EUR/USD" not in candidates
        assert "GBP/USD" not in candidates

    def test_score_threshold_filters_weak_signals(self):
        signals = {
            "EUR/USD": make_sig(-0.20),  # doesn't clear -0.35
            "GBP/USD": make_sig(-0.15),
            "AUD/USD": make_sig(-0.10),
            "NZD/USD": make_sig(-0.05),
            "USD/JPY": make_sig(-0.30),
            "USD/CHF": make_sig(-0.25),
            "USD/CAD": make_sig(-0.10),
        }
        candidates = select_candidates(signals, open_pairs=set(), min_score=-0.35)
        assert candidates == []

    def test_ordered_by_strength(self):
        signals = {
            "EUR/USD": make_sig(-0.40),
            "AUD/USD": make_sig(-0.90),  # strongest
            "USD/JPY": make_sig(-0.60),
        }
        candidates = select_candidates(signals, open_pairs=set(), min_score=-0.35)
        # AUD should be first (most negative)
        assert candidates[0] == "AUD/USD"

    def test_no_candidates_when_all_open(self):
        signals = {p: make_sig(-0.80) for p in
                   ["EUR/USD","GBP/USD","AUD/USD","NZD/USD","USD/JPY","USD/CHF","USD/CAD"]}
        all_open = set(signals.keys())
        candidates = select_candidates(signals, open_pairs=all_open, min_score=-0.35)
        assert candidates == []


# ---------------------------------------------------------------------------
# htf_trend_confirms
# ---------------------------------------------------------------------------

class TestHTFTrendFilter:
    def test_non_inverted_uptrend_confirms(self):
        # EUR/USD rising on H4 → USD weakening → confirm
        closes = make_trend(30, slope=0.001)
        assert htf_trend_confirms("EUR/USD", closes, fast_period=10, slow_period=25) is True

    def test_non_inverted_downtrend_rejects(self):
        # EUR/USD falling on H4 → USD strengthening → reject
        closes = make_trend(30, slope=-0.001)
        assert htf_trend_confirms("EUR/USD", closes, fast_period=10, slow_period=25) is False

    def test_inverted_downtrend_confirms(self):
        # USD/JPY falling on H4 → USD weakening → confirm
        closes = make_trend(30, slope=-0.1)
        assert htf_trend_confirms("USD/JPY", closes, fast_period=10, slow_period=25) is True

    def test_inverted_uptrend_rejects(self):
        # USD/JPY rising on H4 → USD strengthening → reject
        closes = make_trend(30, slope=0.1)
        assert htf_trend_confirms("USD/JPY", closes, fast_period=10, slow_period=25) is False

    def test_insufficient_data_passes(self):
        # Only 5 bars — not enough for slow_period=25 — should not block
        closes = make_trend(5, slope=-0.001)
        assert htf_trend_confirms("EUR/USD", closes, fast_period=10, slow_period=25) is True

    def test_empty_array_passes(self):
        assert htf_trend_confirms("GBP/USD", np.array([]), fast_period=10, slow_period=25) is True


# ---------------------------------------------------------------------------
# EntryCooldown
# ---------------------------------------------------------------------------

class TestEntryCooldown:
    def test_ready_at_startup(self):
        cd = EntryCooldown(cooldown_ticks=15)
        assert cd.ready(current_tick=1) is True

    def test_blocked_after_entry(self):
        cd = EntryCooldown(cooldown_ticks=15)
        cd.record_entry(current_tick=5)
        assert cd.ready(current_tick=6) is False
        assert cd.ready(current_tick=19) is False

    def test_unblocked_after_cooldown(self):
        cd = EntryCooldown(cooldown_ticks=15)
        cd.record_entry(current_tick=5)
        assert cd.ready(current_tick=20) is True

    def test_ticks_remaining(self):
        cd = EntryCooldown(cooldown_ticks=15)
        cd.record_entry(current_tick=10)
        assert cd.ticks_remaining(current_tick=12) == 13
        assert cd.ticks_remaining(current_tick=25) == 0

    def test_multiple_entries_reset_clock(self):
        cd = EntryCooldown(cooldown_ticks=10)
        cd.record_entry(current_tick=1)
        cd.record_entry(current_tick=12)  # second entry
        assert cd.ready(current_tick=21) is False  # clock reset to tick 12
        assert cd.ready(current_tick=22) is True


# ---------------------------------------------------------------------------
# htf_direction
# ---------------------------------------------------------------------------

class TestHTFDirection:
    def test_non_inverted_uptrend_is_sell_usd(self):
        closes = make_trend(30, slope=0.001)
        assert htf_direction("EUR/USD", closes, 10, 25) == "sell_usd"

    def test_non_inverted_downtrend_is_buy_usd(self):
        closes = make_trend(30, slope=-0.001)
        assert htf_direction("EUR/USD", closes, 10, 25) == "buy_usd"

    def test_inverted_uptrend_is_buy_usd(self):
        closes = make_trend(30, slope=0.1)
        assert htf_direction("USD/JPY", closes, 10, 25) == "buy_usd"

    def test_inverted_downtrend_is_sell_usd(self):
        closes = make_trend(30, slope=-0.1)
        assert htf_direction("USD/JPY", closes, 10, 25) == "sell_usd"

    def test_neutral_when_insufficient_data(self):
        assert htf_direction("EUR/USD", make_trend(5, 0.001), 10, 25) == "neutral"

    def test_neutral_on_empty(self):
        assert htf_direction("GBP/USD", np.array([]), 10, 25) == "neutral"


# ---------------------------------------------------------------------------
# select_candidates_bidirectional
# ---------------------------------------------------------------------------

class TestSelectCandidatesBidirectional:
    def _htf(self, direction: str) -> np.ndarray:
        """Return 30-bar H4 history trending in the given USD direction."""
        slope = 0.001 if direction == "sell_usd" else -0.001
        return make_trend(30, slope)

    def test_picks_buy_usd_when_h4_is_strong_usd(self):
        # EUR/USD falling on H4 (buy_usd) + M1 also says buy_usd
        htf_closes = {"EUR/USD": self._htf("buy_usd"), "GBP/USD": self._htf("buy_usd"),
                      "AUD/USD": self._htf("buy_usd"), "NZD/USD": self._htf("buy_usd"),
                      "USD/JPY": make_trend(30, 0.1),  # rising = buy_usd
                      "USD/CHF": make_trend(30, 0.001), "USD/CAD": make_trend(30, 0.001)}
        signals = {
            "EUR/USD": make_sig(-0.50),   # M1 says sell_usd — disagrees with H4
            "GBP/USD": make_sig(+0.60),   # M1 says buy_usd — agrees with H4
            "AUD/USD": make_sig(+0.40),
            "NZD/USD": make_sig(+0.20),
            "USD/JPY": make_sig(+0.70),   # inverted + H4 rising = buy_usd ✓
            "USD/CHF": make_sig(+0.40),
            "USD/CAD": make_sig(+0.10),   # below threshold
        }
        candidates = select_candidates_bidirectional(
            signals, open_pairs=set(), min_score=-0.35,
            htf_closes=htf_closes, htf_fast=10, htf_slow=25,
        )
        pairs = [p for p, _ in candidates]
        actions = {p: a for p, a in candidates}
        assert "GBP/USD" in pairs
        assert actions.get("GBP/USD") == "buy_usd"
        assert "EUR/USD" not in pairs  # M1/H4 disagree

    def test_skips_group_with_open_position(self):
        htf_closes = {p: make_trend(30, -0.001) for p in
                      ["EUR/USD","GBP/USD","AUD/USD","NZD/USD","USD/JPY","USD/CHF","USD/CAD"]}
        signals = {p: make_sig(+0.80) for p in htf_closes}
        candidates = select_candidates_bidirectional(
            signals, open_pairs={"EUR/USD"}, min_score=-0.35,
            htf_closes=htf_closes, htf_fast=10, htf_slow=25,
        )
        pairs = [p for p, _ in candidates]
        assert "EUR/USD" not in pairs
        assert "GBP/USD" not in pairs   # same group

    def test_neutral_h4_skipped(self):
        # Not enough H4 data → neutral → skip
        htf_closes = {p: np.array([1.0, 1.001]) for p in
                      ["EUR/USD","GBP/USD","AUD/USD","NZD/USD","USD/JPY","USD/CHF","USD/CAD"]}
        signals = {p: make_sig(-0.80) for p in htf_closes}
        candidates = select_candidates_bidirectional(
            signals, open_pairs=set(), min_score=-0.35,
            htf_closes=htf_closes, htf_fast=10, htf_slow=25,
        )
        assert candidates == []


# ---------------------------------------------------------------------------
# position_side
# ---------------------------------------------------------------------------

class TestPositionSide:
    def test_sell_usd_non_inverted_is_long(self):
        assert position_side("EUR/USD", "sell_usd") == Side.LONG

    def test_sell_usd_inverted_is_short(self):
        assert position_side("USD/JPY", "sell_usd") == Side.SHORT

    def test_buy_usd_non_inverted_is_short(self):
        assert position_side("GBP/USD", "buy_usd") == Side.SHORT

    def test_buy_usd_inverted_is_long(self):
        assert position_side("USD/CHF", "buy_usd") == Side.LONG
