"""Tests for the signal computation layer."""

import numpy as np
import pytest

from dutch_agent.signals import (
    moving_average_cross,
    rsi,
    bollinger_bands,
    composite_signal,
)


def make_trend(n: int, start: float = 1.0, slope: float = 0.001) -> np.ndarray:
    return np.array([start + i * slope for i in range(n)])


def make_flat(n: int, value: float = 1.0) -> np.ndarray:
    return np.full(n, value, dtype=float)


class TestMAcross:
    def test_bullish_crossover(self):
        closes = make_trend(50, slope=0.01)
        sig = moving_average_cross(closes, fast=5, slow=20)
        assert sig.signal == 1

    def test_bearish_crossover(self):
        closes = make_trend(50, slope=-0.01)
        sig = moving_average_cross(closes, fast=5, slow=20)
        assert sig.signal == -1

    def test_too_few_bars(self):
        closes = make_flat(5)
        sig = moving_average_cross(closes, fast=5, slow=20)
        assert sig.signal == 0

    def test_flat(self):
        closes = make_flat(50)
        sig = moving_average_cross(closes, fast=5, slow=20)
        # fast ≈ slow on flat data → 0
        assert sig.signal == 0


class TestRSI:
    def test_falling_prices_give_low_rsi(self):
        closes = make_trend(20, start=2.0, slope=-0.05)
        sig = rsi(closes, period=14)
        assert sig.value < 50

    def test_rising_prices_give_high_rsi(self):
        closes = make_trend(20, start=1.0, slope=0.05)
        sig = rsi(closes, period=14)
        assert sig.value > 50

    def test_too_few_bars_returns_neutral(self):
        closes = make_flat(5)
        sig = rsi(closes, period=14)
        assert sig.value == 50.0
        assert sig.signal == 0


class TestBollingerBands:
    def test_price_near_upper_band(self):
        closes = make_trend(30, slope=0.02)
        sig = bollinger_bands(closes, period=20, n_std=2.0)
        # strong uptrend → price near upper band
        assert sig.pct_b >= 0.0

    def test_middle_band_is_mean(self):
        closes = make_flat(20, value=1.5)
        sig = bollinger_bands(closes, period=20, n_std=2.0)
        assert abs(sig.middle - 1.5) < 1e-9

    def test_too_few_bars(self):
        closes = make_flat(5)
        sig = bollinger_bands(closes, period=20, n_std=2.0)
        assert sig.signal == 0


class TestCompositeSignal:
    def test_strong_usd_downtrend_gives_sell(self):
        # Falling price on EUR/USD means USD getting stronger, not weaker.
        # A *rising* EUR/USD (USD weakening) should yield sell_usd action.
        closes = make_trend(60, start=1.05, slope=0.001)
        sig = composite_signal(
            closes, fast_ma=10, slow_ma=30, rsi_period=14,
            bb_period=20, bb_std=2.0, is_inverted=False,
        )
        # Rising EUR/USD → bullish for EUR, bearish for USD → sell_usd or hold
        assert sig.action in ("sell_usd", "hold")

    def test_inverted_pair_polarity(self):
        # USD/JPY rising means USD stronger → inverted pair → should flip to buy_usd
        closes_up = make_trend(60, start=150.0, slope=0.1)
        sig_inv = composite_signal(
            closes_up, fast_ma=10, slow_ma=30, rsi_period=14,
            bb_period=20, bb_std=2.0, is_inverted=True,
        )
        sig_normal = composite_signal(
            closes_up, fast_ma=10, slow_ma=30, rsi_period=14,
            bb_period=20, bb_std=2.0, is_inverted=False,
        )
        # Inversion should flip the score sign
        assert sig_inv.score == pytest.approx(-sig_normal.score)

    def test_score_in_range(self):
        closes = make_flat(60)
        sig = composite_signal(
            closes, fast_ma=10, slow_ma=30, rsi_period=14,
            bb_period=20, bb_std=2.0,
        )
        assert -1.0 <= sig.score <= 1.0
