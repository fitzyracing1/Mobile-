"""
Technical signal generators for the Dutch USD-short strategy.

Each function operates on a numpy array of closing prices (or OHLC)
and returns a scalar signal value or a named tuple of indicator values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MACrossSignal:
    fast_ma: float
    slow_ma: float
    # +1 = bullish crossover (fast above slow), -1 = bearish, 0 = flat
    signal: int


@dataclass
class RSISignal:
    value: float   # 0-100
    # -1 = oversold (USD weak, sell USD), +1 = overbought, 0 = neutral
    signal: int


@dataclass
class BollingerSignal:
    upper: float
    middle: float
    lower: float
    pct_b: float   # position within the band, 0=lower, 1=upper
    # -1 = price near upper band (mean-reversion short), +1 = near lower
    signal: int


@dataclass
class CompositeSignal:
    """Aggregated signal across all indicators."""

    ma_cross: MACrossSignal
    rsi: RSISignal
    bollinger: BollingerSignal
    # Weighted vote: negative = sell USD, positive = buy USD, 0 = no trade
    score: float
    action: str  # "sell_usd" | "buy_usd" | "hold"


def _ema(closes: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    alpha = 2.0 / (period + 1)
    ema = np.zeros_like(closes, dtype=float)
    ema[0] = closes[0]
    for i in range(1, len(closes)):
        ema[i] = alpha * closes[i] + (1 - alpha) * ema[i - 1]
    return ema


def moving_average_cross(
    closes: np.ndarray,
    fast: int,
    slow: int,
) -> MACrossSignal:
    """MA crossover signal."""
    if len(closes) < slow:
        return MACrossSignal(fast_ma=float("nan"), slow_ma=float("nan"), signal=0)

    fast_val = float(np.mean(closes[-fast:]))
    slow_val = float(np.mean(closes[-slow:]))

    if fast_val > slow_val:
        sig = 1
    elif fast_val < slow_val:
        sig = -1
    else:
        sig = 0

    return MACrossSignal(fast_ma=fast_val, slow_ma=slow_val, signal=sig)


def rsi(closes: np.ndarray, period: int) -> RSISignal:
    """Wilder's RSI."""
    if len(closes) < period + 1:
        return RSISignal(value=50.0, signal=0)

    deltas = np.diff(closes[-(period + 1) :])
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    avg_gain = float(np.mean(gains))
    avg_loss = float(np.mean(losses))

    if avg_loss == 0:
        rsi_val = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi_val = 100.0 - (100.0 / (1.0 + rs))

    # Pair-direction signal: -1 = pair falling (low momentum), +1 = pair rising.
    # The composite_signal layer translates this into USD direction.
    if rsi_val < 40:
        sig = -1
    elif rsi_val > 60:
        sig = 1
    else:
        sig = 0

    return RSISignal(value=rsi_val, signal=sig)


def bollinger_bands(
    closes: np.ndarray,
    period: int,
    n_std: float,
) -> BollingerSignal:
    """Bollinger Bands with %B position."""
    if len(closes) < period:
        mid = float(closes[-1]) if len(closes) > 0 else 0.0
        return BollingerSignal(
            upper=mid, middle=mid, lower=mid, pct_b=0.5, signal=0
        )

    window = closes[-period:]
    middle = float(np.mean(window))
    std = float(np.std(window, ddof=1))

    upper = middle + n_std * std
    lower = middle - n_std * std

    pct_b = (closes[-1] - lower) / (upper - lower + 1e-12)

    # Price near upper band on a non-inverted USD pair → USD strong → bearish for thesis
    if pct_b > 0.8:
        sig = -1  # mean-reversion: expect price to fall → sell USD
    elif pct_b < 0.2:
        sig = 1   # price near lower band → bounce expected
    else:
        sig = 0

    return BollingerSignal(
        upper=upper, middle=middle, lower=lower, pct_b=pct_b, signal=sig
    )


def composite_signal(
    closes: np.ndarray,
    fast_ma: int,
    slow_ma: int,
    rsi_period: int,
    bb_period: int,
    bb_std: float,
    rsi_oversold: float = 40.0,
    rsi_overbought: float = 60.0,
    is_inverted: bool = False,
) -> CompositeSignal:
    """
    Combine MA cross, RSI, and Bollinger into a single directional score.

    For *inverted* pairs (USD is the base, e.g. USD/JPY) the signal
    polarity is flipped so that "sell USD" always means the same thing.
    """
    mac = moving_average_cross(closes, fast_ma, slow_ma)
    rsi_sig = rsi(closes, rsi_period)
    bb_sig = bollinger_bands(closes, bb_period, bb_std)

    # Override RSI thresholds from config if desired
    rsi_val = rsi_sig.value
    if rsi_val < rsi_oversold:
        rsi_signal_vote = -1
    elif rsi_val > rsi_overbought:
        rsi_signal_vote = 1
    else:
        rsi_signal_vote = 0

    # Translate pair-direction signals into USD-direction signals.
    # Non-inverted pairs (e.g. EUR/USD): rising price = USD weakening → negate.
    # Inverted pairs (e.g. USD/JPY):     rising price = USD strengthening → keep.
    # Final score convention: negative = USD bearish (sell_usd), positive = bullish.
    usd_sign = 1 if is_inverted else -1

    score = (
        0.40 * usd_sign * mac.signal
        + 0.35 * usd_sign * rsi_signal_vote
        + 0.25 * usd_sign * bb_sig.signal
    )

    if score < -0.1:
        action = "sell_usd"
    elif score > 0.1:
        action = "buy_usd"
    else:
        action = "hold"

    return CompositeSignal(
        ma_cross=mac,
        rsi=RSISignal(value=rsi_val, signal=rsi_signal_vote),
        bollinger=bb_sig,
        score=score,
        action=action,
    )
