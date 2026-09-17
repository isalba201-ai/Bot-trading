"""Unit tests for the pure indicator functions. Small hand-built numeric
series, purely to check the math — not market data.
"""

import numpy as np
import pandas as pd

from otc_research.features import indicators


def _series(values) -> pd.Series:
    return pd.Series(list(values), dtype=float)


def test_rsi_is_100_on_pure_uptrend():
    close = _series(range(1, 30))  # strictly increasing, no losses at all
    result = indicators.rsi(close, period=14)
    assert result.iloc[-1] == 100.0


def test_rsi_is_0_on_pure_downtrend():
    close = _series(range(30, 1, -1))  # strictly decreasing, no gains at all
    result = indicators.rsi(close, period=14)
    assert result.iloc[-1] == 0.0


def test_rsi_bounded_between_0_and_100():
    rng = np.random.default_rng(1)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.001, size=50)))
    result = indicators.rsi(close, period=14).dropna()
    assert (result >= 0).all() and (result <= 100).all()


def test_ema_converges_to_constant_value():
    close = _series([2.0] * 20)
    result = indicators.ema(close, span=5)
    assert result.iloc[-1] == 2.0


def test_ema_slope_positive_on_uptrend_negative_on_downtrend():
    up = _series(range(1, 30))
    down = _series(range(30, 1, -1))
    assert indicators.ema_slope(up, span=5, lookback=3).iloc[-1] > 0
    assert indicators.ema_slope(down, span=5, lookback=3).iloc[-1] < 0


def test_roc_matches_hand_computed_value():
    close = _series([100.0] * 10 + [110.0])
    result = indicators.roc(close, period=10)
    assert result.iloc[-1] == 10.0  # (110-100)/100 * 100


def test_atr_is_never_negative():
    high = _series([1.05, 1.06, 1.04, 1.07, 1.08] * 6)
    low = _series([1.01, 1.02, 1.00, 1.03, 1.02] * 6)
    close = _series([1.03, 1.04, 1.02, 1.05, 1.06] * 6)
    result = indicators.atr(high, low, close, period=14).dropna()
    assert (result >= 0).all()


def test_bollinger_bands_ordering_when_volatile():
    rng = np.random.default_rng(2)
    close = pd.Series(1.10 + np.cumsum(rng.normal(0, 0.001, size=40)))
    mid, upper, lower, pct_b = indicators.bollinger_bands(close, period=20, num_std=2.0)
    valid = mid.notna()
    assert (upper[valid] >= mid[valid]).all()
    assert (mid[valid] >= lower[valid]).all()


def test_same_color_streak_counts_and_resets():
    open_ = _series([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    close = _series([1.01, 1.02, 0.99, 0.98, 0.97, 1.0])  # up, up, down, down, down, doji
    result = indicators.same_color_streak(open_, close)
    assert list(result) == [1.0, 2.0, -1.0, -2.0, -3.0, 0.0]


def test_wick_ratios_sum_to_one():
    open_ = _series([1.00])
    high = _series([1.10])
    low = _series([0.95])
    close = _series([1.05])
    upper, lower, body = indicators.wick_ratios(open_, high, low, close)
    total = upper.iloc[0] + lower.iloc[0] + body.iloc[0]
    assert np.isclose(total, 1.0)
    # upper wick = high - max(open, close) = 1.10 - 1.05 = 0.05 -> /0.15
    assert np.isclose(upper.iloc[0], 0.05 / 0.15)
    # lower wick = min(open, close) - low = 1.00 - 0.95 = 0.05 -> /0.15
    assert np.isclose(lower.iloc[0], 0.05 / 0.15)


def test_donchian_high_low_exclude_current_candle():
    high = _series([1.0, 1.1, 1.2, 1.3, 5.0])  # last bar is a huge spike
    low = _series([0.9, 0.8, 0.7, 0.6, 0.1])
    donchian_high = indicators.donchian_high(high, period=4)
    donchian_low = indicators.donchian_low(low, period=4)
    # the spike at the last bar must not appear in its own baseline
    assert donchian_high.iloc[-1] == 1.3
    assert donchian_low.iloc[-1] == 0.6


def test_hour_and_day_of_week():
    timestamps = pd.Series(
        pd.to_datetime(["2026-03-02T14:00:00Z", "2026-03-08T23:00:00Z"])  # Mon, Sun
    )
    assert list(indicators.hour_of_day_utc(timestamps)) == [14.0, 23.0]
    assert list(indicators.day_of_week(timestamps)) == [0.0, 6.0]


def test_trading_session_code_buckets():
    hours = _series([3, 8, 13, 18, 22])  # sydney/tokyo overlap start, tokyo, overlap, ny, sydney
    codes = indicators.trading_session_code(hours)
    assert list(codes) == [1.0, 1.0, 3.0, 4.0, 0.0]


def test_structure_bias_detects_uptrend_swing_pattern():
    # Two confirmed swing lows (index 3 then 9, each higher than the last)
    # and two confirmed swing highs (index 6 then 12, each higher than the
    # last) -- a textbook higher-high/higher-low uptrend structure.
    low = _series([1.00, 0.99, 0.97, 0.90, 0.95, 0.98, 1.00, 0.99, 0.97, 0.93, 0.98, 1.01, 1.03, 1.02, 1.00])
    high = _series([1.05, 1.06, 1.08, 1.10, 1.12, 1.14, 1.20, 1.15, 1.13, 1.16, 1.18, 1.19, 1.25, 1.20, 1.18])
    result = indicators.structure_bias(high, low, wing=2)
    # Once both a higher-low and a higher-high have been confirmed, bias
    # must reflect the uptrend.
    assert result.iloc[-1] == 1.0
