"""Tests for the combined feature set (Phase 3).

The most important test in this module is
``test_features_are_point_in_time_safe``: the whole design principle in
ARCHITECTURE.md/FEATURES.md is that a feature computed "as of" a given
candle must never have peeked at a later candle. This is checked directly
rather than trusted: recompute on a truncated prefix of the same series and
assert the last row is bit-for-bit identical to what the full computation
produced for that same timestamp. If any indicator ever gains a
look-ahead bug (a centered rolling window, a wrongly-signed shift, a
mis-lagged pivot confirmation), this test fails.
"""

import numpy as np
import pandas as pd
import pytest

from otc_research.features.engine import FEATURE_NAMES, compute_features


def _synthetic_ohlc(n: int, seed: int = 7) -> pd.DataFrame:
    """Small deterministic synthetic OHLC series, purely to exercise the
    math — not market data (see other tests' docstrings for the same
    caveat re: DATA.md's synthetic-data rules).
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=n, freq="1min", tz="UTC")
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, size=n))
    open_ = close - rng.normal(0, 0.0002, size=n)
    high = np.maximum(open_, close) + np.abs(rng.normal(0, 0.0003, size=n))
    low = np.minimum(open_, close) - np.abs(rng.normal(0, 0.0003, size=n))
    return pd.DataFrame(
        {"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": close}
    )


def test_feature_columns_match_declared_names():
    df = _synthetic_ohlc(60)
    features = compute_features(df)
    assert list(features.columns) == FEATURE_NAMES


def test_early_rows_are_nan_until_enough_history():
    df = _synthetic_ohlc(5)
    features = compute_features(df)
    row0 = features.iloc[0]
    # indicators with a real lookback (shortest is roc_10/ema_12, needing
    # more than 5 candles) must not be fabricated this early.
    for name in ("ema_12", "rsi_14", "atr_14", "bb_mid_20", "roc_10", "donchian_high_20"):
        assert pd.isna(row0[name]), f"{name} should be NaN with only 5 candles"
    # columns with no lookback (pure calendar/candle-shape facts) are
    # legitimately available from the very first candle.
    for name in ("hour_utc", "day_of_week", "trading_session_code", "same_color_streak"):
        assert pd.notna(row0[name]), f"{name} should not require history"


@pytest.mark.parametrize("cutoff", [40, 55, 59])
def test_features_are_point_in_time_safe(cutoff):
    full_df = _synthetic_ohlc(60)
    full_features = compute_features(full_df)

    truncated_df = full_df.iloc[:cutoff].reset_index(drop=True)
    truncated_features = compute_features(truncated_df)

    last_idx = cutoff - 1
    pd.testing.assert_series_equal(
        truncated_features.iloc[last_idx],
        full_features.iloc[last_idx],
        check_names=False,
    )


def test_wick_ratios_sum_to_one_across_a_real_series():
    df = _synthetic_ohlc(30)
    features = compute_features(df)
    total = (
        features["upper_wick_ratio"] + features["lower_wick_ratio"] + features["body_ratio"]
    ).dropna()
    assert np.allclose(total, 1.0)
