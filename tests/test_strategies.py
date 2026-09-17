"""Unit tests for the Phase 5 baseline strategies: given a hand-built
feature dict, does decide() return the expected direction (or None)?
These only check the entry-trigger logic in isolation — they say nothing
about whether any hypothesis has an edge on real data (see
strategies/__init__.py's module docstring).
"""

import pytest

from otc_research.strategies import BASELINE_STRATEGIES
from otc_research.strategies.h1_streak import H1StreakContinuation
from otc_research.strategies.h2_extreme_range import H2ExtremeRangeReversion
from otc_research.strategies.h3_momentum import H3MomentumContinuation
from otc_research.strategies.h4_bollinger import H4BollingerMeanReversion
from otc_research.strategies.h5_breakout import H5DonchianBreakout
from otc_research.strategies.h6_wick_rejection import H6WickRejection
from otc_research.strategies.h7_rsi_extreme import H7RsiExtremeConfirmed
from otc_research.strategies.h8_volatility_expansion import H8VolatilityExpansion
from otc_research.strategies.h9_session_bias import H9SessionBias
from otc_research.strategies.h10_combined import H10CombinedTrendStructureMomentum


def test_baseline_registry_covers_nine_zero_arg_strategies():
    assert set(BASELINE_STRATEGIES) == {
        "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H10",
    }
    for code, cls in BASELINE_STRATEGIES.items():
        strategy = cls()
        assert strategy.code == code
        assert isinstance(strategy.label, str) and strategy.label
        assert strategy.expiry_seconds > 0
        assert isinstance(strategy.required_features, frozenset) and strategy.required_features


# --- H1 ---------------------------------------------------------------


def test_h1_calls_on_bullish_streak_puts_on_bearish_none_otherwise():
    s = H1StreakContinuation(min_streak=3)
    assert s.decide({"same_color_streak": 3.0}) == "CALL"
    assert s.decide({"same_color_streak": 5.0}) == "CALL"
    assert s.decide({"same_color_streak": -3.0}) == "PUT"
    assert s.decide({"same_color_streak": 2.0}) is None
    assert s.decide({"same_color_streak": 0.0}) is None


# --- H2 ---------------------------------------------------------------


def test_h2_fades_extreme_candles_ignores_normal_ones():
    s = H2ExtremeRangeReversion(range_ratio_threshold=2.0)
    assert s.decide({"range_ratio_20": 2.5, "same_color_streak": 1.0}) == "PUT"
    assert s.decide({"range_ratio_20": 2.5, "same_color_streak": -1.0}) == "CALL"
    assert s.decide({"range_ratio_20": 1.2, "same_color_streak": 1.0}) is None
    assert s.decide({"range_ratio_20": 2.5, "same_color_streak": 0.0}) is None


# --- H3 ---------------------------------------------------------------


def test_h3_requires_both_roc_and_slope_to_agree():
    s = H3MomentumContinuation(roc_threshold=0.05, slope_threshold=0.0001)
    assert s.decide({"roc_10": 0.1, "ema_slope_12_3": 0.001}) == "CALL"
    assert s.decide({"roc_10": -0.1, "ema_slope_12_3": -0.001}) == "PUT"
    assert s.decide({"roc_10": 0.1, "ema_slope_12_3": -0.001}) is None  # disagreement
    assert s.decide({"roc_10": 0.01, "ema_slope_12_3": 0.001}) is None  # roc too small


# --- H4 ---------------------------------------------------------------


def test_h4_reverts_at_band_extremes():
    s = H4BollingerMeanReversion()
    assert s.decide({"bb_pct_b_20": 1.05}) == "PUT"
    assert s.decide({"bb_pct_b_20": -0.05}) == "CALL"
    assert s.decide({"bb_pct_b_20": 0.5}) is None


# --- H5 ---------------------------------------------------------------


def test_h5_breaks_out_using_the_raw_close():
    s = H5DonchianBreakout()
    assert s.decide({"close": 1.20, "donchian_high_20": 1.15, "donchian_low_20": 1.05}) == "CALL"
    assert s.decide({"close": 1.00, "donchian_high_20": 1.15, "donchian_low_20": 1.05}) == "PUT"
    assert s.decide({"close": 1.10, "donchian_high_20": 1.15, "donchian_low_20": 1.05}) is None


# --- H6 ---------------------------------------------------------------


def test_h6_fades_long_wicks():
    s = H6WickRejection(wick_ratio_threshold=0.6)
    assert s.decide({"upper_wick_ratio": 0.7, "lower_wick_ratio": 0.1}) == "PUT"
    assert s.decide({"upper_wick_ratio": 0.1, "lower_wick_ratio": 0.7}) == "CALL"
    assert s.decide({"upper_wick_ratio": 0.3, "lower_wick_ratio": 0.3}) is None


# --- H7 ---------------------------------------------------------------


def test_h7_requires_rsi_extreme_and_confirming_candle():
    s = H7RsiExtremeConfirmed(oversold=30.0, overbought=70.0)
    assert s.decide({"rsi_14": 20.0, "same_color_streak": 1.0}) == "CALL"
    assert s.decide({"rsi_14": 80.0, "same_color_streak": -1.0}) == "PUT"
    assert s.decide({"rsi_14": 20.0, "same_color_streak": -1.0}) is None  # no confirmation
    assert s.decide({"rsi_14": 50.0, "same_color_streak": 1.0}) is None  # not extreme


# --- H8 ---------------------------------------------------------------


def test_h8_needs_expansion_and_uses_slope_for_direction():
    s = H8VolatilityExpansion(expansion_threshold=1.5, slope_threshold=0.0001)
    assert s.decide({"atr_expansion_ratio": 2.0, "ema_slope_12_3": 0.001}) == "CALL"
    assert s.decide({"atr_expansion_ratio": 2.0, "ema_slope_12_3": -0.001}) == "PUT"
    assert s.decide({"atr_expansion_ratio": 1.0, "ema_slope_12_3": 0.001}) is None


# --- H9 ---------------------------------------------------------------


def test_h9_fires_only_at_its_configured_hour():
    s = H9SessionBias(hour_utc=14, direction="CALL")
    assert s.decide({"hour_utc": 14.0}) == "CALL"
    assert s.decide({"hour_utc": 15.0}) is None


def test_h9_rejects_invalid_construction():
    with pytest.raises(ValueError):
        H9SessionBias(hour_utc=24, direction="CALL")
    with pytest.raises(ValueError):
        H9SessionBias(hour_utc=10, direction="SIDEWAYS")


# --- H10 ----------------------------------------------------------------


def test_h10_requires_all_three_signals_to_agree():
    s = H10CombinedTrendStructureMomentum(slope_threshold=0.0001, roc_threshold=0.05)
    agree_up = {"ema_slope_12_3": 0.001, "structure_bias": 1.0, "roc_10": 0.1}
    agree_down = {"ema_slope_12_3": -0.001, "structure_bias": -1.0, "roc_10": -0.1}
    partial = {"ema_slope_12_3": 0.001, "structure_bias": 0.0, "roc_10": 0.1}

    assert s.decide(agree_up) == "CALL"
    assert s.decide(agree_down) == "PUT"
    assert s.decide(partial) is None
