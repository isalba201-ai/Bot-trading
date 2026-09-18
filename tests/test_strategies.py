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
from otc_research.strategies.h11_macd_cross import H11MacdCrossContinuation
from otc_research.strategies.h12_cci_extreme import H12CciExtremeReversion
from otc_research.strategies.h13_rci_extreme import H13RciExtremeReversion
from otc_research.strategies.h14_engulfing import H14EngulfingReversal
from otc_research.strategies.h15_inside_bar_breakout import H15InsideBarBreakout
from otc_research.strategies.h16_macd_rsi_confirmed import H16MacdRsiConfirmed
from otc_research.strategies.h17_bollinger_rci_confirmed import H17BollingerRciConfirmed
from otc_research.strategies.h18_cci_engulfing_confirmed import H18CciEngulfingConfirmed
from otc_research.strategies.h19_trend_pullback import H19TrendPullback
from otc_research.strategies.h20_breakout_volatility_confirmed import (
    H20BreakoutVolatilityConfirmed,
)
from otc_research.strategies.h21_cci_rsi_macd_reversal import H21CciRsiMacdBearishReversal


def test_baseline_registry_covers_twenty_zero_arg_strategies():
    assert set(BASELINE_STRATEGIES) == {
        "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H10",
        "H11", "H12", "H13", "H14", "H15",
        "H16", "H17", "H18", "H19", "H20", "H21",
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


# --- H11 ----------------------------------------------------------------


def test_h11_follows_macd_cross_direction():
    s = H11MacdCrossContinuation()
    assert s.decide({"macd_cross_signal": 1.0}) == "CALL"
    assert s.decide({"macd_cross_signal": -1.0}) == "PUT"
    assert s.decide({"macd_cross_signal": 0.0}) is None


# --- H12 ----------------------------------------------------------------


def test_h12_fades_cci_extremes():
    s = H12CciExtremeReversion(threshold=100.0)
    assert s.decide({"cci_20": 120.0}) == "PUT"
    assert s.decide({"cci_20": -120.0}) == "CALL"
    assert s.decide({"cci_20": 50.0}) is None


# --- H13 ----------------------------------------------------------------


def test_h13_fades_rci_extremes():
    s = H13RciExtremeReversion(threshold=80.0)
    assert s.decide({"rci_9": 90.0}) == "PUT"
    assert s.decide({"rci_9": -90.0}) == "CALL"
    assert s.decide({"rci_9": 0.0}) is None


# --- H14 ----------------------------------------------------------------


def test_h14_follows_engulfing_direction():
    s = H14EngulfingReversal()
    assert s.decide({"engulfing_signal": 1.0}) == "CALL"
    assert s.decide({"engulfing_signal": -1.0}) == "PUT"
    assert s.decide({"engulfing_signal": 0.0}) is None


# --- H15 ----------------------------------------------------------------


def test_h15_follows_inside_bar_breakout_direction():
    s = H15InsideBarBreakout()
    assert s.decide({"inside_bar_breakout_signal": 1.0}) == "CALL"
    assert s.decide({"inside_bar_breakout_signal": -1.0}) == "PUT"
    assert s.decide({"inside_bar_breakout_signal": 0.0}) is None


# --- H16 ----------------------------------------------------------------


def test_h16_requires_both_macd_cross_and_matching_rsi_regime():
    s = H16MacdRsiConfirmed(rsi_midpoint=50.0, rsi_ceiling=80.0)
    assert s.decide({"macd_cross_signal": 1.0, "rsi_14": 60.0}) == "CALL"
    assert s.decide({"macd_cross_signal": -1.0, "rsi_14": 40.0}) == "PUT"
    # cross fires but RSI disagrees (still bearish) -> no signal
    assert s.decide({"macd_cross_signal": 1.0, "rsi_14": 30.0}) is None
    # RSI already too extreme (exhaustion zone) -> no signal
    assert s.decide({"macd_cross_signal": 1.0, "rsi_14": 90.0}) is None
    assert s.decide({"macd_cross_signal": 0.0, "rsi_14": 60.0}) is None


# --- H17 ----------------------------------------------------------------


def test_h17_requires_both_bollinger_and_rci_extremes_to_agree():
    s = H17BollingerRciConfirmed(rci_threshold=80.0)
    assert s.decide({"bb_pct_b_20": 1.05, "rci_9": 90.0}) == "PUT"
    assert s.decide({"bb_pct_b_20": -0.05, "rci_9": -90.0}) == "CALL"
    # only one of the two clears its threshold -> no signal
    assert s.decide({"bb_pct_b_20": 1.05, "rci_9": 20.0}) is None
    assert s.decide({"bb_pct_b_20": 0.5, "rci_9": 90.0}) is None


# --- H18 ----------------------------------------------------------------


def test_h18_requires_cci_extreme_and_matching_engulfing_pattern():
    s = H18CciEngulfingConfirmed(cci_threshold=100.0)
    assert s.decide({"cci_20": 120.0, "engulfing_signal": -1.0}) == "PUT"
    assert s.decide({"cci_20": -120.0, "engulfing_signal": 1.0}) == "CALL"
    # engulfing disagrees with the reversion direction -> no signal
    assert s.decide({"cci_20": 120.0, "engulfing_signal": 1.0}) is None
    assert s.decide({"cci_20": 50.0, "engulfing_signal": -1.0}) is None


# --- H19 ----------------------------------------------------------------


def test_h19_requires_trend_and_shallow_not_extreme_pullback():
    s = H19TrendPullback(slope_threshold=0.0001, pullback_low=35.0, pullback_high=50.0)
    assert s.decide({"ema_slope_12_3": 0.001, "rsi_14": 40.0}) == "CALL"
    assert s.decide({"ema_slope_12_3": -0.001, "rsi_14": 60.0}) == "PUT"
    # trend present but RSI dip too deep (extreme, not shallow) -> no signal
    assert s.decide({"ema_slope_12_3": 0.001, "rsi_14": 20.0}) is None
    # RSI in range but no trend -> no signal
    assert s.decide({"ema_slope_12_3": 0.0, "rsi_14": 40.0}) is None


# --- H20 ----------------------------------------------------------------


def test_h20_requires_breakout_and_volatility_expansion():
    s = H20BreakoutVolatilityConfirmed(expansion_threshold=1.2)
    base = {"donchian_high_20": 1.15, "donchian_low_20": 1.05}
    assert s.decide({**base, "close": 1.20, "atr_expansion_ratio": 1.5}) == "CALL"
    assert s.decide({**base, "close": 1.00, "atr_expansion_ratio": 1.5}) == "PUT"
    # breakout happens but volatility isn't expanding -> no signal
    assert s.decide({**base, "close": 1.20, "atr_expansion_ratio": 1.0}) is None
    # volatility expanding but no breakout -> no signal
    assert s.decide({**base, "close": 1.10, "atr_expansion_ratio": 1.5}) is None


# --- H21 ------------------------------------------------------------------


def test_h21_requires_cci_and_rsi_overbought_plus_bearish_macd_and_red_candle():
    s = H21CciRsiMacdBearishReversal(cci_threshold=100.0, rsi_threshold=70.0)
    all_bearish = {
        "cci_20": 120.0, "rsi_14": 75.0, "macd_histogram": -0.001, "open": 1.1010, "close": 1.1000,
    }
    assert s.decide(all_bearish) == "PUT"

    # CCI not yet overbought -> no signal
    assert s.decide({**all_bearish, "cci_20": 50.0}) is None
    # RSI not yet overbought -> no signal
    assert s.decide({**all_bearish, "rsi_14": 60.0}) is None
    # MACD histogram still positive ("green", not confirming) -> no signal
    assert s.decide({**all_bearish, "macd_histogram": 0.001}) is None
    # candle itself is bullish (close >= open), not "red" -> no signal
    assert s.decide({**all_bearish, "open": 1.1000, "close": 1.1010}) is None
    # PUT-only by design -- no oversold/bullish mirror case exists
    assert s.decide({
        "cci_20": -120.0, "rsi_14": 25.0, "macd_histogram": 0.001, "open": 1.1000, "close": 1.1010,
    }) is None


def test_h21_rejects_invalid_rsi_threshold():
    with pytest.raises(ValueError, match="rsi_threshold"):
        H21CciRsiMacdBearishReversal(rsi_threshold=40.0)
