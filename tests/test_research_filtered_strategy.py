from otc_research.research.filtered_strategy import FilteredStrategy, volatility_contraction_filter
from otc_research.strategies.h1_streak import H1StreakContinuation


def test_filtered_strategy_passes_through_when_predicate_false():
    base = H1StreakContinuation(min_streak=3, expiry_seconds=300)
    wrapped = FilteredStrategy(
        base, lambda features: False, filter_label="always_off"
    )
    features = {"same_color_streak": 5.0, "atr_expansion_ratio": 1.0}
    assert wrapped.decide(features) == base.decide(features) == "CALL"


def test_filtered_strategy_blocks_when_predicate_true():
    base = H1StreakContinuation(min_streak=3, expiry_seconds=300)
    wrapped = FilteredStrategy(
        base, lambda features: True, filter_label="always_on"
    )
    features = {"same_color_streak": 5.0, "atr_expansion_ratio": 1.0}
    assert base.decide(features) == "CALL"
    assert wrapped.decide(features) is None


def test_filtered_strategy_preserves_base_metadata():
    base = H1StreakContinuation(min_streak=3, expiry_seconds=180)
    wrapped = FilteredStrategy(base, lambda f: False, filter_label="vol_contraction")
    assert wrapped.expiry_seconds == 180
    assert wrapped.required_features >= base.required_features
    assert "atr_expansion_ratio" in wrapped.required_features
    assert wrapped.label == "H1_streak_continuation_min3__NOTRADE_vol_contraction"
    assert wrapped.code is None


def test_volatility_contraction_filter_blocks_below_threshold_only():
    predicate = volatility_contraction_filter(threshold=0.8)
    assert predicate({"atr_expansion_ratio": 0.5}) is True
    assert predicate({"atr_expansion_ratio": 0.8}) is True  # boundary: <=, blocked
    assert predicate({"atr_expansion_ratio": 1.5}) is False


def test_filtered_strategy_with_volatility_contraction_filter_end_to_end():
    base = H1StreakContinuation(min_streak=3, expiry_seconds=300)
    wrapped = FilteredStrategy(
        base, volatility_contraction_filter(), filter_label="vol_contraction"
    )
    contracted = {"same_color_streak": 5.0, "atr_expansion_ratio": 0.3}
    normal = {"same_color_streak": 5.0, "atr_expansion_ratio": 1.5}
    assert wrapped.decide(contracted) is None
    assert wrapped.decide(normal) == "CALL"
