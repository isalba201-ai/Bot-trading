import pandas as pd

from otc_research.research.regimes import (
    REGIME_LABELS,
    classify_regime,
)


def test_classify_regime_known_combinations():
    adx = pd.Series([30.0, 30.0, 30.0, 10.0, 10.0, 10.0])
    atr_ratio = pd.Series([1.5, 1.0, 0.5, 1.5, 1.0, 0.5])
    result = classify_regime(adx, atr_ratio)
    assert list(result) == [
        "strong_trend_expanding_vol",
        "strong_trend_normal_vol",
        "strong_trend_contracting_vol",
        "weak_trend_expanding_vol",
        "weak_trend_normal_vol",
        "weak_trend_contracting_vol",
    ]
    assert set(result) <= set(REGIME_LABELS)


def test_classify_regime_boundary_values():
    # ADX exactly at the trend threshold counts as strong (>=); vol ratio
    # exactly at the expansion/contraction thresholds count as expanding/
    # contracting (>=/<=) -- boundaries are inclusive, not ambiguous.
    adx = pd.Series([25.0, 24.999])
    atr_ratio = pd.Series([1.2, 0.8])
    result = classify_regime(adx, atr_ratio)
    assert result.iloc[0] == "strong_trend_expanding_vol"
    assert result.iloc[1] == "weak_trend_contracting_vol"


def test_classify_regime_propagates_nan():
    adx = pd.Series([30.0, float("nan")])
    atr_ratio = pd.Series([float("nan"), 1.0])
    result = classify_regime(adx, atr_ratio)
    assert result.isna().all()
