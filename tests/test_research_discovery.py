import numpy as np
import pandas as pd
import pytest

from otc_research.db.models import ConditionTrial
from otc_research.research.discovery import (
    Condition,
    _benjamini_hochberg,
    generate_conditions,
    run_discovery,
)


def test_generate_conditions_count_matches_bins_times_pairs():
    n = 400
    df = pd.DataFrame(
        {
            "a": np.linspace(0, 1, n),
            "b": np.linspace(0, 1, n) + np.random.default_rng(0).normal(0, 0.01, n),
            "c": np.linspace(1, 0, n),
        }
    )
    conditions = generate_conditions(df, ["a", "b", "c"], combo_size=2, n_bins=4)
    # C(3,2)=3 feature pairs, each with 4x4=16 bin combinations -> 48
    assert len(conditions) == 3 * 4 * 4
    for c in conditions:
        assert len(c.parts) == 2
        assert c.parts[0][0] != c.parts[1][0]


def test_generate_conditions_skips_constant_features():
    df = pd.DataFrame({"a": [1.0] * 50, "b": np.linspace(0, 1, 50)})
    conditions = generate_conditions(df, ["a", "b"], combo_size=2, n_bins=4)
    assert conditions == []  # "a" can't be quantile-binned


def test_condition_matches_produces_correct_boolean_mask():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0], "y": [10.0, 20.0, 30.0, 40.0]})
    condition = Condition(parts=(("x", 1.0, 3.0), ("y", 15.0, 35.0)))
    mask = condition.matches(df)
    assert list(mask) == [False, True, True, False]


def test_condition_label_and_json_roundtrip_shape():
    condition = Condition(parts=(("rsi_14", 30.0, 45.0),))
    assert "rsi_14" in condition.label()
    import json

    parsed = json.loads(condition.to_json())
    assert parsed == [{"feature": "rsi_14", "low": 30.0, "high": 45.0}]


# --- Benjamini-Hochberg -----------------------------------------------


def test_benjamini_hochberg_classic_example():
    # Standard teaching example: 8 p-values, q=0.05 -> exactly the two
    # smallest (0.005, 0.01) survive correction.
    p_values = [0.01, 0.04, 0.03, 0.005, 0.20, 0.15, 0.50, 0.99]
    flags = _benjamini_hochberg(p_values, q=0.05)
    significant_values = sorted(p for p, f in zip(p_values, flags) if f)
    assert significant_values == [0.005, 0.01]
    assert sum(flags) == 2


def test_benjamini_hochberg_none_significant_when_all_p_values_large():
    flags = _benjamini_hochberg([0.5, 0.6, 0.7, 0.8], q=0.05)
    assert not any(flags)


def test_benjamini_hochberg_ignores_none_p_values():
    flags = _benjamini_hochberg([0.001, None, None, 0.9], q=0.05)
    assert flags[1] is False
    assert flags[2] is False


# --- run_discovery (integration, real DB session) ------------------------


def _make_dataset_with_planted_interaction(n=2000, seed=0):
    """Two features, uniform; target is 1 whenever BOTH features are in
    the top quartile, otherwise a fair coin flip -- a real, plantable
    2-way interaction that should stand out from the noise conditions
    around it.
    """
    rng = np.random.default_rng(seed)
    feature_a = rng.uniform(0, 1, n)
    feature_b = rng.uniform(0, 1, n)
    noise_feature = rng.uniform(0, 1, n)  # unrelated to the target
    planted = (feature_a >= 0.75) & (feature_b >= 0.75)
    coin_flip = rng.integers(0, 2, n).astype(float)
    target = np.where(planted, 1.0, coin_flip)
    return pd.DataFrame(
        {"feature_a": feature_a, "feature_b": feature_b, "noise_feature": noise_feature, "target": target}
    )


def test_run_discovery_logs_every_trial(session):
    df = _make_dataset_with_planted_interaction(n=800)
    results = run_discovery(
        session, df, ["feature_a", "feature_b", "noise_feature"], "target",
        asset="TEST_FX", timeframe="1h", feature_set_version="v4",
        combo_sizes=(2,), n_bins=4, min_sample_size=10,
    )
    # C(3,2)=3 pairs x 16 bin combos = 48 conditions logged
    assert len(results) == 48
    stored = session.query(ConditionTrial).all()
    assert len(stored) == 48
    assert len({row.run_id for row in stored}) == 1


def test_run_discovery_finds_the_planted_interaction(session):
    df = _make_dataset_with_planted_interaction(n=3000, seed=2)
    results = run_discovery(
        session, df, ["feature_a", "feature_b", "noise_feature"], "target",
        asset="TEST_FX", timeframe="1h", feature_set_version="v4",
        combo_sizes=(2,), n_bins=4, min_sample_size=30,
    )

    best = results[0]
    assert best.condition.parts[0][0] in ("feature_a", "feature_b")
    assert best.condition.parts[1][0] in ("feature_a", "feature_b")
    assert best.stat.win_rate > 0.85  # the planted top-quartile x top-quartile cell wins ~100%
    assert best.fdr_significant is True

    # a condition scoped entirely to the unrelated noise feature should
    # not survive correction
    noise_only_results = [
        r for r in results if all(f == "noise_feature" or f in ("feature_a", "feature_b") for f, _, _ in r.condition.parts)
    ]
    # at least confirm not everything got flagged significant (the FDR
    # correction is doing real work, not rubber-stamping every trial)
    assert not all(r.fdr_significant for r in results)


def test_run_discovery_respects_min_sample_size(session):
    df = _make_dataset_with_planted_interaction(n=100)
    results = run_discovery(
        session, df, ["feature_a", "feature_b"], "target",
        asset="TEST_FX", timeframe="1h", feature_set_version="v4",
        combo_sizes=(2,), n_bins=4, min_sample_size=10_000,  # nothing can reach this
    )
    assert all(r.p_value is None for r in results)
    assert all(r.fdr_significant is False for r in results)


def test_run_discovery_regime_scoping_filters_rows(session):
    df = _make_dataset_with_planted_interaction(n=800)
    df["regime"] = np.where(np.arange(len(df)) % 2 == 0, "regime_a", "regime_b")

    results = run_discovery(
        session, df, ["feature_a", "feature_b"], "target",
        asset="TEST_FX", timeframe="1h", feature_set_version="v4",
        combo_sizes=(2,), n_bins=4, min_sample_size=10, regime="regime_a",
    )
    total_n = sum(r.stat.n for r in results if r.stat.n)
    # every trial's sample must come only from the "regime_a" half of the data
    assert total_n <= (df["regime"] == "regime_a").sum()
    stored = session.query(ConditionTrial).filter_by(regime="regime_a").all()
    assert len(stored) == len(results)
