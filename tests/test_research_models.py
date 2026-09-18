import numpy as np
import pandas as pd
import pytest

from otc_research.research.models import (
    MODEL_FAMILIES,
    fit_and_evaluate,
    run_model_comparison,
    train_val_frames,
)


def _planted_dataset(n=2000, seed=0):
    """Two informative features and one pure-noise feature; the target is
    1 whenever BOTH informative features are in their top quartile,
    otherwise a fair coin flip -- the same "plantable interaction" shape
    ``test_research_discovery.py`` uses, so a model that actually learns
    the relationship (rather than memorizing noise) should show a real,
    replicated edge on VALIDATION, not just on TRAIN.
    """
    rng = np.random.default_rng(seed)
    feature_a = rng.uniform(0, 1, n)
    feature_b = rng.uniform(0, 1, n)
    noise_feature = rng.uniform(0, 1, n)
    planted = (feature_a >= 0.75) & (feature_b >= 0.75)
    coin_flip = rng.integers(0, 2, n).astype(float)
    target = np.where(planted, 1.0, coin_flip)
    # Chronological order is arbitrary for this synthetic set, but
    # train_val_frames requires a stable row order to slice against --
    # an explicit RangeIndex documents that this is already "sorted".
    return pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "noise_feature": noise_feature,
            "target": target,
        }
    ).reset_index(drop=True)


def _planted_additive_dataset(n=2000, seed=0):
    """A linearly-separable-ish plant: target is 1 whenever
    feature_a + feature_b is high, otherwise a fair coin flip. Logistic
    regression models a linear log-odds boundary, so (unlike the AND-of-
    two-quartiles interaction in ``_planted_dataset``, which needs a
    split/interaction to detect) this is the shape it can actually learn
    -- used only for the logistic-regression-specific recovery test below.
    """
    rng = np.random.default_rng(seed)
    feature_a = rng.uniform(0, 1, n)
    feature_b = rng.uniform(0, 1, n)
    noise_feature = rng.uniform(0, 1, n)
    planted = (feature_a + feature_b) >= 1.5
    coin_flip = rng.integers(0, 2, n).astype(float)
    target = np.where(planted, 1.0, coin_flip)
    return pd.DataFrame(
        {
            "feature_a": feature_a,
            "feature_b": feature_b,
            "noise_feature": noise_feature,
            "target": target,
        }
    ).reset_index(drop=True)


FEATURE_COLS = ["feature_a", "feature_b", "noise_feature"]


def test_train_val_frames_splits_by_row_position_not_shuffled():
    df = pd.DataFrame({"x": range(100)})
    train_df, val_df = train_val_frames(df, train_fraction=0.6, validation_fraction=0.2)
    assert list(train_df["x"]) == list(range(0, 60))
    assert list(val_df["x"]) == list(range(60, 80))
    # the remaining 20% (test) is never returned by this function
    assert 80 not in set(train_df["x"]) | set(val_df["x"])


def test_fit_and_evaluate_rejects_single_class_target():
    df = _planted_dataset(n=200)
    df["target"] = 1.0  # only one outcome class present
    train_df, val_df = train_val_frames(df, 0.6, 0.2)
    with pytest.raises(ValueError, match="only one outcome class"):
        fit_and_evaluate(train_df, val_df, FEATURE_COLS, "target", family="logistic_regression")


def test_fit_and_evaluate_rejects_empty_train_after_dropna():
    df = _planted_dataset(n=50)
    df.loc[:, FEATURE_COLS] = np.nan
    train_df, val_df = train_val_frames(df, 0.6, 0.2)
    with pytest.raises(ValueError, match="no resolved TRAIN rows"):
        fit_and_evaluate(train_df, val_df, FEATURE_COLS, "target", family="logistic_regression")


def test_fit_and_evaluate_rejects_unknown_family():
    df = _planted_dataset(n=200)
    train_df, val_df = train_val_frames(df, 0.6, 0.2)
    with pytest.raises(ValueError, match="unknown model family"):
        fit_and_evaluate(train_df, val_df, FEATURE_COLS, "target", family="neural_network")


@pytest.mark.parametrize("family", ["random_forest", "gradient_boosting"])
def test_fit_and_evaluate_recovers_the_planted_interaction_on_validation(family):
    # The tree-based families can split on both features, so they can
    # actually detect an AND-of-two-quartiles interaction; a linear
    # logistic regression cannot (see the dedicated test below).
    df = _planted_dataset(n=4000, seed=1)
    train_df, val_df = train_val_frames(df, train_fraction=0.6, validation_fraction=0.2)

    result = fit_and_evaluate(
        train_df, val_df, FEATURE_COLS, "target", family=family, probability_threshold=0.7, seed=0
    )

    assert result.family == family
    assert result.train_sample_size == len(train_df)
    # a model that learned the planted interaction should trade a minority
    # of VALIDATION rows (roughly the top-quartile x top-quartile cell,
    # ~6.25% of rows) at high confidence, and win on almost all of them
    assert 0.0 < result.validation_coverage < 0.35
    assert result.validation_stat_at_threshold.n > 0
    assert result.validation_stat_at_threshold.win_rate > 0.75


def test_fit_and_evaluate_logistic_regression_recovers_a_linearly_separable_edge():
    # Logistic regression models a linear log-odds boundary, so it is
    # tested against a plant its model class can actually represent
    # (feature_a + feature_b high), not the AND-interaction above.
    df = _planted_additive_dataset(n=4000, seed=1)
    train_df, val_df = train_val_frames(df, train_fraction=0.6, validation_fraction=0.2)

    result = fit_and_evaluate(
        train_df,
        val_df,
        FEATURE_COLS,
        "target",
        family="logistic_regression",
        probability_threshold=0.7,
        seed=0,
    )

    assert result.train_sample_size == len(train_df)
    assert result.validation_stat_at_threshold.n > 0
    assert result.validation_coverage > 0.0
    assert result.validation_stat_at_threshold.win_rate > 0.75


def test_fit_and_evaluate_feature_importance_ranks_informative_features_above_noise():
    df = _planted_dataset(n=4000, seed=2)
    train_df, val_df = train_val_frames(df, 0.6, 0.2)
    result = fit_and_evaluate(
        train_df, val_df, FEATURE_COLS, "target", family="random_forest", seed=0
    )
    assert set(result.feature_importance.keys()) == set(FEATURE_COLS)
    noise_importance = result.feature_importance["noise_feature"]
    assert result.feature_importance["feature_a"] > noise_importance
    assert result.feature_importance["feature_b"] > noise_importance


def test_fit_and_evaluate_logistic_regression_importance_is_coefficients():
    df = _planted_dataset(n=1000, seed=3)
    train_df, val_df = train_val_frames(df, 0.6, 0.2)
    result = fit_and_evaluate(
        train_df, val_df, FEATURE_COLS, "target", family="logistic_regression", seed=0
    )
    # coefficients can be negative/positive -- just confirm every feature
    # got a real (non-fabricated) coefficient, one per feature
    assert set(result.feature_importance.keys()) == set(FEATURE_COLS)
    assert all(isinstance(v, float) for v in result.feature_importance.values())


def test_run_model_comparison_runs_every_family_in_fixed_order():
    df = _planted_dataset(n=1500, seed=4)
    train_df, val_df = train_val_frames(df, 0.6, 0.2)
    results = run_model_comparison(train_df, val_df, FEATURE_COLS, "target", seed=0)
    assert [r.family for r in results] == list(MODEL_FAMILIES)


def test_run_model_comparison_respects_families_override():
    df = _planted_dataset(n=1500, seed=5)
    train_df, val_df = train_val_frames(df, 0.6, 0.2)
    results = run_model_comparison(
        train_df, val_df, FEATURE_COLS, "target", families=("logistic_regression",), seed=0
    )
    assert [r.family for r in results] == ["logistic_regression"]


def test_fit_and_evaluate_drops_rows_with_unresolved_target():
    df = _planted_dataset(n=500, seed=6)
    df.loc[400:, "target"] = np.nan  # simulate dataset.py's trailing-horizon NaNs
    train_df, val_df = train_val_frames(df, 0.6, 0.2)
    result = fit_and_evaluate(
        train_df, val_df, FEATURE_COLS, "target", family="logistic_regression", seed=0
    )
    assert result.train_sample_size == train_df["target"].notna().sum()
