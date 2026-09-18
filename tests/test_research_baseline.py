import numpy as np
import pandas as pd
import pytest

from otc_research.research.baseline import (
    bootstrap_mean_ci,
    conditional_by_bins,
    conditional_by_category,
    regime_conditional_bins,
    unconditional_baseline,
)


def test_unconditional_baseline_matches_simple_ratio():
    df = pd.DataFrame({"target": [1.0, 1.0, 0.0, 0.0, 1.0, np.nan]})  # nan excluded
    stat = unconditional_baseline(df, "target")
    assert stat.n == 5
    assert stat.wins == 3
    assert stat.win_rate == pytest.approx(0.6)
    assert stat.ci_low is not None and stat.ci_high is not None
    assert stat.ci_low <= stat.win_rate <= stat.ci_high


def test_unconditional_baseline_empty_returns_no_fabricated_stat():
    df = pd.DataFrame({"target": [np.nan, np.nan]})
    stat = unconditional_baseline(df, "target")
    assert stat.n == 0
    assert stat.win_rate is None
    assert stat.ci_low is None


def test_conditional_by_bins_detects_a_perfect_relationship():
    # target is deterministically 1 exactly when feature is in the top half
    n = 200
    feature = np.linspace(0, 1, n)
    target = (feature >= 0.5).astype(float)
    df = pd.DataFrame({"feature": feature, "target": target})

    bins = conditional_by_bins(df, "feature", "target", n_bins=4)
    assert len(bins) == 4
    # lowest bin should have win_rate 0, highest bin win_rate 1
    assert bins[0].stat.win_rate == pytest.approx(0.0)
    assert bins[-1].stat.win_rate == pytest.approx(1.0)
    # bins are sorted ascending by bin_low
    assert [b.bin_low for b in bins] == sorted(b.bin_low for b in bins)


def test_conditional_by_bins_handles_too_few_distinct_values():
    df = pd.DataFrame({"feature": [1.0] * 20, "target": [1.0, 0.0] * 10})
    bins = conditional_by_bins(df, "feature", "target", n_bins=10)
    assert bins == []  # can't quantile-bin a constant column into 10 bins


def test_conditional_by_category_groups_correctly():
    df = pd.DataFrame(
        {
            "regime": ["a", "a", "a", "b", "b"],
            "target": [1.0, 1.0, 0.0, 0.0, 0.0],
        }
    )
    stats = conditional_by_category(df, "regime", "target")
    by_category = {s.category: s.stat for s in stats}
    assert by_category["a"].n == 3
    assert by_category["a"].wins == 2
    assert by_category["b"].win_rate == pytest.approx(0.0)


def test_regime_conditional_bins_splits_by_regime_first():
    n = 100
    feature = np.tile(np.linspace(0, 1, n // 2), 2)
    # in regime "up", target follows feature; in regime "down", it's inverted
    regime = np.array(["up"] * (n // 2) + ["down"] * (n // 2))
    target = np.where(regime == "up", (feature >= 0.5).astype(float), (feature < 0.5).astype(float))
    df = pd.DataFrame({"feature": feature, "target": target, "regime": regime})

    result = regime_conditional_bins(df, "feature", "target", n_bins=4)
    assert set(result.keys()) == {"up", "down"}
    up_bins = result["up"]
    down_bins = result["down"]
    assert up_bins[-1].stat.win_rate == pytest.approx(1.0)  # high feature wins in "up"
    assert down_bins[-1].stat.win_rate == pytest.approx(0.0)  # high feature loses in "down"


def test_bootstrap_mean_ci_tight_around_a_constant():
    values = [0.02] * 50
    ci = bootstrap_mean_ci(values, n_resamples=200, seed=1)
    assert ci is not None
    low, high = ci
    assert low == pytest.approx(0.02, abs=1e-9)
    assert high == pytest.approx(0.02, abs=1e-9)


def test_bootstrap_mean_ci_widens_with_more_variance():
    tight = bootstrap_mean_ci([1.0, 1.0, 1.0, 1.0, 1.0] * 10, n_resamples=500, seed=1)
    wide = bootstrap_mean_ci([-5.0, 5.0, -3.0, 3.0, 1.0] * 10, n_resamples=500, seed=1)
    assert (tight[1] - tight[0]) < (wide[1] - wide[0])


def test_bootstrap_mean_ci_empty_returns_none():
    assert bootstrap_mean_ci([]) is None
