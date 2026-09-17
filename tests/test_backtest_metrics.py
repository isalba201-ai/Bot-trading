import math

import pytest

from otc_research.backtest.metrics import summarize_trades, wilson_confidence_interval


def test_wilson_ci_matches_known_reference_value():
    # Well-known reference point: 50/100 at 95% -> approximately (0.404, 0.596).
    low, high = wilson_confidence_interval(50, 100)
    assert math.isclose(low, 0.4038, abs_tol=0.001)
    assert math.isclose(high, 0.5962, abs_tol=0.001)


def test_wilson_ci_is_none_for_zero_sample():
    assert wilson_confidence_interval(0, 0) is None


def test_wilson_ci_bounds_are_within_0_and_1():
    for wins, n in [(0, 5), (5, 5), (1, 1000), (999, 1000)]:
        low, high = wilson_confidence_interval(wins, n)
        assert 0.0 <= low <= high <= 1.0


def test_wilson_ci_widens_as_sample_shrinks():
    # Same point estimate (50%), smaller sample -> wider interval.
    small_low, small_high = wilson_confidence_interval(5, 10)
    large_low, large_high = wilson_confidence_interval(500, 1000)
    assert (small_high - small_low) > (large_high - large_low)


def test_wilson_ci_rejects_invalid_wins():
    with pytest.raises(ValueError):
        wilson_confidence_interval(11, 10)
    with pytest.raises(ValueError):
        wilson_confidence_interval(-1, 10)


def test_summarize_trades_excludes_void_from_sample_and_expectancy():
    results = ["WIN", "LOSS", "VOID", "WIN", "VOID"]
    pnls = [1.0, -0.5, 0.0, 2.0, 0.0]

    stats = summarize_trades(results, pnls)

    assert stats.sample_size == 3  # 2 WIN + 1 LOSS, VOID excluded
    assert stats.wins == 2
    assert stats.losses == 1
    assert stats.voided == 2
    assert stats.win_rate == pytest.approx(2 / 3)
    assert stats.expectancy_pct == pytest.approx((1.0 - 0.5 + 2.0) / 3)
    assert stats.win_rate_ci_low is not None
    assert stats.win_rate_ci_high is not None


def test_summarize_trades_all_void_yields_no_fabricated_numbers():
    stats = summarize_trades(["VOID", "VOID"], [0.0, 0.0])
    assert stats.sample_size == 0
    assert stats.win_rate is None
    assert stats.win_rate_ci_low is None
    assert stats.win_rate_ci_high is None
    assert stats.expectancy_pct is None


def test_summarize_trades_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        summarize_trades(["WIN"], [1.0, 2.0])
