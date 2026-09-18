import datetime as dt

import numpy as np
import pytest

from otc_research.backtest.engine import BacktestRunResult
from otc_research.backtest.metrics import TradeStats
from otc_research.backtest.robustness import (
    RobustnessVerdict,
    SweepPoint,
    evaluate_robustness,
    run_parameter_sweep,
)
from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import Candle
from otc_research.features.pipeline import compute_and_store
from otc_research.strategies.h4_bollinger import H4BollingerMeanReversion


def _stats(sample_size: int, ci_low: float | None) -> TradeStats:
    return TradeStats(
        sample_size=sample_size,
        wins=0,
        losses=0,
        voided=0,
        win_rate=0.6,
        win_rate_ci_low=ci_low,
        win_rate_ci_high=0.9 if ci_low is not None else None,
        expectancy_pct=0.01,
    )


def _point(sample_size: int, ci_low: float | None, scenario: str = "optimistic") -> SweepPoint:
    result = BacktestRunResult(
        scenario=scenario, stats=_stats(sample_size, ci_low), trades=[], backtest_run_id=1
    )
    return SweepPoint(params={}, window=(None, None), results=[result])


# --- evaluate_robustness (pure classification logic) --------------------


def test_insufficient_data_when_no_point_has_enough_samples():
    points = [_point(5, 0.6), _point(10, 0.7)]
    verdict = evaluate_robustness(points, "optimistic", min_sample_size=30)
    assert verdict.classification == "insufficient_data"
    assert verdict.n_sufficiently_sampled == 0
    assert verdict.fraction_with_edge is None


def test_consistent_direction_when_most_points_clear_the_bar():
    points = [_point(100, 0.55) for _ in range(8)] + [_point(100, 0.45) for _ in range(2)]
    verdict = evaluate_robustness(points, "optimistic", min_sample_size=30, min_edge_fraction=0.7)
    assert verdict.classification == "consistent_direction"
    assert verdict.n_sufficiently_sampled == 10
    assert verdict.n_with_edge == 8
    assert verdict.fraction_with_edge == pytest.approx(0.8)


def test_fragile_when_only_a_lucky_point_or_two_clears_the_bar():
    points = [_point(100, 0.55)] + [_point(100, 0.45) for _ in range(9)]
    verdict = evaluate_robustness(points, "optimistic", min_sample_size=30, min_edge_fraction=0.7)
    assert verdict.classification == "fragile"
    assert verdict.fraction_with_edge == pytest.approx(0.1)


def test_underpowered_points_are_excluded_not_counted_against_it():
    points = [_point(100, 0.55), _point(100, 0.60), _point(5, None)]  # last one underpowered
    verdict = evaluate_robustness(points, "optimistic", min_sample_size=30, min_edge_fraction=0.7)
    assert verdict.n_points == 3
    assert verdict.n_sufficiently_sampled == 2
    assert verdict.classification == "consistent_direction"


def test_evaluate_robustness_only_looks_at_the_requested_scenario():
    points = [_point(100, 0.55, scenario="optimistic"), _point(100, 0.55, scenario="realistic")]
    verdict = evaluate_robustness(points, "realistic", min_sample_size=30)
    assert verdict.n_sufficiently_sampled == 1


# --- run_parameter_sweep (wiring through the real engine) ---------------


def _insert_synthetic_candles(session, n=300, asset="TEST_FX", timeframe="1h"):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rng = np.random.default_rng(5)
    close = 1.10 + np.cumsum(rng.normal(0, 0.002, size=n))
    for i in range(n):
        c = close[i]
        o = close[i - 1] if i > 0 else c
        h = max(o, c) + 0.001
        low = min(o, c) - 0.001
        session.add(
            Candle(
                asset=asset,
                timeframe=timeframe,
                timestamp=start + dt.timedelta(hours=i),
                open=o,
                high=h,
                low=low,
                close=c,
                source="synthetic:test",
                is_synthetic_test_data=True,
            )
        )
    session.commit()
    return start


def _backtest_config() -> BacktestConfig:
    return BacktestConfig(
        train_fraction=0.6,
        validation_fraction=0.2,
        realistic=ExecutionScenarioConfig(
            entry_delay_candles=1, signal_drop_probability=0.0, slippage_pct=0.01
        ),
        pessimistic=ExecutionScenarioConfig(
            entry_delay_candles=2, signal_drop_probability=0.0, slippage_pct=0.03
        ),
    )


def test_run_parameter_sweep_covers_every_grid_point(session):
    _insert_synthetic_candles(session, n=300)
    feature_report = compute_and_store(session, "TEST_FX", "1h")
    assert feature_report.rows_inserted > 0

    param_grid = [
        {"lower_pct_b": 0.0, "upper_pct_b": 1.0, "expiry_seconds": 3600},
        {"lower_pct_b": 0.1, "upper_pct_b": 0.9, "expiry_seconds": 3600},
        {"lower_pct_b": 0.2, "upper_pct_b": 0.8, "expiry_seconds": 3600},
    ]

    points = run_parameter_sweep(
        session,
        H4BollingerMeanReversion,
        param_grid,
        "TEST_FX",
        "1h",
        _backtest_config(),
        feature_set_version=feature_report.feature_set_version,
        split="train",
        rng_seed=1,
    )

    assert len(points) == 3
    for point, expected_params in zip(points, param_grid):
        assert point.params == expected_params
        assert {r.scenario for r in point.results} == {"optimistic", "realistic", "pessimistic"}


def test_run_parameter_sweep_respects_time_windows(session):
    start = _insert_synthetic_candles(session, n=300)
    feature_report = compute_and_store(session, "TEST_FX", "1h")

    full_window = [(None, None)]
    narrow_window = [(start, start + dt.timedelta(hours=60))]

    full_points = run_parameter_sweep(
        session, H4BollingerMeanReversion, [{"expiry_seconds": 3600}], "TEST_FX", "1h",
        _backtest_config(), feature_set_version=feature_report.feature_set_version,
        split="train", windows=full_window, rng_seed=1,
    )
    narrow_points = run_parameter_sweep(
        session, H4BollingerMeanReversion, [{"expiry_seconds": 3600}], "TEST_FX", "1h",
        _backtest_config(), feature_set_version=feature_report.feature_set_version,
        split="train", windows=narrow_window, rng_seed=1,
    )

    full_n = full_points[0].result_for("optimistic").stats.sample_size
    narrow_n = narrow_points[0].result_for("optimistic").stats.sample_size
    voided_full = full_points[0].result_for("optimistic").stats.voided
    voided_narrow = narrow_points[0].result_for("optimistic").stats.voided
    # Restricting to the first 60 (of 300) hours must shrink the eligible
    # candle universe -- proof the window filter actually reached the DB
    # query, not just decoration on the SweepPoint.
    assert (narrow_n + voided_narrow) < (full_n + voided_full)
