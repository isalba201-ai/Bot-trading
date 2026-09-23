import datetime as dt

import numpy as np
import pytest

from otc_research.backtest.engine import BacktestRunResult, compute_split_windows
from otc_research.backtest.metrics import TradeStats
from otc_research.backtest.walkforward import (
    FoldResult,
    WalkForwardFold,
    compute_walk_forward_folds,
    generate_folds,
    run_walk_forward,
    summarize_walk_forward,
)
from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import BacktestRun, Candle
from otc_research.features.pipeline import compute_and_store
from otc_research.strategies.h4_bollinger import H4BollingerMeanReversion

D = dt.timedelta


def _dt(day: int) -> dt.datetime:
    return dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + D(days=day)


# --- generate_folds -------------------------------------------------------


def test_generate_folds_non_overlapping_by_default():
    folds = generate_folds(_dt(0), _dt(100), train_span=D(days=30), test_span=D(days=10))
    assert len(folds) > 0
    for i, fold in enumerate(folds):
        assert fold.fold_index == i
        assert fold.train_window[1] == fold.test_window[0]  # test starts where train ends
        assert fold.test_window[1] - fold.test_window[0] == D(days=10)
    # non-overlapping: each fold's test window starts where the previous
    # fold's train window started + step (== test_span here)
    for a, b in zip(folds, folds[1:]):
        assert b.train_window[0] - a.train_window[0] == D(days=10)


def test_generate_folds_drops_trailing_partial_fold():
    # 100-day history, 30-day train + 10-day test = 40-day fold, step=10
    # (non-overlapping) -> fold i covers train=[10i, 10i+30), test=[10i+30,
    # 10i+40); the last whole fold that fits is i=6 (test ends exactly at
    # day 100); i=7 would need day 110 and must be dropped.
    folds = generate_folds(_dt(0), _dt(100), train_span=D(days=30), test_span=D(days=10))
    assert len(folds) == 7
    assert folds[-1].fold_index == 6
    assert folds[-1].test_window == (_dt(90), _dt(100))
    for fold in folds:
        assert fold.test_window[1] <= _dt(100)


def test_generate_folds_overlapping_with_smaller_step():
    non_overlapping = generate_folds(_dt(0), _dt(100), train_span=D(days=30), test_span=D(days=10))
    overlapping = generate_folds(
        _dt(0), _dt(100), train_span=D(days=30), test_span=D(days=10), step=D(days=5)
    )
    assert len(overlapping) > len(non_overlapping)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_span": D(0), "test_span": D(days=10)},
        {"train_span": D(days=10), "test_span": D(0)},
        {"train_span": D(days=10), "test_span": D(days=10), "step": D(0)},
    ],
)
def test_generate_folds_rejects_non_positive_spans(kwargs):
    with pytest.raises(ValueError):
        generate_folds(_dt(0), _dt(100), **kwargs)


def test_generate_folds_empty_when_history_too_short():
    folds = generate_folds(_dt(0), _dt(20), train_span=D(days=30), test_span=D(days=10))
    assert folds == []


# --- summarize_walk_forward (pure aggregation logic) ----------------------


def _stats(sample_size: int, win_rate: float | None, ci_low: float | None) -> TradeStats:
    return TradeStats(
        sample_size=sample_size, wins=0, losses=0, voided=0, win_rate=win_rate,
        win_rate_ci_low=ci_low, win_rate_ci_high=0.9 if ci_low is not None else None,
        expectancy_pct=0.01,
    )


def _fold_result(fold_index: int, sample_size: int, win_rate: float, ci_low: float | None,
                  scenario: str = "optimistic") -> FoldResult:
    fold = WalkForwardFold(fold_index, (_dt(0), _dt(1)), (_dt(1), _dt(2)))
    result = BacktestRunResult(
        scenario=scenario, stats=_stats(sample_size, win_rate, ci_low), trades=[], backtest_run_id=1
    )
    return FoldResult(fold=fold, results=[result])


def test_summarize_reports_mean_dispersion_and_worst_fold():
    folds = [
        _fold_result(0, 50, 0.60, 0.55),
        _fold_result(1, 50, 0.50, 0.40),
        _fold_result(2, 50, 0.55, 0.48),
    ]
    summary = summarize_walk_forward(folds, "optimistic", min_sample_size=20)

    assert summary.n_folds == 3
    assert summary.n_folds_sufficiently_sampled == 3
    assert summary.mean_win_rate == pytest.approx((0.60 + 0.50 + 0.55) / 3)
    assert summary.worst_fold_win_rate == 0.50
    assert summary.worst_fold_index == 1
    assert summary.n_folds_with_edge == 1  # only fold 0 has ci_low > 0.5
    assert summary.fraction_folds_with_edge == pytest.approx(1 / 3)


def test_summarize_excludes_underpowered_folds():
    folds = [_fold_result(0, 50, 0.60, 0.55), _fold_result(1, 5, 0.90, None)]
    summary = summarize_walk_forward(folds, "optimistic", min_sample_size=20)
    assert summary.n_folds == 2
    assert summary.n_folds_sufficiently_sampled == 1
    assert summary.mean_win_rate == 0.60


def test_summarize_insufficient_data_when_no_fold_qualifies():
    folds = [_fold_result(0, 5, 0.9, None), _fold_result(1, 3, 0.9, None)]
    summary = summarize_walk_forward(folds, "optimistic", min_sample_size=20)
    assert summary.n_folds_sufficiently_sampled == 0
    assert summary.mean_win_rate is None
    assert summary.worst_fold_win_rate is None
    assert summary.fraction_folds_with_edge is None


def test_summarize_only_looks_at_the_requested_scenario():
    folds = [_fold_result(0, 50, 0.6, 0.55, scenario="optimistic"),
             _fold_result(1, 50, 0.6, 0.55, scenario="realistic")]
    summary = summarize_walk_forward(folds, "realistic", min_sample_size=20)
    assert summary.n_folds_sufficiently_sampled == 1


# --- run_walk_forward (wiring through the real engine) --------------------


def _insert_synthetic_candles(session, n=400, asset="TEST_FX", timeframe="1h"):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rng = np.random.default_rng(9)
    close = 1.10 + np.cumsum(rng.normal(0, 0.002, size=n))
    for i in range(n):
        c = close[i]
        o = close[i - 1] if i > 0 else c
        h = max(o, c) + 0.001
        low = min(o, c) - 0.001
        session.add(
            Candle(
                asset=asset, timeframe=timeframe, timestamp=start + dt.timedelta(hours=i),
                open=o, high=h, low=low, close=c,
                source="synthetic:test", is_synthetic_test_data=True,
            )
        )
    session.commit()
    return start


def _backtest_config() -> BacktestConfig:
    return BacktestConfig(
        train_fraction=0.6, validation_fraction=0.2,
        realistic=ExecutionScenarioConfig(entry_delay_candles=1, signal_drop_probability=0.0, slippage_pct=0.01),
        pessimistic=ExecutionScenarioConfig(entry_delay_candles=2, signal_drop_probability=0.0, slippage_pct=0.03),
    )


def test_run_walk_forward_persists_one_row_per_fold_per_scenario(session):
    start = _insert_synthetic_candles(session, n=400)
    feature_report = compute_and_store(session, "TEST_FX", "1h")

    folds = generate_folds(
        start, start + dt.timedelta(hours=400),
        train_span=dt.timedelta(hours=100), test_span=dt.timedelta(hours=50),
    )
    assert len(folds) >= 2

    strategy = H4BollingerMeanReversion(expiry_seconds=3600)
    fold_results = run_walk_forward(
        session, strategy, "TEST_FX", "1h", _backtest_config(), folds,
        feature_set_version=feature_report.feature_set_version, rng_seed=1,
    )

    assert len(fold_results) == len(folds)
    for fr in fold_results:
        assert {r.scenario for r in fr.results} == {"optimistic", "realistic", "pessimistic"}

    stored = session.query(BacktestRun).filter_by(split="walk_forward").all()
    assert len(stored) == len(folds) * 3
    assert {r.fold_index for r in stored} == {f.fold_index for f in folds}


def test_compute_walk_forward_folds_never_reaches_test_start(session):
    # Reproduces the shape of the real bug this function fixes: two
    # disjoint blocks of candles for the same asset/timeframe (a "fresh"
    # research window plus an older block sitting further out in the same
    # table, exactly like EUR_USD/1m after the ML_1M5M experiment). A
    # calendar-time-proportion ESTIMATE of the TRAIN+VALIDATION boundary
    # (the old, buggy approach) drifts under non-uniform candle density;
    # the row-based boundary this function uses must not.
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for i in range(300):
        session.add(Candle(
            asset="TEST_FX", timeframe="1m", timestamp=start + dt.timedelta(minutes=i),
            open=1.1, high=1.1001, low=1.0999, close=1.1,
            source="synthetic:test", is_synthetic_test_data=True,
        ))
    gap_start = start + dt.timedelta(days=60)
    for i in range(100):
        session.add(Candle(
            asset="TEST_FX", timeframe="1m", timestamp=gap_start + dt.timedelta(minutes=i),
            open=1.2, high=1.2001, low=1.1999, close=1.2,
            source="synthetic:test", is_synthetic_test_data=True,
        ))
    session.commit()

    cfg = _backtest_config()
    window_end = start + dt.timedelta(minutes=300)  # excludes the far block entirely
    windows = compute_split_windows(session, "TEST_FX", "1m", cfg, start=start, end=window_end)
    folds = compute_walk_forward_folds(session, "TEST_FX", "1m", cfg, start=start, end=window_end, n_folds=5)

    assert len(folds) > 0
    for fold in folds:
        assert fold.test_window[1] <= windows.test_start
        assert fold.train_window[0] >= windows.train[0]


def test_run_walk_forward_test_windows_do_not_overlap_by_default(session):
    start = _insert_synthetic_candles(session, n=400)
    feature_report = compute_and_store(session, "TEST_FX", "1h")

    folds = generate_folds(
        start, start + dt.timedelta(hours=400),
        train_span=dt.timedelta(hours=100), test_span=dt.timedelta(hours=50),
    )
    for a, b in zip(folds, folds[1:]):
        assert a.test_window[1] == b.test_window[0]
