"""End-to-end integration test: synthetic candles -> real Phase 3 feature
computation -> Phase 4 backtest engine -> a real Phase 5 baseline
strategy. Not a claim that any hypothesis has an edge (the candles are
synthetic, clearly marked is_synthetic_test_data=True) — just proof the
phases actually wire together correctly end to end.
"""

import datetime as dt

import numpy as np

from otc_research.backtest.engine import run_backtest
from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import BacktestRun, Candle
from otc_research.features.pipeline import compute_and_store
from otc_research.strategies.h1_streak import H1StreakContinuation


def _insert_synthetic_candles(session, n=200, asset="TEST_FX", timeframe="1m"):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rng = np.random.default_rng(11)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, size=n))
    for i in range(n):
        c = close[i]
        o = close[i - 1] if i > 0 else c
        h = max(o, c) + 0.0005
        low = min(o, c) - 0.0005
        session.add(
            Candle(
                asset=asset,
                timeframe=timeframe,
                timestamp=start + dt.timedelta(minutes=i),
                open=o,
                high=h,
                low=low,
                close=c,
                source="synthetic:test",
                is_synthetic_test_data=True,
            )
        )
    session.commit()


def test_h1_runs_end_to_end_through_real_features_and_engine(session):
    _insert_synthetic_candles(session, n=200)
    feature_report = compute_and_store(session, "TEST_FX", "1m")
    assert feature_report.rows_inserted > 0

    backtest_config = BacktestConfig(
        train_fraction=0.6,
        validation_fraction=0.2,
        realistic=ExecutionScenarioConfig(
            entry_delay_candles=1, signal_drop_probability=0.02, slippage_pct=0.01
        ),
        pessimistic=ExecutionScenarioConfig(
            entry_delay_candles=2, signal_drop_probability=0.05, slippage_pct=0.03
        ),
    )

    strategy = H1StreakContinuation(min_streak=2, expiry_seconds=60)
    results = run_backtest(
        session,
        strategy,
        "TEST_FX",
        "1m",
        backtest_config,
        feature_set_version=feature_report.feature_set_version,
        split="train",
        rng_seed=1,
    )

    assert {r.scenario for r in results} == {"optimistic", "realistic", "pessimistic"}
    stored = session.query(BacktestRun).filter_by(hypothesis_code="H1").all()
    assert len(stored) == 3
    # win rate, if any trades fired, must always carry its statistical grounding
    for run in stored:
        if run.sample_size > 0:
            assert run.win_rate_ci_low is not None
            assert run.win_rate_ci_high is not None
