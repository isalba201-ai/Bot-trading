import datetime as dt

import pytest

from otc_research.backtest.engine import run_backtest
from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import BacktestRun, Candle, Feature


class _AlwaysCall:
    code = None
    label = "engine_test_strategy"
    expiry_seconds = 60
    required_features = frozenset({"dummy"})

    def decide(self, features):
        return "CALL"


def _insert_candles_and_features(
    session, n=60, asset="TEST_FX", timeframe="1m", feature_set_version="v1"
):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for i in range(n):
        price = 1.00 + i * 0.001
        ts = start + dt.timedelta(minutes=i)
        session.add(
            Candle(
                asset=asset,
                timeframe=timeframe,
                timestamp=ts,
                open=price,
                high=price + 0.0005,
                low=price - 0.0005,
                close=price,
                source="synthetic:test",
                is_synthetic_test_data=True,
            )
        )
        session.add(
            Feature(
                asset=asset,
                timeframe=timeframe,
                timestamp=ts,
                feature_set_version=feature_set_version,
                name="dummy",
                value=1.0,
            )
        )
    session.commit()


def _backtest_config() -> BacktestConfig:
    return BacktestConfig(
        train_fraction=0.6,
        validation_fraction=0.2,
        realistic=ExecutionScenarioConfig(
            entry_delay_candles=1, signal_drop_probability=0.02, slippage_pct=0.01
        ),
        pessimistic=ExecutionScenarioConfig(
            entry_delay_candles=2, signal_drop_probability=0.05, slippage_pct=0.03
        ),
    )


def test_run_backtest_persists_one_row_per_scenario(session):
    _insert_candles_and_features(session)

    results = run_backtest(
        session,
        _AlwaysCall(),
        "TEST_FX",
        "1m",
        _backtest_config(),
        feature_set_version="v1",
        split="train",
        rng_seed=42,
    )

    assert {r.scenario for r in results} == {"optimistic", "realistic", "pessimistic"}
    stored = session.query(BacktestRun).all()
    assert len(stored) == 3
    assert all(r.split == "train" for r in stored)
    assert all(r.rng_seed == 42 for r in stored)
    assert all(r.strategy_label == "engine_test_strategy" for r in stored)
    assert all(r.hypothesis_code is None for r in stored)


def test_run_backtest_is_deterministic_given_same_seed(session):
    _insert_candles_and_features(session)
    cfg = _backtest_config()

    first = run_backtest(
        session, _AlwaysCall(), "TEST_FX", "1m", cfg, feature_set_version="v1",
        split="train", rng_seed=7,
    )
    second = run_backtest(
        session, _AlwaysCall(), "TEST_FX", "1m", cfg, feature_set_version="v1",
        split="train", rng_seed=7,
    )

    for a, b in zip(first, second):
        assert a.stats == b.stats


def test_run_backtest_test_split_logs_a_warning(session, caplog):
    _insert_candles_and_features(session)

    with caplog.at_level("WARNING"):
        run_backtest(
            session, _AlwaysCall(), "TEST_FX", "1m", _backtest_config(),
            feature_set_version="v1", split="test", rng_seed=1,
        )

    assert any("TEST SPLIT ACCESSED" in record.message for record in caplog.records)


def test_run_backtest_rejects_unknown_split(session):
    _insert_candles_and_features(session)
    with pytest.raises(ValueError):
        run_backtest(
            session, _AlwaysCall(), "TEST_FX", "1m", _backtest_config(),
            feature_set_version="v1", split="bogus",
        )


def test_run_backtest_rejects_too_few_candles(session):
    _insert_candles_and_features(session, n=2)
    with pytest.raises(ValueError):
        run_backtest(
            session, _AlwaysCall(), "TEST_FX", "1m", _backtest_config(),
            feature_set_version="v1", split="train",
        )
