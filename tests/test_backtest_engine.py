import datetime as dt

import pytest

from otc_research.backtest.engine import compute_split_windows, run_backtest
from otc_research.backtest.splits import compute_temporal_split
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


# --- compute_split_windows (start/end restriction fix) ---------------------


def test_compute_split_windows_matches_compute_temporal_split_on_full_history(session):
    _insert_candles_and_features(session, n=60)
    cfg = _backtest_config()

    windows = compute_split_windows(session, "TEST_FX", "1m", cfg)

    # SQLite drops tzinfo on round-trip -- compare against naive timestamps
    # to match what compute_split_windows actually reads back.
    start = dt.datetime(2026, 1, 1)
    all_ts = [start + dt.timedelta(minutes=i) for i in range(60)]
    split = compute_temporal_split(60, cfg.train_fraction, cfg.validation_fraction)
    assert windows.train == (all_ts[split.train_slice][0], all_ts[split.train_slice][-1])
    assert windows.validation == (
        all_ts[split.validation_slice][0], all_ts[split.validation_slice][-1]
    )
    assert windows.test == (all_ts[split.test_slice][0], all_ts[split.test_slice][-1])
    assert windows.test_start == windows.test[0]
    assert windows.n_candles == 60


def test_compute_split_windows_restricts_to_the_given_start_end(session):
    # 100 candles total, but only ask for the first 40 -- the returned
    # split must be computed over those 40 alone, not all 100.
    _insert_candles_and_features(session, n=100)
    cfg = _backtest_config()
    start = dt.datetime(2026, 1, 1)
    window_end = start + dt.timedelta(minutes=40)

    windows = compute_split_windows(session, "TEST_FX", "1m", cfg, end=window_end)

    assert windows.n_candles == 40
    # every boundary must fall strictly before the requested end -- none
    # of the 60 excluded candles (minutes 40-99) can leak into any split.
    assert windows.train[1] < window_end
    assert windows.validation[1] < window_end
    assert windows.test[1] < window_end


def test_compute_split_windows_never_sees_candles_outside_the_window(session):
    # Mirrors the real bug this function was added to fix: two disjoint
    # blocks of candles for the same asset/timeframe (e.g. a fresh
    # research window plus an older, already-used block sitting further
    # out in the same table) -- restricting via start/end must make the
    # second block invisible to the split computation entirely, not just
    # unlikely to be selected.
    _insert_candles_and_features(session, n=50, timeframe="1m")
    far_future_start = dt.datetime(2030, 1, 1)
    for i in range(50):
        ts = far_future_start + dt.timedelta(minutes=i)
        session.add(
            Candle(
                asset="TEST_FX", timeframe="1m", timestamp=ts,
                open=2.0, high=2.001, low=1.999, close=2.0,
                source="synthetic:test", is_synthetic_test_data=True,
            )
        )
        session.add(
            Feature(
                asset="TEST_FX", timeframe="1m", timestamp=ts,
                feature_set_version="v1", name="dummy", value=1.0,
            )
        )
    session.commit()

    cfg = _backtest_config()
    near_start = dt.datetime(2026, 1, 1)
    near_end = near_start + dt.timedelta(minutes=50)

    windows = compute_split_windows(session, "TEST_FX", "1m", cfg, start=near_start, end=near_end)

    assert windows.n_candles == 50  # only the near block, not 100
    assert windows.test[1] < near_end
    assert windows.test[1] < far_future_start
