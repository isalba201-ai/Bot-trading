import datetime as dt

import numpy as np

from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import Candle, Hypothesis
from otc_research.features.pipeline import compute_and_store


def _insert_synthetic_candles(session, n=200, asset="TEST_FX", timeframe="1m"):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rng = np.random.default_rng(21)
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


def test_advance_hypothesis_status_moves_registered_to_tested_and_appends_notes(session):
    from scripts import run_baseline_backtests

    session.add(Hypothesis(code="H1", name="test", description="test"))
    session.commit()

    run_baseline_backtests._advance_hypothesis_status(session, "H1", [1, 2, 3])
    row = session.query(Hypothesis).filter_by(code="H1").one()
    assert row.status == "tested"
    assert "backtest_run_ids=[1, 2, 3]" in row.notes
    first_notes = row.notes

    run_baseline_backtests._advance_hypothesis_status(session, "H1", [4])
    row = session.query(Hypothesis).filter_by(code="H1").one()
    assert row.status == "tested"  # not reverted
    assert row.notes != first_notes
    assert first_notes in row.notes  # appended, not overwritten


def test_advance_hypothesis_status_handles_missing_row_without_error(session):
    from scripts import run_baseline_backtests

    run_baseline_backtests._advance_hypothesis_status(session, "H999", [1])  # must not raise


def test_main_runs_every_baseline_strategy_and_updates_status(session, monkeypatch):
    from scripts import run_baseline_backtests

    monkeypatch.setattr(
        run_baseline_backtests, "get_session_factory", lambda engine: (lambda: session)
    )
    monkeypatch.setattr(run_baseline_backtests, "init_db", lambda engine: None)
    monkeypatch.setattr(run_baseline_backtests, "get_engine", lambda url: None)

    class _StubConfig:
        database_url = "sqlite:///:memory:"
        backtest = BacktestConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            realistic=ExecutionScenarioConfig(
                entry_delay_candles=1, signal_drop_probability=0.02, slippage_pct=0.01
            ),
            pessimistic=ExecutionScenarioConfig(
                entry_delay_candles=2, signal_drop_probability=0.05, slippage_pct=0.03
            ),
        )

    monkeypatch.setattr(run_baseline_backtests, "load_config", lambda path: _StubConfig())
    monkeypatch.setattr(
        "sys.argv",
        ["run_baseline_backtests.py", "--pair", "TEST_FX", "--timeframe", "1m"],
    )

    for code in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H10"):
        session.add(Hypothesis(code=code, name=code, description=code))
    session.commit()

    _insert_synthetic_candles(session, n=200)
    feature_report = compute_and_store(session, "TEST_FX", "1m")
    assert feature_report.rows_inserted > 0

    run_baseline_backtests.main()

    for code in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H10"):
        row = session.query(Hypothesis).filter_by(code=code).one()
        assert row.status == "tested", f"{code} was not advanced"
        assert row.notes is not None


def test_main_respects_expiry_seconds_override_for_coarser_timeframes(session, monkeypatch):
    """A 1h timeframe needs an expiry that's a whole multiple of 3600s --
    the default 300s would make every strategy raise. --expiry-seconds
    exists so run_baseline_backtests.py works on any configured timeframe,
    not just 1m/5m.
    """
    from otc_research.db.models import BacktestRun
    from scripts import run_baseline_backtests

    monkeypatch.setattr(
        run_baseline_backtests, "get_session_factory", lambda engine: (lambda: session)
    )
    monkeypatch.setattr(run_baseline_backtests, "init_db", lambda engine: None)
    monkeypatch.setattr(run_baseline_backtests, "get_engine", lambda url: None)

    class _StubConfig:
        database_url = "sqlite:///:memory:"
        backtest = BacktestConfig(
            train_fraction=0.6,
            validation_fraction=0.2,
            realistic=ExecutionScenarioConfig(
                entry_delay_candles=1, signal_drop_probability=0.02, slippage_pct=0.01
            ),
            pessimistic=ExecutionScenarioConfig(
                entry_delay_candles=2, signal_drop_probability=0.05, slippage_pct=0.03
            ),
        )

    monkeypatch.setattr(run_baseline_backtests, "load_config", lambda path: _StubConfig())
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_baseline_backtests.py", "--pair", "TEST_FX", "--timeframe", "1h",
            "--expiry-seconds", "3600",
        ],
    )

    for code in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H10"):
        session.add(Hypothesis(code=code, name=code, description=code))
    session.commit()

    _insert_synthetic_candles(session, n=200, timeframe="1h")
    feature_report = compute_and_store(session, "TEST_FX", "1h")
    assert feature_report.rows_inserted > 0

    run_baseline_backtests.main()

    stored = session.query(BacktestRun).all()
    assert stored  # at least some strategies fired and produced a run
    assert all(r.expiry_seconds == 3600 for r in stored)
