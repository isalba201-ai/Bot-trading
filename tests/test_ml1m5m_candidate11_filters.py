"""Regression tests for the ML_1M5M candidate #11 filter-hypothesis phase
(scripts/rerun_ml1m5m_candidate11_filters.py): the no-TEST-access
guarantee specifically, since that script's whole point is to explore
filters without ever touching the reserved TEST split.
"""

import datetime as dt
import sys

import pytest

from otc_research.backtest.engine import compute_split_windows
from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import Candle

sys.path.insert(0, "scripts")
from rerun_ml1m5m_candidate11_filters import _load_pretest_candles  # noqa: E402


def _backtest_config() -> BacktestConfig:
    return BacktestConfig(
        train_fraction=0.6, validation_fraction=0.2,
        realistic=ExecutionScenarioConfig(entry_delay_candles=1, signal_drop_probability=0.0, slippage_pct=0.01),
        pessimistic=ExecutionScenarioConfig(entry_delay_candles=2, signal_drop_probability=0.0, slippage_pct=0.03),
    )


def _insert_candles(session, n, asset, timeframe, start):
    for i in range(n):
        session.add(Candle(
            asset=asset, timeframe=timeframe, timestamp=start + dt.timedelta(minutes=i),
            open=1.1, high=1.1001, low=1.0999, close=1.1,
            source="synthetic:test", is_synthetic_test_data=True,
        ))
    session.commit()


def test_load_pretest_candles_never_returns_a_test_candle(session, monkeypatch):
    import rerun_ml1m5m_candidate11_filters as filters_mod

    monkeypatch.setattr(filters_mod.exp, "ASSET", "TEST_FX", raising=False)
    monkeypatch.setattr(filters_mod.exp, "TIMEFRAME", "1m", raising=False)

    start = dt.datetime(2026, 1, 1)
    _insert_candles(session, 1000, "TEST_FX", "1m", start)
    cfg = _backtest_config()
    windows = compute_split_windows(session, "TEST_FX", "1m", cfg)

    # Ask for everything, including well past TEST -- the function must
    # clamp to windows.test_start regardless of what end is requested.
    far_future_end = start + dt.timedelta(days=365)
    candles = _load_pretest_candles(session, windows, start=start, end=far_future_end)

    assert len(candles) > 0
    assert all(c.timestamp < windows.test_start for c in candles)
    assert candles[-1].timestamp < windows.test[0]


def test_load_pretest_candles_asserts_if_the_underlying_loader_ever_leaked_a_test_candle(session, monkeypatch):
    # Proves the safety assertion inside _load_pretest_candles is live
    # code (fires when it should), not dead defensive dressing: fakes the
    # underlying candle loader to return a candle strictly at/after
    # windows.test_start despite being asked for an earlier window --
    # exactly the class of bug the assertion exists to catch if
    # _load_candles' own start/end filtering were ever broken.
    import rerun_ml1m5m_candidate11_filters as filters_mod

    monkeypatch.setattr(filters_mod.exp, "ASSET", "TEST_FX3", raising=False)
    monkeypatch.setattr(filters_mod.exp, "TIMEFRAME", "1m", raising=False)

    start = dt.datetime(2026, 1, 1)
    _insert_candles(session, 1000, "TEST_FX3", "1m", start)
    cfg = _backtest_config()
    windows = compute_split_windows(session, "TEST_FX3", "1m", cfg)

    class _LeakedCandle:
        timestamp = windows.test_start  # exactly at the forbidden boundary

    monkeypatch.setattr(filters_mod, "_load_candles", lambda *a, **k: [_LeakedCandle()])

    with pytest.raises(AssertionError, match="TEST candle leaked"):
        _load_pretest_candles(session, windows, start=start, end=windows.validation[1])
