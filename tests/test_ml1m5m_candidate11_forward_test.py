"""Safety tests for the ML_1M5M candidate #11 forward test
(scripts/run_ml1m5m_candidate11_forward_test.py) -- run BEFORE trusting
any result from that script. Covers: frozen thresholds, no TEST/TRAIN
overlap with the forward block, the model is fit exactly once (never
retrained on forward data), and no filter is combined with another.
"""

import datetime as dt
import sys

from otc_research.backtest.engine import _load_candles, compute_split_windows
from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import Candle

sys.path.insert(0, "scripts")
import run_ml1m5m_candidate11_forward_test as fwd  # noqa: E402


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


# --- frozen thresholds -----------------------------------------------------


def test_thresholds_are_the_frozen_phase2_values():
    assert fwd.BASELINE_THRESHOLD == 0.50
    assert fwd.P_FILTER_THRESHOLD == 0.65
    assert fwd.CCI_FILTER_THRESHOLD == -60.0


def test_last_historical_cutoff_is_after_the_sept_block_and_after_test():
    # The cutoff must be strictly after both the historical TEST split end
    # (2026-08-04) AND the unrelated Step10/11 block (ends 2026-09-18
    # 21:01) -- one second after it, specifically.
    assert fwd.LAST_HISTORICAL_CUTOFF == dt.datetime(2026, 9, 18, 21, 1, 1)
    assert fwd.LAST_HISTORICAL_CUTOFF > dt.datetime(2026, 8, 4)
    assert fwd.LAST_HISTORICAL_CUTOFF > dt.datetime(2026, 9, 18, 21, 1, 0)


# --- forward block never overlaps TRAIN/VALIDATION/TEST --------------------


def test_forward_block_loader_never_returns_a_candle_at_or_before_the_cutoff(session):
    start = dt.datetime(2026, 1, 1)
    # 100 candles before the (test-local) cutoff, 50 after.
    cutoff = start + dt.timedelta(minutes=100)
    _insert_candles(session, 150, "TEST_FWD_FX", "1m", start)

    forward_candles = _load_candles(session, "TEST_FWD_FX", "1m", start=cutoff + dt.timedelta(seconds=1), end=None)

    assert len(forward_candles) == 49  # minutes 101..149 inclusive
    assert all(c.timestamp > cutoff for c in forward_candles)


def test_forward_block_is_disjoint_from_train_validation_test(session):
    # Build TRAIN+VALIDATION+TEST (2000 candles) then a forward block
    # starting well after TEST ends -- exactly the shape the real script
    # relies on -- and confirm zero overlap by construction.
    start = dt.datetime(2026, 1, 1)
    _insert_candles(session, 2000, "TEST_FWD_FX2", "1m", start)
    cfg = _backtest_config()
    windows = compute_split_windows(session, "TEST_FWD_FX2", "1m", cfg)

    forward_start = windows.test[1] + dt.timedelta(days=1)
    _insert_candles(session, 500, "TEST_FWD_FX2", "1m", forward_start)

    forward_candles = _load_candles(session, "TEST_FWD_FX2", "1m", start=windows.test[1] + dt.timedelta(seconds=1), end=None)

    assert len(forward_candles) == 500
    assert all(c.timestamp > windows.test[1] for c in forward_candles)
    assert all(c.timestamp < windows.train[0] or c.timestamp > windows.test[1] for c in forward_candles)


# --- model fit exactly once, only on TRAIN ---------------------------------


def test_freeze_model_calls_fit_and_evaluate_exactly_once(session, monkeypatch, tmp_path):
    calls = []
    real_fit_and_evaluate = fwd.fit_and_evaluate

    def counting_fit_and_evaluate(*args, **kwargs):
        calls.append((args, kwargs))
        return real_fit_and_evaluate(*args, **kwargs)

    monkeypatch.setattr(fwd, "fit_and_evaluate", counting_fit_and_evaluate)
    monkeypatch.setattr(fwd, "FROZEN_MODEL_PATH", str(tmp_path / "frozen.joblib"))

    # Build the real windowed EUR_USD/1m dataset -- requires the real DB
    # fixture to already have the historical window ingested; if it
    # doesn't (a bare synthetic session), skip rather than false-fail.
    from otc_research.db.models import Candle as _C
    has_data = session.query(_C).filter(_C.asset == "EUR_USD", _C.timeframe == "1m").first() is not None
    if not has_data:
        import pytest
        pytest.skip("requires the real ingested EUR_USD/1m historical window")

    from otc_research.config import load_config
    config = load_config(None)
    fwd._freeze_model(session, config)
    assert len(calls) == 1


# --- no combined filters ----------------------------------------------------


def test_cci_filter_wraps_the_baseline_alone_never_combined_with_another_filter():
    from sklearn.linear_model import LogisticRegression
    import numpy as np

    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 200)
    y = (x > 0).astype(int)
    model = LogisticRegression().fit(x.reshape(-1, 1), y)

    from otc_research.research.filtered_strategy import FilteredStrategy, cci_extreme_oversold_filter
    from otc_research.research.model_strategy import ModelStrategy

    baseline = ModelStrategy(model, ["x"], "CALL", 300, probability_threshold=0.5)
    cci_wrapped = FilteredStrategy(baseline, cci_extreme_oversold_filter(-60.0), filter_label="cci20_lt_neg60",
                                    filter_required_features=frozenset({"cci_20"}))

    assert cci_wrapped.base is baseline
    assert not isinstance(cci_wrapped.base, FilteredStrategy)  # not wrapping an already-filtered strategy
