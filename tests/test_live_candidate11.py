"""Safety tests for the ML_1M5M candidate #11 manual-live signal
generator (otc_research.live.*, scripts/run_live_signal_monitor.py) --
the user's explicit 14-point pre-flight checklist before ever connecting
this to the real market. Every numbered comment below maps directly to
one item of that checklist.
"""

from __future__ import annotations

import datetime as dt
import sys

import numpy as np
import pytest

from otc_research.backtest.execution import delay_only_scenario
from otc_research.backtest.simulator import SimCandle, simulate
from otc_research.db.models import Feature, LiveEvaluation, Signal
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.live import candidate as cand
from otc_research.live.evaluation import CALL, DATA_ERROR, NO_TRADE, evaluate_candle, resolve_theoretical_result
from otc_research.notifications.console_provider import ConsoleNotificationProvider
from otc_research.research.model_strategy import ModelStrategy
from otc_research.signals.decisions import DID_NOT_TAKE, TOOK_TRADE, record_decision
from otc_research.signals.service import create_signal

sys.path.insert(0, "scripts")
import run_live_signal_monitor as monitor  # noqa: E402


class _FixedProbModel:
    """A deterministic stand-in for the frozen GradientBoostingClassifier:
    always returns the same configured P(CALL) regardless of the feature
    row, and raises if ``fit`` is ever called (checklist item 5/13).
    """

    def __init__(self, p_call: float):
        self.p_call = p_call

    def predict_proba(self, X):
        return np.array([[1 - self.p_call, self.p_call]] * len(X))

    def fit(self, *args, **kwargs):
        raise AssertionError("fit() must never be called during live/replay evaluation")


def _candles(start: dt.datetime, n: int, step_seconds: int = 60) -> list[SimCandle]:
    return [
        SimCandle(
            timestamp=start + dt.timedelta(seconds=i * step_seconds),
            open=1.1000 + i * 0.0001, high=1.1005 + i * 0.0001,
            low=1.0995 + i * 0.0001, close=1.1002 + i * 0.0001,
        )
        for i in range(n)
    ]


def _features_for(candles, names=("x",), value=1.0):
    return {c.timestamp: {n: value for n in names} for c in candles}


def _strategy(p_call: float, feature_cols=("x",), threshold: float = 0.50) -> ModelStrategy:
    return ModelStrategy(_FixedProbModel(p_call), list(feature_cols), "CALL", 300, probability_threshold=threshold)


# --- 1: incomplete candle (missing features) never produces a signal -------


def test_missing_features_produce_data_error_not_a_signal():
    start = dt.datetime(2026, 9, 23, 10, 0)
    candles = _candles(start, 3)
    strategy = _strategy(0.99)  # would fire CALL if features were present
    evaluation = evaluate_candle(
        strategy, candles, 1, features_by_ts={},  # no features stored at all
        timeframe_seconds=60, entry_delay_candles=1, expiry_seconds=300,
    )
    assert evaluation.signal == DATA_ERROR
    assert evaluation.probability_call is None
    assert "missing features" in evaluation.notes


# --- 14: a missing (gapped) candle never produces a fabricated signal ------


def test_gap_before_candle_produces_data_error_not_a_fabricated_signal():
    start = dt.datetime(2026, 9, 23, 10, 0)
    candles = _candles(start, 3)
    # Simulate a dropped candle: candle[1] is really 3 minutes after candle[0].
    candles[1] = SimCandle(
        timestamp=candles[0].timestamp + dt.timedelta(minutes=3),
        open=candles[1].open, high=candles[1].high, low=candles[1].low, close=candles[1].close,
    )
    strategy = _strategy(0.99)
    features = _features_for(candles)
    evaluation = evaluate_candle(
        strategy, candles, 1, features,
        timeframe_seconds=60, entry_delay_candles=1, expiry_seconds=300,
    )
    assert evaluation.signal == DATA_ERROR
    assert evaluation.probability_call is None
    assert "gap detected" in evaluation.notes


# --- 6/7: threshold boundary -------------------------------------------


def test_probability_above_threshold_generates_call():
    start = dt.datetime(2026, 9, 23, 10, 0)
    candles = _candles(start, 3)
    features = _features_for(candles)
    strategy = _strategy(0.501)
    evaluation = evaluate_candle(
        strategy, candles, 1, features, timeframe_seconds=60, entry_delay_candles=1, expiry_seconds=300,
    )
    assert evaluation.signal == CALL
    assert evaluation.probability_call == pytest.approx(0.501)


@pytest.mark.parametrize("p_call", [0.50, 0.499, 0.0])
def test_probability_at_or_below_threshold_generates_no_trade(p_call):
    start = dt.datetime(2026, 9, 23, 10, 0)
    candles = _candles(start, 3)
    features = _features_for(candles)
    strategy = _strategy(p_call)
    evaluation = evaluate_candle(
        strategy, candles, 1, features, timeframe_seconds=60, entry_delay_candles=1, expiry_seconds=300,
    )
    assert evaluation.signal == NO_TRADE


# --- 8: delay=1 entry-time arithmetic ---------------------------------------


def test_delay1_entry_time_is_two_candles_after_signal_candle():
    start = dt.datetime(2026, 9, 23, 10, 0)
    candles = _candles(start, 3)
    features = _features_for(candles)
    strategy = _strategy(0.9)
    evaluation = evaluate_candle(
        strategy, candles, 0, features, timeframe_seconds=60, entry_delay_candles=1, expiry_seconds=300,
    )
    assert evaluation.signal == CALL
    # candle[0] open=10:00; delay=1 -> entry is 2 candles later = 10:02.
    assert evaluation.entry_time == start + dt.timedelta(minutes=2)
    assert evaluation.signal_time == start + dt.timedelta(minutes=1)


# --- 9: expiry arithmetic ---------------------------------------------------


def test_expiry_time_is_entry_time_plus_expiry_seconds():
    start = dt.datetime(2026, 9, 23, 10, 0)
    candles = _candles(start, 3)
    features = _features_for(candles)
    strategy = _strategy(0.9)
    evaluation = evaluate_candle(
        strategy, candles, 0, features, timeframe_seconds=60, entry_delay_candles=1, expiry_seconds=300,
    )
    assert evaluation.expiry_time == evaluation.entry_time + dt.timedelta(seconds=300)


# --- 10: theoretical result uses the exact backtest definition -------------


def test_theoretical_result_matches_backtest_simulator_definition():
    start = dt.datetime(2026, 9, 23, 10, 0)
    candles = _candles(start, 20)
    features = _features_for(candles)
    strategy = _strategy(0.9)

    # Reference: the actual backtest engine, delay_only(1), same candles/strategy.
    reference_trades = simulate(strategy, candles, features, 60, delay_only_scenario(1), rng=None)
    reference = {t.signal_time: t for t in reference_trades}

    for i in range(len(candles)):
        evaluation = evaluate_candle(
            strategy, candles, i, features, timeframe_seconds=60, entry_delay_candles=1, expiry_seconds=300,
        )
        if evaluation.signal != CALL:
            continue
        ref_trade = reference[evaluation.candle_timestamp]
        candles_by_ts = {c.timestamp: c for c in candles}
        entry_candle = candles_by_ts.get(evaluation.entry_time)
        exit_candle = candles_by_ts.get(evaluation.expiry_time)
        if entry_candle is None or exit_candle is None:
            assert ref_trade.result == "VOID"
            continue
        result, _pnl = resolve_theoretical_result(entry_candle.open, exit_candle.close, "CALL")
        assert result == ref_trade.result
        assert entry_candle.open == ref_trade.entry_price
        assert exit_candle.close == ref_trade.exit_price


# --- 4: the loaded model is exactly the frozen artifact ---------------------


def test_frozen_model_hash_matches_recorded_constant():
    import hashlib

    digest = hashlib.sha256(cand.FROZEN_MODEL_PATH.read_bytes()).hexdigest()
    assert digest == cand.EXPECTED_MODEL_SHA256
    # load_frozen_model_payload() re-verifies this itself and would raise
    # FrozenModelIntegrityError if it ever diverged.
    payload = cand.load_frozen_model_payload()
    assert payload["model"] is not None


# --- 5/13: no training happens during live/replay evaluation ---------------


def test_model_is_never_fit_during_evaluation():
    start = dt.datetime(2026, 9, 23, 10, 0)
    candles = _candles(start, 10)
    features = _features_for(candles)
    fake_model = _FixedProbModel(0.9)  # .fit() raises if ever called
    strategy = ModelStrategy(fake_model, ["x"], "CALL", 300, probability_threshold=0.5)

    for i in range(len(candles)):
        evaluate_candle(
            strategy, candles, i, features, timeframe_seconds=60, entry_delay_candles=1, expiry_seconds=300,
        )
    # No AssertionError raised above means fit() was never called.


def test_building_the_strategy_twice_gives_identical_predictions_never_retrained():
    strategy_a = cand.build_strategy()
    strategy_b = cand.build_strategy()
    row = [[0.0] * len(strategy_a.feature_cols)]
    proba_a = strategy_a.model.predict_proba(row)
    proba_b = strategy_b.model.predict_proba(row)
    assert (proba_a == proba_b).all()


# --- 11: a manual decision never mutates the theoretical result ------------


def test_manual_decision_never_mutates_theoretical_result(session):
    signal = create_signal(
        session, asset="EUR_USD", direction="CALL", timeframe="1m", expiry_seconds=300,
        generated_at=dt.datetime(2026, 9, 23, 10, 0), strategy_code=cand.CODE,
        model_version=cand.model_version(), score=0.7, quality_tier="B",
        payout=0.85, payout_is_estimated=True, mode="paper",
    )
    signal.result = "WIN"
    signal.pnl = 0.05
    session.commit()

    record_decision(session, signal.id, DID_NOT_TAKE, dt.datetime(2026, 9, 23, 10, 1))
    session.refresh(signal)
    assert signal.result == "WIN"
    assert signal.pnl == 0.05

    record_decision(session, signal.id, TOOK_TRADE, dt.datetime(2026, 9, 23, 10, 1, 30))
    session.refresh(signal)
    assert signal.result == "WIN"  # still untouched by the second decision too
    assert signal.pnl == 0.05


# --- 12: historical (TEST/forward-test) data is never (re-)evaluated -------


def test_candles_at_or_before_cutoff_are_never_evaluated(session, monkeypatch):
    monkeypatch.setattr(monitor, "cand", cand)
    before = cand.LIVE_TEST_START_AFTER - dt.timedelta(minutes=2)
    after = cand.LIVE_TEST_START_AFTER + dt.timedelta(minutes=2)
    from otc_research.db.models import Candle

    for ts in (before, before + dt.timedelta(minutes=1), after, after + dt.timedelta(minutes=1)):
        session.add(Candle(
            asset=cand.ASSET, timeframe=cand.TIMEFRAME, timestamp=ts,
            open=1.1, high=1.1005, low=1.0995, close=1.1002, source="synthetic:test",
            is_synthetic_test_data=True,
        ))
        session.add(Feature(
            asset=cand.ASSET, timeframe=cand.TIMEFRAME, timestamp=ts,
            feature_set_version=FEATURE_SET_VERSION, name="x", value=1.0,
        ))
    session.commit()

    strategy = ModelStrategy(_FixedProbModel(0.1), ["x"], "CALL", 300, probability_threshold=0.5)
    n_evaluated, _n_calls = monitor._evaluate_new_candles(session, strategy, ConsoleNotificationProvider())

    evaluated_ts = {
        e.candle_timestamp for e in session.query(LiveEvaluation).filter(
            LiveEvaluation.strategy_code == cand.CODE
        )
    }
    assert before not in evaluated_ts
    assert (before + dt.timedelta(minutes=1)) not in evaluated_ts
    assert n_evaluated == 2  # only the two candles strictly after the cutoff


# --- 2: at most one evaluation per candle (dedup / resume) ------------------


def test_one_evaluation_per_candle_dedup(session):
    from otc_research.db.models import Candle

    ts = cand.LIVE_TEST_START_AFTER + dt.timedelta(minutes=5)
    session.add(Candle(
        asset=cand.ASSET, timeframe=cand.TIMEFRAME, timestamp=ts,
        open=1.1, high=1.1005, low=1.0995, close=1.1002, source="synthetic:test",
        is_synthetic_test_data=True,
    ))
    session.add(Feature(
        asset=cand.ASSET, timeframe=cand.TIMEFRAME, timestamp=ts,
        feature_set_version=FEATURE_SET_VERSION, name="x", value=1.0,
    ))
    session.commit()

    strategy = ModelStrategy(_FixedProbModel(0.1), ["x"], "CALL", 300, probability_threshold=0.5)
    notifier = ConsoleNotificationProvider()
    monitor._evaluate_new_candles(session, strategy, notifier)
    monitor._evaluate_new_candles(session, strategy, notifier)  # run again, same candle

    rows = session.query(LiveEvaluation).filter(
        LiveEvaluation.strategy_code == cand.CODE, LiveEvaluation.candle_timestamp == ts
    ).all()
    assert len(rows) == 1


# --- 3: no duplicate CALL signals for the same candle -----------------------


def test_no_duplicate_call_signals_for_same_candle(session):
    from otc_research.db.models import Candle

    ts = cand.LIVE_TEST_START_AFTER + dt.timedelta(minutes=5)
    session.add(Candle(
        asset=cand.ASSET, timeframe=cand.TIMEFRAME, timestamp=ts,
        open=1.1, high=1.1005, low=1.0995, close=1.1002, source="synthetic:test",
        is_synthetic_test_data=True,
    ))
    session.add(Feature(
        asset=cand.ASSET, timeframe=cand.TIMEFRAME, timestamp=ts,
        feature_set_version=FEATURE_SET_VERSION, name="x", value=1.0,
    ))
    session.commit()

    strategy = ModelStrategy(_FixedProbModel(0.99), ["x"], "CALL", 300, probability_threshold=0.5)  # fires CALL
    notifier = ConsoleNotificationProvider()
    monitor._evaluate_new_candles(session, strategy, notifier)
    monitor._evaluate_new_candles(session, strategy, notifier)  # run again, same candle

    signals = session.query(Signal).filter(
        Signal.strategy_code == cand.CODE, Signal.generated_at == ts
    ).all()
    assert len(signals) == 1
