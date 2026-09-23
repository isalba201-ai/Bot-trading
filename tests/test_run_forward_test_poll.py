import datetime as dt

import pytest

from otc_research.db.models import Candle, Feature, Hypothesis, Signal
from otc_research.research.forward_test import ForwardTestCandidate
from scripts.run_forward_test_poll import _check_for_new_signals, _load_candles, _load_features, _resolve_pending

ASSET = "TEST_FX"
TIMEFRAME = "15m"
TF_SECONDS = 900


class _AlwaysCallAtHour6:
    code = None
    label = "test_always_call_at_hour6"
    expiry_seconds = 2 * TF_SECONDS  # h=2
    required_features = frozenset({"hour_utc"})

    def decide(self, features):
        return "CALL" if int(features["hour_utc"]) == 6 else None


@pytest.fixture()
def candidate():
    return ForwardTestCandidate(
        code="TEST_FWD",
        asset=ASSET,
        timeframe=TIMEFRAME,
        expiry_seconds=2 * TF_SECONDS,
        payout=0.85,
        backtest_status="synthetic test candidate",
        forward_test_start=dt.datetime(2026, 1, 1),
        strategy_factory=_AlwaysCallAtHour6,
    )


def _insert_candle(session, ts, close, *, hour_utc):
    session.add(Candle(
        asset=ASSET, timeframe=TIMEFRAME, timestamp=ts,
        open=close, high=close + 0.001, low=close - 0.001, close=close,
        source="synthetic:test", is_synthetic_test_data=True,
    ))
    session.add(Feature(
        asset=ASSET, timeframe=TIMEFRAME, timestamp=ts, feature_set_version="v4",
        name="hour_utc", value=float(hour_utc), computed_at=dt.datetime(2026, 1, 1),
    ))


def test_new_signal_created_when_strategy_fires_on_a_closed_candle(session, candidate):
    session.add(Hypothesis(code="TEST_FWD", name="t", description="t"))
    session.commit()

    signal_ts = dt.datetime(2026, 1, 2, 6, 0, 0)  # hour_utc == 6
    _insert_candle(session, signal_ts, 1.1000, hour_utc=6)
    session.commit()

    candles = _load_candles(session, ASSET, TIMEFRAME)
    features_by_ts = _load_features(session, ASSET, TIMEFRAME)
    created = _check_for_new_signals(session, candidate, candles, {c.timestamp: c for c in candles}, features_by_ts, TF_SECONDS)

    assert created == 1
    sig = session.query(Signal).filter_by(strategy_code="TEST_FWD").one()
    assert sig.direction == "CALL"
    assert sig.generated_at == signal_ts
    assert sig.mode == "paper"
    assert sig.result is None


def test_no_signal_when_strategy_does_not_fire(session, candidate):
    session.add(Hypothesis(code="TEST_FWD", name="t", description="t"))
    session.commit()
    _insert_candle(session, dt.datetime(2026, 1, 2, 7, 0, 0), 1.1000, hour_utc=7)
    session.commit()

    candles = _load_candles(session, ASSET, TIMEFRAME)
    features_by_ts = _load_features(session, ASSET, TIMEFRAME)
    created = _check_for_new_signals(session, candidate, candles, {c.timestamp: c for c in candles}, features_by_ts, TF_SECONDS)
    assert created == 0


def test_no_signal_for_candle_before_forward_test_start(session):
    candidate = ForwardTestCandidate(
        code="TEST_FWD", asset=ASSET, timeframe=TIMEFRAME, expiry_seconds=2 * TF_SECONDS,
        payout=0.85, backtest_status="synthetic",
        forward_test_start=dt.datetime(2026, 6, 1),  # after the candle below
        strategy_factory=_AlwaysCallAtHour6,
    )
    session.add(Hypothesis(code="TEST_FWD", name="t", description="t"))
    session.commit()
    _insert_candle(session, dt.datetime(2026, 1, 2, 6, 0, 0), 1.1000, hour_utc=6)
    session.commit()

    candles = _load_candles(session, ASSET, TIMEFRAME)
    features_by_ts = _load_features(session, ASSET, TIMEFRAME)
    created = _check_for_new_signals(session, candidate, candles, {c.timestamp: c for c in candles}, features_by_ts, TF_SECONDS)
    assert created == 0  # would fire (hour_utc==6) but predates forward_test_start


def test_resolve_pending_fills_entry_then_result(session, candidate):
    session.add(Hypothesis(code="TEST_FWD", name="t", description="t"))
    session.commit()

    # signal candle, then the entry candle (delay=1 -> +2 candles) and the
    # exit candle (expiry h=2 -> +2 more candles from entry).
    t0 = dt.datetime(2026, 1, 2, 6, 0, 0)
    _insert_candle(session, t0, 1.1000, hour_utc=6)
    session.commit()
    candles = _load_candles(session, ASSET, TIMEFRAME)
    features_by_ts = _load_features(session, ASSET, TIMEFRAME)
    _check_for_new_signals(session, candidate, candles, {c.timestamp: c for c in candles}, features_by_ts, TF_SECONDS)

    sig = session.query(Signal).filter_by(strategy_code="TEST_FWD").one()
    entry_ts = t0 + dt.timedelta(seconds=2 * TF_SECONDS)  # 1 (own close) + delay(1)
    exit_ts = t0 + dt.timedelta(seconds=4 * TF_SECONDS)  # entry + expiry_candles(2)

    # only the entry candle has arrived so far
    _insert_candle(session, entry_ts, 1.1010, hour_utc=(entry_ts.hour))
    session.commit()
    candles_by_ts = {c.timestamp: c for c in _load_candles(session, ASSET, TIMEFRAME)}
    resolved = _resolve_pending(session, candidate, candles_by_ts, TF_SECONDS, expiry_candles=2, delay_candles=1)
    session.refresh(sig)
    assert resolved == 0
    assert sig.entry_price == 1.1010
    assert sig.result is None

    # now the exit candle arrives too -- CALL, exit close > entry open -> WIN
    _insert_candle(session, exit_ts, 1.1050, hour_utc=(exit_ts.hour))
    session.commit()
    candles_by_ts = {c.timestamp: c for c in _load_candles(session, ASSET, TIMEFRAME)}
    resolved = _resolve_pending(session, candidate, candles_by_ts, TF_SECONDS, expiry_candles=2, delay_candles=1)
    session.refresh(sig)
    assert resolved == 1
    assert sig.result == "WIN"
    assert sig.expiry_price == 1.1050
    assert sig.pnl > 0


def test_resolve_pending_never_touches_an_already_resolved_signal(session, candidate):
    session.add(Hypothesis(code="TEST_FWD", name="t", description="t"))
    session.commit()
    t0 = dt.datetime(2026, 1, 2, 6, 0, 0)
    _insert_candle(session, t0, 1.1000, hour_utc=6)
    session.commit()
    candles = _load_candles(session, ASSET, TIMEFRAME)
    features_by_ts = _load_features(session, ASSET, TIMEFRAME)
    _check_for_new_signals(session, candidate, candles, {c.timestamp: c for c in candles}, features_by_ts, TF_SECONDS)

    sig = session.query(Signal).filter_by(strategy_code="TEST_FWD").one()
    sig.result = "WIN"
    sig.pnl = 1.0
    sig.entry_price = 1.1000
    session.commit()

    candles_by_ts = {c.timestamp: c for c in _load_candles(session, ASSET, TIMEFRAME)}
    resolved = _resolve_pending(session, candidate, candles_by_ts, TF_SECONDS, expiry_candles=2, delay_candles=1)
    assert resolved == 0  # already resolved, untouched
