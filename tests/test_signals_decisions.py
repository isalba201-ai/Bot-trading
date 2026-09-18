import datetime as dt

import pytest

from otc_research.signals.decisions import (
    ARRIVED_LATE,
    DID_NOT_TAKE,
    TOOK_TRADE,
    record_decision,
)
from otc_research.signals.service import DECIDED, create_signal

GENERATED_AT = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)


def _signal(session, **overrides):
    defaults = dict(
        asset="EUR_USD", direction="CALL", timeframe="1h", expiry_seconds=3600,
        generated_at=GENERATED_AT, strategy_code="H4", model_version="v4",
        score=0.7, quality_tier="A", payout=0.85,
    )
    defaults.update(overrides)
    return create_signal(session, **defaults)


def test_record_decision_rejects_unknown_decision_value(session):
    signal = _signal(session)
    with pytest.raises(ValueError, match="decision must be one of"):
        record_decision(session, signal.id, "MAYBE", GENERATED_AT)


def test_record_decision_rejects_unknown_signal_id(session):
    with pytest.raises(ValueError, match="no Signal with id"):
        record_decision(session, 999999, TOOK_TRADE, GENERATED_AT)


def test_record_decision_persists_actuals_and_marks_signal_decided(session):
    signal = _signal(session)
    decided_at = GENERATED_AT + dt.timedelta(minutes=5)
    row = record_decision(
        session, signal.id, TOOK_TRADE, decided_at,
        entry_price_actual=1.0850, payout_observed=0.82,
        entry_time_actual=decided_at, expiry_used_seconds=3600,
    )
    assert row.decision == TOOK_TRADE
    assert row.entry_price_actual == 1.0850
    assert row.payout_observed == 0.82
    assert signal.status == DECIDED


def test_record_decision_reclassifies_late_took_trade_as_arrived_late(session):
    signal = _signal(session, valid_window_seconds=3600)  # valid_until = +1h
    late = signal.valid_until + dt.timedelta(minutes=1)
    row = record_decision(session, signal.id, TOOK_TRADE, late)
    assert row.decision == ARRIVED_LATE


def test_record_decision_never_reclassifies_an_explicit_did_not_take(session):
    signal = _signal(session, valid_window_seconds=3600)
    late = signal.valid_until + dt.timedelta(minutes=1)
    row = record_decision(session, signal.id, DID_NOT_TAKE, late)
    assert row.decision == DID_NOT_TAKE


def test_record_decision_took_trade_within_window_is_not_reclassified(session):
    signal = _signal(session, valid_window_seconds=3600)
    on_time = signal.valid_until - dt.timedelta(minutes=1)
    row = record_decision(session, signal.id, TOOK_TRADE, on_time)
    assert row.decision == TOOK_TRADE
