import datetime as dt

import pytest

from otc_research.db.models import Signal
from otc_research.notifications.base import NotificationProvider
from otc_research.signals.service import (
    DECIDED,
    EXPIRED,
    PENDING,
    create_signal,
    refresh_expired_signals,
    send_notification,
)

GENERATED_AT = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)


class _RecordingProvider(NotificationProvider):
    def __init__(self):
        self.notified: list[Signal] = []

    def notify(self, signal: Signal) -> None:
        self.notified.append(signal)


class _FailingProvider(NotificationProvider):
    def notify(self, signal: Signal) -> None:
        raise RuntimeError("delivery failed")


def _create(session, **overrides) -> Signal:
    defaults = dict(
        asset="EUR_USD",
        direction="CALL",
        timeframe="1h",
        expiry_seconds=3600,
        generated_at=GENERATED_AT,
        strategy_code="H4",
        model_version="v4",
        score=0.7,
        quality_tier="A",
        payout=0.85,
        historical_sample_size=200,
        historical_win_rate=0.60,
        win_rate_ci_low=0.55,
        win_rate_ci_high=0.65,
    )
    defaults.update(overrides)
    return create_signal(session, **defaults)


# --- create_signal -----------------------------------------------------


def test_create_signal_rejects_invalid_direction(session):
    with pytest.raises(ValueError):
        _create(session, direction="BUY")


def test_create_signal_computes_break_even_margin_and_confidence(session):
    signal = _create(session)
    assert signal.break_even_win_rate == pytest.approx(1 / 1.85)
    assert signal.margin_over_break_even == pytest.approx(0.60 - 1 / 1.85)
    assert signal.expectancy is not None
    assert signal.confidence_label in ("ALTA", "MEDIA", "BAJA")


def test_create_signal_assigns_a_unique_human_readable_ref(session):
    s1 = _create(session)
    s2 = _create(session)
    assert s1.signal_ref.startswith("SIGNAL-2026-")
    assert s1.signal_ref != s2.signal_ref


def test_create_signal_defaults_valid_window_to_expiry_seconds(session):
    signal = _create(session, expiry_seconds=1800)
    assert signal.valid_from == GENERATED_AT
    assert signal.valid_until == GENERATED_AT + dt.timedelta(seconds=1800)


def test_create_signal_respects_explicit_valid_window(session):
    signal = _create(session, valid_window_seconds=60)
    assert signal.valid_until == GENERATED_AT + dt.timedelta(seconds=60)


def test_create_signal_starts_pending(session):
    signal = _create(session)
    assert signal.status == PENDING


def test_create_signal_serializes_json_fields(session):
    signal = _create(
        session,
        features_snapshot={"rsi_14": 32.5},
        conditions_met=["rsi_below_35"],
        reasons_rejected=["adx_below_threshold"],
    )
    assert '"rsi_14"' in signal.features_snapshot
    assert "rsi_below_35" in signal.conditions_met
    assert "adx_below_threshold" in signal.reasons_rejected


def test_create_signal_payout_is_estimated_flag_defaults_true(session):
    signal = _create(session)
    assert signal.payout_is_estimated is True
    signal2 = _create(session, payout_is_estimated=False)
    assert signal2.payout_is_estimated is False


# --- send_notification --------------------------------------------------


def test_send_notification_delivers_and_stamps_sent_at(session):
    signal = _create(session)
    provider = _RecordingProvider()
    sent = send_notification(session, signal, provider, now=GENERATED_AT + dt.timedelta(minutes=5))
    assert sent is True
    assert provider.notified == [signal]
    assert signal.sent_at == GENERATED_AT + dt.timedelta(minutes=5)
    assert signal.status == PENDING


def test_send_notification_expires_instead_of_sending_past_valid_until(session):
    signal = _create(session, expiry_seconds=60, valid_window_seconds=60)
    provider = _RecordingProvider()
    sent = send_notification(session, signal, provider, now=GENERATED_AT + dt.timedelta(minutes=10))
    assert sent is False
    assert provider.notified == []
    assert signal.status == EXPIRED
    assert signal.sent_at is None


def test_send_notification_propagates_provider_failure_without_stamping_sent_at(session):
    signal = _create(session)
    with pytest.raises(RuntimeError):
        send_notification(session, signal, _FailingProvider(), now=GENERATED_AT)
    assert signal.sent_at is None


# --- refresh_expired_signals ---------------------------------------------


def test_refresh_expired_signals_marks_only_stale_pending_rows(session):
    stale = _create(session, expiry_seconds=60, valid_window_seconds=60)
    fresh = _create(session, expiry_seconds=3600, valid_window_seconds=3600)

    count = refresh_expired_signals(session, now=GENERATED_AT + dt.timedelta(minutes=10))

    assert count == 1
    assert stale.status == EXPIRED
    assert fresh.status == PENDING


def test_refresh_expired_signals_never_touches_a_decided_signal(session):
    signal = _create(session, expiry_seconds=60, valid_window_seconds=60)
    signal.status = DECIDED
    session.commit()

    count = refresh_expired_signals(session, now=GENERATED_AT + dt.timedelta(minutes=10))

    assert count == 0
    assert signal.status == DECIDED
