import datetime as dt

from otc_research.notifications.console_provider import ConsoleNotificationProvider
from otc_research.signals.formatting import format_signal_message
from otc_research.signals.latency import (
    is_expired_before_execution,
    latency_to_generate,
    latency_to_notify,
)
from otc_research.signals.service import create_signal, send_notification

GENERATED_AT = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)


def _signal(session, **overrides):
    defaults = dict(
        asset="EUR_USD", direction="CALL", timeframe="1h", expiry_seconds=3600,
        generated_at=GENERATED_AT, strategy_code="H4", model_version="v4",
        score=0.7, quality_tier="A", payout=0.85,
        historical_sample_size=150, historical_win_rate=0.58,
        win_rate_ci_low=0.52, win_rate_ci_high=0.64,
        conditions_met=["bb_pct_b_below_0"], reasons_rejected=["adx_below_25"],
    )
    defaults.update(overrides)
    return create_signal(session, **defaults)


# --- format_signal_message -------------------------------------------------


def test_format_signal_message_includes_key_fields(session):
    signal = _signal(session)
    message = format_signal_message(signal)
    assert signal.signal_ref in message
    assert "EUR_USD" in message
    assert "CALL" in message
    assert signal.confidence_label in message
    assert "bb_pct_b_below_0" in message
    assert "adx_below_25" in message


def test_format_signal_message_never_uses_imperative_language(session):
    signal = _signal(session)
    message = format_signal_message(signal).upper()
    for forbidden in ("EJECUTAR AHORA", "COMPRA YA", "GARANTIZADO", "SEGURO GANAR"):
        assert forbidden not in message


def test_format_signal_message_handles_missing_optional_stats_gracefully(session):
    signal = _signal(
        session, historical_sample_size=None, historical_win_rate=None,
        win_rate_ci_low=None, win_rate_ci_high=None,
    )
    message = format_signal_message(signal)  # must not raise
    assert "n/a" in message


# --- ConsoleNotificationProvider -------------------------------------------


def test_console_provider_sends_and_stamps_sent_at(session, capsys):
    signal = _signal(session)
    provider = ConsoleNotificationProvider()
    sent = send_notification(session, signal, provider, now=GENERATED_AT)
    assert sent is True
    captured = capsys.readouterr()
    assert signal.signal_ref in captured.out


# --- latency -----------------------------------------------------------


def test_latency_to_generate_none_without_data_received_at(session):
    signal = _signal(session, data_received_at=None)
    assert latency_to_generate(signal) is None


def test_latency_to_generate_computed_when_present(session):
    received = GENERATED_AT - dt.timedelta(seconds=2)
    signal = _signal(session, data_received_at=received)
    assert latency_to_generate(signal) == dt.timedelta(seconds=2)


def test_latency_to_notify_none_before_sending(session):
    signal = _signal(session)
    assert latency_to_notify(signal) is None


def test_latency_to_notify_computed_after_sending(session):
    signal = _signal(session)
    send_notification(session, signal, ConsoleNotificationProvider(), now=GENERATED_AT + dt.timedelta(seconds=3))
    assert latency_to_notify(signal) == dt.timedelta(seconds=3)


def test_is_expired_before_execution_reflects_status(session):
    signal = _signal(session, expiry_seconds=60, valid_window_seconds=60)
    assert is_expired_before_execution(signal) is False
    send_notification(
        session, signal, ConsoleNotificationProvider(), now=GENERATED_AT + dt.timedelta(minutes=10)
    )
    assert is_expired_before_execution(signal) is True
