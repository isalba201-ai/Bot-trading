import datetime as dt

from otc_research.signals.decisions import DID_NOT_TAKE, TOOK_TRADE, record_decision
from otc_research.signals.performance import executable_performance, theoretical_performance
from otc_research.signals.service import create_signal

GENERATED_AT = dt.datetime(2026, 1, 1, 12, 0, tzinfo=dt.timezone.utc)


def _signal(session, *, result=None, pnl=None, **overrides):
    defaults = dict(
        asset="EUR_USD", direction="CALL", timeframe="1h", expiry_seconds=3600,
        generated_at=GENERATED_AT, strategy_code="H4", model_version="v4",
        score=0.7, quality_tier="A", payout=0.85,
    )
    defaults.update(overrides)
    signal = create_signal(session, **defaults)
    if result is not None:
        signal.result = result
        signal.pnl = pnl
        session.commit()
    return signal


def test_theoretical_performance_ignores_unresolved_signals(session):
    _signal(session)  # no result at all
    _signal(session, result="WIN", pnl=1.5)
    _signal(session, result="LOSS", pnl=-1.0)

    stats = theoretical_performance(session, asset="EUR_USD", timeframe="1h")
    assert stats.sample_size == 2
    assert stats.wins == 1
    assert stats.losses == 1


def test_theoretical_performance_filters_by_asset_and_timeframe(session):
    _signal(session, result="WIN", pnl=1.0, asset="EUR_USD")
    _signal(session, result="WIN", pnl=1.0, asset="GBP_USD")

    stats = theoretical_performance(session, asset="EUR_USD", timeframe="1h")
    assert stats.sample_size == 1


def test_executable_performance_only_counts_took_trade_decisions(session):
    took = _signal(session, result="WIN", pnl=1.0)
    record_decision(session, took.id, TOOK_TRADE, GENERATED_AT + dt.timedelta(minutes=1))

    declined = _signal(session, result="WIN", pnl=1.0)
    record_decision(session, declined.id, DID_NOT_TAKE, GENERATED_AT + dt.timedelta(minutes=1))

    never_decided = _signal(session, result="WIN", pnl=1.0)  # noqa: F841 -- deliberately no decision

    result = executable_performance(session, asset="EUR_USD", timeframe="1h")
    assert result.stats.sample_size == 1
    assert result.stats.wins == 1


def test_executable_performance_uses_decision_result_when_recorded(session):
    signal = _signal(session, result="WIN", pnl=1.0)  # theoretical: WIN
    decision = record_decision(session, signal.id, TOOK_TRADE, GENERATED_AT + dt.timedelta(minutes=1))
    # real, manually-recorded outcome differs from the theoretical read
    # (e.g. the actual fill missed the entry) -- must win over the fallback
    decision.result = "LOSS"
    decision.pnl = -1.0
    session.commit()

    result = executable_performance(session, asset="EUR_USD", timeframe="1h")
    assert result.stats.sample_size == 1
    assert result.stats.losses == 1
    assert result.n_fallback_to_theoretical == 0


def test_executable_performance_falls_back_to_theoretical_and_flags_it(session):
    signal = _signal(session, result="WIN", pnl=1.0)
    record_decision(session, signal.id, TOOK_TRADE, GENERATED_AT + dt.timedelta(minutes=1))
    # decision has no result/pnl of its own recorded

    result = executable_performance(session, asset="EUR_USD", timeframe="1h")
    assert result.stats.sample_size == 1
    assert result.stats.wins == 1
    assert result.n_fallback_to_theoretical == 1


def test_executable_performance_excludes_took_trade_with_no_resolved_outcome(session):
    signal = _signal(session)  # never resolved (result stays None)
    record_decision(session, signal.id, TOOK_TRADE, GENERATED_AT + dt.timedelta(minutes=1))

    result = executable_performance(session, asset="EUR_USD", timeframe="1h")
    assert result.stats.sample_size == 0
    assert result.n_excluded_not_taken == 1
