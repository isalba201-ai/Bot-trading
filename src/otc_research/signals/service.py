"""Builds and manages the lifecycle of a ``Signal`` row (approved plan
points 8-9): a strategy/model produces a decision, this module turns it
into a persisted, statistically-grounded, human-readable proposal and
hands it to a ``NotificationProvider`` -- it never decides FOR a person,
and never places a trade. ``ConditionEngine``/``research.models`` code
only ever calls ``create_signal`` and hands the result to ``notify``;
nothing upstream of this module needs to know which notification channel
exists, per the approved plan's explicit requirement that swapping
channels later never touches strategy/research code.
"""

from __future__ import annotations

import datetime as dt
import json
from typing import Mapping, Sequence

from sqlalchemy.orm import Session

from otc_research.backtest.metrics import break_even_win_rate, payout_adjusted_expectancy
from otc_research.db.models import Signal
from otc_research.notifications.base import NotificationProvider
from otc_research.signals.confidence import classify_confidence

PENDING = "PENDING"
EXPIRED = "EXPIRED"
DECIDED = "DECIDED"


def create_signal(
    session: Session,
    *,
    asset: str,
    direction: str,
    timeframe: str,
    expiry_seconds: int,
    generated_at: dt.datetime,
    strategy_code: str,
    model_version: str,
    score: float,
    quality_tier: str,
    payout: float,
    payout_is_estimated: bool = True,
    entry_price: float | None = None,
    historical_sample_size: int | None = None,
    historical_win_rate: float | None = None,
    win_rate_ci_low: float | None = None,
    win_rate_ci_high: float | None = None,
    data_received_at: dt.datetime | None = None,
    valid_window_seconds: int | None = None,
    market_regime: str | None = None,
    session_label: str | None = None,
    day_of_week: int | None = None,
    hour: int | None = None,
    features_snapshot: Mapping[str, float] | None = None,
    conditions_met: Sequence[str] | None = None,
    reasons_rejected: Sequence[str] | None = None,
    conditions_summary: str | None = None,
    mode: str = "paper",
) -> Signal:
    """Persists one ``Signal`` row with every statistically-grounded
    field computed here, once, rather than left for a caller (or a
    notification template) to derive inconsistently: break-even win
    rate and margin (``backtest.metrics``, unchanged), the expectancy
    already computed the same way, and the fixed ALTA/MEDIA/BAJA
    confidence rule (``signals.confidence``). ``valid_window_seconds``
    defaults to ``expiry_seconds`` -- the entry window closes no later
    than the trade itself would need to open to still expire on time.
    """
    if direction not in ("CALL", "PUT"):
        raise ValueError("direction must be 'CALL' or 'PUT'")

    be_win_rate = break_even_win_rate(payout)
    margin = historical_win_rate - be_win_rate if historical_win_rate is not None else None
    expectancy = (
        payout_adjusted_expectancy(historical_win_rate, payout)
        if historical_win_rate is not None
        else None
    )
    confidence = classify_confidence(margin, win_rate_ci_low, be_win_rate)

    valid_from = generated_at
    valid_until = generated_at + dt.timedelta(seconds=valid_window_seconds or expiry_seconds)

    row = Signal(
        asset=asset,
        direction=direction,
        timeframe=timeframe,
        expiry_seconds=expiry_seconds,
        generated_at=generated_at,
        entry_price=entry_price,
        payout=payout,
        strategy_code=strategy_code,
        model_version=model_version,
        score=score,
        quality_tier=quality_tier,
        historical_sample_size=historical_sample_size,
        historical_win_rate=historical_win_rate,
        win_rate_ci_low=win_rate_ci_low,
        win_rate_ci_high=win_rate_ci_high,
        expectancy=expectancy,
        market_regime=market_regime,
        session=session_label,
        day_of_week=day_of_week,
        hour=hour,
        features_snapshot=json.dumps(dict(features_snapshot)) if features_snapshot is not None else None,
        conditions_met=json.dumps(list(conditions_met)) if conditions_met is not None else None,
        conditions_summary=conditions_summary,
        mode=mode,
        data_received_at=data_received_at,
        valid_from=valid_from,
        valid_until=valid_until,
        status=PENDING,
        break_even_win_rate=be_win_rate,
        margin_over_break_even=margin,
        payout_is_estimated=payout_is_estimated,
        confidence_label=confidence,
        reasons_rejected=json.dumps(list(reasons_rejected)) if reasons_rejected is not None else None,
    )
    session.add(row)
    session.commit()  # need row.id for signal_ref

    row.signal_ref = f"SIGNAL-{generated_at.year}-{row.id:06d}"
    session.commit()
    return row


def send_notification(
    session: Session,
    signal: Signal,
    provider: NotificationProvider,
    *,
    now: dt.datetime | None = None,
) -> bool:
    """Delivers ``signal`` through ``provider`` unless its entry window
    has already closed -- a signal is never sent once it can no longer be
    acted on in time, even if the send itself would technically succeed.
    Returns whether it was actually sent (False means expired-before-send,
    recorded on the row immediately rather than left to a later sweep).
    """
    now = now if now is not None else dt.datetime.now(dt.timezone.utc)
    if signal.valid_until is not None and now > signal.valid_until:
        signal.status = EXPIRED
        session.commit()
        return False

    provider.notify(signal)
    signal.sent_at = now
    session.commit()
    return True


def refresh_expired_signals(session: Session, *, now: dt.datetime | None = None) -> int:
    """Marks every still-``PENDING`` signal whose entry window has closed
    as ``EXPIRED`` -- the periodic sweep a live poller (or, today, a
    historical replay) runs so a signal nobody acted on doesn't stay
    ``PENDING`` forever. Returns how many were marked. Never touches a
    signal already ``DECIDED`` -- a recorded decision is final.
    """
    now = now if now is not None else dt.datetime.now(dt.timezone.utc)
    stale = (
        session.query(Signal)
        .filter(
            Signal.status == PENDING,
            Signal.valid_until.isnot(None),
            Signal.valid_until < now,
        )
        .all()
    )
    for row in stale:
        row.status = EXPIRED
    session.commit()
    return len(stale)
