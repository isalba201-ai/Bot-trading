"""Records a person's manual decision about a ``Signal`` (approved plan
point 10) -- the ONE write path (a CLI today, since there is no UI yet;
see scripts/record_signal_decision.py). Nothing infers or auto-generates
a decision: a ``Signal`` with no row here was simply never acted on.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy.orm import Session

from otc_research.db.models import Signal, SignalDecision
from otc_research.signals.service import DECIDED

TOOK_TRADE = "TOOK_TRADE"
DID_NOT_TAKE = "DID_NOT_TAKE"
ARRIVED_LATE = "ARRIVED_LATE"

VALID_DECISIONS = (TOOK_TRADE, DID_NOT_TAKE, ARRIVED_LATE)


def record_decision(
    session: Session,
    signal_id: int,
    decision: str,
    decided_at: dt.datetime,
    *,
    entry_price_actual: float | None = None,
    payout_observed: float | None = None,
    entry_time_actual: dt.datetime | None = None,
    expiry_used_seconds: int | None = None,
) -> SignalDecision:
    """A ``TOOK_TRADE`` claim recorded after the signal's entry window
    has already closed is impossible on its face (there was nothing left
    to enter), so it is reclassified to ``ARRIVED_LATE`` automatically --
    approved plan point 11's "a recorded decided_at > valid_until ... is
    marked ... ARRIVED_LATE ... automatically rather than left for manual
    bookkeeping". An explicit ``DID_NOT_TAKE`` is never overridden this
    way -- choosing not to trade is a real decision regardless of timing.
    """
    if decision not in VALID_DECISIONS:
        raise ValueError(f"decision must be one of {VALID_DECISIONS}, got {decision!r}")

    signal = session.get(Signal, signal_id)
    if signal is None:
        raise ValueError(f"no Signal with id={signal_id}")

    if decision == TOOK_TRADE and signal.valid_until is not None and decided_at > signal.valid_until:
        decision = ARRIVED_LATE

    row = SignalDecision(
        signal_id=signal_id,
        decision=decision,
        decided_at=decided_at,
        entry_price_actual=entry_price_actual,
        payout_observed=payout_observed,
        entry_time_actual=entry_time_actual,
        expiry_used_seconds=expiry_used_seconds,
    )
    session.add(row)
    signal.status = DECIDED
    session.commit()
    return row
