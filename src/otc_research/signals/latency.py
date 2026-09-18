"""Latency measurement (approved plan point 11) -- computed on demand
from the timestamps already stored on a ``Signal``, never stored
redundantly as its own column.

**Honest caveat, carried over from the approved plan**: these numbers
are only meaningful once phase (B) is actually running against a live
poller, where ``data_received_at`` reflects a real external event. In a
historical replay or a backtest-derived signal there is no live "data
received" moment to measure against, so a caller feeding synthetic or
backtest-derived timestamps here is measuring the pipeline's own
mechanics, not real-world delivery speed -- this module does not
distinguish the two, the caller must.
"""

from __future__ import annotations

import datetime as dt

from otc_research.db.models import Signal


def latency_to_generate(signal: Signal) -> dt.timedelta | None:
    if signal.data_received_at is None:
        return None
    return signal.generated_at - signal.data_received_at


def latency_to_notify(signal: Signal) -> dt.timedelta | None:
    if signal.sent_at is None:
        return None
    return signal.sent_at - signal.generated_at


def is_expired_before_execution(signal: Signal) -> bool:
    """True when a signal's entry window closed with no decision ever
    recorded on it -- the plan's ``EXPIRED_BEFORE_EXECUTION`` outcome,
    derived rather than stored as its own status value (``Signal.status``
    stays PENDING/EXPIRED/DECIDED; this is EXPIRED specifically with
    nothing in ``signal_decisions`` for it, which the caller checks via
    ``performance.executable_performance``'s exclusion count).
    """
    return signal.status == "EXPIRED"
