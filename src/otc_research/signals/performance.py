"""Theoretical vs. executable performance (approved plan point 12) --
the same signal history read two ways, reported side by side so the gap
between them is explicit rather than buried in one blended number.

Reuses ``backtest.metrics.summarize_trades`` exactly as every other trade
aggregation in this codebase does; this module's only job is picking the
right rows and result/pnl source, not reimplementing the statistics.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from otc_research.backtest.metrics import TradeStats, summarize_trades
from otc_research.db.models import Signal, SignalDecision
from otc_research.signals.decisions import TOOK_TRADE


def _filtered_signals(session: Session, *, asset: str | None, timeframe: str | None):
    query = session.query(Signal).filter(Signal.result.isnot(None))
    if asset is not None:
        query = query.filter(Signal.asset == asset)
    if timeframe is not None:
        query = query.filter(Signal.timeframe == timeframe)
    return query.all()


def theoretical_performance(
    session: Session, *, asset: str | None = None, timeframe: str | None = None
) -> TradeStats:
    """Every ``Signal`` ever generated with a resolved outcome, evaluated
    exactly as if filled at ``entry_price``/``generated_at`` -- this is
    what the existing backtest engine already computes for a strategy;
    here it is read off the persisted signal history instead of re-run.
    """
    signals = _filtered_signals(session, asset=asset, timeframe=timeframe)
    return summarize_trades([s.result for s in signals], [s.pnl or 0.0 for s in signals])


@dataclass(frozen=True)
class ExecutablePerformance:
    stats: TradeStats
    n_fallback_to_theoretical: int  # TOOK_TRADE but no actuals recorded on the decision itself
    n_excluded_not_taken: int  # DID_NOT_TAKE / ARRIVED_LATE / never decided at all


def executable_performance(
    session: Session, *, asset: str | None = None, timeframe: str | None = None
) -> ExecutablePerformance:
    """Filtered to ``SignalDecision.decision == TOOK_TRADE`` only --
    ``DID_NOT_TAKE``, ``ARRIVED_LATE``, and signals nobody ever decided on
    at all are excluded entirely, never counted as a loss or blended in.
    Uses the decision's own ``result``/``pnl`` when recorded; falls back
    to the signal's theoretical result/pnl only when the decision's own
    are missing, and reports how often that fallback happened so the read
    is never silently mixed.
    """
    query = (
        session.query(Signal, SignalDecision)
        .join(SignalDecision, SignalDecision.signal_id == Signal.id)
        .filter(SignalDecision.decision == TOOK_TRADE)
    )
    if asset is not None:
        query = query.filter(Signal.asset == asset)
    if timeframe is not None:
        query = query.filter(Signal.timeframe == timeframe)

    results: list[str] = []
    pnls: list[float] = []
    n_fallback = 0
    n_excluded = 0
    for signal, decision in query.all():
        result = decision.result if decision.result is not None else signal.result
        pnl = decision.pnl if decision.pnl is not None else signal.pnl
        if result is None:
            n_excluded += 1
            continue
        if decision.result is None:
            n_fallback += 1
        results.append(result)
        pnls.append(pnl or 0.0)

    stats = summarize_trades(results, pnls)
    return ExecutablePerformance(
        stats=stats, n_fallback_to_theoretical=n_fallback, n_excluded_not_taken=n_excluded
    )
