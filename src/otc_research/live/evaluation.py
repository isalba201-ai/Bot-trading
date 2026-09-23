"""Pure, deterministic per-candle evaluation logic for the ML_1M5M
candidate #11 manual-live test -- shared VERBATIM by the live poller
(scripts/run_live_signal_monitor.py) and the historical replay tool
(scripts/replay_live_candidate11.py), so the two are structurally
guaranteed to produce identical decisions given identical candles and
features (see LIVE_MANUAL_TEST.md's replay-parity requirement).

This module never fits/trains anything -- it only ever calls the frozen
model's already-fitted ``predict_proba``. It does not call
``ModelStrategy.decide()`` to get the fire/no-fire decision (it needs the
raw probability for display, which ``decide()`` doesn't return), so it
replicates ``decide()``'s exact two-line formula from that class's own
public attributes (``model``, ``feature_cols``, ``probability_threshold``,
``direction``) -- ``otc_research.research.model_strategy`` itself is
never read from except via those attributes, and is never modified. The
runtime assertion at the end of ``evaluate_candle`` cross-checks this
replication against ``strategy.decide()`` on every single call, so any
future divergence between the two would fail loudly rather than silently
producing a different signal than the frozen strategy actually would.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

CALL = "CALL"
NO_TRADE = "NO_TRADE"
DATA_ERROR = "DATA_ERROR"


class CandleLike(Protocol):
    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Evaluation:
    candle_timestamp: dt.datetime  # OPEN time of the candle used for this decision
    signal_time: dt.datetime  # candle_timestamp + one timeframe interval (the candle's close)
    signal: str  # CALL / NO_TRADE / DATA_ERROR
    probability_call: float | None  # None only for DATA_ERROR
    entry_time: dt.datetime | None  # set only when signal == CALL
    expiry_time: dt.datetime | None  # set only when signal == CALL
    notes: str | None = None


def evaluate_candle(
    strategy,
    candles: Sequence[CandleLike],
    index: int,
    features_by_ts: Mapping[dt.datetime, Mapping[str, float]],
    *,
    timeframe_seconds: int,
    entry_delay_candles: int,
    expiry_seconds: int,
) -> Evaluation:
    """Evaluates ``candles[index]`` -- the just-closed candle -- and
    returns a CALL/NO_TRADE/DATA_ERROR decision. Never looks at
    ``candles[index + 1:]`` (no look-ahead): entry/expiry times are
    computed arithmetically from ``candle.timestamp``, not by requiring
    the entry/exit candles to already exist (in live use they are still
    in the future; resolving WIN/LOSS against them happens separately,
    once they arrive -- see ``run_live_signal_monitor.py``).
    """
    candle = candles[index]
    signal_time = candle.timestamp + dt.timedelta(seconds=timeframe_seconds)

    if index > 0:
        expected_prev = candle.timestamp - dt.timedelta(seconds=timeframe_seconds)
        actual_prev = candles[index - 1].timestamp
        if actual_prev != expected_prev:
            return Evaluation(
                candle_timestamp=candle.timestamp,
                signal_time=signal_time,
                signal=DATA_ERROR,
                probability_call=None,
                entry_time=None,
                expiry_time=None,
                notes=(
                    f"gap detected: candle immediately before {candle.timestamp} is "
                    f"{actual_prev}, expected {expected_prev} -- refusing to evaluate "
                    "on a candle series with a missing bar immediately before it"
                ),
            )

    feats: dict[str, float] = {
        "open": candle.open,
        "high": candle.high,
        "low": candle.low,
        "close": candle.close,
        **(features_by_ts.get(candle.timestamp) or {}),
    }
    if not strategy.required_features.issubset(feats.keys()):
        missing = sorted(strategy.required_features - feats.keys())
        return Evaluation(
            candle_timestamp=candle.timestamp,
            signal_time=signal_time,
            signal=DATA_ERROR,
            probability_call=None,
            entry_time=None,
            expiry_time=None,
            notes=f"missing features for {candle.timestamp}: {missing}",
        )

    row = [[feats[f] for f in strategy.feature_cols]]
    probability = float(strategy.model.predict_proba(row)[0, 1])
    fires = probability > strategy.probability_threshold

    # Live cross-check: this must never diverge from calling the actual
    # frozen strategy's own decide() -- if it ever does, something about
    # this replication has drifted from the frozen strategy, and that is
    # exactly the class of bug this assertion exists to catch immediately
    # rather than silently emit a wrong signal.
    assert fires == (strategy.decide(feats) is not None), (
        "evaluate_candle's probability/threshold replication diverged from "
        "strategy.decide() -- refusing to trust this evaluation"
    )

    if not fires:
        return Evaluation(
            candle_timestamp=candle.timestamp,
            signal_time=signal_time,
            signal=NO_TRADE,
            probability_call=probability,
            entry_time=None,
            expiry_time=None,
        )

    entry_time = candle.timestamp + dt.timedelta(seconds=(1 + entry_delay_candles) * timeframe_seconds)
    expiry_time = entry_time + dt.timedelta(seconds=expiry_seconds)
    return Evaluation(
        candle_timestamp=candle.timestamp,
        signal_time=signal_time,
        signal=CALL,
        probability_call=probability,
        entry_time=entry_time,
        expiry_time=expiry_time,
    )


def resolve_theoretical_result(
    entry_price: float | None, exit_price: float | None, direction: str = "CALL"
) -> tuple[str | None, float | None]:
    """WIN/LOSS/pnl_pct for a CALL, using the EXACT same formula as
    ``backtest.simulator._pnl_pct`` under a delay-only (zero-slippage)
    scenario: ``entry_price``/``exit_price`` are used as-is, no
    adjustment. Returns ``(None, None)`` if either price is not yet
    known (still pending) -- never fabricates a result.
    """
    if entry_price is None or exit_price is None:
        return None, None
    if direction == "CALL":
        pnl_pct = (exit_price - entry_price) / entry_price * 100.0
    else:
        pnl_pct = (entry_price - exit_price) / entry_price * 100.0
    return ("WIN" if pnl_pct > 0 else "LOSS"), pnl_pct
