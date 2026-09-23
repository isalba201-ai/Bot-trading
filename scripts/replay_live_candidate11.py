#!/usr/bin/env python
"""Replay/simulation mode for the ML_1M5M candidate #11 manual-live
infrastructure (user's explicit requirement before ever connecting it to
the real market): proves that ``otc_research.live.evaluation`` -- the
exact code the live monitor uses -- reproduces the SAME aggregate result
already published in ML1M5M_CANDIDATE11_FORWARD_TEST_REPORT.md for the
baseline/delay1 configuration, when fed the same historical candles in
the same order.

Reuses the already-ingested, already-disjoint forward-test block
(2026-09-19 09:43 -> 2026-09-22 21:02, strictly after
TRAIN/VALIDATION/TEST and never itself re-touching the historical TEST
reserve) -- no new data is fetched, and nothing about that block's
reservation status changes by reading it again here. This is an
infrastructure-parity check, not a new statistical claim: its numbers are
EXPECTED to exactly match the forward-test report, and the script refuses
to proceed (SystemExit(1)) if they don't.

Never trains anything; never touches Signal/LiveEvaluation tables (this
runs against an in-memory candle list, not through the live poller's
persistence path, so replaying never pollutes the real live history).
"""

from __future__ import annotations

import datetime as dt

from otc_research.config import load_config
from otc_research.db.models import Candle, Feature
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.live import candidate as cand
from otc_research.live.evaluation import CALL, NO_TRADE, evaluate_candle, resolve_theoretical_result
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)

# Exactly the forward-test block already reported on -- see
# ML1M5M_CANDIDATE11_FORWARD_TEST_REPORT.md section 2. Bounded explicitly
# so this replay can never drift into evaluating genuinely new live data
# (that is scripts/run_live_signal_monitor.py's job) or, in the other
# direction, into the historical TRAIN/VALIDATION/TEST window.
REPLAY_START = dt.datetime(2026, 9, 19)
REPLAY_END = dt.datetime(2026, 9, 22, 21, 2, 1)  # exclusive upper bound

# The published baseline/delay1 aggregate this replay must reproduce
# exactly (ML1M5M_CANDIDATE11_FORWARD_TEST_REPORT.md section 3).
EXPECTED_TOTAL_SIGNALS = 2376
EXPECTED_WIN = 1141
EXPECTED_LOSS = 1232
EXPECTED_VOID = 3
EXPECTED_N = 2373


def _load_candles(session):
    return (
        session.query(Candle)
        .filter(
            Candle.asset == cand.ASSET, Candle.timeframe == cand.TIMEFRAME,
            Candle.timestamp >= REPLAY_START, Candle.timestamp < REPLAY_END,
        )
        .order_by(Candle.timestamp.asc())
        .all()
    )


def _load_features(session):
    rows = (
        session.query(Feature.timestamp, Feature.name, Feature.value)
        .filter(
            Feature.asset == cand.ASSET, Feature.timeframe == cand.TIMEFRAME,
            Feature.feature_set_version == FEATURE_SET_VERSION,
        )
        .all()
    )
    by_ts: dict = {}
    for ts, name, value in rows:
        by_ts.setdefault(ts, {})[name] = value
    return by_ts


def run_replay(candles, features_by_ts, strategy) -> dict:
    """Walks ``candles`` in order, evaluating every one via the SAME
    ``evaluate_candle`` function the live monitor calls, and resolves each
    CALL against candles already present in this same list (a closed
    historical block, so "the entry/exit candle never arrives" means
    genuinely VOID here, not merely "not yet" as it would in live use).
    """
    candles_by_ts = {c.timestamp: c for c in candles}
    n_evaluated = 0
    n_call = 0
    n_no_trade = 0
    n_win = 0
    n_loss = 0
    n_void = 0

    for i, candle in enumerate(candles):
        evaluation = evaluate_candle(
            strategy, candles, i, features_by_ts,
            timeframe_seconds=cand.TIMEFRAME_SECONDS,
            entry_delay_candles=cand.ENTRY_DELAY_CANDLES,
            expiry_seconds=cand.EXPIRY_SECONDS,
        )
        n_evaluated += 1

        if evaluation.signal == NO_TRADE:
            n_no_trade += 1
            continue
        if evaluation.signal != CALL:
            continue  # DATA_ERROR -- not expected in this already-validated block

        n_call += 1
        entry_candle = candles_by_ts.get(evaluation.entry_time)
        exit_time = evaluation.expiry_time
        exit_candle = candles_by_ts.get(exit_time)
        if entry_candle is None or exit_candle is None:
            n_void += 1  # insufficient_data -- same semantics as backtest.simulator
            continue

        result, _pnl_pct = resolve_theoretical_result(entry_candle.open, exit_candle.close, "CALL")
        if result == "WIN":
            n_win += 1
        else:
            n_loss += 1

    return {
        "n_evaluated": n_evaluated, "n_call": n_call, "n_no_trade": n_no_trade,
        "n_win": n_win, "n_loss": n_loss, "n_void": n_void, "n_resolved": n_win + n_loss,
    }


def main() -> None:
    config = load_config(None)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    candles = _load_candles(session)
    if not candles:
        raise SystemExit(
            f"no candles found for {cand.ASSET}/{cand.TIMEFRAME} in "
            f"[{REPLAY_START}, {REPLAY_END}) -- the forward-test block must already be ingested."
        )
    features_by_ts = _load_features(session)
    strategy = cand.build_strategy()

    logger.info(
        "replaying %d candles from %s to %s (model=%s)",
        len(candles), candles[0].timestamp, candles[-1].timestamp, cand.model_version(),
    )

    result = run_replay(candles, features_by_ts, strategy)
    logger.info(
        "REPLAY RESULT: evaluated=%d CALL=%d NO_TRADE=%d WIN=%d LOSS=%d VOID=%d n=%d",
        result["n_evaluated"], result["n_call"], result["n_no_trade"],
        result["n_win"], result["n_loss"], result["n_void"], result["n_resolved"],
    )

    matches = (
        result["n_call"] == EXPECTED_TOTAL_SIGNALS
        and result["n_win"] == EXPECTED_WIN
        and result["n_loss"] == EXPECTED_LOSS
        and result["n_void"] == EXPECTED_VOID
        and result["n_resolved"] == EXPECTED_N
    )
    if matches:
        print(
            "REPLAY PARITY: OK -- the live infrastructure reproduces the published "
            "forward-test baseline/delay1 result exactly "
            f"(WIN={EXPECTED_WIN} LOSS={EXPECTED_LOSS} VOID={EXPECTED_VOID} n={EXPECTED_N})."
        )
    else:
        print(
            "REPLAY PARITY: *** MISMATCH *** -- the live infrastructure does NOT reproduce "
            "the published forward-test result. Do not connect this to the live market until "
            "this is resolved."
        )
        print(f"  expected: CALL={EXPECTED_TOTAL_SIGNALS} WIN={EXPECTED_WIN} LOSS={EXPECTED_LOSS} "
              f"VOID={EXPECTED_VOID} n={EXPECTED_N}")
        print(f"  got:      CALL={result['n_call']} WIN={result['n_win']} LOSS={result['n_loss']} "
              f"VOID={result['n_void']} n={result['n_resolved']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
