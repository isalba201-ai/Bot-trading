#!/usr/bin/env python
"""Forward/paper-test poll for the two frozen candidates in
``research.forward_test.FORWARD_TEST_CANDIDATES`` (H9_FWD, ML10_FWD).

Assumes fresh real candles for each candidate's (asset, timeframe) have
ALREADY been ingested into the DB before this runs (via the sanctioned
CSV-import path — see DATA.md/csv_source.py — since fetching live data
here would need network access this script does not have; the calling
agent fetches via an authorized connector, saves a CSV, and ingests it
first). This script only:

1. Recomputes v4 features for any newly-ingested candles
   (``features.pipeline.compute_and_store``, idempotent).
2. Resolves any PENDING Signal for a forward-test candidate whose entry
   candle (signal candle + 1 + DEFAULT_ENTRY_DELAY_CANDLES) and/or exit
   candle (entry + expiry_candles) has now arrived — filling
   entry_price/result/pnl/expiry_price exactly once each, matching the
   same delay-only execution convention every backtest in this project
   uses. A signal is never re-resolved once its result is set.
3. Walks every newly-arrived closed candle (chronological, not-yet-
   checked) for each candidate's (asset, timeframe) and creates a new
   PENDING paper Signal wherever the candidate's frozen strategy fires —
   never more than one Signal per (candidate, signal candle timestamp).

Never touches the candidate's original historical TEST split — this is
exclusively forward, on data ingested after the candidate was frozen.
"""

from __future__ import annotations

import argparse
import datetime as dt

from otc_research.config import load_config
from otc_research.db.models import Candle, Feature, Signal
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.features.pipeline import compute_and_store
from otc_research.research.candidacy import DEFAULT_ENTRY_DELAY_CANDLES
from otc_research.research.forward_test import FORWARD_TEST_CANDIDATES
from otc_research.signals.service import create_signal, refresh_expired_signals
from otc_research.utils.logging import get_logger
from otc_research.utils.timeframes import timeframe_to_seconds

logger = get_logger(__name__)


def _load_candles(session, asset, timeframe):
    return (
        session.query(Candle)
        .filter(Candle.asset == asset, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.asc())
        .all()
    )


def _load_features(session, asset, timeframe):
    rows = (
        session.query(Feature.timestamp, Feature.name, Feature.value)
        .filter(
            Feature.asset == asset, Feature.timeframe == timeframe,
            Feature.feature_set_version == FEATURE_SET_VERSION,
        )
        .all()
    )
    by_ts: dict = {}
    for ts, name, value in rows:
        by_ts.setdefault(ts, {})[name] = value
    return by_ts


def _resolve_pending(session, candidate, candles_by_ts, timeframe_seconds, expiry_candles, delay_candles):
    pending = (
        session.query(Signal)
        .filter(Signal.strategy_code == candidate.code, Signal.result.is_(None))
        .all()
    )
    resolved = 0
    for sig in pending:
        entry_ts = sig.generated_at + dt.timedelta(seconds=(1 + delay_candles) * timeframe_seconds)
        exit_ts = sig.generated_at + dt.timedelta(seconds=(1 + delay_candles + expiry_candles) * timeframe_seconds)

        if sig.entry_price is None:
            entry_candle = candles_by_ts.get(entry_ts)
            if entry_candle is not None:
                sig.entry_price = entry_candle.open
                session.commit()

        if sig.entry_price is not None and sig.result is None:
            exit_candle = candles_by_ts.get(exit_ts)
            if exit_candle is not None:
                exit_price = exit_candle.close
                if sig.direction == "CALL":
                    pnl_pct = (exit_price - sig.entry_price) / sig.entry_price * 100.0
                else:
                    pnl_pct = (sig.entry_price - exit_price) / sig.entry_price * 100.0
                sig.expiry_price = exit_price
                sig.pnl = pnl_pct
                sig.result = "WIN" if pnl_pct > 0 else "LOSS"
                session.commit()
                resolved += 1
                logger.info(
                    "RESOLVED %s %s signal_ref=%s generated_at=%s -> %s (pnl=%.4f%%)",
                    candidate.code, candidate.asset, sig.signal_ref, sig.generated_at,
                    sig.result, pnl_pct,
                )
    return resolved


def _check_for_new_signals(session, candidate, candles, candles_by_ts, features_by_ts, timeframe_seconds):
    strategy = candidate.strategy
    already_signaled = {
        row.generated_at
        for row in session.query(Signal.generated_at).filter(Signal.strategy_code == candidate.code)
    }
    created = 0
    now = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None)
    for candle in candles:
        # Never process a candle from before this candidate's forward-test
        # start -- that data was already used for discovery/candidacy
        # (including, for some candles, the frozen TEST split itself) and
        # must never be silently re-touched.
        if candle.timestamp < candidate.forward_test_start:
            continue
        if candle.timestamp in already_signaled:
            continue
        # Only ever act on a candle that has actually closed -- its open
        # time plus one full interval must already be in the past.
        if candle.timestamp + dt.timedelta(seconds=timeframe_seconds) > now:
            continue
        feats = features_by_ts.get(candle.timestamp)
        merged = {
            "open": candle.open, "high": candle.high, "low": candle.low, "close": candle.close,
            **(feats or {}),
        }
        if not strategy.required_features.issubset(merged.keys()):
            continue
        direction = strategy.decide(merged)
        if direction is None:
            continue

        signal = create_signal(
            session,
            asset=candidate.asset,
            direction=direction,
            timeframe=candidate.timeframe,
            expiry_seconds=candidate.expiry_seconds,
            generated_at=candle.timestamp,
            strategy_code=candidate.code,
            model_version=FEATURE_SET_VERSION,
            score=0.5,
            quality_tier="B",
            payout=candidate.payout,
            payout_is_estimated=True,
            data_received_at=candle.timestamp + dt.timedelta(seconds=timeframe_seconds),
            conditions_met=[strategy.label],
            conditions_summary=candidate.backtest_status,
            mode="paper",
        )
        created += 1
        logger.info(
            "NEW SIGNAL %s %s/%s %s generated_at=%s signal_ref=%s",
            candidate.code, candidate.asset, candidate.timeframe, direction,
            candle.timestamp, signal.signal_ref,
        )
    return created


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    for candidate in FORWARD_TEST_CANDIDATES:
        report = compute_and_store(session, candidate.asset, candidate.timeframe)
        logger.info(
            "%s features: %d inserted, %d already existed",
            candidate.code, report.rows_inserted, report.rows_skipped_existing,
        )

        timeframe_seconds = timeframe_to_seconds(candidate.timeframe)
        expiry_candles = candidate.expiry_seconds // timeframe_seconds

        candles = _load_candles(session, candidate.asset, candidate.timeframe)
        candles_by_ts = {c.timestamp: c for c in candles}
        features_by_ts = _load_features(session, candidate.asset, candidate.timeframe)

        n_resolved = _resolve_pending(
            session, candidate, candles_by_ts, timeframe_seconds, expiry_candles, DEFAULT_ENTRY_DELAY_CANDLES
        )
        n_created = _check_for_new_signals(
            session, candidate, candles, candles_by_ts, features_by_ts, timeframe_seconds
        )
        logger.info("%s: resolved=%d new_signals=%d", candidate.code, n_resolved, n_created)

    n_expired = refresh_expired_signals(session)
    logger.info("expired_swept=%d", n_expired)


if __name__ == "__main__":
    main()
