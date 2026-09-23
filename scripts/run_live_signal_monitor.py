#!/usr/bin/env python
"""Manual-live signal monitor for ML_1M5M candidate #11 baseline
(``P(CALL) > 0.50``, delay=1, EUR/USD 1m, 300s expiry). See
LIVE_MANUAL_TEST.md for full usage, setup, and how to interpret output.

This script NEVER places an order and NEVER trains or modifies the
strategy. On every poll it:

1. Fetches recent EUR_USD 1m candles from Twelve Data
   (``TWELVEDATA_API_KEY`` must be set -- see LIVE_MANUAL_TEST.md).
2. Ingests them via ``data.ingestion.ingest`` (duplicate/out-of-order/gap
   detection already built in there -- a candle failing validation is
   skipped, not inserted).
3. Recomputes v4 features (idempotent -- already-computed rows are
   skipped, never overwritten).
4. Evaluates every newly-closed candle strictly after
   ``live.candidate.LIVE_TEST_START_AFTER`` via
   ``live.evaluation.evaluate_candle`` -- the SAME function
   ``scripts/replay_live_candidate11.py`` uses against historical data,
   so live and replay are structurally guaranteed to agree.
5. Logs one ``LiveEvaluation`` row per evaluation (CALL, NO_TRADE, or
   DATA_ERROR) -- never just the ones that fire.
6. On CALL: persists a ``Signal`` row and sends a notification (console
   by default; sound/desktop/Telegram opt-in via ``--notify``). This is a
   proposal for a person to review, never an executed trade.
7. Resolves any still-pending ``Signal`` whose entry/exit candle has now
   arrived, using the exact same WIN/LOSS formula as
   ``backtest.simulator.simulate`` under a delay-only (zero-slippage)
   scenario.
8. Sweeps expired signals.

Usage:
    python scripts/run_live_signal_monitor.py --once
    python scripts/run_live_signal_monitor.py --loop --poll-interval-seconds 20
    python scripts/run_live_signal_monitor.py --loop --notify console,sound,telegram

Stopping: Ctrl+C (SIGINT) in --loop mode exits cleanly after the current
poll cycle finishes; nothing is left running in the background.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time

from otc_research.config import load_config
from otc_research.data.ingestion import ingest
from otc_research.data.sources.twelvedata_source import TwelveDataSource
from otc_research.db.models import Candle, Feature, LiveEvaluation, Signal
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.features.pipeline import compute_and_store
from otc_research.live import candidate as cand
from otc_research.live.evaluation import CALL, DATA_ERROR, evaluate_candle, resolve_theoretical_result
from otc_research.live.formatting import format_call_signal_from_row, format_status_line
from otc_research.notifications.composite_provider import CompositeNotificationProvider
from otc_research.notifications.console_provider import ConsoleNotificationProvider
from otc_research.notifications.desktop_provider import DesktopNotificationProvider
from otc_research.notifications.sound_provider import SoundNotificationProvider
from otc_research.notifications.telegram_provider import TelegramNotificationProvider
from otc_research.signals.service import create_signal, refresh_expired_signals, send_notification
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)

FETCH_COUNT = 30  # recent candles per poll -- generous relative to the 1-minute cadence
INGEST_MAX_RETRIES = 3
INGEST_RETRY_BACKOFF_SECONDS = 3


def _build_notifier(channels: list[str]):
    factories = {
        "console": lambda: ConsoleNotificationProvider(message_formatter=format_call_signal_from_row),
        "sound": lambda: SoundNotificationProvider(message_formatter=format_call_signal_from_row),
        "desktop": lambda: DesktopNotificationProvider(message_formatter=format_call_signal_from_row),
        "telegram": lambda: TelegramNotificationProvider(message_formatter=format_call_signal_from_row),
    }
    providers = []
    for ch in channels:
        if ch not in factories:
            raise ValueError(f"unknown --notify channel {ch!r}; expected one of {sorted(factories)}")
        providers.append(factories[ch]())
    return providers[0] if len(providers) == 1 else CompositeNotificationProvider(providers)


def _ingest_with_retries(session):
    last_exc: Exception | None = None
    for attempt in range(1, INGEST_MAX_RETRIES + 1):
        try:
            return ingest(session, TwelveDataSource(), cand.ASSET, cand.TIMEFRAME, count=FETCH_COUNT)
        except Exception as exc:  # noqa: BLE001 -- any API/network failure, logged and retried
            last_exc = exc
            logger.error("data fetch attempt %d/%d failed: %s", attempt, INGEST_MAX_RETRIES, exc)
            if attempt < INGEST_MAX_RETRIES:
                time.sleep(INGEST_RETRY_BACKOFF_SECONDS * attempt)
    logger.error("all %d data-fetch attempts failed -- skipping this poll cycle (DATA_ERROR)", INGEST_MAX_RETRIES)
    raise last_exc  # let the caller decide whether to abort or continue looping


def _load_candles(session):
    return (
        session.query(Candle)
        .filter(Candle.asset == cand.ASSET, Candle.timeframe == cand.TIMEFRAME)
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


def _last_processed(session):
    return (
        session.query(LiveEvaluation.candle_timestamp)
        .filter(LiveEvaluation.strategy_code == cand.CODE)
        .order_by(LiveEvaluation.candle_timestamp.desc())
        .limit(1)
        .scalar()
    )


def _resolve_pending(session) -> int:
    pending = session.query(Signal).filter(Signal.strategy_code == cand.CODE, Signal.result.is_(None)).all()
    if not pending:
        return 0
    candles_by_ts = {c.timestamp: c for c in _load_candles(session)}
    resolved = 0
    for sig in pending:
        entry_ts = sig.generated_at + dt.timedelta(
            seconds=(1 + cand.ENTRY_DELAY_CANDLES) * cand.TIMEFRAME_SECONDS
        )
        exit_ts = entry_ts + dt.timedelta(seconds=cand.EXPIRY_SECONDS)

        if sig.entry_price is None:
            entry_candle = candles_by_ts.get(entry_ts)
            if entry_candle is not None:
                sig.entry_price = entry_candle.open
                session.commit()

        if sig.entry_price is not None and sig.result is None:
            exit_candle = candles_by_ts.get(exit_ts)
            if exit_candle is not None:
                result, pnl_pct = resolve_theoretical_result(sig.entry_price, exit_candle.close, "CALL")
                sig.expiry_price = exit_candle.close
                sig.pnl = pnl_pct
                sig.result = result
                session.commit()
                resolved += 1
                logger.info(
                    "RESOLVED signal_ref=%s generated_at=%s -> %s (pnl=%.4f%%)",
                    sig.signal_ref, sig.generated_at, result, pnl_pct,
                )
    return resolved


def _evaluate_new_candles(session, strategy, notifier) -> tuple[int, int]:
    candles = _load_candles(session)
    if not candles:
        logger.info("no candles ingested yet for %s/%s", cand.ASSET, cand.TIMEFRAME)
        return 0, 0
    features_by_ts = _load_features(session)

    last_processed = _last_processed(session)
    already_evaluated = {
        row.candle_timestamp
        for row in session.query(LiveEvaluation.candle_timestamp).filter(
            LiveEvaluation.strategy_code == cand.CODE
        )
    }

    n_evaluated = 0
    n_calls = 0
    for i, candle in enumerate(candles):
        # Never evaluate historical data this candidate's forward test
        # already reported on (or anything before it) -- see
        # live.candidate.LIVE_TEST_START_AFTER's docstring.
        if candle.timestamp <= cand.LIVE_TEST_START_AFTER:
            continue
        # Duplicate protection (point 11): a candle already evaluated for
        # this strategy is never re-evaluated, even across restarts --
        # both checks below are redundant with the DB UniqueConstraint,
        # kept so we skip cheaply instead of relying on a caught IntegrityError.
        if last_processed is not None and candle.timestamp <= last_processed:
            continue
        if candle.timestamp in already_evaluated:
            continue

        evaluation = evaluate_candle(
            strategy, candles, i, features_by_ts,
            timeframe_seconds=cand.TIMEFRAME_SECONDS,
            entry_delay_candles=cand.ENTRY_DELAY_CANDLES,
            expiry_seconds=cand.EXPIRY_SECONDS,
        )
        n_evaluated += 1

        signal_id = None
        if evaluation.signal == CALL:
            signal_row = create_signal(
                session,
                asset=cand.ASSET, direction="CALL", timeframe=cand.TIMEFRAME,
                expiry_seconds=cand.EXPIRY_SECONDS, generated_at=evaluation.candle_timestamp,
                strategy_code=cand.CODE, model_version=cand.model_version(),
                score=evaluation.probability_call, quality_tier="B",
                payout=cand.PAYOUT, payout_is_estimated=cand.PAYOUT_IS_ESTIMATED,
                data_received_at=evaluation.signal_time,
                conditions_met=[strategy.label],
                conditions_summary=cand.HISTORICAL_REFERENCE_NOTE,
                mode="paper",
            )
            signal_id = signal_row.id
            sent = send_notification(session, signal_row, notifier, now=evaluation.signal_time)
            logger.info(
                "CALL signal_ref=%s candle=%s prob=%.4f sent=%s",
                signal_row.signal_ref, evaluation.candle_timestamp, evaluation.probability_call, sent,
            )
            n_calls += 1
        elif evaluation.signal == DATA_ERROR:
            logger.warning("DATA_ERROR at %s: %s", evaluation.candle_timestamp, evaluation.notes)
        else:
            logger.debug("NO_TRADE at %s prob=%s", evaluation.candle_timestamp, evaluation.probability_call)

        session.add(
            LiveEvaluation(
                strategy_code=cand.CODE, asset=cand.ASSET, timeframe=cand.TIMEFRAME,
                candle_timestamp=evaluation.candle_timestamp, signal_time=evaluation.signal_time,
                probability_call=evaluation.probability_call, signal=evaluation.signal,
                model_version=cand.model_version(), delay_candles=cand.ENTRY_DELAY_CANDLES,
                entry_time=evaluation.entry_time, expiry_time=evaluation.expiry_time,
                signal_id=signal_id, notes=evaluation.notes,
            )
        )
        session.commit()

    return n_evaluated, n_calls


def poll_once(session, strategy, notifier) -> None:
    try:
        report = _ingest_with_retries(session)
        logger.info(
            "ingest: seen=%d inserted=%d skipped=%d issues=%d",
            report.candles_seen, report.candles_inserted, report.candles_skipped, report.issues_logged,
        )
    except Exception:
        logger.error("DATA_ERROR / NO_SIGNAL -- could not fetch new candles this cycle")
        return

    feat_report = compute_and_store(session, cand.ASSET, cand.TIMEFRAME)
    logger.info(
        "features: inserted=%d skipped_existing=%d",
        feat_report.rows_inserted, feat_report.rows_skipped_existing,
    )

    n_resolved = _resolve_pending(session)
    n_evaluated, n_calls = _evaluate_new_candles(session, strategy, notifier)
    n_expired = refresh_expired_signals(session)
    logger.info(
        "poll complete: evaluated=%d calls=%d resolved=%d expired=%d",
        n_evaluated, n_calls, n_resolved, n_expired,
    )

    candles = _load_candles(session)
    if candles:
        latest = candles[-1]
        last_eval = (
            session.query(LiveEvaluation)
            .filter(LiveEvaluation.strategy_code == cand.CODE)
            .order_by(LiveEvaluation.candle_timestamp.desc())
            .first()
        )
        print(
            format_status_line(
                last_candle_timestamp=latest.timestamp, last_close=latest.close,
                last_probability=last_eval.probability_call if last_eval else None,
                last_signal=last_eval.signal if last_eval else None,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--notify", default="console",
        help="comma-separated channels: console,sound,desktop,telegram (default: console)",
    )
    parser.add_argument("--loop", action="store_true", help="run continuously until Ctrl+C")
    parser.add_argument("--once", action="store_true", help="run a single poll cycle and exit (default)")
    parser.add_argument("--poll-interval-seconds", type=int, default=20)
    args = parser.parse_args()

    if not args.loop:
        args.once = True

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    strategy = cand.build_strategy()
    notifier = _build_notifier([c.strip() for c in args.notify.split(",") if c.strip()])

    logger.info(
        "ML_1M5M candidate #11 LIVE manual signal monitor -- model=%s threshold=%.2f "
        "delay=%d expiry=%ds asset=%s/%s -- NO automatic order placement, manual entry only",
        cand.model_version(), cand.PROBABILITY_THRESHOLD, cand.ENTRY_DELAY_CANDLES,
        cand.EXPIRY_SECONDS, cand.ASSET, cand.TIMEFRAME,
    )

    if args.once:
        poll_once(session, strategy, notifier)
        return

    logger.info("looping every %ds -- Ctrl+C to stop", args.poll_interval_seconds)
    try:
        while True:
            try:
                poll_once(session, strategy, notifier)
            except Exception:
                logger.exception("poll cycle failed unexpectedly, will retry next interval")
            time.sleep(args.poll_interval_seconds)
    except KeyboardInterrupt:
        logger.info("stopped by user")


if __name__ == "__main__":
    main()
