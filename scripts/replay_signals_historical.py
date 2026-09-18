#!/usr/bin/env python
"""Historical replay: proves the Phase 9 signal lifecycle end-to-end --
create_signal -> send_notification -> record_decision ->
refresh_expired_signals -> theoretical/executable performance -- against
already-ingested candles, fed through in timestamp order. Explicitly NOT
a live run (approved plan implementation order step 8's own requirement):
no network access, no broker, no live polling. Outcomes are resolved
directly from already-ingested future candles purely so this replay's own
performance report is non-empty; a live run would instead depend on a
person recording what actually happened.
"""

from __future__ import annotations

import argparse
import datetime as dt
import random

from otc_research.config import load_config
from otc_research.db.models import Candle, Feature
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.notifications.console_provider import ConsoleNotificationProvider
from otc_research.signals.decisions import DID_NOT_TAKE, TOOK_TRADE, record_decision
from otc_research.signals.performance import executable_performance, theoretical_performance
from otc_research.signals.service import create_signal, refresh_expired_signals, send_notification
from otc_research.strategies.h4_bollinger import H4BollingerMeanReversion
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


def _load_features(session, asset, timeframe, feature_set_version):
    rows = (
        session.query(Feature.timestamp, Feature.name, Feature.value)
        .filter(
            Feature.asset == asset,
            Feature.timeframe == timeframe,
            Feature.feature_set_version == feature_set_version,
        )
        .all()
    )
    by_ts: dict = {}
    for ts, name, value in rows:
        by_ts.setdefault(ts, {})[name] = value
    return by_ts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair", default="EUR_USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--expiry-seconds", type=int, default=3600)
    parser.add_argument("--payout", type=float, default=0.85)
    parser.add_argument("--max-signals", type=int, default=20)
    parser.add_argument("--rng-seed", type=int, default=0)
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    candles = _load_candles(session, args.pair, args.timeframe)
    features_by_ts = _load_features(session, args.pair, args.timeframe, FEATURE_SET_VERSION)
    if not candles:
        raise SystemExit(f"no candles ingested for {args.pair}/{args.timeframe}")

    timeframe_seconds = timeframe_to_seconds(args.timeframe)
    if args.expiry_seconds % timeframe_seconds != 0:
        raise SystemExit("--expiry-seconds must be a whole number of candles")
    expiry_candles = args.expiry_seconds // timeframe_seconds

    strategy = H4BollingerMeanReversion(expiry_seconds=args.expiry_seconds)
    provider = ConsoleNotificationProvider()
    rng = random.Random(args.rng_seed)

    created = 0
    for i, candle in enumerate(candles):
        if created >= args.max_signals:
            break
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
            asset=args.pair,
            direction=direction,
            timeframe=args.timeframe,
            expiry_seconds=args.expiry_seconds,
            generated_at=candle.timestamp,
            strategy_code=strategy.code,
            model_version=FEATURE_SET_VERSION,
            score=0.5,
            quality_tier="B",
            payout=args.payout,
            payout_is_estimated=True,
            entry_price=candle.close,
            data_received_at=candle.timestamp,
            conditions_met=[strategy.label],
            mode="paper",
        )
        created += 1
        sent = send_notification(session, signal, provider, now=candle.timestamp)
        logger.info("signal %s sent=%s", signal.signal_ref, sent)

        decided_at = candle.timestamp + dt.timedelta(seconds=30)
        took_trade = rng.random() < 0.5
        record_decision(
            session, signal.id, TOOK_TRADE if took_trade else DID_NOT_TAKE, decided_at
        )

        exit_idx = i + expiry_candles
        if exit_idx < len(candles):
            exit_close = candles[exit_idx].close
            pnl_pct = (
                (exit_close - candle.close) / candle.close * 100.0
                if direction == "CALL"
                else (candle.close - exit_close) / candle.close * 100.0
            )
            signal.result = "WIN" if pnl_pct > 0 else "LOSS"
            signal.pnl = pnl_pct
            session.commit()

    n_expired = refresh_expired_signals(
        session, now=candles[-1].timestamp + dt.timedelta(days=3650)
    )
    logger.info("created=%d expired_swept=%d", created, n_expired)

    theo = theoretical_performance(session, asset=args.pair, timeframe=args.timeframe)
    exe = executable_performance(session, asset=args.pair, timeframe=args.timeframe)
    logger.info(
        "THEORETICAL: n=%d win_rate=%s",
        theo.sample_size,
        f"{theo.win_rate:.3f}" if theo.win_rate is not None else "n/a",
    )
    logger.info(
        "EXECUTABLE:  n=%d win_rate=%s (fallback_to_theoretical=%d, excluded_not_taken=%d)",
        exe.stats.sample_size,
        f"{exe.stats.win_rate:.3f}" if exe.stats.win_rate is not None else "n/a",
        exe.n_fallback_to_theoretical,
        exe.n_excluded_not_taken,
    )


if __name__ == "__main__":
    main()
