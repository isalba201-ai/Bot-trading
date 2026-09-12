#!/usr/bin/env python
"""Fetch real Forex candles from OANDA and store them.

Requires OANDA_API_TOKEN to be set in the environment (never pass it as a
CLI argument — that would leak it into your shell history).

Examples:
    # most recent 500 closed 1-minute candles
    python scripts/fetch_oanda.py --pair EUR_USD --timeframe 1m --count 500

    # a specific historical window (for backtesting)
    python scripts/fetch_oanda.py --pair EUR_USD --timeframe 5m \\
        --start 2026-01-01T00:00:00Z --end 2026-01-08T00:00:00Z
"""

from __future__ import annotations

import argparse
import datetime as dt

from otc_research.config import load_config
from otc_research.data.ingestion import ingest
from otc_research.data.sources.oanda_source import OandaDataSource
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)


def _parse_iso(value: str) -> dt.datetime:
    return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, help='OANDA instrument, e.g. "EUR_USD"')
    parser.add_argument("--timeframe", required=True, help='"1m", "5m", "15m", "30m", "1h", ...')
    parser.add_argument("--count", type=int, default=None, help="Most recent N closed candles")
    parser.add_argument("--start", default=None, help="ISO8601 start (requires --end)")
    parser.add_argument("--end", default=None, help="ISO8601 end (requires --start)")
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    if (args.start is None) != (args.end is None):
        parser.error("--start and --end must be given together")

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    source = OandaDataSource(environment=config.market.oanda_environment)

    start = _parse_iso(args.start) if args.start else None
    end = _parse_iso(args.end) if args.end else None

    # Same validated ingestion path as CSV imports — no separate/looser
    # code path just because this source is live.
    report = ingest(
        session,
        source,
        asset=args.pair,
        timeframe=args.timeframe,
        max_suspicious_move_pct=config.data_validation.max_suspicious_move_pct,
        start=start,
        end=end,
        count=args.count,
    )

    logger.info(
        "OANDA fetch %s/%s: seen=%d inserted=%d skipped=%d issues_logged=%d",
        args.pair,
        args.timeframe,
        report.candles_seen,
        report.candles_inserted,
        report.candles_skipped,
        report.issues_logged,
    )


if __name__ == "__main__":
    main()
