#!/usr/bin/env python
"""Compute point-in-time features over already-ingested, validated candles.

Reads from the local database only — run scripts/fetch_market_data.py or
scripts/import_csv.py first to populate candles for the pair/timeframe.
Safe to re-run: rows already computed for the given feature-set version are
never overwritten (see FEATURES.md).

Example:
    python scripts/compute_features.py --pair EUR_USD --timeframe 5m
"""

from __future__ import annotations

import argparse

from otc_research.config import load_config
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.features.engine import FEATURE_SET_VERSION
from otc_research.features.pipeline import compute_and_store
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", required=True, help='e.g. "EUR_USD"')
    parser.add_argument("--timeframe", required=True, help='"1m", "5m", "15m", ...')
    parser.add_argument(
        "--feature-set-version",
        default=FEATURE_SET_VERSION,
        help=f"Defaults to the current version ({FEATURE_SET_VERSION}).",
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    report = compute_and_store(
        session,
        asset=args.pair,
        timeframe=args.timeframe,
        feature_set_version=args.feature_set_version,
    )

    logger.info(
        "features %s/%s v=%s: candles_seen=%d inserted=%d "
        "skipped_existing=%d skipped_insufficient_history=%d",
        args.pair,
        args.timeframe,
        report.feature_set_version,
        report.candles_seen,
        report.rows_inserted,
        report.rows_skipped_existing,
        report.rows_skipped_insufficient_history,
    )


if __name__ == "__main__":
    main()
