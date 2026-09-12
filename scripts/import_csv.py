#!/usr/bin/env python
"""Import a CSV file of real candle data for one asset/timeframe.

Example:
    python scripts/import_csv.py --asset EURUSD_OTC --timeframe 30s \\
        --csv data/raw/eurusd_otc_30s.csv

The CSV must have columns: timestamp, open, high, low, close.
This script never invents data — see otc_research.data.ingestion for the
exact validation/rejection rules applied.
"""

from __future__ import annotations

import argparse

from otc_research.config import load_config
from otc_research.data.ingestion import ingest
from otc_research.data.sources.csv_source import CsvDataSource
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="Path to the CSV file")
    parser.add_argument("--asset", required=True, help='e.g. "EURUSD_OTC"')
    parser.add_argument("--timeframe", required=True, help='e.g. "30s", "1m", "5m"')
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    source = CsvDataSource(args.csv)
    report = ingest(
        session,
        source,
        asset=args.asset,
        timeframe=args.timeframe,
        max_suspicious_move_pct=config.data_validation.max_suspicious_move_pct,
    )

    logger.info(
        "Ingested %s/%s from %s: seen=%d inserted=%d skipped=%d issues_logged=%d",
        args.asset,
        args.timeframe,
        source.name,
        report.candles_seen,
        report.candles_inserted,
        report.candles_skipped,
        report.issues_logged,
    )


if __name__ == "__main__":
    main()
