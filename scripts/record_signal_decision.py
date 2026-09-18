#!/usr/bin/env python
"""Record a person's manual decision about a signal (approved plan point
10). This is the one write path for a ``SignalDecision`` row -- there is
no UI yet, so this CLI is it.

Example:
    python scripts/record_signal_decision.py --signal-id 42 \\
        --decision TOOK_TRADE --entry-price-actual 1.0842 \\
        --payout-observed 0.82
"""

from __future__ import annotations

import argparse
import datetime as dt

from otc_research.config import load_config
from otc_research.db.session import get_engine, get_session_factory, init_db
from otc_research.signals.decisions import VALID_DECISIONS, record_decision
from otc_research.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--signal-id", type=int, required=True)
    parser.add_argument("--decision", required=True, choices=VALID_DECISIONS)
    parser.add_argument(
        "--decided-at", default=None,
        help="ISO 8601 timestamp; defaults to now (UTC).",
    )
    parser.add_argument("--entry-price-actual", type=float, default=None)
    parser.add_argument("--payout-observed", type=float, default=None)
    parser.add_argument(
        "--entry-time-actual", default=None, help="ISO 8601 timestamp, if known."
    )
    parser.add_argument("--expiry-used-seconds", type=int, default=None)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    decided_at = (
        dt.datetime.fromisoformat(args.decided_at)
        if args.decided_at
        else dt.datetime.now(dt.timezone.utc)
    )
    entry_time_actual = (
        dt.datetime.fromisoformat(args.entry_time_actual) if args.entry_time_actual else None
    )

    row = record_decision(
        session,
        args.signal_id,
        args.decision,
        decided_at,
        entry_price_actual=args.entry_price_actual,
        payout_observed=args.payout_observed,
        entry_time_actual=entry_time_actual,
        expiry_used_seconds=args.expiry_used_seconds,
    )

    logger.info(
        "signal_id=%d decision=%s (requested=%s) decided_at=%s",
        args.signal_id, row.decision, args.decision, row.decided_at,
    )


if __name__ == "__main__":
    main()
