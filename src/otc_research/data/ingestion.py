"""Ingestion: DataSource -> validation -> storage.

Policy, deliberately strict:

* Duplicate timestamps, out-of-order input, and internally-impossible OHLC
  values are logged as issues AND the offending candle is skipped (not
  inserted). Silently accepting contradictory data would poison every
  feature/backtest built on top of it.
* Gaps and suspicious moves are logged but do NOT block insertion of the
  surrounding valid candles — a gap is a fact about the world (the source
  didn't have that bar), not a defect in the candle you did receive.
* A candle that already exists (same asset/timeframe/timestamp) is
  skipped, not overwritten, unless ``allow_reingest`` is explicitly set —
  history should not change under you silently.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from sqlalchemy.orm import Session

from otc_research.data.sources.base import DataSource, RawCandle
from otc_research.data.validation import (
    find_duplicate_timestamps,
    find_gaps,
    find_impossible_values,
    find_out_of_order,
    find_source_changes,
    find_suspicious_moves,
)
from otc_research.db.models import Candle, DataQualityIssue


def _normalize_ts(ts: dt.datetime) -> dt.datetime:
    """SQLite drops tzinfo on round-trip, so all timestamp comparisons in this
    module go through this to avoid aware-vs-naive mismatches silently
    breaking duplicate detection (which would otherwise let the same candle
    be inserted twice).
    """
    if ts.tzinfo is not None:
        ts = ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return ts


@dataclass
class IngestionReport:
    asset: str
    timeframe: str
    candles_seen: int
    candles_inserted: int
    candles_skipped: int
    issues_logged: int


def ingest(
    session: Session,
    source: DataSource,
    asset: str,
    timeframe: str,
    *,
    max_suspicious_move_pct: float = 5.0,
    is_synthetic_test_data: bool = False,
) -> IngestionReport:
    raw_candles: list[RawCandle] = list(source.fetch(asset, timeframe))

    all_findings = []
    all_findings += find_duplicate_timestamps(raw_candles)
    all_findings += find_out_of_order(raw_candles)
    all_findings += find_impossible_values(raw_candles)
    all_findings += find_gaps(raw_candles, timeframe)
    all_findings += find_suspicious_moves(raw_candles, max_suspicious_move_pct)

    existing_source = (
        session.query(Candle.source)
        .filter(Candle.asset == asset, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.desc())
        .limit(1)
        .scalar()
    )
    all_findings += find_source_changes(raw_candles, existing_source, source.name)

    for finding in all_findings:
        session.add(
            DataQualityIssue(
                asset=asset,
                timeframe=timeframe,
                issue_type=finding.issue_type,
                detail=finding.detail,
                candle_timestamp=finding.candle_timestamp,
                source=finding.source or source.name,
            )
        )

    reject_timestamps = {
        _normalize_ts(f.candle_timestamp)
        for f in all_findings
        if f.issue_type in ("duplicate_timestamp", "out_of_order", "impossible_value")
    }

    existing_timestamps = {
        _normalize_ts(ts)
        for (ts,) in session.query(Candle.timestamp).filter(
            Candle.asset == asset, Candle.timeframe == timeframe
        )
    }

    inserted = 0
    skipped = 0
    seen_this_batch: set = set()
    for c in raw_candles:
        key = _normalize_ts(c.timestamp)
        if key in reject_timestamps or key in existing_timestamps or key in seen_this_batch:
            skipped += 1
            continue
        session.add(
            Candle(
                asset=c.asset,
                timeframe=c.timeframe,
                timestamp=c.timestamp,
                open=c.open,
                high=c.high,
                low=c.low,
                close=c.close,
                source=source.name,
                is_synthetic_test_data=is_synthetic_test_data,
            )
        )
        seen_this_batch.add(key)
        inserted += 1

    session.commit()

    return IngestionReport(
        asset=asset,
        timeframe=timeframe,
        candles_seen=len(raw_candles),
        candles_inserted=inserted,
        candles_skipped=skipped,
        issues_logged=len(all_findings),
    )
