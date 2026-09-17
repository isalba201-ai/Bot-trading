"""Phase 3 orchestration: stored Candle rows -> engine.compute_features ->
stored Feature rows.

Reads only from the ``candles`` table (never a live data source directly —
see ARCHITECTURE.md, so validation can never be silently bypassed) and
writes to ``features``. Idempotent like data/ingestion.py: a
(asset, timeframe, timestamp, feature_set_version, name) row that already
exists is skipped, never overwritten. If the computation logic changes,
bump FEATURE_SET_VERSION instead of mutating history in place.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import pandas as pd
from sqlalchemy.orm import Session

from otc_research.db.models import Candle, Feature
from otc_research.features.engine import FEATURE_SET_VERSION, compute_features


def _normalize_ts(ts: dt.datetime) -> dt.datetime:
    """Same rationale as data/ingestion.py: SQLite drops tzinfo on
    round-trip, so comparisons go through this to avoid aware-vs-naive
    mismatches silently breaking duplicate detection.
    """
    if ts.tzinfo is not None:
        ts = ts.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return ts


@dataclass
class FeatureComputationReport:
    asset: str
    timeframe: str
    feature_set_version: str
    candles_seen: int
    rows_inserted: int
    rows_skipped_existing: int
    rows_skipped_insufficient_history: int


def compute_and_store(
    session: Session,
    asset: str,
    timeframe: str,
    *,
    feature_set_version: str = FEATURE_SET_VERSION,
) -> FeatureComputationReport:
    candles = (
        session.query(Candle)
        .filter(Candle.asset == asset, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.asc())
        .all()
    )

    if not candles:
        return FeatureComputationReport(
            asset=asset,
            timeframe=timeframe,
            feature_set_version=feature_set_version,
            candles_seen=0,
            rows_inserted=0,
            rows_skipped_existing=0,
            rows_skipped_insufficient_history=0,
        )

    df = pd.DataFrame(
        {
            "timestamp": [c.timestamp for c in candles],
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
        }
    )
    features_df = compute_features(df)

    existing = {
        (_normalize_ts(ts), name)
        for (ts, name) in session.query(Feature.timestamp, Feature.name).filter(
            Feature.asset == asset,
            Feature.timeframe == timeframe,
            Feature.feature_set_version == feature_set_version,
        )
    }

    inserted = 0
    skipped_existing = 0
    skipped_insufficient_history = 0

    for i, candle in enumerate(candles):
        ts_key = _normalize_ts(candle.timestamp)
        row = features_df.iloc[i]
        for name in features_df.columns:
            value = row[name]
            if pd.isna(value):
                # Not enough history yet for this indicator's lookback at
                # this point in time — never stored as a fabricated 0/None.
                skipped_insufficient_history += 1
                continue
            key = (ts_key, name)
            if key in existing:
                skipped_existing += 1
                continue
            session.add(
                Feature(
                    asset=asset,
                    timeframe=timeframe,
                    timestamp=candle.timestamp,
                    feature_set_version=feature_set_version,
                    name=name,
                    value=float(value),
                )
            )
            existing.add(key)
            inserted += 1

    session.commit()

    return FeatureComputationReport(
        asset=asset,
        timeframe=timeframe,
        feature_set_version=feature_set_version,
        candles_seen=len(candles),
        rows_inserted=inserted,
        rows_skipped_existing=skipped_existing,
        rows_skipped_insufficient_history=skipped_insufficient_history,
    )
