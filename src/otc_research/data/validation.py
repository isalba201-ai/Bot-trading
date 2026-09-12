"""Data quality validation.

Every function here inspects a sequence of candles and *reports* problems.
None of them repair, interpolate, drop-silently, or otherwise invent data.
The caller (ingestion.py) decides what to do with the reported issues —
currently: log them all, and refuse to insert anything that is internally
contradictory (duplicate timestamp, disordered, or an impossible OHLC
relationship), while still recording *why* it was refused.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

from otc_research.data.sources.base import RawCandle

_TIMEFRAME_SECONDS = {
    "30s": 30,
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "1h": 3600,
}


def timeframe_to_seconds(timeframe: str) -> int:
    try:
        return _TIMEFRAME_SECONDS[timeframe]
    except KeyError as exc:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}; known values: {sorted(_TIMEFRAME_SECONDS)}"
        ) from exc


@dataclass(frozen=True)
class DataQualityFinding:
    issue_type: str
    detail: str
    candle_timestamp: dt.datetime | None
    source: str | None = None


def find_duplicate_timestamps(candles: Sequence[RawCandle]) -> list[DataQualityFinding]:
    seen: set[dt.datetime] = set()
    findings = []
    for c in candles:
        if c.timestamp in seen:
            findings.append(
                DataQualityFinding(
                    issue_type="duplicate_timestamp",
                    detail=f"Duplicate candle timestamp for {c.asset}/{c.timeframe}",
                    candle_timestamp=c.timestamp,
                )
            )
        seen.add(c.timestamp)
    return findings


def find_out_of_order(candles: Sequence[RawCandle]) -> list[DataQualityFinding]:
    findings = []
    prev_ts: dt.datetime | None = None
    for c in candles:
        if prev_ts is not None and c.timestamp < prev_ts:
            findings.append(
                DataQualityFinding(
                    issue_type="out_of_order",
                    detail=f"Candle timestamp {c.timestamp} arrived after {prev_ts}",
                    candle_timestamp=c.timestamp,
                )
            )
        else:
            prev_ts = c.timestamp
    return findings


def find_gaps(candles: Sequence[RawCandle], timeframe: str) -> list[DataQualityFinding]:
    """Flag any interval between consecutive candles that isn't exactly one bar.

    Assumes candles are already sorted ascending; run after
    ``find_out_of_order`` / sort, not before.
    """
    step = dt.timedelta(seconds=timeframe_to_seconds(timeframe))
    findings = []
    sorted_candles = sorted(candles, key=lambda c: c.timestamp)
    for prev, cur in zip(sorted_candles, sorted_candles[1:]):
        delta = cur.timestamp - prev.timestamp
        if delta > step:
            missing_bars = int(delta / step) - 1
            findings.append(
                DataQualityFinding(
                    issue_type="gap",
                    detail=(
                        f"Gap of {delta} between {prev.timestamp} and {cur.timestamp} "
                        f"(~{missing_bars} missing bar(s) at {timeframe})"
                    ),
                    candle_timestamp=cur.timestamp,
                )
            )
    return findings


def find_impossible_values(candles: Sequence[RawCandle]) -> list[DataQualityFinding]:
    findings = []
    for c in candles:
        reasons = []
        if c.high < c.low:
            reasons.append("high < low")
        if c.high < c.open or c.high < c.close:
            reasons.append("high is not the max of open/close")
        if c.low > c.open or c.low > c.close:
            reasons.append("low is not the min of open/close")
        if any(v <= 0 for v in (c.open, c.high, c.low, c.close)):
            reasons.append("non-positive price")
        if reasons:
            findings.append(
                DataQualityFinding(
                    issue_type="impossible_value",
                    detail=f"{c.asset}/{c.timeframe} @ {c.timestamp}: {'; '.join(reasons)}",
                    candle_timestamp=c.timestamp,
                )
            )
    return findings


def find_suspicious_moves(
    candles: Sequence[RawCandle], max_pct: float
) -> list[DataQualityFinding]:
    """Flag (but do not reject) candle-to-candle closes that jump more than
    ``max_pct`` percent. OTC synthetic feeds can legitimately be jumpy, so
    this is informational, not a rejection criterion.
    """
    findings = []
    sorted_candles = sorted(candles, key=lambda c: c.timestamp)
    for prev, cur in zip(sorted_candles, sorted_candles[1:]):
        if prev.close == 0:
            continue
        pct_move = abs(cur.close / prev.close - 1) * 100
        if pct_move > max_pct:
            findings.append(
                DataQualityFinding(
                    issue_type="suspicious_move",
                    detail=(
                        f"{cur.asset}/{cur.timeframe} close moved {pct_move:.3f}% "
                        f"between {prev.timestamp} and {cur.timestamp}"
                    ),
                    candle_timestamp=cur.timestamp,
                )
            )
    return findings


def find_source_changes(
    candles: Sequence[RawCandle], previous_source: str | None, new_source: str
) -> list[DataQualityFinding]:
    """Flag when new candles for an asset/timeframe come from a different
    provenance than what is already stored. Not an error by itself, but it
    must be visible/auditable since it can change data characteristics
    mid-history.
    """
    if previous_source is not None and previous_source != new_source and candles:
        return [
            DataQualityFinding(
                issue_type="source_change",
                detail=f"Source changed from {previous_source!r} to {new_source!r}",
                candle_timestamp=candles[0].timestamp,
                source=new_source,
            )
        ]
    return []
