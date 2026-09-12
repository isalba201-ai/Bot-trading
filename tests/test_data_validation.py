"""Validators are tested against hand-built synthetic candle lists.

These candles are NOT market data of any kind — they exist only to exercise
one validation rule at a time and must never be confused with, or used as
a substitute for, real Forex history.
"""

import datetime as dt

from otc_research.data.sources.base import RawCandle
from otc_research.data.validation import (
    find_duplicate_timestamps,
    find_gaps,
    find_impossible_values,
    find_out_of_order,
    find_source_changes,
    find_suspicious_moves,
    timeframe_to_seconds,
)

BASE = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)


def _candle(seconds_offset: int, o=1.0, h=1.01, l=0.99, c=1.0) -> RawCandle:
    return RawCandle(
        asset="TEST_FX",
        timeframe="1m",
        timestamp=BASE + dt.timedelta(seconds=seconds_offset),
        open=o,
        high=h,
        low=l,
        close=c,
    )


def test_timeframe_to_seconds():
    assert timeframe_to_seconds("1m") == 60
    assert timeframe_to_seconds("5m") == 300


def test_find_duplicate_timestamps():
    candles = [_candle(0), _candle(0), _candle(60)]
    findings = find_duplicate_timestamps(candles)
    assert len(findings) == 1
    assert findings[0].issue_type == "duplicate_timestamp"


def test_find_out_of_order():
    candles = [_candle(60), _candle(0)]
    findings = find_out_of_order(candles)
    assert len(findings) == 1
    assert findings[0].issue_type == "out_of_order"


def test_find_gaps_detects_missing_bar():
    # 1m timeframe but candles are 180s apart -> 2 missing bars
    candles = [_candle(0), _candle(180)]
    findings = find_gaps(candles, "1m")
    assert len(findings) == 1
    assert "2 missing bar" in findings[0].detail


def test_find_gaps_no_gap_when_contiguous():
    candles = [_candle(0), _candle(60), _candle(120)]
    findings = find_gaps(candles, "1m")
    assert findings == []


def test_find_impossible_values_high_below_low():
    candles = [_candle(0, o=1.0, h=0.5, l=0.9, c=1.0)]
    findings = find_impossible_values(candles)
    assert len(findings) == 1
    assert "high < low" in findings[0].detail


def test_find_impossible_values_negative_price():
    candles = [_candle(0, o=-1.0, h=1.0, l=-2.0, c=0.5)]
    findings = find_impossible_values(candles)
    assert any("non-positive price" in f.detail for f in findings)


def test_find_impossible_values_accepts_valid_candle():
    candles = [_candle(0, o=1.0, h=1.05, l=0.95, c=1.02)]
    assert find_impossible_values(candles) == []


def test_find_suspicious_moves():
    candles = [_candle(0, c=1.0), _candle(60, c=2.0)]  # +100% close-to-close
    findings = find_suspicious_moves(candles, max_pct=5.0)
    assert len(findings) == 1
    assert findings[0].issue_type == "suspicious_move"


def test_find_suspicious_moves_within_threshold_not_flagged():
    candles = [_candle(0, c=1.0), _candle(60, c=1.01)]  # +1%
    assert find_suspicious_moves(candles, max_pct=5.0) == []


def test_find_source_changes_flags_new_source():
    candles = [_candle(0)]
    findings = find_source_changes(candles, previous_source="csv:old.csv", new_source="csv:new.csv")
    assert len(findings) == 1
    assert findings[0].issue_type == "source_change"


def test_find_source_changes_no_flag_when_same_source():
    candles = [_candle(0)]
    assert find_source_changes(candles, previous_source="csv:a.csv", new_source="csv:a.csv") == []


def test_find_source_changes_no_flag_when_no_previous_source():
    candles = [_candle(0)]
    assert find_source_changes(candles, previous_source=None, new_source="csv:a.csv") == []
