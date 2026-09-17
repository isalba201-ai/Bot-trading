"""Integration tests for features/pipeline.py: Candle rows in the DB ->
Feature rows in the DB. Uses the same in-memory-SQLite ``session`` fixture
as test_ingestion.py, and small synthetic candles inserted directly
(is_synthetic_test_data=True) rather than going through a real data
source, since this module only cares about DB orchestration, not data
fetching.
"""

import datetime as dt

import numpy as np

from otc_research.db.models import Candle, Feature
from otc_research.features.engine import FEATURE_NAMES, FEATURE_SET_VERSION
from otc_research.features.pipeline import compute_and_store


def _insert_synthetic_candles(session, asset="TEST_FX", timeframe="1m", n=40):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rng = np.random.default_rng(3)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, size=n))
    for i in range(n):
        c = close[i]
        o = close[i - 1] if i > 0 else c
        h = max(o, c) + 0.001
        low = min(o, c) - 0.001
        session.add(
            Candle(
                asset=asset,
                timeframe=timeframe,
                timestamp=start + dt.timedelta(minutes=i),
                open=o,
                high=h,
                low=low,
                close=c,
                source="synthetic:test",
                is_synthetic_test_data=True,
            )
        )
    session.commit()


def test_compute_and_store_inserts_expected_feature_rows(session):
    _insert_synthetic_candles(session, n=40)

    report = compute_and_store(session, "TEST_FX", "1m")

    assert report.candles_seen == 40
    assert report.rows_inserted > 0
    stored = session.query(Feature).filter_by(asset="TEST_FX", timeframe="1m").all()
    assert len(stored) == report.rows_inserted
    assert {f.name for f in stored} <= set(FEATURE_NAMES)
    assert all(f.feature_set_version == FEATURE_SET_VERSION for f in stored)


def test_rerunning_is_idempotent_no_duplicates(session):
    _insert_synthetic_candles(session, n=40)

    first = compute_and_store(session, "TEST_FX", "1m")
    second = compute_and_store(session, "TEST_FX", "1m")

    assert second.rows_inserted == 0
    assert second.rows_skipped_existing == first.rows_inserted
    total_rows = session.query(Feature).filter_by(asset="TEST_FX", timeframe="1m").count()
    assert total_rows == first.rows_inserted


def test_new_candles_only_add_new_feature_rows_not_recompute_old(session):
    _insert_synthetic_candles(session, n=40)
    first = compute_and_store(session, "TEST_FX", "1m")

    # Append candles with new, later timestamps to simulate new data
    # arriving (inserting at the same timestamps again would violate the
    # candles table's unique constraint, same as in ingestion.py).
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(minutes=40)
    for i in range(5):
        session.add(
            Candle(
                asset="TEST_FX",
                timeframe="1m",
                timestamp=start + dt.timedelta(minutes=i),
                open=1.12,
                high=1.13,
                low=1.11,
                close=1.125,
                source="synthetic:test",
                is_synthetic_test_data=True,
            )
        )
    session.commit()

    second = compute_and_store(session, "TEST_FX", "1m")

    assert second.rows_inserted > 0
    assert second.rows_skipped_existing == first.rows_inserted
    total_rows = session.query(Feature).filter_by(asset="TEST_FX", timeframe="1m").count()
    assert total_rows == first.rows_inserted + second.rows_inserted


def test_no_candles_returns_empty_report_without_error(session):
    report = compute_and_store(session, "NO_SUCH_PAIR", "1m")
    assert report.candles_seen == 0
    assert report.rows_inserted == 0
