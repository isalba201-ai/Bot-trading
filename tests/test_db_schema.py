import datetime as dt

import pytest
from sqlalchemy.exc import IntegrityError

from otc_research.db.models import Candle, Hypothesis


def _ts(i: int = 0) -> dt.datetime:
    return dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(seconds=30 * i)


def test_insert_candle(session):
    session.add(
        Candle(
            asset="EURUSD_OTC",
            timeframe="30s",
            timestamp=_ts(0),
            open=1.1000,
            high=1.1005,
            low=1.0998,
            close=1.1002,
            source="csv:test.csv",
        )
    )
    session.commit()

    stored = session.query(Candle).one()
    assert stored.asset == "EURUSD_OTC"
    assert stored.is_synthetic_test_data is False


def test_duplicate_asset_timeframe_timestamp_rejected(session):
    kwargs = dict(
        asset="EURUSD_OTC",
        timeframe="30s",
        timestamp=_ts(0),
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.0,
        source="csv:test.csv",
    )
    session.add(Candle(**kwargs))
    session.commit()

    session.add(Candle(**kwargs))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()


def test_hypothesis_code_unique(session):
    session.add(Hypothesis(code="H1", name="continuation", description="test"))
    session.commit()

    session.add(Hypothesis(code="H1", name="dup", description="test"))
    with pytest.raises(IntegrityError):
        session.commit()
    session.rollback()
