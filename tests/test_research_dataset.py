"""Integration tests for research/dataset.py: real Phase 3 feature
computation -> point-in-time dataset builder. Synthetic candles, clearly
marked as such (is_synthetic_test_data=True) -- these tests only check
the dataset builder's plumbing and point-in-time contract, not any claim
about real market behavior.
"""

import datetime as dt

import numpy as np
import pandas as pd

from otc_research.db.models import Candle
from otc_research.features.engine import FEATURE_NAMES, FEATURE_SET_VERSION
from otc_research.features.pipeline import compute_and_store
from otc_research.research.dataset import (
    NON_FEATURE_COLUMNS,
    REAL_FOREX_DATA,
    build_dataset,
    target_columns,
)


def _insert_synthetic_candles(session, n=400, asset="TEST_FX", timeframe="1h", seed=17):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rng = np.random.default_rng(seed)
    close = 1.10 + np.cumsum(rng.normal(0, 0.001, size=n))
    for i in range(n):
        c = close[i]
        o = close[i - 1] if i > 0 else c
        h = max(o, c) + 0.0005
        low = min(o, c) - 0.0005
        session.add(
            Candle(
                asset=asset, timeframe=timeframe, timestamp=start + dt.timedelta(hours=i),
                open=o, high=h, low=low, close=c,
                source="synthetic:test", is_synthetic_test_data=True,
            )
        )
    session.commit()
    return start, close


def test_build_dataset_basic_shape_and_labels(session):
    _insert_synthetic_candles(session, n=400)
    compute_and_store(session, "TEST_FX", "1h")

    df = build_dataset(session, "TEST_FX", "1h", feature_set_version=FEATURE_SET_VERSION)

    assert len(df) > 0
    assert list(df.columns) == [*NON_FEATURE_COLUMNS, *FEATURE_NAMES, *target_columns()]
    assert (df["data_source_label"] == REAL_FOREX_DATA).all()
    assert (df["asset"] == "TEST_FX").all()
    assert (df["timeframe"] == "1h").all()
    # every feature column must be fully populated -- rows with an
    # incomplete vector are dropped, never NaN-filled
    assert df[list(FEATURE_NAMES)].isna().sum().sum() == 0
    # regime is derived from adx_14/atr_expansion_ratio, both always
    # present in a row that made it into the dataset -- so regime must be too
    assert df["regime"].notna().all()


def test_build_dataset_targets_match_manual_pnl_calculation(session):
    _insert_synthetic_candles(session, n=400)
    compute_and_store(session, "TEST_FX", "1h")
    df = build_dataset(session, "TEST_FX", "1h", feature_set_version=FEATURE_SET_VERSION, horizons=(1,))

    close = df["close_price"].to_numpy()
    for i in range(len(df) - 1):
        expected_call = 1.0 if close[i + 1] > close[i] else (0.0 if close[i + 1] < close[i] else None)
        actual = df["call_wins_1"].iloc[i]
        if expected_call is None:
            assert pd.isna(actual)
        else:
            assert actual == expected_call


def test_build_dataset_targets_are_nan_near_the_end_not_fabricated(session):
    _insert_synthetic_candles(session, n=400)
    compute_and_store(session, "TEST_FX", "1h")
    df = build_dataset(session, "TEST_FX", "1h", feature_set_version=FEATURE_SET_VERSION, horizons=(1, 5))

    # the very last row can't resolve any horizon -- no future candle exists
    assert pd.isna(df["call_wins_1"].iloc[-1])
    assert pd.isna(df["put_wins_1"].iloc[-1])
    assert pd.isna(df["call_wins_5"].iloc[-1])
    # but a row 5 candles from the end CAN resolve horizon 1, just not horizon 5
    assert pd.notna(df["call_wins_1"].iloc[-5])
    assert pd.isna(df["call_wins_5"].iloc[-1])


def test_build_dataset_call_put_mutual_exclusivity(session):
    _insert_synthetic_candles(session, n=400)
    compute_and_store(session, "TEST_FX", "1h")
    df = build_dataset(session, "TEST_FX", "1h", feature_set_version=FEATURE_SET_VERSION, horizons=(1,))

    resolved = df.dropna(subset=["call_wins_1", "put_wins_1"])
    # never both 1 (a tie would make both NaN, never both winning)
    assert not ((resolved["call_wins_1"] == 1.0) & (resolved["put_wins_1"] == 1.0)).any()
    # exactly one of the two must be 1 for every resolved (non-tie) row
    assert ((resolved["call_wins_1"] + resolved["put_wins_1"]) == 1.0).all()


def test_build_dataset_is_point_in_time_safe_for_features(session):
    """Truncating the candle history must not change any FEATURE column
    already computed for an earlier prefix -- the same invariant
    test_feature_engine.py checks at the indicator level, verified here
    at the dataset-builder level too.
    """
    _insert_synthetic_candles(session, n=400)
    compute_and_store(session, "TEST_FX", "1h")
    full_df = build_dataset(session, "TEST_FX", "1h", feature_set_version=FEATURE_SET_VERSION)

    # Build a second, independent DB with only the first 300 candles.
    from otc_research.db.session import get_engine, get_session_factory, init_db

    truncated_engine = get_engine("sqlite:///:memory:")
    init_db(truncated_engine)
    truncated_session = get_session_factory(truncated_engine)()
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rng = np.random.default_rng(17)
    close = 1.10 + np.cumsum(rng.normal(0, 0.001, size=400))
    for i in range(300):
        c = close[i]
        o = close[i - 1] if i > 0 else c
        h = max(o, c) + 0.0005
        low = min(o, c) - 0.0005
        truncated_session.add(
            Candle(
                asset="TEST_FX", timeframe="1h", timestamp=start + dt.timedelta(hours=i),
                open=o, high=h, low=low, close=c,
                source="synthetic:test", is_synthetic_test_data=True,
            )
        )
    truncated_session.commit()
    compute_and_store(truncated_session, "TEST_FX", "1h")
    truncated_df = build_dataset(
        truncated_session, "TEST_FX", "1h", feature_set_version=FEATURE_SET_VERSION
    )

    shared_ts = truncated_df["timestamp"].iloc[-1]
    full_row = full_df[full_df["timestamp"] == shared_ts].iloc[0]
    truncated_row = truncated_df[truncated_df["timestamp"] == shared_ts].iloc[0]
    for name in FEATURE_NAMES:
        assert np.isclose(full_row[name], truncated_row[name]), name


def test_build_dataset_returns_empty_frame_for_unknown_pair(session):
    df = build_dataset(session, "NO_SUCH_PAIR", "1h", feature_set_version=FEATURE_SET_VERSION)
    assert len(df) == 0
    assert list(df.columns) == [*NON_FEATURE_COLUMNS, *FEATURE_NAMES, *target_columns()]
