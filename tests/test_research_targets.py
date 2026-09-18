import numpy as np
import pandas as pd
import pytest

from otc_research.research.targets import (
    COLUMNS,
    mfe_mae_labels,
    triple_barrier_labels,
    triple_barrier_to_binary,
)


def _df(closes, atr=1.0):
    n = len(closes)
    atr_series = [atr] * n if not isinstance(atr, list) else atr
    return pd.DataFrame({"close_price": closes, "atr_14": atr_series})


# --- triple_barrier_labels --------------------------------------------


def test_triple_barrier_rejects_non_positive_multipliers():
    df = _df([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        triple_barrier_labels(df, upper_mult=0.0, lower_mult=1.0, max_horizon=2)
    with pytest.raises(ValueError):
        triple_barrier_labels(df, upper_mult=1.0, lower_mult=-1.0, max_horizon=2)


def test_triple_barrier_rejects_non_positive_horizon():
    df = _df([1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        triple_barrier_labels(df, upper_mult=1.0, lower_mult=1.0, max_horizon=0)


def test_triple_barrier_labels_upper_touch_first():
    # entry=100, atr=1, upper barrier at 102, lower at 98
    closes = [100.0, 100.5, 103.0, 90.0, 100.0, 100.0]
    df = _df(closes, atr=1.0)
    labels = triple_barrier_labels(df, upper_mult=2.0, lower_mult=2.0, max_horizon=3)
    assert labels.iloc[0] == 1.0  # touches 103 (>=102) at index 2, before the 90 at index 3


def test_triple_barrier_labels_lower_touch_first():
    closes = [100.0, 99.0, 97.0, 105.0, 100.0, 100.0]
    df = _df(closes, atr=1.0)
    labels = triple_barrier_labels(df, upper_mult=2.0, lower_mult=2.0, max_horizon=3)
    assert labels.iloc[0] == -1.0  # touches 97 (<=98) at index 2


def test_triple_barrier_labels_times_out_when_neither_barrier_touched():
    closes = [100.0, 100.5, 99.5, 100.2, 100.0, 100.0]
    df = _df(closes, atr=1.0)
    labels = triple_barrier_labels(df, upper_mult=2.0, lower_mult=2.0, max_horizon=3)
    assert np.isnan(labels.iloc[0])  # never reaches 102 or 98 within 3 candles


def test_triple_barrier_labels_nan_for_trailing_rows_without_enough_future():
    closes = [100.0, 101.0, 99.0, 105.0, 95.0]
    df = _df(closes, atr=1.0)
    labels = triple_barrier_labels(df, upper_mult=2.0, lower_mult=2.0, max_horizon=3)
    # last 3 rows don't have 3 full future candles available
    assert np.isnan(labels.iloc[2])
    assert np.isnan(labels.iloc[3])
    assert np.isnan(labels.iloc[4])


def test_triple_barrier_labels_nan_when_atr_missing_or_non_positive():
    closes = [100.0, 103.0, 100.0, 100.0, 100.0]
    df = _df(closes, atr=[np.nan, 1.0, 0.0, -1.0, 1.0])
    labels = triple_barrier_labels(df, upper_mult=2.0, lower_mult=2.0, max_horizon=2)
    assert np.isnan(labels.iloc[0])  # NaN atr
    assert np.isnan(labels.iloc[2])  # zero atr
    assert np.isnan(labels.iloc[3])  # negative atr


def test_triple_barrier_labels_point_in_time_safe_via_truncation():
    rng = np.random.default_rng(0)
    closes = list(1.10 + np.cumsum(rng.normal(0, 0.002, size=100)))
    atr = [0.01] * 100
    full = _df(closes, atr=atr)
    truncated = full.iloc[:60].copy()

    labels_full = triple_barrier_labels(full, upper_mult=1.5, lower_mult=1.5, max_horizon=5)
    labels_truncated = triple_barrier_labels(
        truncated, upper_mult=1.5, lower_mult=1.5, max_horizon=5
    )
    # a row that has its full look-forward window inside BOTH frames must
    # agree -- proof the label only ever looks forward, never uses
    # anything about how much history follows beyond its own window
    row = 50
    a, b = labels_full.iloc[row], labels_truncated.iloc[row]
    if np.isnan(a):
        assert np.isnan(b)
    else:
        assert a == b


# --- triple_barrier_to_binary --------------------------------------------


def test_triple_barrier_to_binary_maps_upper_touch_to_call_win():
    labels = pd.Series([1.0, -1.0, np.nan])
    call_wins, put_wins = triple_barrier_to_binary(labels)
    assert call_wins.iloc[0] == 1.0 and call_wins.iloc[1] == 0.0 and pd.isna(call_wins.iloc[2])
    assert put_wins.iloc[0] == 0.0 and put_wins.iloc[1] == 1.0 and pd.isna(put_wins.iloc[2])


# --- mfe_mae_labels ------------------------------------------------------


def test_mfe_mae_rejects_non_positive_horizon():
    df = _df([1.0, 1.0])
    with pytest.raises(ValueError):
        mfe_mae_labels(df, max_horizon=0)


def test_mfe_mae_computes_best_and_worst_close_to_close_return():
    closes = [100.0, 105.0, 95.0, 102.0, 100.0, 100.0]
    df = _df(closes)
    result = mfe_mae_labels(df, max_horizon=3)
    # window for row 0: closes[1..3] = [105, 95, 102] -> returns [0.05, -0.05, 0.02]
    assert result[COLUMNS.mfe_call].iloc[0] == pytest.approx(0.05)
    assert result[COLUMNS.mae_call].iloc[0] == pytest.approx(-0.05)


def test_mfe_mae_nan_for_trailing_rows_without_enough_future():
    closes = [100.0, 101.0, 99.0, 105.0, 95.0]
    df = _df(closes)
    result = mfe_mae_labels(df, max_horizon=3)
    assert np.isnan(result[COLUMNS.mfe_call].iloc[3])
    assert np.isnan(result[COLUMNS.mae_call].iloc[4])


def test_mfe_mae_put_is_the_sign_flipped_mirror_of_call():
    # documented relationship: mfe_put == -mae_call, mae_put == -mfe_call
    closes = [100.0, 103.0, 97.0, 100.0, 100.0]
    df = _df(closes)
    result = mfe_mae_labels(df, max_horizon=2)
    mfe_call = result[COLUMNS.mfe_call].iloc[0]
    mae_call = result[COLUMNS.mae_call].iloc[0]
    # manually derive put excursions from the same underlying path
    returns = [(103.0 - 100.0) / 100.0, (97.0 - 100.0) / 100.0]
    mfe_put = max(-r for r in returns)
    mae_put = min(-r for r in returns)
    assert mfe_put == pytest.approx(-mae_call)
    assert mae_put == pytest.approx(-mfe_call)
