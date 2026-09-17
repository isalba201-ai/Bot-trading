import datetime as dt
import random

import pytest

from otc_research.backtest.execution import ExecutionScenario, OPTIMISTIC
from otc_research.backtest.simulator import SimCandle, simulate


class _AlwaysCall:
    code = None
    label = "always_call_test_strategy"
    expiry_seconds = 60
    required_features = frozenset({"x"})

    def decide(self, features):
        return "CALL"


class _NeedsMissingFeature:
    code = None
    label = "needs_missing_feature_test_strategy"
    expiry_seconds = 60
    required_features = frozenset({"does_not_exist"})

    def decide(self, features):  # pragma: no cover - never reached
        return "CALL"


def _make_candles(n: int) -> list[SimCandle]:
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    candles = []
    for i in range(n):
        price = 1.00 + i * 0.01
        candles.append(
            SimCandle(
                timestamp=start + dt.timedelta(minutes=i),
                open=price,
                high=price + 0.001,
                low=price - 0.001,
                close=price,
            )
        )
    return candles


def _features_for(candles: list[SimCandle]) -> dict:
    return {c.timestamp: {"x": 1.0} for c in candles}


def test_rising_market_produces_wins_for_call_strategy():
    candles = _make_candles(10)
    features = _features_for(candles)

    trades = simulate(_AlwaysCall(), candles, features, timeframe_seconds=60, scenario=OPTIMISTIC)

    assert len(trades) == 10  # every candle produced a signal
    voided = [t for t in trades if t.result == "VOID"]
    resolved = [t for t in trades if t.result != "VOID"]
    assert len(voided) == 2  # last two candles can't resolve (no data left)
    assert all(t.void_reason == "insufficient_data" for t in voided)
    assert len(resolved) == 8
    assert all(t.result == "WIN" for t in resolved)  # strictly rising market


def test_first_trade_uses_next_candle_open_and_following_close():
    candles = _make_candles(10)
    features = _features_for(candles)

    trades = simulate(_AlwaysCall(), candles, features, timeframe_seconds=60, scenario=OPTIMISTIC)
    first = trades[0]

    assert first.entry_price == pytest.approx(candles[1].open)
    assert first.exit_price == pytest.approx(candles[2].close)
    assert first.entry_time == candles[1].timestamp
    assert first.exit_time == candles[2].timestamp


def test_entry_delay_and_slippage_can_flip_a_win_into_a_loss():
    candles = _make_candles(10)
    features = _features_for(candles)
    harsh = ExecutionScenario(
        name="harsh", entry_delay_candles=1, signal_drop_probability=0.0, slippage_pct=1.0
    )

    trades = simulate(_AlwaysCall(), candles, features, timeframe_seconds=60, scenario=harsh)
    first = trades[0]

    # entry pushed to candles[2] (1 extra delay candle), 1% adverse slippage
    # on a market only moving ~1% per candle wipes out the edge.
    expected_entry = candles[2].open * 1.01
    assert first.entry_price == pytest.approx(expected_entry)
    assert first.result == "LOSS"


def test_signal_drop_probability_one_voids_every_signal():
    candles = _make_candles(10)
    features = _features_for(candles)
    always_drop = ExecutionScenario(
        name="always_drop", entry_delay_candles=0, signal_drop_probability=1.0, slippage_pct=0.0
    )

    trades = simulate(
        _AlwaysCall(), candles, features, timeframe_seconds=60, scenario=always_drop,
        rng=random.Random(0),
    )

    assert len(trades) == 10
    assert all(t.result == "VOID" and t.void_reason == "signal_dropped" for t in trades)


def test_missing_required_feature_produces_no_signal():
    candles = _make_candles(10)
    features = _features_for(candles)

    trades = simulate(
        _NeedsMissingFeature(), candles, features, timeframe_seconds=60, scenario=OPTIMISTIC
    )

    assert trades == []


def test_expiry_not_a_whole_number_of_candles_raises():
    candles = _make_candles(10)
    features = _features_for(candles)

    class _BadExpiry(_AlwaysCall):
        expiry_seconds = 90  # not a multiple of a 60s timeframe

    with pytest.raises(ValueError):
        simulate(_BadExpiry(), candles, features, timeframe_seconds=60, scenario=OPTIMISTIC)
