import datetime as dt

import numpy as np
import pytest

from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import Candle
from otc_research.features.pipeline import compute_and_store
from otc_research.research.decomposition import (
    A_NO_PREDICTABILITY,
    B_TOO_SMALL_FOR_PAYOUT,
    C_EXECUTION_DESTROYS_IT,
    INSUFFICIENT_DATA,
    SURVIVES_LADDER,
    DecompositionRung,
    classify,
    delay_only_scenario,
    run_decomposition,
    slippage_only_scenario,
)
from otc_research.research.discovery import Condition

PAYOUT = 0.85  # break_even_win_rate ~= 0.5405


def _rung(name, sample_size, win_rate, ci_low=None):
    be = 1 / (1 + PAYOUT)
    margin = win_rate - be if win_rate is not None else None
    return DecompositionRung(
        name=name, sample_size=sample_size, win_rate=win_rate, win_rate_ci_low=ci_low,
        break_even_win_rate=be, margin_over_break_even=margin,
        expectancy=(win_rate * PAYOUT - (1 - win_rate)) if win_rate is not None else None,
    )


# --- classify() pure logic -------------------------------------------


def test_classify_insufficient_data_when_raw_sample_too_small():
    rungs = [_rung("raw", 5, 0.9)]
    assert classify(rungs) == INSUFFICIENT_DATA


def test_classify_a_when_raw_is_indistinguishable_from_coin_flip():
    rungs = [_rung("raw", 500, 0.505), _rung("realistic", 400, 0.48)]
    assert classify(rungs) == A_NO_PREDICTABILITY


def test_classify_survives_ladder_when_realistic_clears_margin():
    rungs = [
        _rung("raw", 500, 0.75),
        _rung("optimistic", 480, 0.70),
        _rung("realistic", 400, 0.62),
    ]
    assert classify(rungs) == SURVIVES_LADDER


def test_classify_c_when_optimistic_clears_but_realistic_does_not():
    rungs = [
        _rung("raw", 500, 0.75),
        _rung("optimistic", 480, 0.70),  # clears margin (0.70 - 0.5405 = 0.16 > 0.03)
        _rung("realistic", 400, 0.50),  # collapses under friction
    ]
    assert classify(rungs) == C_EXECUTION_DESTROYS_IT


def test_classify_c_when_even_optimistic_never_clears_despite_raw_skew():
    # raw shows a real skew, but the real simulator's own entry/exit
    # mechanics already erase it at zero friction (the label-vs-simulator
    # timing mismatch the module docstring describes)
    rungs = [
        _rung("raw", 500, 0.75),
        _rung("optimistic", 480, 0.50),
        _rung("realistic", 400, 0.45),
    ]
    assert classify(rungs) == C_EXECUTION_DESTROYS_IT


def test_classify_b_when_skew_survives_but_margin_never_clears_threshold():
    # small, consistent margin at every rung (never collapses, never large
    # enough to clear min_margin either) -- real but too small vs. payout
    rungs = [
        _rung("raw", 500, 0.555),
        _rung("optimistic", 480, 0.55),
        _rung("realistic", 400, 0.545),  # barely above break-even (0.5405), margin < 0.03 throughout
    ]
    assert classify(rungs) == B_TOO_SMALL_FOR_PAYOUT


def test_classify_respects_custom_thresholds():
    rungs = [_rung("raw", 500, 0.60), _rung("realistic", 400, 0.545)]
    # with a much smaller required margin, this now survives
    assert classify(rungs, min_margin=0.001) == SURVIVES_LADDER


# --- scenario builders -------------------------------------------------


def test_delay_only_scenario_zeroes_out_other_friction():
    s = delay_only_scenario(3)
    assert s.entry_delay_candles == 3
    assert s.signal_drop_probability == 0.0
    assert s.slippage_pct == 0.0
    assert s.name == "delay_only_3"


def test_slippage_only_scenario_zeroes_out_delay():
    s = slippage_only_scenario(0.02)
    assert s.entry_delay_candles == 0
    assert s.signal_drop_probability == 0.0
    assert s.slippage_pct == 0.02


# --- run_decomposition integration (real DB + engine) ----------------


def _insert_deterministic_uptrend_candles(session, n=400, asset="TEST_FX", timeframe="1m"):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    closes = [1.10000]
    for i in range(1, n):
        closes.append(closes[-1] * 1.0008)  # steady, deterministic uptrend
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h = max(o, c) + 1e-6
        low = min(o, c) - 1e-6
        session.add(
            Candle(
                asset=asset, timeframe=timeframe, timestamp=start + dt.timedelta(minutes=i),
                open=o, high=h, low=low, close=c, source="synthetic:test", is_synthetic_test_data=True,
            )
        )
    session.commit()


def _backtest_config() -> BacktestConfig:
    return BacktestConfig(
        train_fraction=0.6, validation_fraction=0.2,
        realistic=ExecutionScenarioConfig(entry_delay_candles=1, signal_drop_probability=0.0, slippage_pct=0.01),
        pessimistic=ExecutionScenarioConfig(entry_delay_candles=2, signal_drop_probability=0.0, slippage_pct=0.03),
    )


def test_run_decomposition_produces_every_rung_and_a_classification(session):
    _insert_deterministic_uptrend_candles(session, n=400)
    feature_report = compute_and_store(session, "TEST_FX", "1m")

    condition = Condition(parts=(("rsi_14", 0.0, 100.0),))  # matches almost every candle
    result = run_decomposition(
        session, condition, "CALL", 60, "TEST_FX", "1m", _backtest_config(),
        feature_set_version=feature_report.feature_set_version,
        payout=PAYOUT, raw_win_rate=0.95, raw_sample_size=200,
    )

    rung_names = [r.name for r in result.rungs]
    assert rung_names[0] == "raw"
    assert "optimistic" in rung_names
    assert "delay_only_1" in rung_names
    assert "delay_only_5" in rung_names
    assert "slippage_only" in rung_names
    assert "realistic" in rung_names
    assert "pessimistic" in rung_names
    # a steady, deterministic uptrend with a near-universal CALL condition
    # should survive every rung of the ladder
    assert result.classification == SURVIVES_LADDER
