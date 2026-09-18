import datetime as dt

import numpy as np
import pytest

from otc_research.backtest.engine import run_backtest
from otc_research.backtest.strategy import Strategy
from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import Candle
from otc_research.features.pipeline import compute_and_store
from otc_research.research.condition_strategy import ConditionStrategy, perturb_condition
from otc_research.research.discovery import Condition

# --- perturb_condition ----------------------------------------------------


def test_perturb_condition_zero_pct_returns_same_object():
    condition = Condition(parts=(("rsi_14", 30.0, 45.0),))
    assert perturb_condition(condition, 0.0) is condition


def test_perturb_condition_widens_symmetrically_for_positive_pct():
    condition = Condition(parts=(("rsi_14", 30.0, 50.0),))  # width 20
    widened = perturb_condition(condition, 0.5)  # +50% each side
    (feature, low, high) = widened.parts[0]
    assert feature == "rsi_14"
    assert low == pytest.approx(30.0 - 10.0)
    assert high == pytest.approx(50.0 + 10.0)


def test_perturb_condition_narrows_for_negative_pct():
    condition = Condition(parts=(("rsi_14", 30.0, 50.0),))  # width 20
    narrowed = perturb_condition(condition, -0.25)
    (_, low, high) = narrowed.parts[0]
    assert low == pytest.approx(30.0 + 5.0)
    assert high == pytest.approx(50.0 - 5.0)


def test_perturb_condition_applies_independently_to_every_part():
    condition = Condition(parts=(("a", 0.0, 10.0), ("b", 0.0, 100.0)))
    widened = perturb_condition(condition, 0.1)
    a_low, a_high = widened.parts[0][1], widened.parts[0][2]
    b_low, b_high = widened.parts[1][1], widened.parts[1][2]
    assert a_low == pytest.approx(-1.0)
    assert a_high == pytest.approx(11.0)
    assert b_low == pytest.approx(-10.0)
    assert b_high == pytest.approx(110.0)


# --- ConditionStrategy ------------------------------------------------


def _condition():
    return Condition(parts=(("rsi_14", 30.0, 45.0), ("adx_14", 25.0, 40.0)))


def test_condition_strategy_rejects_invalid_direction():
    with pytest.raises(ValueError):
        ConditionStrategy(_condition(), "BUY", 300)


def test_condition_strategy_required_features_matches_condition_parts():
    strategy = ConditionStrategy(_condition(), "CALL", 300)
    assert strategy.required_features == frozenset({"rsi_14", "adx_14"})


def test_condition_strategy_decide_returns_direction_when_condition_matches():
    strategy = ConditionStrategy(_condition(), "CALL", 300)
    decision = strategy.decide({"rsi_14": 35.0, "adx_14": 30.0})
    assert decision == "CALL"


def test_condition_strategy_decide_returns_none_when_condition_does_not_match():
    strategy = ConditionStrategy(_condition(), "CALL", 300)
    decision = strategy.decide({"rsi_14": 60.0, "adx_14": 30.0})
    assert decision is None


def test_condition_strategy_decide_respects_direction_put():
    strategy = ConditionStrategy(_condition(), "PUT", 300)
    decision = strategy.decide({"rsi_14": 35.0, "adx_14": 30.0})
    assert decision == "PUT"


def test_condition_strategy_edge_perturbation_widens_the_effective_condition():
    tight = ConditionStrategy(_condition(), "CALL", 300, edge_perturbation_pct=0.0)
    wide = ConditionStrategy(_condition(), "CALL", 300, edge_perturbation_pct=1.0)
    # A value just outside the original rsi_14 bin is rejected by the tight
    # strategy but accepted once the bin has been widened.
    features = {"rsi_14": 29.0, "adx_14": 30.0}
    assert tight.decide(features) is None
    assert wide.decide(features) == "CALL"


def test_condition_strategy_code_is_always_none():
    strategy = ConditionStrategy(_condition(), "CALL", 300)
    assert strategy.code is None


def test_condition_strategy_label_defaults_to_a_descriptive_string():
    strategy = ConditionStrategy(_condition(), "CALL", 300)
    assert "CALL" in strategy.label
    assert "rsi_14" in strategy.label


def test_condition_strategy_label_override():
    strategy = ConditionStrategy(_condition(), "CALL", 300, label="my_custom_label")
    assert strategy.label == "my_custom_label"


def test_condition_strategy_satisfies_strategy_protocol():
    strategy = ConditionStrategy(_condition(), "CALL", 300)
    assert isinstance(strategy, Strategy)


# --- integration: ConditionStrategy runs through the real backtest engine -


def _insert_synthetic_candles(session, n=300, asset="TEST_FX", timeframe="1h"):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rng = np.random.default_rng(11)
    close = 1.10 + np.cumsum(rng.normal(0, 0.002, size=n))
    for i in range(n):
        c = close[i]
        o = close[i - 1] if i > 0 else c
        h = max(o, c) + 0.001
        low = min(o, c) - 0.001
        session.add(
            Candle(
                asset=asset,
                timeframe=timeframe,
                timestamp=start + dt.timedelta(hours=i),
                open=o,
                high=h,
                low=low,
                close=c,
                source="synthetic:test",
                is_synthetic_test_data=True,
            )
        )
    session.commit()


def _backtest_config() -> BacktestConfig:
    return BacktestConfig(
        train_fraction=0.6,
        validation_fraction=0.2,
        realistic=ExecutionScenarioConfig(
            entry_delay_candles=1, signal_drop_probability=0.0, slippage_pct=0.01
        ),
        pessimistic=ExecutionScenarioConfig(
            entry_delay_candles=2, signal_drop_probability=0.0, slippage_pct=0.03
        ),
    )


def test_condition_strategy_runs_through_run_backtest(session):
    _insert_synthetic_candles(session, n=300)
    feature_report = compute_and_store(session, "TEST_FX", "1h")

    condition = Condition(parts=(("rsi_14", 0.0, 100.0),))  # matches almost every candle
    strategy = ConditionStrategy(condition, "CALL", 3600, label="test_condition")

    results = run_backtest(
        session,
        strategy,
        "TEST_FX",
        "1h",
        _backtest_config(),
        feature_set_version=feature_report.feature_set_version,
        split="train",
        rng_seed=1,
    )
    assert {r.scenario for r in results} == {"optimistic", "realistic", "pessimistic"}
    for r in results:
        assert r.stats.sample_size > 0
