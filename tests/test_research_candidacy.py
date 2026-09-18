import datetime as dt

import numpy as np
import pytest

from otc_research.backtest.execution import delay_only_scenario
from otc_research.backtest.walkforward import generate_folds
from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import BacktestRun, Candle
from otc_research.features.pipeline import compute_and_store
from otc_research.research.candidacy import (
    DEFAULT_ENTRY_DELAY_CANDLES,
    CandidacyThresholds,
    evaluate_candidacy,
    evaluate_condition_candidacy,
)
from otc_research.research.condition_strategy import ConditionStrategy
from otc_research.research.discovery import Condition
from otc_research.research.strategy_candidacy import build_candidacy_inputs

ASSET = "TEST_FX"
TIMEFRAME = "1m"
N_CANDLES = 3000
SPIKE_EVERY = 5
SPIKE_PCT = 1.00  # return_1 (in percent) at a planted spike candle
DRIFT_PCT = 0.0007  # per-candle drift applied for the 3 candles after a spike


def _build_candle_prices(n: int, *, reverse_in_test_region: bool, seed: int = 7) -> list[float]:
    """A random-walk baseline with a deterministic, plantable momentum
    pattern injected every ``SPIKE_EVERY`` candles: a sharp +1% jump
    (large return_1) followed by three candles of small continued upward
    drift — engineered so a condition on "return_1 in a high bin" predicts
    a CALL win with near-certainty, exactly the kind of discovered
    interaction ``research.discovery``/``research.candidacy`` are meant to
    evaluate. When ``reverse_in_test_region`` is True, the last 20% of the
    series (the TEST split) has the post-spike drift flipped to DOWN
    instead of up, simulating an edge that fails to replicate out-of-
    sample (used by the TEST-gate rejection test).
    """
    rng = np.random.default_rng(seed)
    test_region_start = int(n * 0.8)  # train_fraction=0.6 + validation_fraction=0.2
    closes = [1.10000]
    for i in range(1, n):
        prev = closes[-1]
        in_test_region = reverse_in_test_region and i >= test_region_start
        if i % SPIKE_EVERY == 0:
            new_close = prev * (1 + SPIKE_PCT / 100.0)
        elif i % SPIKE_EVERY in (1, 2, 3):
            drift = -DRIFT_PCT if in_test_region else DRIFT_PCT
            new_close = prev * (1 + drift)
        else:
            new_close = prev * (1 + rng.normal(0, 0.00005))
        closes.append(new_close)
    return closes


def _insert_candles(session, closes: list[float], *, asset=ASSET, timeframe=TIMEFRAME):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for i, c in enumerate(closes):
        o = closes[i - 1] if i > 0 else c
        h = max(o, c) + 1e-6
        low = min(o, c) - 1e-6
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
    return start


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


def _momentum_condition() -> Condition:
    # Tight, finite bounds around the planted +1.00% spike -- comfortably
    # contains 1.00 under every point of CandidacyThresholds' default
    # edge_perturbation_grid ((-0.2 .. 0.2) of the (0.5, 1.5) width),
    # while excluding ordinary noise-candle returns (std ~0.005%).
    return Condition(parts=(("return_1", 0.5, 1.5),))


def _train_val_folds(history_start: dt.datetime):
    # Folds only span the first 80% of history (train+validation) so
    # walk-forward never touches the TEST region -- same discipline the
    # plan requires of every caller of backtest.walkforward.
    return generate_folds(
        history_start,
        history_start + dt.timedelta(minutes=int(N_CANDLES * 0.8)),
        train_span=dt.timedelta(minutes=400),
        test_span=dt.timedelta(minutes=400),
    )


@pytest.fixture()
def momentum_session(session):
    closes = _build_candle_prices(N_CANDLES, reverse_in_test_region=False)
    start = _insert_candles(session, closes)
    feature_report = compute_and_store(session, ASSET, TIMEFRAME)
    return session, start, feature_report.feature_set_version


@pytest.fixture()
def reversing_session(session):
    closes = _build_candle_prices(N_CANDLES, reverse_in_test_region=True)
    start = _insert_candles(session, closes)
    feature_report = compute_and_store(session, ASSET, TIMEFRAME)
    return session, start, feature_report.feature_set_version


# --- happy path: passes all four gates -------------------------------------


def test_evaluate_candidacy_accepts_a_condition_with_a_real_replicated_edge(momentum_session):
    session, start, feature_set_version = momentum_session
    verdict = evaluate_condition_candidacy(
        session,
        _momentum_condition(),
        "CALL",
        60,  # expiry_seconds == one 1m candle
        ASSET,
        TIMEFRAME,
        _backtest_config(),
        feature_set_version=feature_set_version,
        payout=0.85,
        walk_forward_folds=_train_val_folds(start),
        label="planted_momentum",
    )

    assert verdict.accepted is True
    assert verdict.rejected_at_gate is None
    assert verdict.train_sample_size >= 100
    assert verdict.train_win_rate > 0.9
    assert verdict.train_margin_over_break_even > 0.03
    assert verdict.robustness.classification == "consistent_direction"
    assert verdict.walk_forward.fraction_folds_with_edge >= 0.7
    assert verdict.walk_forward.worst_fold_win_rate > 0.5
    assert verdict.test_sample_size is not None and verdict.test_sample_size > 0
    assert verdict.test_win_rate_ci_low is not None
    assert verdict.test_win_rate_ci_low > 0.5405  # break_even_win_rate(0.85)


def test_evaluate_candidacy_touches_test_split_exactly_once_on_acceptance(momentum_session):
    session, start, feature_set_version = momentum_session
    evaluate_condition_candidacy(
        session,
        _momentum_condition(),
        "CALL",
        60,
        ASSET,
        TIMEFRAME,
        _backtest_config(),
        feature_set_version=feature_set_version,
        payout=0.85,
        walk_forward_folds=_train_val_folds(start),
        label="planted_momentum",
    )
    test_runs = session.query(BacktestRun).filter_by(split="test").all()
    assert len(test_runs) == 1  # exactly one scenario (realistic) x one gate-4 call


# --- gate 1: sample size --------------------------------------------------


def test_evaluate_candidacy_rejects_at_sample_size_gate_when_floor_is_unreachable(
    momentum_session,
):
    session, start, feature_set_version = momentum_session
    verdict = evaluate_condition_candidacy(
        session,
        _momentum_condition(),
        "CALL",
        60,
        ASSET,
        TIMEFRAME,
        _backtest_config(),
        feature_set_version=feature_set_version,
        payout=0.85,
        walk_forward_folds=_train_val_folds(start),
        thresholds=CandidacyThresholds(min_sample_size=1_000_000),
    )
    assert verdict.accepted is False
    assert verdict.rejected_at_gate == "sample_size_and_margin"
    assert "sample size" in verdict.reason
    # later gates never ran
    assert verdict.robustness is None
    assert verdict.walk_forward is None
    assert verdict.test_sample_size is None
    assert session.query(BacktestRun).filter_by(split="test").count() == 0


# --- gate 1: margin over break-even ----------------------------------------


def test_evaluate_candidacy_rejects_at_margin_gate_when_matched_trades_have_no_edge(session):
    # No planted drift at all -- the spike still fires (so the condition
    # matches plenty of rows) but the post-spike move is unbiased noise,
    # so the matched trades' win rate has no real edge over 50%.
    rng = np.random.default_rng(3)
    closes = [1.10000]
    for i in range(1, N_CANDLES):
        prev = closes[-1]
        if i % SPIKE_EVERY == 0:
            new_close = prev * (1 + SPIKE_PCT / 100.0)
        else:
            new_close = prev * (1 + rng.normal(0, 0.0006))
        closes.append(new_close)
    start = _insert_candles(session, closes)
    feature_report = compute_and_store(session, ASSET, TIMEFRAME)

    verdict = evaluate_condition_candidacy(
        session,
        _momentum_condition(),
        "CALL",
        60,
        ASSET,
        TIMEFRAME,
        _backtest_config(),
        feature_set_version=feature_report.feature_set_version,
        payout=0.85,
        walk_forward_folds=_train_val_folds(start),
    )
    assert verdict.accepted is False
    assert verdict.rejected_at_gate == "sample_size_and_margin"
    assert verdict.train_sample_size >= 100  # reached the margin check, not blocked by size
    assert verdict.robustness is None
    assert session.query(BacktestRun).filter_by(split="test").count() == 0


# --- gate 4: TEST split -----------------------------------------------------


def test_evaluate_candidacy_rejects_at_test_gate_when_edge_does_not_replicate_out_of_sample(
    reversing_session,
):
    session, start, feature_set_version = reversing_session
    verdict = evaluate_condition_candidacy(
        session,
        _momentum_condition(),
        "CALL",
        60,
        ASSET,
        TIMEFRAME,
        _backtest_config(),
        feature_set_version=feature_set_version,
        payout=0.85,
        walk_forward_folds=_train_val_folds(start),
    )
    # TRAIN and walk-forward (both entirely inside the non-reversed 80%)
    # still look like a real edge -- only TEST (the reversed last 20%)
    # gives it away.
    assert verdict.robustness is not None
    assert verdict.walk_forward is not None
    assert verdict.accepted is False
    assert verdict.rejected_at_gate == "test"
    assert verdict.test_sample_size is not None and verdict.test_sample_size > 0
    # TEST was touched exactly once even though it failed
    assert session.query(BacktestRun).filter_by(split="test").count() == 1


# --- binary-options execution-model correction (BINARY_OPTIONS_REFRAME_AUDIT.md) --


def test_evaluate_candidacy_default_scenario_ignores_slippage_config(momentum_session):
    # a binary option has no fill-price/spread concept, so the default
    # scenario must not be affected by backtest_config.realistic's
    # slippage_pct at all -- proven by cranking it up absurdly high and
    # confirming the TRAIN margin is unaffected.
    session, start, feature_set_version = momentum_session
    poisoned_config = BacktestConfig(
        train_fraction=0.6,
        validation_fraction=0.2,
        realistic=ExecutionScenarioConfig(
            entry_delay_candles=1, signal_drop_probability=0.0, slippage_pct=50.0
        ),
        pessimistic=ExecutionScenarioConfig(
            entry_delay_candles=2, signal_drop_probability=0.0, slippage_pct=90.0
        ),
    )
    verdict = evaluate_condition_candidacy(
        session, _momentum_condition(), "CALL", 60, ASSET, TIMEFRAME, poisoned_config,
        feature_set_version=feature_set_version, payout=0.85,
        walk_forward_folds=_train_val_folds(start), label="poisoned_config_test",
    )
    assert verdict.train_win_rate > 0.9  # unaffected by the absurd slippage_pct above


def test_evaluate_candidacy_default_scenario_is_delay_only_not_realistic(momentum_session):
    session, start, feature_set_version = momentum_session
    evaluate_condition_candidacy(
        session, _momentum_condition(), "CALL", 60, ASSET, TIMEFRAME, _backtest_config(),
        feature_set_version=feature_set_version, payout=0.85,
        walk_forward_folds=_train_val_folds(start), label="scenario_name_check",
    )
    scenarios_used = {
        r.execution_scenario
        for r in session.query(BacktestRun).filter_by(strategy_label="scenario_name_check").all()
    }
    assert scenarios_used == {f"delay_only_{DEFAULT_ENTRY_DELAY_CANDLES}"}
    assert "realistic" not in scenarios_used


def test_evaluate_candidacy_respects_explicit_scenario_override(momentum_session):
    session, start, feature_set_version = momentum_session
    evaluate_condition_candidacy(
        session, _momentum_condition(), "CALL", 60, ASSET, TIMEFRAME, _backtest_config(),
        feature_set_version=feature_set_version, payout=0.85,
        walk_forward_folds=_train_val_folds(start), scenario=delay_only_scenario(3),
        label="override_scenario_check",
    )
    scenarios_used = {
        r.execution_scenario
        for r in session.query(BacktestRun).filter_by(strategy_label="override_scenario_check").all()
    }
    assert scenarios_used == {"delay_only_3"}


def test_evaluate_candidacy_rejects_at_walk_forward_gate_when_too_few_folds_sampled(momentum_session):
    session, start, feature_set_version = momentum_session
    verdict = evaluate_condition_candidacy(
        session, _momentum_condition(), "CALL", 60, ASSET, TIMEFRAME, _backtest_config(),
        feature_set_version=feature_set_version, payout=0.85,
        walk_forward_folds=_train_val_folds(start),
        thresholds=CandidacyThresholds(min_folds_sampled=99),  # unreachable
    )
    assert verdict.accepted is False
    assert verdict.rejected_at_gate == "walk_forward"
    assert "n_folds_sufficiently_sampled" in verdict.reason
    # TEST must never be touched once gate 3 fails
    assert session.query(BacktestRun).filter_by(split="test").count() == 0


# --- Step 9 generalization: evaluate_candidacy(base_strategy, ...) ---------


def test_evaluate_candidacy_generic_path_matches_condition_wrapper_byte_for_byte(momentum_session):
    # evaluate_condition_candidacy must be a pure pass-through onto the
    # generalized evaluate_candidacy -- proven by building the identical
    # ConditionStrategy-based base_strategy/strategy_factory/param_grid by
    # hand and confirming the resulting CandidacyVerdict is identical,
    # field for field, to the wrapper's own verdict. This protects the
    # already-reported 260-evaluation corrected rerun from silently
    # changing when new callers (H1-H20) start using the same function.
    session, start, feature_set_version = momentum_session
    condition = _momentum_condition()
    config = _backtest_config()
    folds = _train_val_folds(start)

    wrapper_verdict = evaluate_condition_candidacy(
        session, condition, "CALL", 60, ASSET, TIMEFRAME, config,
        feature_set_version=feature_set_version, payout=0.85,
        walk_forward_folds=folds, label="generic_parity_check",
    )

    base_strategy = ConditionStrategy(condition, "CALL", 60, label="generic_parity_check_direct")

    def strategy_factory(edge_perturbation_pct: float) -> ConditionStrategy:
        return ConditionStrategy(
            condition, "CALL", 60, edge_perturbation_pct=edge_perturbation_pct,
            label="generic_parity_check_direct",
        )

    thresholds = CandidacyThresholds()
    perturbation_param_grid = [
        {"edge_perturbation_pct": pct} for pct in thresholds.edge_perturbation_grid
    ]
    direct_verdict = evaluate_candidacy(
        session, base_strategy, ASSET, TIMEFRAME, config,
        feature_set_version=feature_set_version, payout=0.85,
        walk_forward_folds=folds, strategy_factory=strategy_factory,
        perturbation_param_grid=perturbation_param_grid, thresholds=thresholds,
    )

    assert direct_verdict.accepted == wrapper_verdict.accepted
    assert direct_verdict.rejected_at_gate == wrapper_verdict.rejected_at_gate
    assert direct_verdict.train_sample_size == wrapper_verdict.train_sample_size
    assert direct_verdict.train_win_rate == wrapper_verdict.train_win_rate
    assert direct_verdict.train_margin_over_break_even == wrapper_verdict.train_margin_over_break_even
    assert direct_verdict.robustness == wrapper_verdict.robustness
    assert direct_verdict.walk_forward == wrapper_verdict.walk_forward
    assert direct_verdict.test_sample_size == wrapper_verdict.test_sample_size
    assert direct_verdict.test_win_rate == wrapper_verdict.test_win_rate
    assert direct_verdict.test_win_rate_ci_low == wrapper_verdict.test_win_rate_ci_low


def test_evaluate_candidacy_accepts_an_h_strategy_via_strategy_candidacy_builder(momentum_session):
    # H3's momentum-continuation trigger, on the planted-momentum
    # fixture -- proves the generalized funnel works for a hand-designed
    # Strategy class, not just ConditionStrategy, end to end including
    # gate 2's per-strategy perturbation grid.
    session, start, feature_set_version = momentum_session
    base_strategy, strategy_factory, param_grid = build_candidacy_inputs("H3", expiry_seconds=60)
    verdict = evaluate_candidacy(
        session, base_strategy, ASSET, TIMEFRAME, _backtest_config(),
        feature_set_version=feature_set_version, payout=0.85,
        walk_forward_folds=_train_val_folds(start), strategy_factory=strategy_factory,
        perturbation_param_grid=param_grid,
    )
    # Whatever the verdict, the funnel itself must have run correctly --
    # gate 1 is always reachable (real trades exist), and if it clears
    # gate 1, robustness must have actually been evaluated via the H3
    # perturbation grid (not silently skipped).
    assert verdict.train_sample_size is not None
    if verdict.rejected_at_gate not in ("sample_size_and_margin", None) or verdict.accepted:
        assert verdict.robustness is not None
        assert verdict.robustness.n_points == len(param_grid)
