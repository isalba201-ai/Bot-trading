import datetime as dt

import numpy as np
import pytest

from otc_research.backtest.walkforward import generate_folds
from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import BacktestRun, Candle
from otc_research.features.pipeline import compute_and_store
from otc_research.research.candidacy import CandidacyThresholds, evaluate_candidacy
from otc_research.research.discovery import Condition

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
    verdict = evaluate_candidacy(
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
    evaluate_candidacy(
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
    verdict = evaluate_candidacy(
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

    verdict = evaluate_candidacy(
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
    verdict = evaluate_candidacy(
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
