"""Step 5 (approved plan point 6): the pre-registered accept/reject bar
for a discovered condition — fixed BEFORE any condition is evaluated,
reusing the EXISTING robustness/walk-forward machinery (Phases 6-7)
unchanged, pointed at a discovered ``Condition`` via
``research.condition_strategy.ConditionStrategy`` instead of a
hand-designed H1-H20 strategy. This gives a discovered condition exactly
the same rigor H1-H20 already got — no separate, lighter-weight path.

Four gates, run in order, each a precondition for the next — cheapest and
most informative first, and TEST is the very last thing touched:

  1. TRAIN sample size + payout-adjusted margin over break-even
     (``backtest.metrics``, unchanged).
  2. TRAIN parameter-sensitivity sweep over ``edge_perturbation_grid``
     (``backtest.robustness.run_parameter_sweep`` / ``evaluate_robustness``,
     unchanged) — must classify "consistent_direction".
  3. Walk-forward on the ORIGINAL (unperturbed) condition, realistic
     scenario only (``backtest.walkforward``, unchanged) —
     ``fraction_folds_with_edge`` must clear the threshold AND the worst
     fold must still clear break-even, not just the mean.
  4. Only if 1-3 all pass: TEST split, touched exactly once via the same
     ``run_backtest(split="test")`` path (and its loud log warning) every
     other split-touching caller in this codebase uses — must be
     positive and Wilson-CI-clearing.

A condition that fails a gate is rejected there; later gates never run,
so TEST is never reached unless everything else already passed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sqlalchemy.orm import Session

from otc_research.backtest.engine import run_backtest
from otc_research.backtest.execution import realistic_scenario
from otc_research.backtest.metrics import break_even_win_rate, payout_adjusted_expectancy
from otc_research.backtest.robustness import (
    RobustnessVerdict,
    evaluate_robustness,
    run_parameter_sweep,
)
from otc_research.backtest.walkforward import (
    WalkForwardFold,
    WalkForwardSummary,
    run_walk_forward,
    summarize_walk_forward,
)
from otc_research.config import BacktestConfig
from otc_research.research.condition_strategy import ConditionStrategy
from otc_research.research.discovery import Condition

#: The only execution scenario candidacy is judged under — BACKTESTING.md's
#: "realistic", never "optimistic" (that would understate real friction)
#: and never "pessimistic" (that's for a separate stress read, not the bar
#: itself).
REALISTIC = "realistic"


@dataclass(frozen=True)
class CandidacyThresholds:
    """Every default here is the approved plan's own proposed value (point
    6 / "Open decisions") — adjustable, but never silently: changing one
    changes what "candidate" means for every condition evaluated against
    it, so a caller that overrides a default should say why.
    """

    min_sample_size: int = 100
    min_margin_over_break_even: float = 0.03
    min_edge_fraction_robustness: float = 0.7
    min_fraction_folds_with_edge: float = 0.7
    edge_perturbation_grid: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)
    robustness_min_sample_size: int = 30
    walk_forward_min_sample_size: int = 20


@dataclass(frozen=True)
class CandidacyVerdict:
    accepted: bool
    #: None when accepted; otherwise which gate stopped it:
    #: "sample_size_and_margin" / "robustness" / "walk_forward" / "test".
    rejected_at_gate: str | None
    reason: str
    train_sample_size: int | None = None
    train_win_rate: float | None = None
    train_margin_over_break_even: float | None = None
    robustness: RobustnessVerdict | None = None
    walk_forward: WalkForwardSummary | None = None
    test_sample_size: int | None = None
    test_win_rate: float | None = None
    test_win_rate_ci_low: float | None = None


def evaluate_candidacy(
    session: Session,
    condition: Condition,
    direction: str,
    expiry_seconds: int,
    asset: str,
    timeframe: str,
    backtest_config: BacktestConfig,
    *,
    feature_set_version: str,
    payout: float,
    walk_forward_folds: Sequence[WalkForwardFold],
    thresholds: CandidacyThresholds = CandidacyThresholds(),
    rng_seed: int = 0,
    label: str | None = None,
) -> CandidacyVerdict:
    """``walk_forward_folds`` is the caller's responsibility to generate
    (``backtest.walkforward.generate_folds``) — candidacy.py has no
    opinion about fold spans/steps, same separation of concerns as every
    other module in this package. ``payout`` is the fixed binary-options
    payout the condition is judged against (see ``BACKTESTING.md``); it is
    not looked up from anywhere, since the real broker payout at signal
    time is not known ahead of a live run (see ``Signal.payout_is_estimated``
    in the approved plan's point 9).
    """
    break_even = break_even_win_rate(payout)
    scenarios = [realistic_scenario(backtest_config.realistic)]
    base_strategy = ConditionStrategy(condition, direction, expiry_seconds, label=label)

    # --- gate 1: TRAIN sample size + payout-adjusted margin ---------------
    train_result = run_backtest(
        session,
        base_strategy,
        asset,
        timeframe,
        backtest_config,
        feature_set_version=feature_set_version,
        split="train",
        scenarios=scenarios,
        rng_seed=rng_seed,
    )[0]
    train_stats = train_result.stats

    if train_stats.sample_size < thresholds.min_sample_size:
        return CandidacyVerdict(
            accepted=False,
            rejected_at_gate="sample_size_and_margin",
            reason=(
                f"TRAIN sample size {train_stats.sample_size} < required "
                f"{thresholds.min_sample_size}"
            ),
            train_sample_size=train_stats.sample_size,
            train_win_rate=train_stats.win_rate,
        )

    if train_stats.win_rate is None:
        return CandidacyVerdict(
            accepted=False,
            rejected_at_gate="sample_size_and_margin",
            reason="TRAIN win rate undefined (no resolved trades)",
            train_sample_size=train_stats.sample_size,
        )

    margin = train_stats.win_rate - break_even
    expectancy = payout_adjusted_expectancy(train_stats.win_rate, payout)
    if expectancy <= 0 or margin < thresholds.min_margin_over_break_even:
        return CandidacyVerdict(
            accepted=False,
            rejected_at_gate="sample_size_and_margin",
            reason=(
                f"TRAIN margin over break-even {margin:.4f} < required "
                f"{thresholds.min_margin_over_break_even:.4f} "
                f"(win_rate={train_stats.win_rate:.4f}, "
                f"break_even={break_even:.4f}, expectancy={expectancy:.4f})"
            ),
            train_sample_size=train_stats.sample_size,
            train_win_rate=train_stats.win_rate,
            train_margin_over_break_even=margin,
        )

    # --- gate 2: TRAIN parameter-sensitivity sweep -------------------------
    def factory(edge_perturbation_pct: float) -> ConditionStrategy:
        return ConditionStrategy(
            condition,
            direction,
            expiry_seconds,
            edge_perturbation_pct=edge_perturbation_pct,
            label=label,
        )

    param_grid = [{"edge_perturbation_pct": pct} for pct in thresholds.edge_perturbation_grid]
    sweep_points = run_parameter_sweep(
        session,
        factory,
        param_grid,
        asset,
        timeframe,
        backtest_config,
        feature_set_version=feature_set_version,
        split="train",
        scenarios=scenarios,
        rng_seed=rng_seed,
    )
    robustness = evaluate_robustness(
        sweep_points,
        REALISTIC,
        min_sample_size=thresholds.robustness_min_sample_size,
        min_edge_fraction=thresholds.min_edge_fraction_robustness,
    )
    if robustness.classification != "consistent_direction":
        return CandidacyVerdict(
            accepted=False,
            rejected_at_gate="robustness",
            reason=(
                f"parameter-sweep robustness classified "
                f"{robustness.classification!r}, not consistent_direction"
            ),
            train_sample_size=train_stats.sample_size,
            train_win_rate=train_stats.win_rate,
            train_margin_over_break_even=margin,
            robustness=robustness,
        )

    # --- gate 3: walk-forward on the ORIGINAL (unperturbed) condition -----
    fold_results = run_walk_forward(
        session,
        base_strategy,
        asset,
        timeframe,
        backtest_config,
        walk_forward_folds,
        feature_set_version=feature_set_version,
        scenarios=scenarios,
        rng_seed=rng_seed,
    )
    wf_summary = summarize_walk_forward(
        fold_results, REALISTIC, min_sample_size=thresholds.walk_forward_min_sample_size
    )
    fraction = wf_summary.fraction_folds_with_edge
    worst = wf_summary.worst_fold_win_rate
    if (
        fraction is None
        or fraction < thresholds.min_fraction_folds_with_edge
        or worst is None
        or worst <= break_even
    ):
        return CandidacyVerdict(
            accepted=False,
            rejected_at_gate="walk_forward",
            reason=(
                f"walk-forward fraction_folds_with_edge="
                f"{fraction!r} (required >= "
                f"{thresholds.min_fraction_folds_with_edge}), "
                f"worst_fold_win_rate={worst!r} "
                f"(required > break_even={break_even:.4f})"
            ),
            train_sample_size=train_stats.sample_size,
            train_win_rate=train_stats.win_rate,
            train_margin_over_break_even=margin,
            robustness=robustness,
            walk_forward=wf_summary,
        )

    # --- gate 4: TEST, touched exactly once, only after 1-3 all pass ------
    test_stats = run_backtest(
        session,
        base_strategy,
        asset,
        timeframe,
        backtest_config,
        feature_set_version=feature_set_version,
        split="test",
        scenarios=scenarios,
        rng_seed=rng_seed,
    )[0].stats

    test_passes = (
        test_stats.win_rate is not None
        and test_stats.win_rate_ci_low is not None
        and test_stats.win_rate_ci_low > break_even
    )
    common_fields = dict(
        train_sample_size=train_stats.sample_size,
        train_win_rate=train_stats.win_rate,
        train_margin_over_break_even=margin,
        robustness=robustness,
        walk_forward=wf_summary,
        test_sample_size=test_stats.sample_size,
        test_win_rate=test_stats.win_rate,
        test_win_rate_ci_low=test_stats.win_rate_ci_low,
    )
    if not test_passes:
        return CandidacyVerdict(
            accepted=False,
            rejected_at_gate="test",
            reason=(
                f"TEST win_rate_ci_low={test_stats.win_rate_ci_low!r} "
                f"does not clear break_even={break_even:.4f}"
            ),
            **common_fields,
        )

    return CandidacyVerdict(
        accepted=True,
        rejected_at_gate=None,
        reason="passed all four gates: sample size/margin, robustness, walk-forward, TEST",
        **common_fields,
    )
