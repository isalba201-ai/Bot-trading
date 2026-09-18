"""Phase 6: robustness / parameter-sensitivity sweeps.

BACKTESTING.md: "Every promising strategy is deliberately stress-tested
by varying its parameters, the assets, the time period, the time of day,
the payout, the expiration, and the sample size. A strategy that only
works at one exact parameter value ... and breaks under small
perturbations is classified OVERFITTED / FRAGILE, regardless of how good
its original backtest looked."

This module operationalizes that rule: run the same strategy shape across
a grid of constructor parameters and, optionally, several time windows,
under the Phase 4 engine, and report whether a statistically credible
edge (Wilson CI lower bound above 50%) shows up consistently across
nearby variations or only at one lucky point.

**Vocabulary discipline**: the verdicts here ("insufficient_data",
"fragile", "consistent_direction") are deliberately NOT the same words as
BACKTESTING.md's final edge classification (NO EDGE / WEAK EDGE /
PROMISING / ROBUST EDGE). That classification needs walk-forward (Phase
7) and Monte Carlo (Phase 8) as well, neither of which exists yet — a
"consistent_direction" verdict here is one necessary input to that later
classification, not a substitute for it.

The 70%-of-points threshold below is this project's own choice, not a
universal statistical standard — it's a parameter (``min_edge_fraction``)
precisely so it can be reconsidered.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from sqlalchemy.orm import Session

from otc_research.backtest.engine import BacktestRunResult, run_backtest
from otc_research.backtest.execution import ExecutionScenario
from otc_research.backtest.strategy import Strategy
from otc_research.config import BacktestConfig

TimeWindow = tuple[dt.datetime | None, dt.datetime | None]


@dataclass(frozen=True)
class SweepPoint:
    params: dict[str, object]
    window: TimeWindow
    results: list[BacktestRunResult]

    def result_for(self, scenario: str) -> BacktestRunResult | None:
        for r in self.results:
            if r.scenario == scenario:
                return r
        return None


def run_parameter_sweep(
    session: Session,
    strategy_factory: Callable[..., Strategy],
    param_grid: Sequence[Mapping[str, object]],
    asset: str,
    timeframe: str,
    backtest_config: BacktestConfig,
    *,
    feature_set_version: str,
    split: str = "train",
    windows: Sequence[TimeWindow] | None = None,
    scenarios: Sequence[ExecutionScenario] | None = None,
    rng_seed: int = 0,
) -> list[SweepPoint]:
    """Runs ``strategy_factory(**params)`` through ``run_backtest`` once
    per (params, window) combination. ``windows`` defaults to a single
    ``(None, None)`` window (the whole ingested history). Never touches
    ``split="test"`` implicitly any more than a single ``run_backtest``
    call would — the same discipline applies here, multiplied by however
    many grid points you pass, so keep sweeps on train/validation.
    """
    active_windows = list(windows) if windows is not None else [(None, None)]
    points: list[SweepPoint] = []
    for params in param_grid:
        strategy = strategy_factory(**params)
        for start, end in active_windows:
            results = run_backtest(
                session,
                strategy,
                asset,
                timeframe,
                backtest_config,
                feature_set_version=feature_set_version,
                split=split,
                scenarios=scenarios,
                rng_seed=rng_seed,
                start=start,
                end=end,
            )
            points.append(SweepPoint(params=dict(params), window=(start, end), results=results))
    return points


@dataclass(frozen=True)
class RobustnessVerdict:
    scenario: str
    min_sample_size: int
    min_edge_fraction: float
    n_points: int
    n_sufficiently_sampled: int
    n_with_edge: int
    fraction_with_edge: float | None
    classification: str  # "insufficient_data" / "fragile" / "consistent_direction"


def evaluate_robustness(
    points: Sequence[SweepPoint],
    scenario: str,
    *,
    min_sample_size: int = 30,
    min_edge_fraction: float = 0.7,
) -> RobustnessVerdict:
    """Looks only at points with at least ``min_sample_size`` resolved
    trades under ``scenario`` (an underpowered point is neither evidence
    for nor against robustness — it's just excluded, never treated as a
    failure). Among those, "has an edge" means the Wilson 95% CI lower
    bound clears 50% — a point estimate above 50% alone isn't enough,
    same discipline as everywhere else in this project.

    ``classification``:
      - "insufficient_data" — no point had enough samples to judge at all.
      - "fragile" — an edge shows up at some points but not
        ``min_edge_fraction`` of the sufficiently-sampled ones.
      - "consistent_direction" — at least ``min_edge_fraction`` of the
        sufficiently-sampled points show an edge in the same direction.
        Still not "ROBUST EDGE" — see the module docstring.
    """
    sufficiently_sampled = [
        result
        for point in points
        if (result := point.result_for(scenario)) is not None
        and result.stats.sample_size >= min_sample_size
    ]

    n_sufficiently = len(sufficiently_sampled)
    if n_sufficiently == 0:
        return RobustnessVerdict(
            scenario=scenario,
            min_sample_size=min_sample_size,
            min_edge_fraction=min_edge_fraction,
            n_points=len(points),
            n_sufficiently_sampled=0,
            n_with_edge=0,
            fraction_with_edge=None,
            classification="insufficient_data",
        )

    n_with_edge = sum(
        1
        for r in sufficiently_sampled
        if r.stats.win_rate_ci_low is not None and r.stats.win_rate_ci_low > 0.5
    )
    fraction = n_with_edge / n_sufficiently
    classification = "consistent_direction" if fraction >= min_edge_fraction else "fragile"

    return RobustnessVerdict(
        scenario=scenario,
        min_sample_size=min_sample_size,
        min_edge_fraction=min_edge_fraction,
        n_points=len(points),
        n_sufficiently_sampled=n_sufficiently,
        n_with_edge=n_with_edge,
        fraction_with_edge=fraction,
        classification=classification,
    )
