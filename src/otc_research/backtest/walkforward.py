"""Phase 7: walk-forward analysis.

BACKTESTING.md: "In addition to the single train/val/test split,
strategies that pass it are re-evaluated with walk-forward analysis:
train on a window, test on the following out-of-sample window, slide
forward, repeat. Results are reported aggregated across all folds (mean,
dispersion, worst fold) — a strategy that only works in one fold is not
robust."

None of the strategies implemented so far (STRATEGIES.md's H1-H10
baselines) fit any parameter from data — every threshold is a fixed
constructor argument. So today, a fold's "train" window is not actually
used for anything; it exists so the fold shape already matches the
standard walk-forward pattern, and so a future parameter-fitting strategy
(e.g. a Phase 12 ML model) can slot into the exact same fold sequence
without this module changing. What the fold's "test" window does today:
it's evaluated exactly like backtest.robustness's time-windows, except
there are many small, contiguous, non-overlapping (by default) folds
covering the whole history instead of one or two large halves — a much
finer-grained version of the same "does the edge hold up over time?"
question, and the one BACKTESTING.md specifically asks for before any
edge classification.

**Vocabulary discipline**: same rule as backtest/robustness.py — nothing
here is allowed to say NO EDGE / WEAK EDGE / PROMISING / ROBUST EDGE.
That still needs Monte Carlo (Phase 8) as well.
"""

from __future__ import annotations

import datetime as dt
import statistics
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy.orm import Session

from otc_research.backtest.engine import BacktestRunResult, compute_split_windows, run_backtest
from otc_research.backtest.execution import ExecutionScenario
from otc_research.backtest.strategy import Strategy
from otc_research.config import BacktestConfig


@dataclass(frozen=True)
class WalkForwardFold:
    fold_index: int
    train_window: tuple[dt.datetime, dt.datetime]
    test_window: tuple[dt.datetime, dt.datetime]


def generate_folds(
    history_start: dt.datetime,
    history_end: dt.datetime,
    train_span: dt.timedelta,
    test_span: dt.timedelta,
    step: dt.timedelta | None = None,
) -> list[WalkForwardFold]:
    """Slides a (train_span, test_span) window pair across
    [history_start, history_end), advancing by ``step`` each time
    (defaults to ``test_span``, giving non-overlapping test windows — the
    standard shape; a smaller step overlaps test windows, which trades
    fold independence for more folds out of the same history).

    A fold is only included if its full test window fits within
    ``history_end`` — a trailing partial window is dropped rather than
    silently evaluated on less data than every other fold (which would
    make it an outlier for a reason that has nothing to do with the
    strategy).
    """
    if train_span <= dt.timedelta(0):
        raise ValueError("train_span must be positive")
    if test_span <= dt.timedelta(0):
        raise ValueError("test_span must be positive")
    active_step = step if step is not None else test_span
    if active_step <= dt.timedelta(0):
        raise ValueError("step must be positive")

    folds: list[WalkForwardFold] = []
    fold_index = 0
    train_start = history_start
    while True:
        train_end = train_start + train_span
        test_start = train_end
        test_end = test_start + test_span
        if test_end > history_end:
            break
        folds.append(
            WalkForwardFold(
                fold_index=fold_index,
                train_window=(train_start, train_end),
                test_window=(test_start, test_end),
            )
        )
        fold_index += 1
        train_start = train_start + active_step

    return folds


def compute_walk_forward_folds(
    session: Session,
    asset: str,
    timeframe: str,
    backtest_config: BacktestConfig,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    n_folds: int = 5,
) -> list[WalkForwardFold]:
    """The correct way to generate walk-forward folds for a candidacy
    evaluation: bounded strictly to TRAIN+VALIDATION, using the same
    row-based split boundary ``run_backtest``/``evaluate_candidacy``
    themselves use (via ``backtest.engine.compute_split_windows``) —
    never a calendar-time-proportion *estimate* of where TRAIN+VALIDATION
    ends, which does not line up with the true row-based boundary
    whenever candle density isn't perfectly uniform across the window
    (this is exactly what let one walk-forward fold overflow ~2 hours
    into the TEST split in the ML_1M5M experiment — see
    ``ML1M5M_EXPERIMENT_REPORT.md`` and the follow-up investigation).

    ``windows.test_start`` is used as the folds' exclusive upper bound:
    ``generate_folds`` already drops any fold whose test window would
    exceed it, so a fold can structurally never include a TEST candle,
    not just "usually doesn't" by calendar-arithmetic luck.
    """
    windows = compute_split_windows(session, asset, timeframe, backtest_config, start=start, end=end)
    history_start = windows.train[0]
    history_end = windows.test_start
    fold_span = (history_end - history_start) / n_folds
    return generate_folds(history_start, history_end, train_span=fold_span, test_span=fold_span)


@dataclass(frozen=True)
class FoldResult:
    fold: WalkForwardFold
    results: list[BacktestRunResult]

    def result_for(self, scenario: str) -> BacktestRunResult | None:
        for r in self.results:
            if r.scenario == scenario:
                return r
        return None


def run_walk_forward(
    session: Session,
    strategy: Strategy,
    asset: str,
    timeframe: str,
    backtest_config: BacktestConfig,
    folds: Sequence[WalkForwardFold],
    *,
    feature_set_version: str,
    scenarios: Sequence[ExecutionScenario] | None = None,
    rng_seed: int = 0,
) -> list[FoldResult]:
    """Evaluates ``strategy`` on each fold's TEST window only (see module
    docstring for why the train window isn't used yet), via
    ``run_backtest(split="walk_forward", ...)`` — which, unlike
    ``split="train"/"validation"/"test"``, does not apply any further
    60/20/20 subdivision to what it's given.
    """
    fold_results: list[FoldResult] = []
    for fold in folds:
        test_start, test_end = fold.test_window
        results = run_backtest(
            session,
            strategy,
            asset,
            timeframe,
            backtest_config,
            feature_set_version=feature_set_version,
            split="walk_forward",
            scenarios=scenarios,
            rng_seed=rng_seed,
            start=test_start,
            end=test_end,
            fold_index=fold.fold_index,
        )
        fold_results.append(FoldResult(fold=fold, results=results))
    return fold_results


@dataclass(frozen=True)
class WalkForwardSummary:
    scenario: str
    min_sample_size: int
    n_folds: int
    n_folds_sufficiently_sampled: int
    mean_win_rate: float | None
    stdev_win_rate: float | None
    worst_fold_win_rate: float | None
    worst_fold_index: int | None
    n_folds_with_edge: int
    fraction_folds_with_edge: float | None


def summarize_walk_forward(
    fold_results: Sequence[FoldResult], scenario: str, *, min_sample_size: int = 20
) -> WalkForwardSummary:
    """Aggregates per-fold results per BACKTESTING.md: mean, dispersion
    (population stdev), and worst fold — never just the best-looking fold
    or an average that hides how bad the worst one was. Folds under
    ``min_sample_size`` resolved trades are excluded from every statistic
    below (too little signal to be evidence either way), same convention
    as backtest/robustness.py.
    """
    sufficiently_sampled = [
        (fr.fold.fold_index, r)
        for fr in fold_results
        if (r := fr.result_for(scenario)) is not None and r.stats.sample_size >= min_sample_size
    ]

    if not sufficiently_sampled:
        return WalkForwardSummary(
            scenario=scenario,
            min_sample_size=min_sample_size,
            n_folds=len(fold_results),
            n_folds_sufficiently_sampled=0,
            mean_win_rate=None,
            stdev_win_rate=None,
            worst_fold_win_rate=None,
            worst_fold_index=None,
            n_folds_with_edge=0,
            fraction_folds_with_edge=None,
        )

    win_rates = [r.stats.win_rate for _, r in sufficiently_sampled if r.stats.win_rate is not None]
    mean_win_rate = statistics.mean(win_rates) if win_rates else None
    stdev_win_rate = statistics.pstdev(win_rates) if len(win_rates) > 1 else 0.0 if win_rates else None

    worst_index, worst_result = min(
        sufficiently_sampled, key=lambda pair: pair[1].stats.win_rate
        if pair[1].stats.win_rate is not None else float("inf"),
    )
    worst_fold_win_rate = worst_result.stats.win_rate

    n_with_edge = sum(
        1
        for _, r in sufficiently_sampled
        if r.stats.win_rate_ci_low is not None and r.stats.win_rate_ci_low > 0.5
    )
    fraction_with_edge = n_with_edge / len(sufficiently_sampled)

    return WalkForwardSummary(
        scenario=scenario,
        min_sample_size=min_sample_size,
        n_folds=len(fold_results),
        n_folds_sufficiently_sampled=len(sufficiently_sampled),
        mean_win_rate=mean_win_rate,
        stdev_win_rate=stdev_win_rate,
        worst_fold_win_rate=worst_fold_win_rate,
        worst_fold_index=worst_index,
        n_folds_with_edge=n_with_edge,
        fraction_folds_with_edge=fraction_with_edge,
    )
