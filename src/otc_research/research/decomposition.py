"""Step 8 (audit+research addendum, see
``/root/.claude/plans/sequential-sparking-candle.md``): decomposes a
discovered condition's win rate through a LADDER of execution-realism
scenarios, to answer "where exactly does the edge disappear" (the plan's
Sections 1-3) mechanically rather than by inspection.

Reuses ``backtest.execution.ExecutionScenario`` and
``backtest.engine.run_backtest`` completely unchanged — no changes to
the core engine anywhere. ``research.candidacy``'s gate 1 only ever runs
the single bundled "realistic" scenario (delay + slippage + drop
together); this module runs the SAME condition through several more
scenario instances so delay-only and slippage-only effects can be told
apart, which candidacy.py deliberately never needed to do (its job is a
pass/fail bar, not a diagnostic breakdown).

The delay-only rungs are spaced in whole CANDLES on this project's finest
available data (1-minute), the approved proxy for a true sub-minute
delay sweep — see ``PREDICTABILITY_AUDIT.md`` for why: no tick/sub-minute
Forex data exists in this project (Twelve Data's REST API tops out at
1-minute candles), and the user chose this proxy over sourcing tick data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from sqlalchemy.orm import Session

from otc_research.backtest.engine import run_backtest
from otc_research.backtest.execution import (
    OPTIMISTIC,
    ExecutionScenario,
    pessimistic_scenario,
    realistic_scenario,
)
from otc_research.backtest.metrics import TradeStats, break_even_win_rate, payout_adjusted_expectancy
from otc_research.config import BacktestConfig
from otc_research.research.candidacy import CandidacyThresholds
from otc_research.research.condition_strategy import ConditionStrategy
from otc_research.research.discovery import Condition

#: Whole-candle delay rungs on 1-minute data — the approved proxy for
#: ~0/60/120/180/300 seconds of manual-signal reaction time.
DELAY_ONLY_CANDLES: tuple[int, ...] = (0, 1, 2, 3, 5)

#: The three failure modes the user's plan explicitly asked to keep
#: separate, plus two housekeeping outcomes that are NOT failure modes:
#: "survives_ladder" (the condition clears every rung — not a failure at
#: all, still subject to candidacy.py's full 4-gate bar before being a
#: real candidate) and "insufficient_data" (too few resolved trades at
#: the raw stage to classify anything, never treated as evidence either
#: way).
A_NO_PREDICTABILITY = "A_no_predictability"
B_TOO_SMALL_FOR_PAYOUT = "B_too_small_for_payout"
C_EXECUTION_DESTROYS_IT = "C_execution_destroys_it"
SURVIVES_LADDER = "survives_ladder"
INSUFFICIENT_DATA = "insufficient_data"


def delay_only_scenario(entry_delay_candles: int) -> ExecutionScenario:
    return ExecutionScenario(
        name=f"delay_only_{entry_delay_candles}",
        entry_delay_candles=entry_delay_candles,
        signal_drop_probability=0.0,
        slippage_pct=0.0,
    )


def slippage_only_scenario(slippage_pct: float) -> ExecutionScenario:
    return ExecutionScenario(
        name="slippage_only",
        entry_delay_candles=0,
        signal_drop_probability=0.0,
        slippage_pct=slippage_pct,
    )


@dataclass(frozen=True)
class DecompositionRung:
    name: str
    sample_size: int
    win_rate: float | None
    win_rate_ci_low: float | None
    break_even_win_rate: float
    margin_over_break_even: float | None
    expectancy: float | None  # payout-adjusted


@dataclass(frozen=True)
class DecompositionResult:
    condition_label: str
    direction: str
    asset: str
    timeframe: str
    rungs: tuple[DecompositionRung, ...]
    classification: str


def _rung(name: str, sample_size: int, win_rate: float | None, ci_low: float | None, payout: float) -> DecompositionRung:
    be = break_even_win_rate(payout)
    margin = win_rate - be if win_rate is not None else None
    expectancy = payout_adjusted_expectancy(win_rate, payout) if win_rate is not None else None
    return DecompositionRung(
        name=name,
        sample_size=sample_size,
        win_rate=win_rate,
        win_rate_ci_low=ci_low,
        break_even_win_rate=be,
        margin_over_break_even=margin,
        expectancy=expectancy,
    )


def _rung_from_stats(name: str, stats: TradeStats, payout: float) -> DecompositionRung:
    return _rung(name, stats.sample_size, stats.win_rate, stats.win_rate_ci_low, payout)


def _clears_margin(rung: DecompositionRung | None, min_margin: float, min_sample_size: int) -> bool:
    return (
        rung is not None
        and rung.sample_size >= min_sample_size
        and rung.margin_over_break_even is not None
        and rung.margin_over_break_even >= min_margin
    )


def classify(
    rungs: Sequence[DecompositionRung],
    *,
    min_margin: float = CandidacyThresholds().min_margin_over_break_even,
    min_sample_size: int = 30,
    raw_null_tolerance: float = 0.02,
) -> str:
    """Mechanically assigns exactly one of the three failure modes (or
    ``SURVIVES_LADDER`` / ``INSUFFICIENT_DATA``) from the rung table:

    - **A**: the raw (naive-label, zero-execution-model) win rate is
      already indistinguishable from 50% — no predictability was ever
      there to lose.
    - **C**: raw shows a real directional skew, but it is gone (or never
      clears the payout-adjusted margin) once the REAL simulator is
      involved — either because of zero-friction "optimistic" execution
      mechanics alone (the naive label's implied trade window doesn't
      match what the simulator actually executes — see the module
      docstring's note on entry-timing mismatch) or because delay/
      slippage/signal-drop specifically erode it. Both are execution-
      mechanics stories, deliberately bundled under one letter per the
      user's own three-way split.
    - **B**: the skew survives every execution rung but never clears the
      payout-adjusted break-even margin by ``min_margin`` — real, but too
      small to trade profitably at this payout.
    """
    by_name = {r.name: r for r in rungs}
    raw = by_name.get("raw")
    if raw is None or raw.sample_size < min_sample_size or raw.win_rate is None:
        return INSUFFICIENT_DATA
    if abs(raw.win_rate - 0.5) < raw_null_tolerance:
        return A_NO_PREDICTABILITY

    realistic = by_name.get("realistic")
    if _clears_margin(realistic, min_margin, min_sample_size):
        return SURVIVES_LADDER

    optimistic = by_name.get("optimistic")
    if _clears_margin(optimistic, min_margin, min_sample_size):
        return C_EXECUTION_DESTROYS_IT
    if _clears_margin(raw, min_margin, min_sample_size=1):
        # raw shows a real skew, but it's already gone by the time the
        # real simulator's own entry/exit mechanics are involved, even
        # before any friction is applied.
        return C_EXECUTION_DESTROYS_IT

    return B_TOO_SMALL_FOR_PAYOUT


def run_decomposition(
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
    raw_win_rate: float,
    raw_sample_size: int,
    split: str = "train",
    rng_seed: int = 0,
    label: str | None = None,
) -> DecompositionResult:
    """``raw_win_rate``/``raw_sample_size`` come from the condition's
    ALREADY-STORED ``ConditionTrial`` row (the naive dataset-label win
    rate computed by ``research.discovery.run_discovery``, zero execution
    model) — this function never recomputes them itself, so it can never
    drift from what discovery.py actually found for this exact condition.
    """
    strategy = ConditionStrategy(condition, direction, expiry_seconds, label=label)

    scenarios: list[ExecutionScenario] = [OPTIMISTIC]
    for candles in DELAY_ONLY_CANDLES[1:]:  # 0 candles == OPTIMISTIC, already included
        scenarios.append(delay_only_scenario(candles))
    scenarios.append(slippage_only_scenario(backtest_config.realistic.slippage_pct))
    scenarios.append(realistic_scenario(backtest_config.realistic))
    scenarios.append(pessimistic_scenario(backtest_config.pessimistic))

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
    )

    rungs = [_rung("raw", raw_sample_size, raw_win_rate, None, payout)]
    rungs.extend(_rung_from_stats(r.scenario, r.stats, payout) for r in results)

    return DecompositionResult(
        condition_label=condition.label(),
        direction=direction,
        asset=asset,
        timeframe=timeframe,
        rungs=tuple(rungs),
        classification=classify(rungs),
    )
