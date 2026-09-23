"""Orchestrates one backtest run: stored Candle+Feature rows -> temporal
split -> simulator, under all three execution scenarios -> statistically
grounded metrics -> a persisted, auditable BacktestRun row per scenario.

Deliberately does NOT run all three splits automatically. The caller picks
one split per call (default "train") specifically so touching "test" is a
visible, separate decision — never an accidental side effect of running
train/validation — matching BACKTESTING.md's rule that the out-of-sample
test split is frozen the moment training begins and touched exactly once,
after every development decision is final.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from typing import Sequence

from sqlalchemy.orm import Session

from otc_research.backtest.execution import (
    OPTIMISTIC,
    ExecutionScenario,
    pessimistic_scenario,
    realistic_scenario,
)
from otc_research.backtest.metrics import TradeStats, summarize_trades
from otc_research.backtest.simulator import SimCandle, Trade, simulate
from otc_research.backtest.splits import compute_temporal_split
from otc_research.backtest.strategy import Strategy
from otc_research.config import BacktestConfig
from otc_research.db.models import BacktestRun, Candle, Feature
from otc_research.utils.logging import get_logger
from otc_research.utils.timeframes import timeframe_to_seconds

logger = get_logger(__name__)

VALID_SPLITS = ("train", "validation", "test", "walk_forward")


@dataclass(frozen=True)
class BacktestRunResult:
    scenario: str
    stats: TradeStats
    trades: list[Trade]
    backtest_run_id: int


@dataclass(frozen=True)
class SplitWindows:
    """The actual candle timestamps each split resolves to, for a given
    ``[start, end)`` candle universe — the same row-based
    ``compute_temporal_split`` arithmetic ``run_backtest`` uses
    internally, exposed so a caller can see (and reuse, e.g. for
    walk-forward fold generation) exactly where TRAIN ends and TEST
    begins, rather than re-deriving it by hand from a calendar-time
    proportion, which need not line up with the true row-based boundary
    when candle density isn't perfectly uniform across the window.

    ``test_start`` is the safe, exclusive upper bound for anything that
    must never touch TEST (TRAIN, VALIDATION, or walk-forward folds
    carved from TRAIN+VALIDATION): loading candles with ``end=test_start``
    is guaranteed to load zero TEST candles.
    """

    train: tuple[dt.datetime, dt.datetime]
    validation: tuple[dt.datetime, dt.datetime]
    test: tuple[dt.datetime, dt.datetime]
    n_candles: int

    @property
    def test_start(self) -> dt.datetime:
        return self.test[0]


def compute_split_windows(
    session: Session,
    asset: str,
    timeframe: str,
    backtest_config: BacktestConfig,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> SplitWindows:
    """Loads candles in ``[start, end)`` (defaulting to the full ingested
    history for ``asset``/``timeframe`` when both are ``None`` — the same
    default every other unwindowed caller here already had), applies the
    identical ``compute_temporal_split`` arithmetic ``run_backtest`` uses
    for ``split="train"/"validation"/"test"``, and returns each split's
    first/last actual candle timestamp.

    This exists so TRAIN/VALIDATION/TEST boundaries are computed exactly
    once, from real data, and every caller that needs to know them
    (``evaluate_candidacy``'s gate 1/2/4, walk-forward fold generation)
    reads the SAME boundary — never a separately-estimated one that could
    silently drift into a different split (see
    ``ML1M5M_EXPERIMENT_REPORT.md``'s split-mismatch finding, which this
    function was added specifically to fix).
    """
    candles = _load_candles(session, asset, timeframe, start=start, end=end)
    if len(candles) < 3:
        raise ValueError(
            f"not enough candles for {asset}/{timeframe} in the requested window "
            f"(have {len(candles)}, need at least 3 to form train/validation/test)"
        )
    split = compute_temporal_split(
        len(candles), backtest_config.train_fraction, backtest_config.validation_fraction
    )
    timestamps = [c.timestamp for c in candles]
    train_ts = timestamps[split.train_slice]
    validation_ts = timestamps[split.validation_slice]
    test_ts = timestamps[split.test_slice]
    return SplitWindows(
        train=(train_ts[0], train_ts[-1]),
        validation=(validation_ts[0], validation_ts[-1]),
        test=(test_ts[0], test_ts[-1]),
        n_candles=len(candles),
    )


def _load_candles(
    session: Session,
    asset: str,
    timeframe: str,
    *,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
) -> list[SimCandle]:
    query = session.query(Candle).filter(Candle.asset == asset, Candle.timeframe == timeframe)
    if start is not None:
        query = query.filter(Candle.timestamp >= start)
    if end is not None:
        query = query.filter(Candle.timestamp < end)
    rows = query.order_by(Candle.timestamp.asc()).all()
    return [
        SimCandle(timestamp=r.timestamp, open=r.open, high=r.high, low=r.low, close=r.close)
        for r in rows
    ]


def _load_features(
    session: Session, asset: str, timeframe: str, feature_set_version: str
) -> dict[dt.datetime, dict[str, float]]:
    rows = (
        session.query(Feature.timestamp, Feature.name, Feature.value)
        .filter(
            Feature.asset == asset,
            Feature.timeframe == timeframe,
            Feature.feature_set_version == feature_set_version,
        )
        .all()
    )
    by_timestamp: dict[dt.datetime, dict[str, float]] = {}
    for timestamp, name, value in rows:
        by_timestamp.setdefault(timestamp, {})[name] = value
    return by_timestamp


def _default_scenarios(backtest_config: BacktestConfig) -> list[ExecutionScenario]:
    return [
        OPTIMISTIC,
        realistic_scenario(backtest_config.realistic),
        pessimistic_scenario(backtest_config.pessimistic),
    ]


def run_backtest(
    session: Session,
    strategy: Strategy,
    asset: str,
    timeframe: str,
    backtest_config: BacktestConfig,
    *,
    feature_set_version: str,
    split: str = "train",
    scenarios: Sequence[ExecutionScenario] | None = None,
    rng_seed: int = 0,
    start: dt.datetime | None = None,
    end: dt.datetime | None = None,
    fold_index: int | None = None,
) -> list[BacktestRunResult]:
    """``start``/``end`` restrict the candle universe to a sub-window
    BEFORE the train/validation/test split is computed — i.e. the split
    fractions apply within that window, not the full history. This exists
    for Phase 6's robustness sweeps (BACKTESTING.md: "varying ... the time
    period ... the sample size"), not for everyday use — most callers
    should leave both as None and use the strategy's full ingested
    history.

    ``split="walk_forward"`` (Phase 7, see backtest/walkforward.py) is
    different in kind, not just another slice: it skips the internal
    60/20/20 split entirely and evaluates every candle in
    ``[start, end)`` as one block — the caller (walkforward.py) is
    expected to pass one fold's own test window via ``start``/``end`` and
    that fold's number via ``fold_index``, which is stored on the row so
    many folds from one sweep can be told apart later.
    """
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {VALID_SPLITS}, got {split!r}")

    if split == "test":
        logger.warning(
            "TEST SPLIT ACCESSED for strategy=%s asset=%s timeframe=%s — "
            "BACKTESTING.md requires this split be touched exactly once, "
            "after every development decision (hypothesis selection, "
            "parameters) is already final. Review this run's justification "
            "before trusting or acting on its result.",
            strategy.label,
            asset,
            timeframe,
        )

    candles = _load_candles(session, asset, timeframe, start=start, end=end)
    min_candles = 1 if split == "walk_forward" else 3
    if len(candles) < min_candles:
        raise ValueError(
            f"not enough candles for {asset}/{timeframe} "
            f"(have {len(candles)}, need at least {min_candles})"
        )
    features_by_timestamp = _load_features(session, asset, timeframe, feature_set_version)

    if split == "walk_forward":
        split_candles = candles
    else:
        temporal_split = compute_temporal_split(
            len(candles), backtest_config.train_fraction, backtest_config.validation_fraction
        )
        split_slice = {
            "train": temporal_split.train_slice,
            "validation": temporal_split.validation_slice,
            "test": temporal_split.test_slice,
        }[split]
        split_candles = candles[split_slice]

    timeframe_seconds = timeframe_to_seconds(timeframe)
    active_scenarios = list(scenarios) if scenarios is not None else _default_scenarios(
        backtest_config
    )

    results: list[BacktestRunResult] = []
    for scenario in active_scenarios:
        rng = random.Random(rng_seed)
        trades = simulate(
            strategy, split_candles, features_by_timestamp, timeframe_seconds, scenario, rng
        )
        stats = summarize_trades([t.result for t in trades], [t.pnl_pct for t in trades])

        run_row = BacktestRun(
            hypothesis_code=strategy.code,
            strategy_label=strategy.label,
            asset=asset,
            timeframe=timeframe,
            feature_set_version=feature_set_version,
            split=split,
            execution_scenario=scenario.name,
            expiry_seconds=strategy.expiry_seconds,
            sample_size=stats.sample_size,
            wins=stats.wins,
            losses=stats.losses,
            voided=stats.voided,
            win_rate=stats.win_rate,
            win_rate_ci_low=stats.win_rate_ci_low,
            win_rate_ci_high=stats.win_rate_ci_high,
            expectancy_pct=stats.expectancy_pct,
            rng_seed=rng_seed,
            fold_index=fold_index,
        )
        session.add(run_row)
        session.commit()

        results.append(
            BacktestRunResult(
                scenario=scenario.name, stats=stats, trades=trades, backtest_run_id=run_row.id
            )
        )

    return results
