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

VALID_SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class BacktestRunResult:
    scenario: str
    stats: TradeStats
    trades: list[Trade]
    backtest_run_id: int


def _load_candles(session: Session, asset: str, timeframe: str) -> list[SimCandle]:
    rows = (
        session.query(Candle)
        .filter(Candle.asset == asset, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.asc())
        .all()
    )
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
) -> list[BacktestRunResult]:
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

    candles = _load_candles(session, asset, timeframe)
    if len(candles) < 3:
        raise ValueError(
            f"not enough candles for {asset}/{timeframe} to split "
            f"(have {len(candles)}, need at least 3)"
        )
    features_by_timestamp = _load_features(session, asset, timeframe, feature_set_version)

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
        )
        session.add(run_row)
        session.commit()

        results.append(
            BacktestRunResult(
                scenario=scenario.name, stats=stats, trades=trades, backtest_run_id=run_row.id
            )
        )

    return results
