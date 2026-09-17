"""Walks a candle series bar by bar, asks a Strategy for a decision at
every point using only the point-in-time features available for that
candle, and resolves each signal into a trade under a given
ExecutionScenario.

Nothing here ever fabricates a price: if the data needed to enter or
resolve a trade doesn't exist (end of the series, a candle the feature
pipeline skipped), the trade is recorded as VOID, never guessed at — the
same "if it can't be obtained, it has nothing for that moment" rule
DATA.md applies to candles applies here to trades.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from typing import Mapping, Protocol, Sequence

from otc_research.backtest.execution import ExecutionScenario
from otc_research.backtest.strategy import Direction, Strategy


class RandomSource(Protocol):
    def random(self) -> float: ...


@dataclass(frozen=True)
class SimCandle:
    timestamp: dt.datetime
    open: float
    high: float
    low: float
    close: float


@dataclass(frozen=True)
class Trade:
    signal_time: dt.datetime  # close of the candle that produced the decision
    direction: Direction
    expiry_seconds: int
    result: str  # "WIN" / "LOSS" / "VOID"
    entry_time: dt.datetime | None = None
    entry_price: float | None = None
    exit_time: dt.datetime | None = None
    exit_price: float | None = None
    pnl_pct: float = 0.0
    void_reason: str | None = None  # "signal_dropped" / "insufficient_data"


def _apply_slippage(price: float, direction: Direction, slippage_pct: float) -> float:
    adjustment = slippage_pct / 100.0
    if direction == "CALL":
        return price * (1 + adjustment)  # worse (higher) entry for a buyer
    return price * (1 - adjustment)  # worse (lower) entry for a seller


def _pnl_pct(entry_price: float, exit_price: float, direction: Direction) -> float:
    if direction == "CALL":
        return (exit_price - entry_price) / entry_price * 100.0
    return (entry_price - exit_price) / entry_price * 100.0


def simulate(
    strategy: Strategy,
    candles: Sequence[SimCandle],
    features_by_timestamp: Mapping[dt.datetime, Mapping[str, float]],
    timeframe_seconds: int,
    scenario: ExecutionScenario,
    rng: RandomSource | None = None,
) -> list[Trade]:
    """Runs ``strategy`` over ``candles`` (already sorted ascending,
    belonging to one split — train/validation/test are simulated
    independently, never concatenated, so a trade can never straddle a
    split boundary or use data from a different split to resolve).

    ``rng`` controls the ``signal_drop_probability`` draw — pass a seeded
    ``random.Random(seed)`` for reproducible results (see
    backtest/engine.py, which records the seed used on every BacktestRun).
    Defaults to an unseeded ``random.Random()`` when not given.
    """
    if strategy.expiry_seconds % timeframe_seconds != 0:
        raise ValueError(
            f"strategy expiry_seconds={strategy.expiry_seconds} is not a whole number of "
            f"{timeframe_seconds}s candles"
        )
    expiry_candles = strategy.expiry_seconds // timeframe_seconds
    rng = rng if rng is not None else random.Random()

    trades: list[Trade] = []
    n = len(candles)

    for i, candle in enumerate(candles):
        stored_feats = features_by_timestamp.get(candle.timestamp)
        # Reserved keys always available from the candle itself (raw price,
        # not a computed/stored Feature row) — needed by strategies like
        # H5's breakout, which compares the current close to a baseline.
        # Stored features win on a name clash (there shouldn't be one).
        feats: dict[str, float] = {
            "open": candle.open,
            "high": candle.high,
            "low": candle.low,
            "close": candle.close,
            **(stored_feats or {}),
        }
        if not strategy.required_features.issubset(feats.keys()):
            continue  # not enough history yet for this strategy's features

        direction = strategy.decide(feats)
        if direction is None:
            continue

        if rng.random() < scenario.signal_drop_probability:
            trades.append(
                Trade(
                    signal_time=candle.timestamp,
                    direction=direction,
                    expiry_seconds=strategy.expiry_seconds,
                    result="VOID",
                    void_reason="signal_dropped",
                )
            )
            continue

        entry_idx = i + 1 + scenario.entry_delay_candles
        exit_idx = entry_idx + expiry_candles
        if exit_idx >= n:
            trades.append(
                Trade(
                    signal_time=candle.timestamp,
                    direction=direction,
                    expiry_seconds=strategy.expiry_seconds,
                    result="VOID",
                    void_reason="insufficient_data",
                )
            )
            continue

        entry_candle = candles[entry_idx]
        exit_candle = candles[exit_idx]
        entry_price = _apply_slippage(entry_candle.open, direction, scenario.slippage_pct)
        exit_price = exit_candle.close
        pnl_pct = _pnl_pct(entry_price, exit_price, direction)

        trades.append(
            Trade(
                signal_time=candle.timestamp,
                direction=direction,
                expiry_seconds=strategy.expiry_seconds,
                result="WIN" if pnl_pct > 0 else "LOSS",
                entry_time=entry_candle.timestamp,
                entry_price=entry_price,
                exit_time=exit_candle.timestamp,
                exit_price=exit_price,
                pnl_pct=pnl_pct,
            )
        )

    return trades
