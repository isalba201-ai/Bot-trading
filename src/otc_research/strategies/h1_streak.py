"""H1 — Continuation after a same-color candle streak (STRATEGIES.md).

Entry trigger: ``min_streak`` or more consecutive same-color candles have
just closed -> bet on continuation in that same direction.

Invalidation: implicitly, anything short of ``min_streak`` (including a
doji, which resets the underlying feature to 0) produces no signal at all
— there is no separate override condition modeled in this first version.

Market regime / session applicability: not modeled here — per
STRATEGIES.md this must be measured empirically (BACKTESTING.md), not
assumed, so this class only implements the entry trigger itself.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H1StreakContinuation:
    code = "H1"

    def __init__(self, min_streak: int = 3, expiry_seconds: int = 300):
        if min_streak < 1:
            raise ValueError("min_streak must be >= 1")
        self.min_streak = min_streak
        self.expiry_seconds = expiry_seconds
        self.label = f"H1_streak_continuation_min{min_streak}"
        self.required_features = frozenset({"same_color_streak"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        streak = features["same_color_streak"]
        if streak >= self.min_streak:
            return "CALL"
        if streak <= -self.min_streak:
            return "PUT"
        return None
