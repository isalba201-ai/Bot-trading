"""H9 — Hour-of-day / session bias, independent of any price-action signal
(STRATEGIES.md).

Unlike H1-H8/H10, this hypothesis has no price-action trigger at all by
definition — it's a claim that a specific hour has a systematic
directional bias on its own. That means there is no single "H9 strategy"
to write: hardcoding e.g. "14:00 UTC -> CALL" from intuition would be
exactly the kind of unproven assumption STRATEGIES.md and BACKTESTING.md
exist to prevent. Instead, this class tests ONE candidate (hour,
direction) pair at a time — the actual hypothesis test is running it
across all 24 hours x both directions (Phase 6's parameter-sensitivity
sweep is the natural place for that) and seeing whether any specific
pair's backtested win rate clears STRATEGIES.md's bar, not assuming one
does.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction

_VALID_DIRECTIONS = ("CALL", "PUT")


class H9SessionBias:
    code = "H9"

    def __init__(self, hour_utc: int, direction: Direction, expiry_seconds: int = 300):
        if not 0 <= hour_utc <= 23:
            raise ValueError("hour_utc must be between 0 and 23")
        if direction not in _VALID_DIRECTIONS:
            raise ValueError(f"direction must be one of {_VALID_DIRECTIONS}")
        self.hour_utc = hour_utc
        self.direction: Direction = direction
        self.expiry_seconds = expiry_seconds
        self.label = f"H9_session_bias_hour{hour_utc}_{direction}"
        self.required_features = frozenset({"hour_utc"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        if int(features["hour_utc"]) == self.hour_utc:
            return self.direction
        return None
