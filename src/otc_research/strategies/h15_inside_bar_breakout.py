"""H15 — Inside-bar ("mother bar") breakout predicts continuation
(STRATEGIES.md).

Entry trigger: ``inside_bar_breakout_signal`` fires (+1 = close just
broke above the most recent inside-bar consolidation's mother-bar high,
-1 = broke below its low) -> bet on continuation in the breakout
direction.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H15InsideBarBreakout:
    code = "H15"

    def __init__(self, expiry_seconds: int = 300):
        self.expiry_seconds = expiry_seconds
        self.label = "H15_inside_bar_breakout"
        self.required_features = frozenset({"inside_bar_breakout_signal"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        signal = features["inside_bar_breakout_signal"]
        if signal > 0:
            return "CALL"
        if signal < 0:
            return "PUT"
        return None
