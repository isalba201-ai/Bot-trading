"""H11 — MACD signal-line crossover predicts continuation (STRATEGIES.md).

Entry trigger: ``macd_cross_signal`` fires (the MACD line just crossed
its own signal line) -> bet on continuation in the crossing direction.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H11MacdCrossContinuation:
    code = "H11"

    def __init__(self, expiry_seconds: int = 300):
        self.expiry_seconds = expiry_seconds
        self.label = "H11_macd_cross_continuation"
        self.required_features = frozenset({"macd_cross_signal"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        cross = features["macd_cross_signal"]
        if cross > 0:
            return "CALL"
        if cross < 0:
            return "PUT"
        return None
