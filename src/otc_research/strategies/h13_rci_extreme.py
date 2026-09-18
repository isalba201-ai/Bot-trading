"""H13 — RCI extreme predicts reversion (STRATEGIES.md).

Entry trigger: ``rci_9`` clears ``threshold`` on the high side (price has
risen on almost every candle in the window, in rank terms) -> bet on
reversion down; clears ``-threshold`` on the low side -> bet on
reversion up. Mechanistically distinct from H7's RSI extreme: RCI is
rank-based (did price consistently rise/fall in order), not
magnitude-based (how big were the gains/losses).
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H13RciExtremeReversion:
    code = "H13"

    def __init__(self, threshold: float = 80.0, expiry_seconds: int = 300):
        if not 0 < threshold <= 100:
            raise ValueError("threshold must be between 0 and 100")
        self.threshold = threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H13_rci_extreme_reversion_thr{threshold}"
        self.required_features = frozenset({"rci_9"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        rci = features["rci_9"]
        if rci >= self.threshold:
            return "PUT"
        if rci <= -self.threshold:
            return "CALL"
        return None
