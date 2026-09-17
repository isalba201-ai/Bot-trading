"""H6 — Level rejection via a long wick against the prevailing range
(STRATEGIES.md).

Entry trigger: the just-closed candle's upper wick makes up at least
``wick_ratio_threshold`` of its full range (price pushed up and rejected
back down) -> bet on reversion down; symmetric for a long lower wick ->
bet on reversion up.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H6WickRejection:
    code = "H6"

    def __init__(self, wick_ratio_threshold: float = 0.6, expiry_seconds: int = 300):
        if not 0.0 < wick_ratio_threshold < 1.0:
            raise ValueError("wick_ratio_threshold must be between 0 and 1")
        self.wick_ratio_threshold = wick_ratio_threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H6_wick_rejection_thr{wick_ratio_threshold}"
        self.required_features = frozenset({"upper_wick_ratio", "lower_wick_ratio"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        if features["upper_wick_ratio"] >= self.wick_ratio_threshold:
            return "PUT"
        if features["lower_wick_ratio"] >= self.wick_ratio_threshold:
            return "CALL"
        return None
