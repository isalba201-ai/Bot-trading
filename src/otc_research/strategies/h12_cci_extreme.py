"""H12 — CCI extreme predicts reversion (STRATEGIES.md).

Entry trigger: ``cci_20`` clears ``threshold`` on the high side -> bet on
reversion down; clears ``-threshold`` on the low side -> bet on
reversion up.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H12CciExtremeReversion:
    code = "H12"

    def __init__(self, threshold: float = 100.0, expiry_seconds: int = 300):
        if threshold <= 0:
            raise ValueError("threshold must be positive")
        self.threshold = threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H12_cci_extreme_reversion_thr{threshold}"
        self.required_features = frozenset({"cci_20"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        cci = features["cci_20"]
        if cci >= self.threshold:
            return "PUT"
        if cci <= -self.threshold:
            return "CALL"
        return None
