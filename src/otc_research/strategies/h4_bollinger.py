"""H4 — Mean reversion from Bollinger Band extremes (STRATEGIES.md).

Entry trigger: close sits at or beyond the upper band (``bb_pct_b_20`` >=
``upper_pct_b``) -> bet on reversion down; at or beyond the lower band
(``bb_pct_b_20`` <= ``lower_pct_b``) -> bet on reversion up.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H4BollingerMeanReversion:
    code = "H4"

    def __init__(
        self, lower_pct_b: float = 0.0, upper_pct_b: float = 1.0, expiry_seconds: int = 300
    ):
        if lower_pct_b >= upper_pct_b:
            raise ValueError("lower_pct_b must be < upper_pct_b")
        self.lower_pct_b = lower_pct_b
        self.upper_pct_b = upper_pct_b
        self.expiry_seconds = expiry_seconds
        self.label = f"H4_bollinger_mean_reversion_{lower_pct_b}_{upper_pct_b}"
        self.required_features = frozenset({"bb_pct_b_20"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        pct_b = features["bb_pct_b_20"]
        if pct_b >= self.upper_pct_b:
            return "PUT"
        if pct_b <= self.lower_pct_b:
            return "CALL"
        return None
