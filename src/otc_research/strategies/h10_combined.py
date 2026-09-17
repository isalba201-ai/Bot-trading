"""H10 — Combined trend + market structure + momentum confirmation
(STRATEGIES.md).

Entry trigger: all three must agree — EMA slope, confirmed swing
structure (see FEATURES.md's ``structure_bias``), and rate-of-change
momentum all pointing the same direction -> bet on continuation. This is
this project's most conservative baseline by construction: three
independent confirmations required, not one.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H10CombinedTrendStructureMomentum:
    code = "H10"

    def __init__(
        self,
        slope_threshold: float = 0.0001,
        roc_threshold: float = 0.05,
        expiry_seconds: int = 300,
    ):
        self.slope_threshold = slope_threshold
        self.roc_threshold = roc_threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H10_combined_trend_structure_momentum_{slope_threshold}_{roc_threshold}"
        self.required_features = frozenset({"ema_slope_12_3", "structure_bias", "roc_10"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        slope = features["ema_slope_12_3"]
        structure = features["structure_bias"]
        roc = features["roc_10"]

        if slope > self.slope_threshold and structure == 1.0 and roc > self.roc_threshold:
            return "CALL"
        if slope < -self.slope_threshold and structure == -1.0 and roc < -self.roc_threshold:
            return "PUT"
        return None
