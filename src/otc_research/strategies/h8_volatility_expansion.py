"""H8 — Volatility expansion following contraction, an ATR regime change
(STRATEGIES.md).

Entry trigger: ``atr_expansion_ratio`` (current ATR vs. its own preceding,
calmer baseline — see FEATURES.md) crosses above ``expansion_threshold``
-> volatility has visibly expanded relative to the recent past. Raw
volatility expanding has no direction of its own, so the trade direction
comes from the EMA slope at the same instant (see FEATURES.md).

This baseline operationalizes "expansion following contraction" as
simply "current ATR is well above its own recent (already-baked-in)
average" — a high ratio already implies the recent baseline was calmer —
rather than requiring a separately-detected prior contraction phase.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H8VolatilityExpansion:
    code = "H8"

    def __init__(
        self,
        expansion_threshold: float = 1.5,
        slope_threshold: float = 0.0001,
        expiry_seconds: int = 300,
    ):
        if expansion_threshold <= 1.0:
            raise ValueError("expansion_threshold must be > 1.0 to mean 'expansion'")
        self.expansion_threshold = expansion_threshold
        self.slope_threshold = slope_threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H8_volatility_expansion_thr{expansion_threshold}"
        self.required_features = frozenset({"atr_expansion_ratio", "ema_slope_12_3"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        if features["atr_expansion_ratio"] < self.expansion_threshold:
            return None
        slope = features["ema_slope_12_3"]
        if slope > self.slope_threshold:
            return "CALL"
        if slope < -self.slope_threshold:
            return "PUT"
        return None
