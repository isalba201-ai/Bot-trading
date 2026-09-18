"""H20 — Donchian breakout confirmed by volatility expansion
(STRATEGIES.md).

Entry trigger: the current close breaks the Donchian channel (as H5)
AND ``atr_expansion_ratio`` clears ``expansion_threshold`` at the same
time -> bet on continuation. The volatility filter is meant to reject
breakouts that happen during otherwise-quiet conditions (more likely to
be noise) and only trust ones that coincide with genuinely expanding
volatility.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H20BreakoutVolatilityConfirmed:
    code = "H20"

    def __init__(self, expansion_threshold: float = 1.2, expiry_seconds: int = 300):
        if expansion_threshold <= 1.0:
            raise ValueError("expansion_threshold must be > 1.0 to mean 'expansion'")
        self.expansion_threshold = expansion_threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H20_breakout_volatility_confirmed_thr{expansion_threshold}"
        self.required_features = frozenset(
            {"donchian_high_20", "donchian_low_20", "atr_expansion_ratio", "close"}
        )

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        if features["atr_expansion_ratio"] < self.expansion_threshold:
            return None
        close = features["close"]
        if close > features["donchian_high_20"]:
            return "CALL"
        if close < features["donchian_low_20"]:
            return "PUT"
        return None
