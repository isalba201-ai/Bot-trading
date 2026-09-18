"""H16 — MACD cross confirmed by RSI regime (STRATEGIES.md).

Entry trigger: ``macd_cross_signal`` fires AND ``rsi_14`` is already on
the same side of 50 as the cross (bullish cross + RSI > 50, or bearish
cross + RSI < 50) -> bet on continuation. The RSI filter is meant to
avoid a cross that's either premature (RSI still on the other side) or
already exhausted (RSI at a hard overbought/oversold extreme) — see
``rsi_ceiling``.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H16MacdRsiConfirmed:
    code = "H16"

    def __init__(self, rsi_midpoint: float = 50.0, rsi_ceiling: float = 80.0, expiry_seconds: int = 300):
        if not 0.0 < rsi_midpoint < 100.0:
            raise ValueError("rsi_midpoint must be between 0 and 100")
        if not rsi_midpoint < rsi_ceiling <= 100.0:
            raise ValueError("rsi_ceiling must be between rsi_midpoint and 100")
        self.rsi_midpoint = rsi_midpoint
        self.rsi_ceiling = rsi_ceiling
        self.expiry_seconds = expiry_seconds
        self.label = f"H16_macd_rsi_confirmed_{rsi_midpoint}_{rsi_ceiling}"
        self.required_features = frozenset({"macd_cross_signal", "rsi_14"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        cross = features["macd_cross_signal"]
        rsi = features["rsi_14"]
        if cross > 0 and self.rsi_midpoint <= rsi <= self.rsi_ceiling:
            return "CALL"
        if cross < 0 and (100.0 - self.rsi_ceiling) <= rsi <= self.rsi_midpoint:
            return "PUT"
        return None
