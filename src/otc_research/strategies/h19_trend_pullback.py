"""H19 — Trend + shallow RSI pullback (STRATEGIES.md).

Entry trigger: an established trend (``ema_slope_12_3`` clears
``slope_threshold``) with RSI currently sitting in a SHALLOW pullback
band (``pullback_low``..``pullback_high`` below 50 for an uptrend,
mirrored above 50 for a downtrend) -> bet on continuation with the
trend. Deliberately excludes a deep/extreme RSI reading — that's H7's
territory (an exhaustion-reversal bet), not this hypothesis's "buy the
dip within a trend" mechanism.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H19TrendPullback:
    code = "H19"

    def __init__(
        self,
        slope_threshold: float = 0.0001,
        pullback_low: float = 35.0,
        pullback_high: float = 50.0,
        expiry_seconds: int = 300,
    ):
        if not 0.0 <= pullback_low < pullback_high <= 50.0:
            raise ValueError("must have 0 <= pullback_low < pullback_high <= 50")
        self.slope_threshold = slope_threshold
        self.pullback_low = pullback_low
        self.pullback_high = pullback_high
        self.expiry_seconds = expiry_seconds
        self.label = f"H19_trend_pullback_{pullback_low}_{pullback_high}"
        self.required_features = frozenset({"ema_slope_12_3", "rsi_14"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        slope = features["ema_slope_12_3"]
        rsi = features["rsi_14"]

        if slope > self.slope_threshold and self.pullback_low <= rsi <= self.pullback_high:
            return "CALL"
        if slope < -self.slope_threshold and (100.0 - self.pullback_high) <= rsi <= (
            100.0 - self.pullback_low
        ):
            return "PUT"
        return None
