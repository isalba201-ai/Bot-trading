"""H14 — Engulfing candle predicts a reversal (STRATEGIES.md).

Entry trigger: ``engulfing_signal`` fires (+1 bullish engulfing -> bet up,
-1 bearish engulfing -> bet down).
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H14EngulfingReversal:
    code = "H14"

    def __init__(self, expiry_seconds: int = 300):
        self.expiry_seconds = expiry_seconds
        self.label = "H14_engulfing_reversal"
        self.required_features = frozenset({"engulfing_signal"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        signal = features["engulfing_signal"]
        if signal > 0:
            return "CALL"
        if signal < 0:
            return "PUT"
        return None
