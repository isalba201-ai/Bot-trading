"""H18 — CCI extreme confirmed by an engulfing candle (STRATEGIES.md).

Entry trigger: ``cci_20`` clears an extreme threshold AND
``engulfing_signal`` fires in the reversion direction (an oscillator
extreme AND actual price-action reversal behavior, not just the
oscillator alone — the same logic as H7's RSI+streak confirmation, with
a different oscillator/pattern pair).
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H18CciEngulfingConfirmed:
    code = "H18"

    def __init__(self, cci_threshold: float = 100.0, expiry_seconds: int = 300):
        if cci_threshold <= 0:
            raise ValueError("cci_threshold must be positive")
        self.cci_threshold = cci_threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H18_cci_engulfing_confirmed_thr{cci_threshold}"
        self.required_features = frozenset({"cci_20", "engulfing_signal"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        cci = features["cci_20"]
        engulfing = features["engulfing_signal"]
        if cci >= self.cci_threshold and engulfing < 0:
            return "PUT"
        if cci <= -self.cci_threshold and engulfing > 0:
            return "CALL"
        return None
