"""H7 — RSI extreme combined with a price-action confirmation
(STRATEGIES.md).

Entry trigger: RSI is oversold AND the most recently closed candle is
already bullish (the "price-action confirmation" — the reversal has
visibly started, not just implied by RSI alone) -> bet on continuation up.
Symmetric for overbought + a bearish confirmation candle -> bet down.

Requiring the confirmation candle (rather than acting on RSI alone) is
this baseline's operationalization of "combined with a price-action
confirmation" from STRATEGIES.md.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H7RsiExtremeConfirmed:
    code = "H7"

    def __init__(
        self, oversold: float = 30.0, overbought: float = 70.0, expiry_seconds: int = 300
    ):
        if not 0.0 <= oversold < overbought <= 100.0:
            raise ValueError("must have 0 <= oversold < overbought <= 100")
        self.oversold = oversold
        self.overbought = overbought
        self.expiry_seconds = expiry_seconds
        self.label = f"H7_rsi_extreme_confirmed_{oversold}_{overbought}"
        self.required_features = frozenset({"rsi_14", "same_color_streak"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        rsi = features["rsi_14"]
        streak = features["same_color_streak"]
        if rsi <= self.oversold and streak > 0:
            return "CALL"
        if rsi >= self.overbought and streak < 0:
            return "PUT"
        return None
