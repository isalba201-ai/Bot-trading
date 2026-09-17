"""H3 — Short-term momentum predicts next-candle direction (STRATEGIES.md).

Entry trigger: both a rate-of-change measure and an EMA-slope measure
agree, in magnitude and direction, above their respective thresholds ->
bet on continuation. Requiring both to agree (rather than either alone)
is this baseline's confirmation rule; STRATEGIES.md notes such supporting
factors are only worth keeping if they measurably help out-of-sample
(BACKTESTING.md), which this class doesn't decide on its own — it just
implements the two-factor version so the ablation can actually be tested.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H3MomentumContinuation:
    code = "H3"

    def __init__(
        self,
        roc_threshold: float = 0.05,
        slope_threshold: float = 0.0001,
        expiry_seconds: int = 300,
    ):
        self.roc_threshold = roc_threshold
        self.slope_threshold = slope_threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H3_momentum_continuation_roc{roc_threshold}_slope{slope_threshold}"
        self.required_features = frozenset({"roc_10", "ema_slope_12_3"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        roc = features["roc_10"]
        slope = features["ema_slope_12_3"]
        if roc > self.roc_threshold and slope > self.slope_threshold:
            return "CALL"
        if roc < -self.roc_threshold and slope < -self.slope_threshold:
            return "PUT"
        return None
