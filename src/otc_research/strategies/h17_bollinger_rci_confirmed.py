"""H17 — Bollinger extreme confirmed by RCI extreme (STRATEGIES.md).

Entry trigger: ``bb_pct_b_20`` AND ``rci_9`` both clear their extreme
thresholds on the SAME side -> bet on reversion. Two mechanistically
distinct exhaustion measures (a volatility-band measure and a
rank-correlation measure) agreeing is the whole point — either alone is
H4 or H13.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H17BollingerRciConfirmed:
    code = "H17"

    def __init__(
        self,
        lower_pct_b: float = 0.0,
        upper_pct_b: float = 1.0,
        rci_threshold: float = 80.0,
        expiry_seconds: int = 300,
    ):
        if lower_pct_b >= upper_pct_b:
            raise ValueError("lower_pct_b must be < upper_pct_b")
        if not 0.0 < rci_threshold <= 100.0:
            raise ValueError("rci_threshold must be between 0 and 100")
        self.lower_pct_b = lower_pct_b
        self.upper_pct_b = upper_pct_b
        self.rci_threshold = rci_threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H17_bollinger_rci_confirmed_{lower_pct_b}_{upper_pct_b}_{rci_threshold}"
        self.required_features = frozenset({"bb_pct_b_20", "rci_9"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        pct_b = features["bb_pct_b_20"]
        rci = features["rci_9"]
        if pct_b >= self.upper_pct_b and rci >= self.rci_threshold:
            return "PUT"
        if pct_b <= self.lower_pct_b and rci <= -self.rci_threshold:
            return "CALL"
        return None
