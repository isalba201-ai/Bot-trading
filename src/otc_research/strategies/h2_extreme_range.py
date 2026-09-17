"""H2 — Reversion after an extreme-range candle (STRATEGIES.md).

Entry trigger: the just-closed candle's range is at least
``range_ratio_threshold`` times the average range of the preceding 20
candles (an "extreme-range candle"), and it had a clear color -> bet on
reversion against that candle's direction (a large bullish candle fades
down, a large bearish candle fades up).

Invalidation: an extreme-range doji (open == close despite a large
high-low range) has no directional read and produces no signal.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H2ExtremeRangeReversion:
    code = "H2"

    def __init__(self, range_ratio_threshold: float = 2.0, expiry_seconds: int = 300):
        if range_ratio_threshold <= 1.0:
            raise ValueError("range_ratio_threshold must be > 1.0 to mean 'extreme'")
        self.range_ratio_threshold = range_ratio_threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H2_extreme_range_reversion_thr{range_ratio_threshold}"
        self.required_features = frozenset({"range_ratio_20", "same_color_streak"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        if features["range_ratio_20"] < self.range_ratio_threshold:
            return None
        streak = features["same_color_streak"]
        if streak > 0:
            return "PUT"  # extreme bullish candle -> fade down
        if streak < 0:
            return "CALL"  # extreme bearish candle -> fade up
        return None
