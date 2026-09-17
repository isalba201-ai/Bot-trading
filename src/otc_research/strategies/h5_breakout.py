"""H5 — Breakout of a recent N-candle high/low range (STRATEGIES.md).

Entry trigger: the current close is beyond the highest high (or lowest
low) of the preceding 20 candles -> bet on continuation in the breakout's
direction.

Needs the current close price itself, not just computed features — see
backtest/strategy.py's reserved ``"close"`` key, merged in by the
simulator from the raw candle.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H5DonchianBreakout:
    code = "H5"

    def __init__(self, expiry_seconds: int = 300):
        self.expiry_seconds = expiry_seconds
        self.label = "H5_donchian_breakout"
        self.required_features = frozenset({"donchian_high_20", "donchian_low_20", "close"})

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        close = features["close"]
        if close > features["donchian_high_20"]:
            return "CALL"
        if close < features["donchian_low_20"]:
            return "PUT"
        return None
