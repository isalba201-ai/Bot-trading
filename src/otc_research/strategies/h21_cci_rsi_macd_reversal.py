"""H21 — CCI + RSI overbought confluence confirmed by a bearish MACD
histogram AND a red candle, betting on a reversal down (user-requested,
2026-09-18 — not from STRATEGIES.md's original H1-H20 batch).

Entry trigger, ALL of the following true on the same candle:
  1. ``cci_20 >= cci_threshold`` (default 100, CCI's own canonical
     overbought line — same default H12 already uses).
  2. ``rsi_14 >= rsi_threshold`` (default 70, RSI's own canonical
     overbought line — same default H7 already uses).
  3. ``macd_histogram < 0`` — the MACD histogram bar is below zero
     ("red" on every charting platform that colors histogram bars by
     sign; this is what the user's "el MACD presenta una vela roja ...
     en el volumen del MAC" describes — MACD's histogram IS drawn as a
     bar/volume-style chart, colored red when negative).
  4. ``close < open`` — the candle itself is bearish ("esa vela roja").

All four together -> PUT (bet on continuation of the reversal down).
Deliberately PUT-only: the user described only the overbought/bearish
case, not its oversold/bullish mirror, and "no quiero que te inventes
nada" -- a symmetric CALL variant was not requested and is not added here.

No smoothing/confirmation-candle-count logic beyond this: the "count 2 or
4 candles" the user described is the EXPIRY (h=2 or h=4 candles after the
signal), handled by ``expiry_seconds`` at construction time exactly like
every other Strategy here -- never a second condition inside decide().
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction


class H21CciRsiMacdBearishReversal:
    code = "H21"

    def __init__(
        self,
        cci_threshold: float = 100.0,
        rsi_threshold: float = 70.0,
        expiry_seconds: int = 300,
    ):
        if cci_threshold <= 0:
            raise ValueError("cci_threshold must be positive")
        if not 50.0 < rsi_threshold <= 100.0:
            raise ValueError("rsi_threshold must be between 50 and 100 to mean 'overbought'")
        self.cci_threshold = cci_threshold
        self.rsi_threshold = rsi_threshold
        self.expiry_seconds = expiry_seconds
        self.label = f"H21_cci_rsi_macd_bearish_reversal_cci{cci_threshold}_rsi{rsi_threshold}"
        self.required_features = frozenset(
            {"cci_20", "rsi_14", "macd_histogram", "open", "close"}
        )

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        if features["cci_20"] < self.cci_threshold:
            return None
        if features["rsi_14"] < self.rsi_threshold:
            return None
        if features["macd_histogram"] >= 0:
            return None
        if features["close"] >= features["open"]:
            return None
        return "PUT"
