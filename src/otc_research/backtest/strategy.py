"""The interface every hypothesis strategy (Phase 5: H1-H10) implements to
be runnable by this engine. No concrete H1-H10 strategy exists yet — see
STRATEGIES.md; this module only fixes the shape they must have, so Phase 4
(this engine) can be built and tested against a throwaway example without
waiting on Phase 5.

A Strategy is deliberately given nothing but a dict of feature values for
one candle — never a data source, never anything about what happened
after that candle. That's what makes engine-level look-ahead bias
structurally hard to introduce: the strategy physically cannot see the
future because the engine never hands it any.

The dict always carries the reserved keys ``"open"``/``"high"``/
``"low"``/``"close"`` (the current candle's own raw OHLC — this is
needed by e.g. a breakout strategy comparing the close to a computed
baseline), plus whatever already-computed, already point-in-time-safe
``Feature`` rows exist for that timestamp (see FEATURES.md). List any of
the reserved keys in ``required_features`` too if your strategy needs
them — they're merged in by the engine, not stored in the database.
"""

from __future__ import annotations

from typing import Mapping, Protocol, runtime_checkable

Direction = str  # "CALL" or "PUT"


@runtime_checkable
class Strategy(Protocol):
    #: Hypothesis code this strategy implements, e.g. "H3" (STRATEGIES.md).
    #: None for an ad-hoc/example strategy not registered as a hypothesis.
    code: str | None

    #: Human-readable label, always set (used for BacktestRun.strategy_label
    #: even when `code` is None).
    label: str

    #: Target holding time in seconds for a trade this strategy opens.
    expiry_seconds: int

    #: Feature names (see FEATURES.md) this strategy needs to make a
    #: decision. The engine only calls decide() once every one of these is
    #: present (not NaN/missing) for a given candle — a strategy should
    #: never need to defensively check for missing keys itself.
    required_features: frozenset[str]

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        """Return "CALL", "PUT", or None (no setup) given feature values
        for exactly one point in time. Must be a pure function of
        ``features`` — no internal state that depends on call order, so
        that the same features always produce the same decision regardless
        of what the engine has done before or will do after.
        """
        ...
