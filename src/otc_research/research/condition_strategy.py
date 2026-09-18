"""Adapts a discovered ``discovery.Condition`` into the Phase 4
``backtest.strategy.Strategy`` protocol, so a condition found by the
statistical-discovery search can run through the EXISTING, already
battle-tested backtest engine — ``run_backtest``, ``robustness``,
``walkforward`` — completely unchanged. This is the bridge point
``research.candidacy`` uses to apply H1-H20's exact same rigor to
discovered conditions instead of hand-designed strategies.
"""

from __future__ import annotations

from typing import Mapping

from otc_research.backtest.strategy import Direction
from otc_research.research.discovery import Condition


def perturb_condition(condition: Condition, pct: float) -> Condition:
    """Widens (``pct`` > 0) or narrows (``pct`` < 0) every part's bin by
    ``pct`` of that bin's width, symmetrically. Used to test whether a
    condition's edge survives small threshold perturbations — BACKTESTING.md's
    OVERFITTED/FRAGILE check, applied to a discovered condition instead of
    a hand-picked strategy parameter.
    """
    if pct == 0.0:
        return condition
    new_parts = tuple(
        (feature, low - (high - low) * pct, high + (high - low) * pct)
        for feature, low, high in condition.parts
    )
    return Condition(parts=new_parts)


class ConditionStrategy:
    """A discovered condition, in one fixed direction, as a Strategy.
    Not tied to a pre-registered STRATEGIES.md hypothesis (``code`` stays
    None) — its provenance is the ``ConditionTrial`` row it came from,
    referenced via ``label`` instead.
    """

    code: str | None = None

    def __init__(
        self,
        condition: Condition,
        direction: Direction,
        expiry_seconds: int,
        *,
        edge_perturbation_pct: float = 0.0,
        label: str | None = None,
    ):
        if direction not in ("CALL", "PUT"):
            raise ValueError("direction must be 'CALL' or 'PUT'")
        self.direction: Direction = direction
        self.expiry_seconds = expiry_seconds
        self.edge_perturbation_pct = edge_perturbation_pct
        self._condition = perturb_condition(condition, edge_perturbation_pct)
        self.required_features = frozenset(f for f, _, _ in self._condition.parts)
        self.label = label or f"condition_{direction}_{condition.label()}_pert{edge_perturbation_pct}"

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        return self.direction if self._condition.matches_row(features) else None
