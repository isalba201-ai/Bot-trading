"""Step 9 addendum, point 17: the NO_TRADE filter as its own, separate
hypothesis — never a post-hoc TRAIN-only comparison. ``FilteredStrategy``
wraps an existing base ``Strategy`` (H1-H20 or a discovered
``ConditionStrategy``) unchanged and blocks its signal whenever a regime
predicate is active, answering "does adding this filter to an ALREADY-
DEFINED strategy actually improve it," never "do quiet periods have small
moves" (a different, already-answered question — see
``PREDICTABILITY_AUDIT.md``'s no-trade-filter finding).

Each ``FilteredStrategy`` instance is evaluated through the exact same
``research.candidacy.evaluate_candidacy`` four-gate funnel as its
unfiltered base — it is a genuinely new, separately pre-registered
hypothesis (new ``label``, own fresh TEST touch if it gets there), never a
modification of an already-TESTed verdict.
"""

from __future__ import annotations

from typing import Callable, Mapping

from otc_research.backtest.strategy import Direction, Strategy
from otc_research.research.regimes import VOL_CONTRACTION_THRESHOLD

Predicate = Callable[[Mapping[str, float]], bool]


def volatility_contraction_filter(
    threshold: float = VOL_CONTRACTION_THRESHOLD,
) -> Predicate:
    """True (block trading) whenever ``atr_expansion_ratio <= threshold``
    — the same "contracting_vol" boundary ``research.regimes`` already
    uses, reused here rather than redefined, so this filter and the
    baseline regime labels never silently drift apart.
    """

    def predicate(features: Mapping[str, float]) -> bool:
        return features["atr_expansion_ratio"] <= threshold

    return predicate


class FilteredStrategy:
    """``base.decide(features)`` unchanged, except suppressed (returns
    None) whenever ``block_predicate(features)`` is True. Requires every
    feature both ``base`` and ``block_predicate`` need — the engine
    already only calls ``decide()`` once every required feature is
    present (see ``backtest.strategy.Strategy``'s own contract), so this
    wrapper adds ``block_predicate``'s own feature needs to
    ``required_features`` rather than defensively checking for them here.
    """

    code: str | None = None

    def __init__(
        self,
        base: Strategy,
        block_predicate: Predicate,
        *,
        filter_label: str,
        filter_required_features: frozenset[str] = frozenset({"atr_expansion_ratio"}),
    ):
        self.base = base
        self.block_predicate = block_predicate
        self.expiry_seconds = base.expiry_seconds
        self.required_features = frozenset(base.required_features) | filter_required_features
        self.label = f"{base.label}__NOTRADE_{filter_label}"

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        if self.block_predicate(features):
            return None
        return self.base.decide(features)
