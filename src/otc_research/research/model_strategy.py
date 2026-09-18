"""Step 10 (extended systematic search): adapts a fitted ML model
(``research.models.fit_and_evaluate``) into the ``backtest.strategy.
Strategy`` protocol, so a model that shows a promising VALIDATION-only
read can go through the EXACT SAME four-gate ``research.candidacy.
evaluate_candidacy`` funnel every other strategy in this project does —
sample size/margin, robustness (here: sensitivity to the probability
threshold), walk-forward, and a genuinely held-out TEST touch. Without
this, a model's VALIDATION read would stay purely informational (as it
was in Step 7's pipeline) and never receive the same TEST-once discipline
a discovered condition or a hand-designed strategy already gets — this
module closes that gap.

The model itself is fit exactly ONCE, on TRAIN rows only
(``research.models.fit_and_evaluate``), before being wrapped here — this
class never refits; it only calls ``predict_proba`` on whatever feature
row the backtest engine hands it. Gate 1's own TRAIN re-run is therefore
an in-sample read (same as it already is for a discovered ``Condition``,
which is likewise mined on the full TRAIN split before being re-checked
there) — the walk-forward and TEST gates are what actually test out-of-
sample behavior, exactly as for every other strategy family here.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from otc_research.backtest.strategy import Direction


class ModelStrategy:
    """A fitted sklearn classifier, read as a binary CALL/PUT signal:
    predicts P(win) for ``target_col`` (``call_wins_h`` or ``put_wins_h``)
    and bets ``direction`` whenever that probability clears
    ``probability_threshold``. Not tied to a STRATEGIES.md hypothesis
    code (``code`` stays None) — provenance is the ``label`` (model
    family + target + feature set), same convention as
    ``ConditionStrategy``.
    """

    code: str | None = None

    def __init__(
        self,
        model,
        feature_cols: Sequence[str],
        direction: Direction,
        expiry_seconds: int,
        *,
        probability_threshold: float = 0.5,
        label: str | None = None,
    ):
        if direction not in ("CALL", "PUT"):
            raise ValueError("direction must be 'CALL' or 'PUT'")
        self.model = model
        self.feature_cols = tuple(feature_cols)
        self.direction: Direction = direction
        self.expiry_seconds = expiry_seconds
        self.probability_threshold = probability_threshold
        self.required_features = frozenset(self.feature_cols)
        self.label = label or f"model_{direction}_thr{probability_threshold}"

    def decide(self, features: Mapping[str, float]) -> Direction | None:
        row = [[features[f] for f in self.feature_cols]]
        probability = self.model.predict_proba(row)[0, 1]
        return self.direction if probability > self.probability_threshold else None
