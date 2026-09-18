"""The fixed, documented confidence-label rule (approved plan point 9:
"ALTA/MEDIA/BAJA, derived from margin+CI width via a fixed, documented
rule ... never hand-waved per-signal").

Thresholds are deliberately tied to values already used elsewhere in this
project rather than invented fresh: MEDIA's floor is
``research.candidacy.CandidacyThresholds.min_margin_over_break_even``
(0.03), and ALTA additionally requires the Wilson CI lower bound itself
to clear break-even -- the same "CI lower bound, not just the point
estimate" discipline every edge classification in this codebase already
follows (``backtest.robustness``, ``backtest.walkforward``).
"""

from __future__ import annotations

ALTA = "ALTA"
MEDIA = "MEDIA"
BAJA = "BAJA"

#: Mirrors research.candidacy.CandidacyThresholds.min_margin_over_break_even
#: -- a signal below this margin has not even cleared the bar this
#: project already requires before treating a condition as a candidate.
MEDIA_MIN_MARGIN = 0.03
ALTA_MIN_MARGIN = 0.05


def classify_confidence(
    margin_over_break_even: float | None,
    win_rate_ci_low: float | None,
    break_even_win_rate: float | None,
) -> str:
    """Never fabricates ALTA/MEDIA from incomplete inputs -- any missing
    grounding value (no margin, no CI, no break-even figure) is BAJA,
    the same "insufficient information is not evidence" rule
    ``backtest.metrics``/``research.baseline`` already apply to win
    rates and p-values.
    """
    if margin_over_break_even is None or win_rate_ci_low is None or break_even_win_rate is None:
        return BAJA
    if margin_over_break_even >= ALTA_MIN_MARGIN and win_rate_ci_low > break_even_win_rate:
        return ALTA
    if margin_over_break_even >= MEDIA_MIN_MARGIN:
        return MEDIA
    return BAJA
