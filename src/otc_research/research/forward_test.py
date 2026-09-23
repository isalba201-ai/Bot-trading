"""Forward/paper testing of the two candidates that mechanically cleared
every candidacy gate (`BINARY_OPTIONS_BACKTEST_REPORT.md`'s H9 finding
and `STEP10_EXTENDED_SEARCH_REPORT.md`'s ML finding) — both were flagged
as fragile (razor-thin TEST margin, high payout-crossover, provenance
from a large multi-candidate search) and explicitly NOT treated as
validated. The only statistically honest way to find out whether either
is real is to check it against data it has never touched, as that data
actually arrives — never re-touching the original historical TEST split.

Each candidate here is FROZEN exactly as it was when originally evaluated
-- the same fixed rule (H9) or the same fitted model object (ML10,
reloaded from the pickle in data/forward_test_models/, never refit on
new data as it arrives, which would silently turn this into a moving,
retrained target and defeat the whole point of a held-out check).

This module only decides WHETHER a candidate's condition fires on a given
point-in-time feature row -- it has no opinion about fetching data,
persisting Signal rows, or scheduling; see
scripts/run_forward_test_poll.py for that.
"""

from __future__ import annotations

import datetime as dt
import functools
from dataclasses import dataclass
from pathlib import Path

from otc_research.backtest.strategy import Strategy
from otc_research.research.discovery import Condition
from otc_research.research.model_strategy import ModelStrategy

FORWARD_TEST_MODELS_DIR = Path("data/forward_test_models")


@dataclass(frozen=True)
class ForwardTestCandidate:
    #: Hypothesis code this is registered under (see scripts/seed_hypotheses.py).
    code: str
    asset: str
    timeframe: str
    expiry_seconds: int
    payout: float
    #: Frozen, deterministic status string from the backtest report this
    #: candidate came from -- never updated by forward-test results
    #: (those go in the SignalDecision/performance tables instead).
    backtest_status: str
    #: Only a candle strictly at or after this timestamp may ever produce
    #: a forward-test signal -- guards against ever silently reprocessing
    #: the historical TRAIN/VALIDATION/TEST data this candidate was
    #: already discovered and evaluated on (that data ends 2026-09-18 for
    #: both candidates; this is fixed once here, not derived at runtime
    #: from "now", so it never drifts).
    forward_test_start: dt.datetime
    strategy_factory: "callable[[], Strategy]"

    @property
    def strategy(self) -> Strategy:
        return self.strategy_factory()


def _h9_fwd_strategy() -> Strategy:
    from otc_research.research.condition_strategy import ConditionStrategy

    condition = Condition(parts=(("hour_utc", 5.0, 6.0),))  # hour_utc == 6 (UTC)
    return ConditionStrategy(condition, "CALL", 4500, label="H9_FWD_hour_utc_eq_6_CALL")


@functools.lru_cache(maxsize=1)
def _load_ml10_fwd_model():
    import joblib

    path = FORWARD_TEST_MODELS_DIR / "ml10_fwd_usdjpy_1h_call_wins_3_randomforest.joblib"
    return joblib.load(path)


def _ml10_fwd_strategy() -> Strategy:
    payload = _load_ml10_fwd_model()
    return ModelStrategy(
        payload["model"], payload["feature_cols"], "CALL", 10800,
        probability_threshold=0.5, label="ML10_FWD_USDJPY_1h_randomforest",
    )


#: The two, and only two, candidates under forward test -- both flagged
#: fragile at the time they were frozen; see each report cross-referenced
#: above. Adding a third requires the same discipline: it must have
#: already cleared all four backtest gates first, never a shortcut.
FORWARD_TEST_CANDIDATES: tuple[ForwardTestCandidate, ...] = (
    ForwardTestCandidate(
        code="H9_FWD",
        asset="EUR_USD",
        timeframe="15m",
        expiry_seconds=4500,
        payout=0.85,
        backtest_status=(
            "TEST 70.0%, n=40, CI low=54.57% (break-even 54.05%) -- ACCEPTED "
            "mechanically, flagged fragile: payout crossover 83.25%, thin n. "
            "See BINARY_OPTIONS_BACKTEST_REPORT.md Section 3."
        ),
        forward_test_start=dt.datetime(2026, 9, 19),
        strategy_factory=_h9_fwd_strategy,
    ),
    ForwardTestCandidate(
        code="ML10_FWD",
        asset="USD_JPY",
        timeframe="1h",
        expiry_seconds=10800,
        payout=0.85,
        backtest_status=(
            "TEST 57.67%, n=730, CI low=54.06% (break-even 54.05%) -- ACCEPTED "
            "mechanically, flagged fragile: payout crossover 85.0%, margin ~0, "
            "1-of-11 correlated TEST arrivals from the same dataset/window. "
            "See STEP10_EXTENDED_SEARCH_REPORT.md Section 3."
        ),
        forward_test_start=dt.datetime(2026, 9, 19),
        strategy_factory=_ml10_fwd_strategy,
    ),
)
