import datetime as dt

import numpy as np
import pytest

from otc_research.config import BacktestConfig, ExecutionScenarioConfig
from otc_research.db.models import Candle
from otc_research.research.candidacy import CandidacyVerdict


def _config() -> BacktestConfig:
    return BacktestConfig(
        train_fraction=0.6,
        validation_fraction=0.2,
        realistic=ExecutionScenarioConfig(
            entry_delay_candles=1, signal_drop_probability=0.02, slippage_pct=0.01
        ),
        pessimistic=ExecutionScenarioConfig(
            entry_delay_candles=2, signal_drop_probability=0.05, slippage_pct=0.03
        ),
    )


def _verdict(accepted: bool, rejected_at_gate: str | None) -> CandidacyVerdict:
    return CandidacyVerdict(accepted=accepted, rejected_at_gate=rejected_at_gate, reason="test")


# --- _classify_conclusion (pure logic) -------------------------------------


def test_classify_conclusion_is_d_when_nothing_significant_anywhere():
    from scripts import run_research_pipeline as rp

    results = [{"n_significant": 0, "candidacy_verdicts": []}]
    assert rp._classify_conclusion(results) == "D"


def test_classify_conclusion_is_a_when_any_verdict_is_accepted():
    from scripts import run_research_pipeline as rp

    results = [
        {
            "n_significant": 2,
            "candidacy_verdicts": [
                {"verdict": _verdict(False, "sample_size_and_margin")},
                {"verdict": _verdict(True, None)},
            ],
        }
    ]
    assert rp._classify_conclusion(results) == "A"


def test_classify_conclusion_is_c_when_a_verdict_reaches_and_fails_test_gate():
    from scripts import run_research_pipeline as rp

    results = [
        {
            "n_significant": 1,
            "candidacy_verdicts": [{"verdict": _verdict(False, "test")}],
        }
    ]
    assert rp._classify_conclusion(results) == "C"


def test_classify_conclusion_is_b_when_significant_but_nothing_reaches_test():
    from scripts import run_research_pipeline as rp

    results = [
        {
            "n_significant": 3,
            "candidacy_verdicts": [
                {"verdict": _verdict(False, "sample_size_and_margin")},
                {"verdict": _verdict(False, "robustness")},
            ],
        }
    ]
    assert rp._classify_conclusion(results) == "B"


def test_classify_conclusion_is_b_when_significant_but_no_candidacy_verdicts_at_all():
    from scripts import run_research_pipeline as rp

    results = [{"n_significant": 5, "candidacy_verdicts": []}]
    assert rp._classify_conclusion(results) == "B"


def test_classify_conclusion_aggregates_across_multiple_datasets():
    from scripts import run_research_pipeline as rp

    results = [
        {"n_significant": 0, "candidacy_verdicts": []},
        {
            "n_significant": 4,
            "candidacy_verdicts": [{"verdict": _verdict(False, "walk_forward")}],
        },
    ]
    assert rp._classify_conclusion(results) == "B"


# --- _walk_forward_folds (wiring through real DB + config) -----------------


def _insert_synthetic_candles(session, n, asset="TEST_FX", timeframe="1m"):
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    rng = np.random.default_rng(9)
    close = 1.10 + np.cumsum(rng.normal(0, 0.0005, size=n))
    for i in range(n):
        c = close[i]
        o = close[i - 1] if i > 0 else c
        h = max(o, c) + 0.0005
        low = min(o, c) - 0.0005
        session.add(
            Candle(
                asset=asset, timeframe=timeframe,
                timestamp=start + dt.timedelta(minutes=i),
                open=o, high=h, low=low, close=c,
                source="synthetic:test", is_synthetic_test_data=True,
            )
        )
    session.commit()
    return start


def test_walk_forward_folds_stay_within_the_train_plus_validation_region(session):
    from scripts import run_research_pipeline as rp

    n = 1000
    start = _insert_synthetic_candles(session, n).replace(tzinfo=None)
    config = _config()
    folds = rp._walk_forward_folds(session, "TEST_FX", "1m", config, n_folds=5)

    history_end = start + dt.timedelta(minutes=n - 1)
    train_val_end = start + (history_end - start) * (
        config.train_fraction + config.validation_fraction
    )

    assert len(folds) > 0
    for fold in folds:
        assert fold.test_window[1] <= train_val_end


def test_walk_forward_folds_raises_for_unknown_asset(session):
    from scripts import run_research_pipeline as rp

    with pytest.raises(ValueError, match="no candles ingested"):
        rp._walk_forward_folds(session, "NOPE", "1m", _config())
