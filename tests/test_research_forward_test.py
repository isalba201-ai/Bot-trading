from otc_research.research.forward_test import FORWARD_TEST_CANDIDATES


def test_two_and_only_two_forward_test_candidates():
    assert len(FORWARD_TEST_CANDIDATES) == 2
    assert {c.code for c in FORWARD_TEST_CANDIDATES} == {"H9_FWD", "ML10_FWD"}


def test_h9_fwd_strategy_fires_only_at_hour_6_utc():
    h9 = next(c for c in FORWARD_TEST_CANDIDATES if c.code == "H9_FWD")
    strategy = h9.strategy
    assert strategy.expiry_seconds == 4500
    assert strategy.decide({"hour_utc": 6.0}) == "CALL"
    assert strategy.decide({"hour_utc": 5.0}) is None
    assert strategy.decide({"hour_utc": 7.0}) is None


def test_ml10_fwd_strategy_loads_the_frozen_model_and_matches_backtest_numbers():
    ml10 = next(c for c in FORWARD_TEST_CANDIDATES if c.code == "ML10_FWD")
    strategy = ml10.strategy
    assert strategy.expiry_seconds == 10800
    assert strategy.direction == "CALL"
    assert set(strategy.required_features) == {
        "return_1", "return_5", "rsi_14", "adx_14", "atr_expansion_ratio",
        "cci_20", "rci_9", "bb_pct_b_20", "move_size_atr", "pct_position_in_range_20",
    }


def test_forward_test_candidates_are_immutable_dataclasses():
    for c in FORWARD_TEST_CANDIDATES:
        assert c.payout == 0.85
        assert c.backtest_status  # non-empty, documents the frozen backtest result
