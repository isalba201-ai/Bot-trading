import pytest

from otc_research.research.strategy_candidacy import SUPPORTED_CODES, build_candidacy_inputs
from otc_research.strategies import BASELINE_STRATEGIES


def test_supported_codes_match_baseline_strategies_registry():
    # H9 is deliberately excluded from both registries for the same
    # reason (see strategies/__init__.py and strategy_candidacy.py's
    # module docstrings) -- this test guards that the two never drift
    # apart silently (e.g. a future H21 added to one but not the other).
    assert set(SUPPORTED_CODES) == set(BASELINE_STRATEGIES.keys())


@pytest.mark.parametrize("code", SUPPORTED_CODES)
def test_build_candidacy_inputs_base_strategy_matches_factory_midpoint(code):
    base_strategy, strategy_factory, param_grid = build_candidacy_inputs(code, expiry_seconds=180)
    midpoint = param_grid[len(param_grid) // 2]
    reconstructed = strategy_factory(**midpoint)
    assert base_strategy.label == reconstructed.label
    assert base_strategy.expiry_seconds == 180 == reconstructed.expiry_seconds
    assert base_strategy.code == code


@pytest.mark.parametrize("code", SUPPORTED_CODES)
def test_build_candidacy_inputs_every_grid_point_constructs_a_valid_strategy(code):
    _, strategy_factory, param_grid = build_candidacy_inputs(code, expiry_seconds=60)
    for params in param_grid:
        strategy = strategy_factory(**params)
        assert strategy.expiry_seconds == 60
        assert strategy.required_features  # never empty


def test_build_candidacy_inputs_perturbation_grid_actually_varies_thresholds():
    # H4 has two continuous band-edge params -- confirm the grid isn't
    # accidentally collapsed to the same point 5 times (which would make
    # gate 2's robustness sweep measure nothing).
    _, strategy_factory, param_grid = build_candidacy_inputs("H4", expiry_seconds=300)
    labels = {strategy_factory(**p).label for p in param_grid}
    assert len(labels) == len(param_grid)


def test_build_candidacy_inputs_rejects_h9():
    with pytest.raises(ValueError, match="H9"):
        build_candidacy_inputs("H9", expiry_seconds=300)


def test_build_candidacy_inputs_single_point_grid_for_threshold_free_strategies():
    for code in ("H5", "H11", "H14", "H15"):
        _, _, param_grid = build_candidacy_inputs(code, expiry_seconds=300)
        assert param_grid == [{}]
