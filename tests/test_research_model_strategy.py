import numpy as np
from sklearn.linear_model import LogisticRegression

from otc_research.research.model_strategy import ModelStrategy


def _fitted_model():
    # Planted: feature "x" above 0 -> class 1 (win), below -> class 0.
    rng = np.random.default_rng(0)
    x = rng.uniform(-1, 1, 500)
    y = (x > 0).astype(int)
    model = LogisticRegression()
    model.fit(x.reshape(-1, 1), y)
    return model


def test_model_strategy_calls_when_probability_clears_threshold():
    model = _fitted_model()
    strategy = ModelStrategy(model, ["x"], "CALL", expiry_seconds=300, probability_threshold=0.5)
    assert strategy.decide({"x": 0.9}) == "CALL"
    assert strategy.decide({"x": -0.9}) is None


def test_model_strategy_uses_configured_direction():
    model = _fitted_model()
    strategy = ModelStrategy(model, ["x"], "PUT", expiry_seconds=300, probability_threshold=0.5)
    assert strategy.decide({"x": 0.9}) == "PUT"


def test_model_strategy_threshold_changes_sensitivity():
    model = _fitted_model()
    lenient = ModelStrategy(model, ["x"], "CALL", expiry_seconds=300, probability_threshold=0.1)
    strict = ModelStrategy(model, ["x"], "CALL", expiry_seconds=300, probability_threshold=0.99)
    # a mild positive x clears a lenient threshold but not a near-1.0 one
    assert lenient.decide({"x": 0.05}) == "CALL"
    assert strict.decide({"x": 0.05}) is None


def test_model_strategy_rejects_invalid_direction():
    model = _fitted_model()
    try:
        ModelStrategy(model, ["x"], "UP", expiry_seconds=300)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_model_strategy_metadata():
    model = _fitted_model()
    strategy = ModelStrategy(model, ["x", "y"], "CALL", expiry_seconds=600, label="my_model")
    assert strategy.expiry_seconds == 600
    assert strategy.required_features == frozenset({"x", "y"})
    assert strategy.label == "my_model"
    assert strategy.code is None
