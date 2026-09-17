import pytest

from otc_research.backtest.splits import compute_temporal_split, split_sequence


def test_split_is_strictly_temporal_and_covers_everything():
    items = list(range(100))
    train, val, test = split_sequence(items, train_fraction=0.6, validation_fraction=0.2)

    assert train == list(range(0, 60))
    assert val == list(range(60, 80))
    assert test == list(range(80, 100))
    # nothing lost, nothing duplicated, nothing reordered
    assert train + val + test == items


def test_split_fractions_are_approximate_but_close():
    split = compute_temporal_split(1000, train_fraction=0.6, validation_fraction=0.2)
    assert split.train_end == 600
    assert split.validation_end == 800


@pytest.mark.parametrize("n", [3, 4, 5, 10])
def test_every_split_gets_at_least_one_item_when_possible(n):
    split = compute_temporal_split(n, train_fraction=0.6, validation_fraction=0.2)
    assert split.train_end >= 1
    assert split.validation_end > split.train_end
    assert split.n > split.validation_end


def test_rejects_non_positive_n():
    with pytest.raises(ValueError):
        compute_temporal_split(0, 0.6, 0.2)


def test_rejects_fractions_leaving_no_test_slice():
    with pytest.raises(ValueError):
        compute_temporal_split(100, 0.8, 0.2)


def test_rejects_out_of_range_fractions():
    with pytest.raises(ValueError):
        compute_temporal_split(100, 1.5, 0.2)
    with pytest.raises(ValueError):
        compute_temporal_split(100, 0.6, -0.1)
