"""Strict temporal train/validation/test split (BACKTESTING.md).

Never shuffled. The boundaries are simple index cuts over an
already-ascending sequence — anything else would let information from
later in time leak into an earlier split, which is exactly the bias this
project's design goes out of its way to avoid (see FEATURES.md's
point-in-time contract, which this split relies on rather than duplicates).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TemporalSplit:
    train_end: int  # exclusive upper bound of the train slice
    validation_end: int  # exclusive upper bound of the validation slice (train+val)
    n: int

    @property
    def train_slice(self) -> slice:
        return slice(0, self.train_end)

    @property
    def validation_slice(self) -> slice:
        return slice(self.train_end, self.validation_end)

    @property
    def test_slice(self) -> slice:
        return slice(self.validation_end, self.n)


def compute_temporal_split(
    n: int, train_fraction: float, validation_fraction: float
) -> TemporalSplit:
    """Pure index arithmetic: given ``n`` items already sorted ascending by
    time, returns the (train_end, validation_end) cut points. The
    remaining fraction (``1 - train_fraction - validation_fraction``) is
    the out-of-sample test slice, touched exactly once — see
    backtest/engine.py.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must leave a nonzero test slice")

    train_end = round(n * train_fraction)
    validation_end = round(n * (train_fraction + validation_fraction))

    # Guarantee every split gets at least one item once n is large enough
    # to make that possible, rather than silently handing a split zero
    # rows due to rounding at small n.
    train_end = max(1, min(train_end, n - 2)) if n >= 3 else train_end
    validation_end = max(train_end + 1, min(validation_end, n - 1)) if n >= 3 else validation_end

    return TemporalSplit(train_end=train_end, validation_end=validation_end, n=n)


def split_sequence(
    items: Sequence[T], train_fraction: float, validation_fraction: float
) -> tuple[Sequence[T], Sequence[T], Sequence[T]]:
    """Convenience wrapper: split an already time-ordered sequence directly
    into (train, validation, test) sub-sequences.
    """
    split = compute_temporal_split(len(items), train_fraction, validation_fraction)
    return (
        items[split.train_slice],
        items[split.validation_slice],
        items[split.test_slice],
    )
