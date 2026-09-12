"""Abstract interface every data source must implement.

Why this exists: Pocket Option does not publish an official API for OTC
candle data. This project deliberately does NOT scrape the platform, bypass
login/2FA, or reverse-engineer private endpoints — see DATA.md for the
reasoning. Instead, the system is built around this interface so that:

* real historical data you provide (e.g. exported CSV files) can be
  ingested today, and
* a legitimate, authorized data feed can be plugged in later without
  touching anything downstream (validation, storage, features, ...).

The one hard rule for every implementation: if a candle cannot be
obtained, it is omitted. Never interpolate, repeat, or otherwise invent a
candle to fill a gap.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class RawCandle:
    asset: str
    timeframe: str
    timestamp: dt.datetime  # candle OPEN time, must be tz-aware UTC
    open: float
    high: float
    low: float
    close: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError(
                f"RawCandle timestamp must be timezone-aware (got naive {self.timestamp!r})"
            )


class DataSource(ABC):
    """Something that can supply raw OHLC candles for an asset/timeframe."""

    #: short machine-readable identifier stored as Candle.source provenance
    name: str

    @abstractmethod
    def fetch(self, asset: str, timeframe: str) -> Iterable[RawCandle]:
        """Yield candles in strictly ascending timestamp order.

        Implementations must not fabricate missing candles. Ordering and
        duplicate/gap detection is re-verified independently by the
        validation layer, but sources should still do their best to yield
        clean, sorted data.
        """
        raise NotImplementedError
