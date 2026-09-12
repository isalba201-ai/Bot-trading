"""Abstract interface every data source must implement.

The system analyzes real Forex market data (not Pocket Option's OTC
instruments — see DATA.md for why). Sources implemented against this
interface: ``CsvDataSource`` (any user-provided historical export) and
``OandaDataSource`` (real market data via OANDA's v20 API, using a free
practice/demo account and personal access token — never scraping,
never bypassing login/2FA on anything).

The one hard rule for every implementation: if a candle cannot be
obtained, it is omitted. Never interpolate, repeat, or otherwise invent a
candle to fill a gap. And never return a candle that is still forming —
only fully closed bars.
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
    def fetch(
        self,
        asset: str,
        timeframe: str,
        *,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        count: int | None = None,
    ) -> Iterable[RawCandle]:
        """Yield candles in strictly ascending timestamp order.

        ``start``/``end`` request a specific closed time range; ``count``
        requests the most recent N candles when a range isn't given. A
        source that doesn't support range/count filtering (e.g. a fixed
        CSV file) may ignore these and always return everything it has.

        Implementations must not fabricate missing candles, and must never
        yield a candle that is still forming (not yet closed). Ordering
        and duplicate/gap detection is re-verified independently by the
        validation layer, but sources should still do their best to yield
        clean, sorted data.
        """
        raise NotImplementedError
