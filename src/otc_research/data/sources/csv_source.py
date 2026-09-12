"""CSV-based data source.

This is currently the *only* sanctioned way to bring real OTC (or any
other) candle data into the system: you export/obtain history yourself
(from the broker's own export feature, a data vendor, a trading journal,
etc.) as a CSV file, and this source parses it. It performs no network
access and fabricates nothing.

Expected columns (case-insensitive, order does not matter):
    timestamp, open, high, low, close

``timestamp`` must be parseable by ``pandas.to_datetime``. If it has no
timezone, it is assumed to already be UTC (make sure that is actually true
of your export before importing).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Iterable, Iterator

import pandas as pd

from otc_research.data.sources.base import DataSource, RawCandle

REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close"}


class CsvDataSource(DataSource):
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.name = f"csv:{self.path.name}"

    def fetch(self, asset: str, timeframe: str) -> Iterable[RawCandle]:
        return list(self._iter_rows(asset, timeframe))

    def _iter_rows(self, asset: str, timeframe: str) -> Iterator[RawCandle]:
        df = pd.read_csv(self.path)
        df.columns = [c.strip().lower() for c in df.columns]

        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(
                f"{self.path}: missing required column(s) {sorted(missing)}; "
                f"found {list(df.columns)}"
            )

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")

        for row in df.itertuples(index=False):
            ts = row.timestamp.to_pydatetime()
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=dt.timezone.utc)
            yield RawCandle(
                asset=asset,
                timeframe=timeframe,
                timestamp=ts,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
            )
