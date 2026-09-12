"""Real Forex market data via OANDA's v20 REST API.

Requires a free OANDA practice (demo) account and a personal access token
(create one at OANDA's account management portal after signing up). The
token is never hardcoded or committed: pass it explicitly or set the
``OANDA_API_TOKEN`` environment variable.

This source only ever returns candles OANDA itself marks ``complete``
(closed). The still-forming current candle is always discarded — using it
would mean analyzing a bar whose close hasn't happened yet, which is
exactly the kind of look-ahead bias this project explicitly avoids.

Timeframe support is intentionally limited to 1 minute and above (no
sub-minute granularity) to match the project's own decision to drop
30-second analysis.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Iterable, Iterator

import requests

from otc_research.data.sources.base import DataSource, RawCandle

TIMEFRAME_TO_GRANULARITY = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "4h": "H4",
    "1d": "D",
}

_HOSTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

MAX_COUNT_PER_REQUEST = 5000
DEFAULT_COUNT = 500


class OandaDataSource(DataSource):
    def __init__(self, api_token: str | None = None, environment: str = "practice"):
        self.api_token = api_token or os.environ.get("OANDA_API_TOKEN")
        if not self.api_token:
            raise ValueError(
                "OANDA API token required: pass api_token=... or set the "
                "OANDA_API_TOKEN environment variable (never hardcode it)."
            )
        if environment not in _HOSTS:
            raise ValueError(
                f"Unknown OANDA environment {environment!r}; use 'practice' or 'live'."
            )
        self.environment = environment
        self.name = f"oanda:{environment}"

    def fetch(
        self,
        asset: str,
        timeframe: str,
        *,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        count: int | None = None,
    ) -> Iterable[RawCandle]:
        return list(self._iter_candles(asset, timeframe, start, end, count))

    def _iter_candles(
        self,
        asset: str,
        timeframe: str,
        start: dt.datetime | None,
        end: dt.datetime | None,
        count: int | None,
    ) -> Iterator[RawCandle]:
        granularity = TIMEFRAME_TO_GRANULARITY.get(timeframe)
        if granularity is None:
            raise ValueError(
                f"OANDA source does not support timeframe {timeframe!r}; "
                f"supported: {sorted(TIMEFRAME_TO_GRANULARITY)}"
            )

        params: dict[str, str | int] = {"granularity": granularity, "price": "M"}
        if start is not None or end is not None:
            if start is None or end is None:
                raise ValueError("Both start and end must be given together, or neither.")
            params["from"] = start.astimezone(dt.timezone.utc).isoformat()
            params["to"] = end.astimezone(dt.timezone.utc).isoformat()
        else:
            params["count"] = min(count or DEFAULT_COUNT, MAX_COUNT_PER_REQUEST)

        url = f"{_HOSTS[self.environment]}/v3/instruments/{asset}/candles"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        response = requests.get(url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        for c in payload.get("candles", []):
            if not c.get("complete", False):
                continue  # never use the still-forming candle
            mid = c["mid"]
            yield RawCandle(
                asset=asset,
                timeframe=timeframe,
                timestamp=dt.datetime.fromisoformat(c["time"].replace("Z", "+00:00")),
                open=float(mid["o"]),
                high=float(mid["h"]),
                low=float(mid["l"]),
                close=float(mid["c"]),
            )
