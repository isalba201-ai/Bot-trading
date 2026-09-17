"""Real Forex market data via Twelve Data's REST API.

Twelve Data is a pure market-data reseller: signup only requires an email
address and gets you an API key immediately — no broker account, no KYC,
no country-based trading-account approval (unlike OANDA, which can and did
reject applications on that basis). This is the primary source for anyone
who can't get an OANDA account.

Free tier constraints (verify current values at twelvedata.com/pricing
before relying on them): roughly 8 requests/minute, 800/day, 5000 data
points per request. This class does not itself rate-limit — callers must
not hammer it faster than the plan allows.

Twelve Data does not flag "still forming" candles the way OANDA does, so
this source computes each bar's implied close time (open + timeframe
duration) and drops any bar that hasn't actually closed yet — the same
guarantee as the OANDA source, just computed manually.
"""

from __future__ import annotations

import datetime as dt
import os
from typing import Iterable, Iterator

import requests

from otc_research.data.sources.base import DataSource, RawCandle
from otc_research.utils.timeframes import timeframe_to_seconds

TIMEFRAME_TO_INTERVAL = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1day",
}

BASE_URL = "https://api.twelvedata.com/time_series"
MAX_COUNT_PER_REQUEST = 5000
DEFAULT_COUNT = 500


class TwelveDataSource(DataSource):
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("TWELVEDATA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Twelve Data API key required: pass api_key=... or set the "
                "TWELVEDATA_API_KEY environment variable (never hardcode it)."
            )
        self.name = "twelvedata"

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
        interval = TIMEFRAME_TO_INTERVAL.get(timeframe)
        if interval is None:
            raise ValueError(
                f"Twelve Data source does not support timeframe {timeframe!r}; "
                f"supported: {sorted(TIMEFRAME_TO_INTERVAL)}"
            )

        # Internal asset codes use OANDA-style underscores (EUR_USD); Twelve
        # Data expects a slash (EUR/USD).
        symbol = asset.replace("_", "/")

        params: dict[str, str | int] = {
            "symbol": symbol,
            "interval": interval,
            "apikey": self.api_key,
            "timezone": "UTC",
            "order": "ASC",
        }
        if start is not None or end is not None:
            if start is None or end is None:
                raise ValueError("Both start and end must be given together, or neither.")
            params["start_date"] = start.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
            params["end_date"] = end.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        else:
            params["outputsize"] = min(count or DEFAULT_COUNT, MAX_COUNT_PER_REQUEST)

        response = requests.get(BASE_URL, params=params, timeout=30)
        response.raise_for_status()
        payload = response.json()

        if payload.get("status") == "error":
            raise RuntimeError(f"Twelve Data error for {symbol}/{interval}: {payload.get('message', payload)}")

        bar_seconds = timeframe_to_seconds(timeframe)
        now = dt.datetime.now(dt.timezone.utc)

        for row in payload.get("values", []):
            bar_open = _parse_datetime(row["datetime"])
            bar_close = bar_open + dt.timedelta(seconds=bar_seconds)
            if bar_close > now:
                continue  # still-forming bar — never use it
            yield RawCandle(
                asset=asset,
                timeframe=timeframe,
                timestamp=bar_open,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
            )


def _parse_datetime(value: str) -> dt.datetime:
    try:
        parsed = dt.datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        parsed = dt.datetime.strptime(value, "%Y-%m-%d")
    return parsed.replace(tzinfo=dt.timezone.utc)
