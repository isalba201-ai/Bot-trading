"""Timeframe string <-> seconds conversion.

Deliberately dependency-free (stdlib only) so it can be imported from
anywhere in the codebase — data validation, feature engineering,
backtesting — without risking a circular import through
``otc_research.data.*``.
"""

from __future__ import annotations

_TIMEFRAME_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
}


def timeframe_to_seconds(timeframe: str) -> int:
    try:
        return _TIMEFRAME_SECONDS[timeframe]
    except KeyError as exc:
        raise ValueError(
            f"Unknown timeframe {timeframe!r}; known values: {sorted(_TIMEFRAME_SECONDS)}"
        ) from exc
