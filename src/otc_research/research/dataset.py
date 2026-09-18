"""Point-in-time feature+target dataset builder for the statistical
discovery layer.

Reads only already-validated ``Candle``/``Feature`` rows (never a live
source, never recomputing indicators ad hoc) — the same layering rule
``backtest/engine.py`` already follows. The ONE column family that is
deliberately NOT point-in-time is the target columns (``call_wins_h``/
``put_wins_h``): by definition they look ``h`` candles into the future to
know whether a hypothetical trade opened at this row would have won.
This is safe specifically because targets are never treated as inputs —
they are excluded from every "features available at T" computation
downstream (baseline.py/discovery.py/models.py), and
``NON_FEATURE_COLUMNS``/``target_columns()`` below are the single source
of truth for which columns those modules must exclude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from otc_research.db.models import Candle, Feature
from otc_research.features.engine import FEATURE_NAMES
from otc_research.research.regimes import classify_regime

DEFAULT_HORIZONS: tuple[int, ...] = (1, 2, 3, 5)

#: The only data source this project has ever used for anything that
#: reached a Signal or a report — see STRATEGIES.md point 25. Never
#: silently swapped for a different meaning.
REAL_FOREX_DATA = "REAL_FOREX_DATA"

#: Columns present for context/grouping, never fed to discovery/models as
#: a bin-able numeric feature.
NON_FEATURE_COLUMNS = (
    "timestamp",
    "asset",
    "timeframe",
    "data_source_label",
    "close_price",
    "regime",
)


def target_columns(horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> list[str]:
    """Names of every target column for the given horizons — the
    authoritative exclude-list every statistical/model function downstream
    must subtract from a dataset's columns before treating the rest as
    features.
    """
    cols: list[str] = []
    for h in horizons:
        cols.append(f"call_wins_{h}")
        cols.append(f"put_wins_{h}")
    return cols


def _empty_dataset(horizons: tuple[int, ...]) -> pd.DataFrame:
    columns = [*NON_FEATURE_COLUMNS, *FEATURE_NAMES, *target_columns(horizons)]
    return pd.DataFrame(columns=columns)


def _load_candles(session: Session, asset: str, timeframe: str) -> list[Candle]:
    return (
        session.query(Candle)
        .filter(Candle.asset == asset, Candle.timeframe == timeframe)
        .order_by(Candle.timestamp.asc())
        .all()
    )


def _load_features(
    session: Session, asset: str, timeframe: str, feature_set_version: str
) -> dict:
    rows = (
        session.query(Feature.timestamp, Feature.name, Feature.value)
        .filter(
            Feature.asset == asset,
            Feature.timeframe == timeframe,
            Feature.feature_set_version == feature_set_version,
        )
        .all()
    )
    by_timestamp: dict = {}
    for timestamp, name, value in rows:
        by_timestamp.setdefault(timestamp, {})[name] = value
    return by_timestamp


def build_dataset(
    session: Session,
    asset: str,
    timeframe: str,
    *,
    feature_set_version: str,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> pd.DataFrame:
    """One row per candle timestamp that has a COMPLETE feature vector
    (every name in ``FEATURE_NAMES`` present, non-NaN) — rows with
    insufficient history are dropped, same convention as
    ``features/pipeline.py`` never storing a fabricated value.

    Returns a DataFrame sorted ascending by timestamp with columns:
    ``NON_FEATURE_COLUMNS``, every name in ``FEATURE_NAMES``, and
    ``target_columns(horizons)``. A target column is NaN for the trailing
    rows where not enough future candles exist to resolve that horizon —
    never fabricated. A tied outcome (future close exactly equal to this
    row's close) is also NaN for both ``call_wins_h`` and ``put_wins_h``
    at that horizon — a tie is neither a win nor a loss for either
    direction, so it must not silently count as one.
    """
    candles = _load_candles(session, asset, timeframe)
    if not candles:
        return _empty_dataset(horizons)

    features_by_ts = _load_features(session, asset, timeframe, feature_set_version)

    rows = []
    for c in candles:
        feats = features_by_ts.get(c.timestamp)
        if feats is None or not all(name in feats for name in FEATURE_NAMES):
            continue
        row = {"timestamp": c.timestamp, "close_price": c.close}
        row.update({name: feats[name] for name in FEATURE_NAMES})
        rows.append(row)

    if not rows:
        return _empty_dataset(horizons)

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    df.insert(1, "asset", asset)
    df.insert(2, "timeframe", timeframe)
    df.insert(3, "data_source_label", REAL_FOREX_DATA)

    df["regime"] = classify_regime(df["adx_14"], df["atr_expansion_ratio"])

    close = df["close_price"]
    for h in horizons:
        future_close = close.shift(-h)
        pnl = future_close - close
        call_wins = pd.Series(
            np.where(pnl > 0, 1.0, np.where(pnl < 0, 0.0, np.nan)), index=df.index
        )
        put_wins = pd.Series(
            np.where(pnl < 0, 1.0, np.where(pnl > 0, 0.0, np.nan)), index=df.index
        )
        still_unresolved = future_close.isna()
        call_wins[still_unresolved] = np.nan
        put_wins[still_unresolved] = np.nan
        df[f"call_wins_{h}"] = call_wins
        df[f"put_wins_{h}"] = put_wins

    ordered_columns = [*NON_FEATURE_COLUMNS, *FEATURE_NAMES, *target_columns(horizons)]
    return df[ordered_columns]
