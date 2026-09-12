"""SQLAlchemy models for the OTC research database.

Design notes (read before adding columns):

* Raw market data (``Candle``) is kept completely separate from anything
  derived (``Feature``) or anything that represents a decision
  (``Signal``). Never blend them into one wide table — it makes it too
  easy to accidentally leak derived/future information back into what is
  supposed to be raw, immutable history.
* Nothing in this file "fills in" missing data. Gaps are recorded in
  ``DataQualityIssue``, never synthesized as candles.
* ``Hypothesis`` exists so that every strategy idea tested against data is
  logged BEFORE results are known, so that later we can account for
  multiple-comparisons / data-mining bias instead of only reporting
  whichever hypothesis happened to look best.
"""

from __future__ import annotations

import datetime as dt

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base

Base = declarative_base()


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class Candle(Base):
    """One raw OHLC candle for one asset/timeframe, as observed.

    ``timestamp`` is the candle's OPEN time in UTC. A candle is only ever
    inserted once it is fully closed — see data validation / ingestion,
    which is also where look-ahead protection starts.
    """

    __tablename__ = "candles"

    id = Column(Integer, primary_key=True)
    asset = Column(String(32), nullable=False)
    timeframe = Column(String(8), nullable=False)  # "30s", "1m", "5m", ...
    timestamp = Column(DateTime(timezone=True), nullable=False)

    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)

    # Provenance of this candle, e.g. "csv:my_export_2026-01-05.csv".
    # Required so a source change mid-history can be detected and audited.
    source = Column(String(128), nullable=False)

    # Explicit, permanent flag. Synthetic data is only ever used for testing
    # the pipeline itself and must never be mixed into research/backtests.
    is_synthetic_test_data = Column(Boolean, nullable=False, default=False)

    ingested_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint("asset", "timeframe", "timestamp", name="uq_candle_asset_tf_ts"),
        Index("ix_candle_lookup", "asset", "timeframe", "timestamp"),
    )


class DataQualityIssue(Base):
    """A logged data-quality finding. Never causes automatic data repair."""

    __tablename__ = "data_quality_issues"

    id = Column(Integer, primary_key=True)
    asset = Column(String(32), nullable=False)
    timeframe = Column(String(8), nullable=False)

    issue_type = Column(String(64), nullable=False)
    # one of: duplicate_timestamp, out_of_order, gap, impossible_value,
    # suspicious_move, source_change

    detail = Column(Text, nullable=False)
    candle_timestamp = Column(DateTime(timezone=True), nullable=True)
    source = Column(String(128), nullable=True)

    detected_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_dqi_lookup", "asset", "timeframe", "issue_type"),
    )


class Feature(Base):
    """A single computed feature value for one candle timestamp.

    ``feature_set_version`` must be bumped whenever the computation logic
    changes, so old backtests remain reproducible against the feature
    version they were actually run with.

    Point-in-time contract: the value stored here must be computable using
    only candles with timestamp <= this row's timestamp (i.e. up to and
    including the close of this candle). Anything else is look-ahead bias.
    """

    __tablename__ = "features"

    id = Column(Integer, primary_key=True)
    asset = Column(String(32), nullable=False)
    timeframe = Column(String(8), nullable=False)
    timestamp = Column(DateTime(timezone=True), nullable=False)

    feature_set_version = Column(String(32), nullable=False)
    name = Column(String(64), nullable=False)
    value = Column(Float, nullable=True)

    computed_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        UniqueConstraint(
            "asset", "timeframe", "timestamp", "feature_set_version", "name",
            name="uq_feature_point",
        ),
        Index("ix_feature_lookup", "asset", "timeframe", "timestamp", "feature_set_version"),
    )


class Hypothesis(Base):
    """Registry of every trading hypothesis considered, tested or not.

    This table exists specifically to counter data-mining / multiple
    testing bias (see BACKTESTING.md, section on multiple comparisons):
    every idea gets a row here BEFORE its results are known, so a later
    audit can see how many hypotheses were tried, not just the winner.
    """

    __tablename__ = "hypotheses"

    id = Column(Integer, primary_key=True)
    code = Column(String(32), nullable=False, unique=True)  # "H1", "H2", ...
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=False)

    registered_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    # registered -> tested -> one of: no_edge, weak_edge, promising, robust_edge
    status = Column(String(32), nullable=False, default="registered")
    notes = Column(Text, nullable=True)


class Signal(Base):
    """A generated (paper or live) signal and, once resolved, its outcome.

    Rows are never mutated after the outcome is known except to fill in
    the result/pnl fields once the expiry is reached — the inputs that led
    to the signal (features_snapshot, conditions_met, score, ...) are
    write-once so a signal can always be reconstructed and audited later.
    """

    __tablename__ = "signals"

    id = Column(Integer, primary_key=True)

    asset = Column(String(32), nullable=False)
    direction = Column(String(8), nullable=False)  # CALL / PUT
    timeframe = Column(String(8), nullable=False)
    expiry_seconds = Column(Integer, nullable=False)

    generated_at = Column(DateTime(timezone=True), nullable=False)
    entry_price = Column(Float, nullable=True)
    expiry_price = Column(Float, nullable=True)
    payout = Column(Float, nullable=False)  # e.g. 0.92 for 92%

    strategy_code = Column(String(32), ForeignKey("hypotheses.code"), nullable=False)
    model_version = Column(String(32), nullable=False)

    score = Column(Float, nullable=False)
    quality_tier = Column(String(8), nullable=False)  # "A+", "A", "B", "NO_TRADE"

    market_regime = Column(String(32), nullable=True)
    session = Column(String(16), nullable=True)
    day_of_week = Column(Integer, nullable=True)  # 0=Monday ... 6=Sunday
    hour = Column(Integer, nullable=True)  # UTC hour of day

    features_snapshot = Column(Text, nullable=True)  # JSON: {feature_name: value}
    conditions_met = Column(Text, nullable=True)  # JSON list of condition names

    mode = Column(String(16), nullable=False, default="paper")  # paper / live
    result = Column(String(8), nullable=True)  # WIN / LOSS / VOID
    pnl = Column(Float, nullable=True)

    reason_discarded = Column(Text, nullable=True)

    created_at = Column(DateTime(timezone=True), nullable=False, default=_utcnow)

    __table_args__ = (
        Index("ix_signal_lookup", "asset", "timeframe", "generated_at"),
    )
