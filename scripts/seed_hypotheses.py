#!/usr/bin/env python
"""Register the initial hypotheses (see STRATEGIES.md) in the database.

Idempotent: re-running does not duplicate or reset existing rows/status.
"""

from __future__ import annotations

from otc_research.config import load_config
from otc_research.db.models import Hypothesis
from otc_research.db.session import get_engine, get_session_factory, init_db

INITIAL_HYPOTHESES = [
    ("H1", "Same-color streak continuation",
     "Continuation after a same-color candle streak (1..6+ candles), tested empirically."),
    ("H2", "Extreme-range reversion",
     "Reversion after an extreme-range candle relative to recent average/ATR."),
    ("H3", "Short-term momentum",
     "Rate of change / EMA slope predicts next candle direction."),
    ("H4", "Bollinger mean reversion",
     "Mean reversion from Bollinger Band extremes (touch/close outside bands)."),
    ("H5", "N-candle range breakout",
     "Breakout of a recent N-candle high/low range."),
    ("H6", "Level rejection",
     "Long wick against the prevailing range, followed by reversion."),
    ("H7", "RSI extreme + confirmation",
     "RSI overbought/oversold combined with a price-action confirmation."),
    ("H8", "Volatility expansion after squeeze",
     "ATR regime change: expansion following a contraction/squeeze."),
    ("H9", "Hour-of-day / session bias",
     "Time-of-day or session bias independent of any price-action signal."),
    ("H10", "Trend + structure + momentum",
     "Combined EMA-slope trend, market structure, and momentum confirmation."),
]


def main() -> None:
    config = load_config()
    engine = get_engine(config.database_url)
    init_db(engine)
    session = get_session_factory(engine)()

    existing_codes = {code for (code,) in session.query(Hypothesis.code)}
    added = 0
    for code, name, description in INITIAL_HYPOTHESES:
        if code in existing_codes:
            continue
        session.add(Hypothesis(code=code, name=name, description=description))
        added += 1
    session.commit()
    print(f"Registered {added} new hypotheses ({len(existing_codes)} already present).")


if __name__ == "__main__":
    main()
