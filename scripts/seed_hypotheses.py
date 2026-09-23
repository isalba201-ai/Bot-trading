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
    ("H11", "MACD signal-line crossover",
     "MACD line crossing its signal line predicts continuation in the crossing direction."),
    ("H12", "CCI extreme reversion",
     "Commodity Channel Index beyond +-100 predicts reversion."),
    ("H13", "RCI extreme reversion",
     "Rank Correlation Index beyond +-80 predicts reversion."),
    ("H14", "Engulfing candle reversal",
     "A bullish/bearish engulfing candle predicts a reversal."),
    ("H15", "Inside-bar breakout continuation",
     "Breakout of an inside-bar (mother bar) range predicts continuation."),
    ("H16", "MACD cross confirmed by RSI regime",
     "MACD signal-line cross confirmed by RSI being on the same side of 50 predicts continuation."),
    ("H17", "Bollinger extreme confirmed by RCI extreme",
     "Bollinger Band extreme confirmed by an RCI extreme (two independent oscillators agreeing) predicts reversion."),
    ("H18", "CCI extreme confirmed by engulfing candle",
     "CCI extreme confirmed by an engulfing candle in the same direction predicts reversion."),
    ("H19", "Trend + shallow RSI pullback",
     "An established EMA-slope trend with a shallow (non-extreme) RSI pullback predicts continuation with the trend."),
    ("H20", "Donchian breakout confirmed by volatility expansion",
     "A Donchian channel breakout confirmed by ATR expansion predicts continuation."),
    ("H21", "CCI+RSI overbought confirmed by bearish MACD histogram + red candle",
     "CCI(20) and RSI(14) both overbought, confirmed by a bearish MACD histogram bar "
     "and a bearish candle on the same candle, predicts a PUT reversal."),
    ("H9_FWD", "H9 session-bias candidate, forward/paper test",
     "EUR_USD 15m, hour_utc==6 UTC -> CALL, h=5 (75 min expiry). Frozen candidate from "
     "the corrected binary-options backtest (BINARY_OPTIONS_BACKTEST_REPORT.md Section 3) "
     "that mechanically cleared all four gates but was flagged fragile; forward-tested "
     "against data it has never touched, never re-evaluated against its original TEST split."),
    ("ML10_FWD", "Step 10 ML candidate, forward/paper test",
     "USD_JPY 1h, random forest on call_wins h=3 (3h expiry) -> CALL. Frozen candidate "
     "from Step 10's extended search (STEP10_EXTENDED_SEARCH_REPORT.md Section 3) that "
     "mechanically cleared all four gates but was flagged fragile; the exact fitted model "
     "is frozen (never refit) and forward-tested against data it has never touched."),
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
