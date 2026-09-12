# Strategies (hypotheses under investigation)

Every entry below is registered in the `hypotheses` table
(`src/otc_research/db/models.py: Hypothesis`) **before** any test is run
against it, so the full list of what was tried is always auditable — see
BACKTESTING.md's section on multiple-testing control.

None of these are strategies yet. They are hypotheses to be tested against
real, validated data starting in Phase 4/5 (backtesting engine + baseline
strategies). Nothing below should be read as a claim that it works.

| Code | Hypothesis | Status |
|---|---|---|
| H1 | Continuation after a same-color candle streak (1, 2, 3, 4, 5, 6+ candles) — tested empirically, not assumed | registered |
| H2 | Reversion after an extreme-range candle (range far above recent average/ATR) | registered |
| H3 | Short-term momentum (rate of change / EMA slope) predicts next candle direction | registered |
| H4 | Mean reversion from Bollinger Band extremes (touch/close outside bands) | registered |
| H5 | Breakout of a recent N-candle high/low range | registered |
| H6 | Level rejection (long wick against the prevailing range) → reversion | registered |
| H7 | RSI extreme (overbought/oversold) combined with a price-action confirmation | registered |
| H8 | Volatility expansion following a contraction/squeeze (ATR regime change) | registered |
| H9 | Hour-of-day / session bias, independent of any price-action signal | registered |
| H10 | Combined trend (EMA slope) + market structure + momentum confirmation | registered |

## Status values

- **registered** — hypothesis defined, not yet tested against data.
- **tested** — backtested; see BACKTESTING.md classification below for the
  outcome once one exists.
- **no_edge / weak_edge / promising / robust_edge** — final classification
  after out-of-sample + walk-forward + Monte Carlo + robustness testing.

## What every hypothesis must define, not just its entry trigger

Per the project brief, a hypothesis is not just "condition X → CALL/PUT". At
minimum, each one (when it reaches Phase 4/5 backtesting) must specify:

- **Entry conditions** — the exact, testable trigger.
- **Invalidation conditions** — under what circumstances the setup is
  considered void even if the trigger technically fired (e.g. against a
  strong opposing trend, during anomalous volatility, near a major level).
- **Market regime applicability** — whether it's expected to work in trend,
  range, high-volatility, or low-volatility conditions, verified empirically
  rather than assumed (see BACKTESTING.md).
- **Time-of-day/session performance** — measured, not assumed; a hypothesis
  can be restricted to the hours/sessions where it actually shows an edge.
- **Quality filter** — what separates an A+ setup from a merely-acceptable
  one for this specific hypothesis.

Supporting factors considered across hypotheses (price action, trend,
momentum, volatility, market structure, support/resistance, candlestick
patterns) are only kept in a hypothesis's final rule set if they measurably
improve out-of-sample results — see BACKTESTING.md's feature/ablation
approach. None are included by default just because they're common
technical-analysis concepts.

## Rules for adding a new hypothesis

1. Register it here and in the `hypotheses` table with a `registered_at`
   timestamp *before* running it against validation or test data.
2. State the mechanism in plain terms (why would this pattern predict
   anything, in principle) — not just "let's try RSI 73".
3. It must be tested against the same discipline as everything else: no
   peeking at the out-of-sample split until development is final.

## Explicitly forbidden language

Never describe any hypothesis or strategy, at any status, as: "infallible",
"guaranteed", "90% guaranteed", "winning bot", "sure profit", or similar. The
only vocabulary allowed for a strategy's status is the classification in
BACKTESTING.md (NO EDGE / WEAK EDGE / PROMISING / ROBUST EDGE).
