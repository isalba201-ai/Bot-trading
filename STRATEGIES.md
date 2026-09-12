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
