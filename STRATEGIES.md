# Strategies (hypotheses under investigation)

Every entry below is registered in the `hypotheses` table
(`src/otc_research/db/models.py: Hypothesis`) **before** any test is run
against it, so the full list of what was tried is always auditable — see
BACKTESTING.md's section on multiple-testing control.

**Status: Phase 5 (baseline strategy code) is implemented** — every
hypothesis below has a concrete, testable trigger in
`src/otc_research/strategies/`, runnable through the Phase 4 engine (see
the table after the hypothesis list). **None of them has actually been
run against real market data yet.** Every row's `status` in the database
stays `registered` until someone actually fetches real candles
(`scripts/fetch_market_data.py`), computes features
(`scripts/compute_features.py`), and runs
`scripts/run_baseline_backtests.py` against them — at that point it
becomes `tested`, meaning "has been run through the engine," never a
claim that it works. See BACKTESTING.md's edge classification: nothing
in this repository is entitled to be called NO EDGE / WEAK EDGE /
PROMISING / ROBUST EDGE yet — that needs walk-forward + Monte Carlo +
robustness testing (Phases 6-8), none of which exist yet either.

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

## Baseline implementation (Phase 5)

| Code | Implementation | Entry trigger (first version — see the file's docstring for the exact reasoning) |
|---|---|---|
| H1 | `strategies/h1_streak.py::H1StreakContinuation` | `same_color_streak` reaches `min_streak` (default 3) in either direction |
| H2 | `strategies/h2_extreme_range.py::H2ExtremeRangeReversion` | `range_ratio_20` clears a threshold (default 2.0) → fade the extreme candle's color |
| H3 | `strategies/h3_momentum.py::H3MomentumContinuation` | `roc_10` AND `ema_slope_12_3` agree in sign and clear their thresholds |
| H4 | `strategies/h4_bollinger.py::H4BollingerMeanReversion` | `bb_pct_b_20` at or beyond 0/1 |
| H5 | `strategies/h5_breakout.py::H5DonchianBreakout` | current close beyond `donchian_high_20` / `donchian_low_20` |
| H6 | `strategies/h6_wick_rejection.py::H6WickRejection` | `upper_wick_ratio` / `lower_wick_ratio` clears a threshold (default 0.6) |
| H7 | `strategies/h7_rsi_extreme.py::H7RsiExtremeConfirmed` | `rsi_14` extreme AND a confirming `same_color_streak` |
| H8 | `strategies/h8_volatility_expansion.py::H8VolatilityExpansion` | `atr_expansion_ratio` clears a threshold; direction from `ema_slope_12_3` |
| H9 | `strategies/h9_session_bias.py::H9SessionBias` | fires only at one explicit `(hour_utc, direction)` pair — see the file's docstring for why this one can't be a single fixed rule |
| H10 | `strategies/h10_combined.py::H10CombinedTrendStructureMomentum` | `ema_slope_12_3`, `structure_bias`, and `roc_10` all agree |

Every threshold above is a constructor parameter, not a hardcoded
constant, specifically so Phase 6's robustness/sensitivity sweeps can
vary it — see that module's docstring and BACKTESTING.md's
OVERFITTED/FRAGILE classification.

## Explicitly forbidden language

Never describe any hypothesis or strategy, at any status, as: "infallible",
"guaranteed", "90% guaranteed", "winning bot", "sure profit", or similar. The
only vocabulary allowed for a strategy's status is the classification in
BACKTESTING.md (NO EDGE / WEAK EDGE / PROMISING / ROBUST EDGE).
