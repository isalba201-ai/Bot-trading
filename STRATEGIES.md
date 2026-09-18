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
| H11 | MACD line crossing its signal line predicts continuation in the crossing direction | registered |
| H12 | CCI extreme (beyond ±100) → reversion | registered |
| H13 | RCI (rank correlation index) extreme (beyond ±80) → reversion | registered |
| H14 | Bullish/bearish engulfing candle → reversal | registered |
| H15 | Breakout of an inside-bar ("mother bar") range → continuation | registered |

H11-H15 were added after H1-H10 had already been run against real data
and shown no robust edge (see "What real data has actually shown so
far" below) — registered here, in this table, before any of them were
run against anything, same discipline as H1-H10. H11 (MACD) and H12/H13
(CCI/RCI) round out the indicator families used across H1-H10 (trend,
momentum-by-magnitude, volatility, bands) with a classic
trend-following/momentum oscillator (MACD) and two more overbought/
oversold oscillators (CCI, RCI) that are mechanistically distinct from
RSI (RCI in particular is rank-based, not magnitude-based). H14/H15 round
out the price-action side (H1/H2/H6 already covered streaks, extreme
range, and wick rejection) with the two most commonly cited two-candle
and consolidation-breakout patterns that weren't covered yet.

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
| H11 | `strategies/h11_macd_cross.py::H11MacdCrossContinuation` | `macd_cross_signal` fires (+1/-1) |
| H12 | `strategies/h12_cci_extreme.py::H12CciExtremeReversion` | `cci_20` clears a threshold (default ±100) → fade |
| H13 | `strategies/h13_rci_extreme.py::H13RciExtremeReversion` | `rci_9` clears a threshold (default ±80) → fade |
| H14 | `strategies/h14_engulfing.py::H14EngulfingReversal` | `engulfing_signal` fires (+1/-1) |
| H15 | `strategies/h15_inside_bar_breakout.py::H15InsideBarBreakout` | `inside_bar_breakout_signal` fires (+1/-1) |

Every threshold above is a constructor parameter, not a hardcoded
constant, specifically so Phase 6's robustness/sensitivity sweeps can
vary it — see that module's docstring and BACKTESTING.md's
OVERFITTED/FRAGILE classification.

## What real data has actually shown so far (not a final classification)

The only data run through the engine so far: real EUR/USD candles from
Twelve Data (5-minute, ~17 days; 1-hour, ~7 months), train/validation
splits only, never test.

- **None of H1, H2, H3, H5, H6, H8, H10 showed a credible edge** on
  either timeframe — most Wilson 95% CIs on the optimistic scenario span
  50%, and a few (H5, H6 at 1h; H3, H6 at 5m) sat entirely *below* 50%
  with large samples, i.e. statistically significant *underperformance*
  vs. a coin flip. That is itself informative — it will not be
  "corrected" by flipping the rule after the fact, since that would be
  exactly the kind of post-hoc data-mining this document's multiple-testing
  section exists to prevent — but it isn't evidence for anything currently
  registered.
- **H4 (Bollinger mean reversion) looked promising at first** on 1h data:
  56.9% (train, n=415) and 58.9% (validation, n=124), both with CIs
  clearing 50%. A Phase 6 robustness sweep then showed this is fragile,
  not robust: a threshold sweep held up (`consistent_direction`, 4/5
  points), but an expiry sweep did not (edge present at 1h-2h, gone by
  3h-4h) and a coarse 2-window time-period sweep did not either (edge
  present in the second half of the ingested window, absent in the
  first half).
  A Phase 7 walk-forward (12 non-overlapping 14-day folds across the same
  ~7 months) painted a more nuanced picture than that 2-window split:
  under the optimistic scenario, 11/12 folds had a positive point
  estimate (mean 56.9%, stdev 8.1pp across folds) but only 3/12 folds
  individually cleared the 95% CI edge bar, and the worst fold (Aug
  11-25) sat at 40.5% — clearly a losing stretch. Under the realistic
  scenario, the mean drops to 47.2% (stdev 4.8pp) and **0 of 12 folds**
  show an edge. See `Hypothesis.notes` for H4 (via `scripts/` or a DB
  query) for the exact per-fold figures.
- **Execution cost dominates at 5-minute granularity**: EUR/USD's typical
  5-minute move is ~0.01% (about a pip), comparable to or smaller than
  the default realistic/pessimistic slippage assumptions in
  `config.yaml`. Every hypothesis's win rate collapsed under those
  scenarios at 5m; the effect was much smaller at 1h, where the typical
  move is larger relative to the same cost assumptions.

None of this is a ROBUST EDGE, a PROMISING classification, or grounds to
build the Phase 9 signal UI on top of H4 specifically. It's exactly what
Phases 6-7 are for: catching an apparent edge before it reaches further
phases on the strength of one lucky split or one lucky fold. The
realistic-scenario walk-forward result (0/12 folds) is the most decisive
finding yet: under costs a real trader would actually pay, H4 has not
shown an edge in any two-week stretch of the ~7 months tested.

### Cross-pair check (GBP/USD, USD/JPY) and a full H9 scan

Two further exploratory checks, both on real Twelve Data 1h candles,
train split:

- **H4 across three pairs**: the optimistic-scenario tendency above 50%
  shows up on all three (EUR/USD 56.9%, GBP/USD 58.0%, USD/JPY 54.3%,
  all n>340) — directionally consistent, which sounds encouraging. But
  it does **not** replicate cleanly on each pair's own validation split
  (GBP/USD drops to 51.2%; USD/JPY holds better at 59.2%), and the
  realistic-execution scenario erases or reverses it on every pair
  (GBP/USD 50.0%, USD/JPY 41.2%). Read together with the Phase 6/7
  results above: this looks like a real, mild, structural tendency in
  how these pairs behave around Bollinger-band extremes intraday — not
  an exploitable edge once real execution costs are priced in.
- **H9, done properly this time**: rather than guess an hour, all 24 UTC
  hours × both directions (48 configs) were run against real EUR/USD/1h.
  3 of 48 individually cleared a naive 95% confidence interval — almost
  exactly the ~2.4 false positives that pure chance predicts from 48
  comparisons at 95% confidence, and **0 of 48** cleared a
  Bonferroni-adjusted interval that actually accounts for testing 48
  hypotheses at once. This is the multiple-testing section above,
  demonstrated rather than just asserted: naive per-config significance
  testing on a wide scan manufactures "signals" that a correction for
  the number of comparisons makes disappear. **No hour-of-day bias found.**

### H11-H15 (MACD, CCI, RCI, engulfing, inside-bar breakout)

Registered above, then run against real EUR/USD, GBP/USD, and USD/JPY
1h data, train split, optimistic scenario (15 pair×hypothesis
combinations — see the multiple-testing note above: at 95% confidence,
pure chance predicts roughly 1 "hit" out of 15):

- **H11 (MACD cross), H14 (engulfing), H15 (inside-bar breakout): no
  signal on any pair.** Every 95% CI straddled or sat below 50%.
- **H12 (CCI extreme) and H13 (RCI extreme) each cleared a 95% CI on
  GBP/USD only** (H12: 53.3%, n=1134; H13: 54.8%, n=809) — out of 15
  tests, finding 2 borderline hits is within what chance alone predicts.
  Both were checked against GBP/USD's own validation split before being
  taken seriously, exactly like H4 was: **neither replicated** (H12 drops
  to 50.5%; H13 to 52.4%, both no longer clearing 50%), and the realistic
  execution scenario erases both regardless (H12 45.1%, H13 42.2%).

**Combined verdict for H11-H15: no credible edge on any of the three
pairs.** See `Hypothesis.notes` for the exact per-pair figures.

### Where this leaves the project

15 hypotheses (H1-H15), 3 real currency pairs, two timeframes for the
original 10, a full 48-way session-bias scan, cross-pair checks, and
robustness/walk-forward testing on the single most promising candidate
found (H4) — none has produced a credible, replicated edge once
execution costs and independent validation are applied. This is not a
sign the methodology is broken; consistent, well-behaved null results
across genuinely different indicator families (trend, momentum,
oscillators, volatility, price action, rank-based, time-of-day) are
exactly what "there is no easily-found edge in liquid spot Forex at
1-hour granularity with these tools" looks like when tested honestly,
and is itself a legitimate research conclusion — see BACKTESTING.md's
edge classification, none of which any hypothesis here has earned.

## Explicitly forbidden language

Never describe any hypothesis or strategy, at any status, as: "infallible",
"guaranteed", "90% guaranteed", "winning bot", "sure profit", or similar. The
only vocabulary allowed for a strategy's status is the classification in
BACKTESTING.md (NO EDGE / WEAK EDGE / PROMISING / ROBUST EDGE).
