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
| H16 | MACD cross CONFIRMED by RSI regime (RSI on the same side of 50 as the cross) → continuation | registered |
| H17 | Bollinger Band extreme CONFIRMED by RCI extreme (two independent oscillators agreeing on exhaustion) → reversion | registered |
| H18 | CCI extreme CONFIRMED by an engulfing candle in the same direction → reversion | registered |
| H19 | Pullback entry: EMA-slope trend direction + RSI dipping into a shallow (not extreme) pullback zone → continuation with the trend | registered |
| H20 | Donchian breakout CONFIRMED by volatility expansion (ATR expansion ratio) → continuation | registered |

H16-H20 were requested explicitly: combine multiple indicators from
different families rather than testing any one alone, on binary-options-
realistic short timeframes (1-5 minutes, not 1h). Each stacks two
already-registered single-indicator mechanisms as an AND condition
(never invented from scratch) — the rationale in each case is
"single indicator X showed a coin-flip result; does requiring a second,
mechanistically different indicator to agree filter out the noise and
leave a real signal?":

- **H16** stacks H11 (MACD cross) with an RSI regime filter — the classic
  "don't fade an already-exhausted move" confirmation: only take a
  bullish MACD cross when RSI is already on the bullish side of 50 (not
  overbought, which would suggest exhaustion, and not still bearish,
  which would suggest the cross is premature).
- **H17** stacks H4 (Bollinger extreme) with H13 (RCI extreme) — two
  mechanistically distinct exhaustion signals (a volatility-band
  measure and a rank-correlation measure) agreeing, rather than either
  alone.
- **H18** stacks H12 (CCI extreme) with H14 (engulfing) — an oscillator
  extreme confirmed by actual price-action reversal behavior, the same
  logic as H7 (RSI + streak) but with a different oscillator/pattern
  pair.
- **H19** is a distinct mechanism from H3/H10: not "is momentum currently
  strong", but "is there an established trend (EMA slope) that has just
  had a shallow, non-extreme RSI dip" — the "buy the pullback, not the
  extreme" entry style common in real discretionary trading, which none
  of H1-H15 tests.
- **H20** stacks H5 (Donchian breakout) with H8's volatility-expansion
  measure — only trust a breakout that coincides with expanding
  volatility (a standard real-trading filter against false/low-
  conviction breakouts).

No new features were needed for H16-H20 — every one is built entirely
from `feature_set_version = "v3"` columns already computed for H1-H15.

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
| H16 | `strategies/h16_macd_rsi_confirmed.py::H16MacdRsiConfirmed` | `macd_cross_signal` fires AND `rsi_14` is on the same side of 50 |
| H17 | `strategies/h17_bollinger_rci_confirmed.py::H17BollingerRciConfirmed` | `bb_pct_b_20` AND `rci_9` both clear their extreme thresholds in the same direction |
| H18 | `strategies/h18_cci_engulfing_confirmed.py::H18CciEngulfingConfirmed` | `cci_20` extreme AND `engulfing_signal` agrees in direction |
| H19 | `strategies/h19_trend_pullback.py::H19TrendPullback` | `ema_slope_12_3` established AND `rsi_14` in a shallow pullback band |
| H20 | `strategies/h20_breakout_volatility_confirmed.py::H20BreakoutVolatilityConfirmed` | Donchian breakout AND `atr_expansion_ratio` clears a threshold |

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

### H16-H20 (combined indicators, binary-options timeframes: 1m and 5m)

Requested explicitly: combine multiple indicators (never just one), and
use short timeframes (1-5 minutes) matching real binary-options contract
durations rather than the 1h data used for H1-H15. Run on real EUR/USD
(1-minute candles, 60s and 300s expiry), EUR/USD, GBP/USD, and USD/JPY
(5-minute candles, 300s expiry) — 5 runs × 5 hypotheses, train split,
optimistic scenario:

- **H17, H18, H19, H20: no credible edge anywhere.** H19 didn't even
  fire on 3 of 5 series (its EMA-slope threshold is too strict for
  short-timeframe price moves). H18's joint CCI+engulfing condition is
  rare enough (n=5-24 per run) that no run had a trustworthy sample.
  H17 and H20 hovered at or below 50% on every run with no consistent
  direction.
- **H16 (MACD+RSI) looked like a hit at first**: 57.1% (CI[0.502,0.637],
  n=205) on 1-minute EUR/USD at a 60-second expiry — the single strongest
  optimistic-scenario result of the entire investigation, and the exact
  kind of number a less careful analysis would report as "found one."
  It **did not replicate**: on EUR/USD's own validation split it drops
  to 55.4% with a CI of [0.441, 0.662] — no longer clearing 50% — and at
  every other timeframe/pair tested (EUR/USD 1m at a 300s expiry, and
  5-minute EUR/USD, GBP/USD, USD/JPY) it sits at 36-55%, including three
  results *significantly below* 50%. A result that reverses sign across
  nearby timeframes on the same underlying mechanism is the textbook
  OVERFITTED/FRAGILE pattern BACKTESTING.md defines, not a validated edge.

**The most important finding for binary options specifically is not
about any one hypothesis — it's about timeframe itself.** Going shorter
did not make execution costs matter less; it made them dominate more
completely. At 1-hour resolution (H1-H15), the realistic-execution
scenario typically cut win rates by 15-25 percentage points. At 1- and
5-minute resolution (H16-H20), it typically cut them by 30-50 points —
several runs above collapsed from 50-60% optimistic to under 15%
realistic, and to single digits or zero under the pessimistic scenario.
This makes sense mechanically: expected price movement shrinks roughly
with the square root of time, but fixed costs (spread, latency, slippage)
don't shrink proportionally, so they eat a larger fraction of the
available move the shorter the timeframe gets.

**Binary options add a second, independent problem on top of that: the
break-even bar itself is higher than 50%.** With a typical binary-options
payout of 0.80-0.90, break-even is 52.6%-55.6%, not 50% (see
BACKTESTING.md's payout math). Even H16's best, non-replicating,
optimistic-only number (57.1%) only barely cleared that stricter bar —
and that was before any execution cost was applied at all. Once realistic
execution is included, every configuration tested here, at every
timeframe from 5 minutes to 1 hour, falls far short of any realistic
binary-options break-even.

**Combined verdict for H16-H20, and for the timeframe question that
motivated them: no combination of the indicators available in this
codebase has shown a credible, replicated edge at binary-options-length
expiries, and the shorter the expiry, the worse the execution-cost
problem gets, not better.**

### Relaxing the timeframe limit to 15-60 minutes

After the 1-5 minute result above, the constraint was relaxed to test
whether H16-H20 fare better where execution costs matter proportionally
less. They were re-run on the already-ingested real 1h data (EUR/USD,
GBP/USD, USD/JPY, train split) and on a freshly-fetched real EUR/USD
15-minute series (a different ~52-day window, at 15/30/60-minute
expiries):

- **H16, H18, H19, H20 at 1h: no credible edge**, consistent with every
  shorter timeframe already tested.
- **H17 (Bollinger + RCI) at 1h produced the single strongest,
  most cross-pair-consistent result of the entire investigation**:
  60.8% (EUR/USD), 60.6% (GBP/USD), 58.6% (USD/JPY) on the optimistic
  scenario, train split, all three clearing a 95% CI above 50% with
  reasonable samples (n=145-209). This is exactly the kind of number
  that would look like "found it" without the rest of this
  methodology. It was run through every remaining check before being
  taken seriously:
  - **Validation split**: only 2/3 pairs still clear 50% on the
    optimistic scenario (EUR/USD 62.3%, USD/JPY 61.5%; GBP/USD drops to
    49.4%), and **0/3 pairs clear 50% on the realistic scenario**
    (EUR/USD 46.7%, GBP/USD 32.9%, USD/JPY 36.0%).
  - **12-fold walk-forward on EUR/USD/1h**: optimistic mean win rate
    57.8% (stdev 11.2pp) with 11/12 folds individually positive — the
    most directionally consistent walk-forward result of the whole
    investigation — but only 1 of 8 sufficiently-sampled folds
    individually significant. Under the **realistic scenario, the mean
    drops to 48.1% and 0/8 folds show an edge.**
  - **Fresh out-of-window check**: run again on a different, more
    recent EUR/USD data window (15-minute candles, 15/30/60-minute
    expiries) — 50.7%-52.9%, no longer clearing 50% at all.

  H17 does not survive any of the three independent checks it was put
  through (validation, walk-forward, out-of-window replication) once
  realistic execution costs are applied, despite being the best-looking
  candidate across every earlier, less rigorous check. **No credible
  edge.**

### Where this leaves the timeframe question

Across 1-minute, 5-minute, 15-minute, and 1-hour expiries, on three real
currency pairs, with single and combined indicators: **the pattern is
the same at every timeframe tested, including the ones with the most
favorable cost-to-signal ratio.** Longer timeframes do measurably reduce
how much realistic execution costs erode an apparent edge (roughly
15-25 points at 1h vs. 30-50 points at 1-5 minutes) — but in every case
tested so far, including the strongest candidate found at the most
favorable timeframe, the underlying apparent edge itself did not survive
independent replication even before execution costs were the deciding
factor. This is no longer a statement about any one hypothesis; it is
the accumulated result of the entire investigation to date.

## Pivot: systematic statistical discovery (H1-H20 superseded, not extended)

Every hypothesis above (H1-H20) was hand-picked: a person chose an
indicator or combination, then it was tested. After H1-H20 showed no
credible, replicated edge at any timeframe (see above), the project was
explicitly redirected away from more hand-picked combinations and toward
a bottom-up, systematic search: baseline win-rate statistics →
quantile-binned single-feature conditioning → a combinatorial 2-3-way
interaction search over binned features, with every combination tried
logged and Benjamini-Hochberg FDR-corrected before anything is treated as
a candidate → only a fixed, pre-registered accept/reject bar
(`otc_research/research/candidacy.py`) decides whether a discovered
condition earns the out-of-sample TEST split, touched at most once per
candidate. See `/root/.claude/plans/sequential-sparking-candle.md` for
the full design; this section reports the first real-data run of it.

### Step 7 run: scope

- **Data**: real, already-ingested EUR/USD (1m, 5m, 15m), GBP/USD (5m),
  USD/JPY (5m) — 5 asset/timeframe combinations, ~5,000 candles each,
  labeled `REAL_FOREX_DATA` throughout (`research/dataset.py`).
- **Features searched**: a deliberately restricted 10-feature core subset
  spanning momentum (`return_1`, `return_5`), trend (`rsi_14`, `adx_14`),
  mean-reversion (`cci_20`, `rci_9`, `bb_pct_b_20`), volatility
  (`atr_expansion_ratio`, `move_size_atr`), and position-in-range
  (`pct_position_in_range_20`) — not all ~40 v4 features at once, and
  2-way interactions only (not yet 3-way) — a runtime-budget choice, not
  a methodological one; both are natural next steps if this subset had
  found something worth extending.
- **Targets**: `call_wins_h`/`put_wins_h` for h ∈ {1, 2, 3, 5} candles —
  40 target searches total (5 datasets × 4 horizons × 2 directions), each
  a 45-pair × 16-bin-combination = 720-condition grid, all on the TRAIN
  split only. **28,800 condition trials total, every one logged to the
  new `condition_trials` table before correction.**

### Step 7 run: results

- **2,902 of 28,800 trials (10.1%) survived Benjamini-Hochberg FDR
  correction** (q=0.05) — well above the ~5% a pure null would produce,
  so the search is finding real, non-random structure in how these
  features relate to short-horizon price direction in the naive,
  frictionless sense `research/dataset.py`'s targets measure (immediate
  close-to-close comparison, no execution model at all).
- **Every FDR-significant condition's top 3 (by p-value) per target were
  re-evaluated by `research/candidacy.py` — 120 evaluations — through
  the same realistic-execution simulator (Phase 4) used for H1-H20: 1
  candle of entry delay, 0.01% slippage, 2% signal-drop probability
  (`config.yaml`'s `backtest.execution.realistic`).**
  - **0 of 120 passed the first gate** (TRAIN sample size + payout-
    adjusted margin over break-even at a 0.85 payout, break-even 54.05%).
    **119 of 120 were rejected there**, with a median simulated TRAIN win
    rate of **20.7%** (min 5.6%, max 65.8%) — not merely "no edge", but
    *the labeled direction losing on the clear majority of matched
    trades* once the trade is actually simulated with entry delay and
    slippage, rather than read straight off the next candle's close.
  - The 1 condition that cleared gate 1 (USD/JPY 5m, CALL, h=5,
    `return_5∈(-0.762,-0.0309] AND pct_position_in_range_20∈(0.476,0.749]`,
    n=111, TRAIN win rate 65.8%) was rejected at **gate 2**: its
    parameter-sensitivity sweep classified `"fragile"`, not
    `"consistent_direction"` — the same overfitting signature every
    fragile H1-H20 candidate showed.
  - **0 conditions reached walk-forward or the TEST split.**

### Why the naive search and the realistic simulator disagree this sharply

Checking CALL and PUT win rates for the *same* condition (same matched
candles, same entry/exit indices, same rng draws — they differ only in
direction and the sign of the slippage adjustment) shows both well below
50% simultaneously in the typical case (e.g. USD/JPY 5m h=3,
`return_1∈(0.000125,0.0127] AND return_5∈(3.52e-05,0.0251]`: CALL 18.9%,
PUT 12.7% — CALL+PUT should sum to roughly 100% minus a negligible tie
rate if the underlying price move were decisive either way). The
explanation is execution cost, not a bug: at these timeframes, a large
fraction of the matched short-horizon moves are smaller than the
round-trip slippage cost, so **both directions lose** against costs on
the same trade. `research/dataset.py`'s exploratory target (a bare
close-to-close comparison) has no way to see this — it is intentionally
a cheap, fast proxy for the discovery search's combinatorics, never a
claim about tradeable performance. That gap is exactly why
`research/candidacy.py` exists as a mandatory, separate re-validation
gate through the same battle-tested simulator every hand-picked
hypothesis already had to clear, rather than trusting discovery's own
output directly — and this run is the first real evidence that gate is
doing necessary work, not rubber-stamping.

### Conclusion: **B — Promising but insufficient**

Per the plan's four-way classification: discovery found statistically
significant (FDR-corrected) structure — this is not "no evidence" in the
literal sense — but **none of it survived the very first re-validation
gate against realistic trade execution**, let alone walk-forward or an
out-of-sample TEST read. No condition from this run is a candidate.

This reaches the same practical conclusion as H1-H20 (no credible,
execution-cost-adjusted, replicated edge yet found in this data with the
indicators and combinations tried), but by an independent, systematic,
pre-registered method rather than hand-picked hypotheses — which makes it
corroborating evidence, not a repeat of the same test.

**Scope not yet covered, and the natural next steps if resumed**: the
remaining ~30 v4 features, 3-way interactions, the 1h timeframe (already
extensively covered by H1-H20's own methodology), and — the most
promising lead from the mismatch above — redefining `research/dataset.py`'s
exploratory target to match the realistic simulator's own entry-delay/
expiry convention, so the discovery search itself stops surfacing
patterns that were only ever visible in a frictionless, zero-delay read.

## Explicitly forbidden language

Never describe any hypothesis or strategy, at any status, as: "infallible",
"guaranteed", "90% guaranteed", "winning bot", "sure profit", or similar. The
only vocabulary allowed for a strategy's status is the classification in
BACKTESTING.md (NO EDGE / WEAK EDGE / PROMISING / ROBUST EDGE).
