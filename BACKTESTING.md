# Backtesting methodology

**Status: Phases 4-7 are implemented** — `src/otc_research/backtest/` runs
a strategy over stored candles/features under the temporal split and the
three execution-realism scenarios below and reports Wilson-CI-grounded
win rate and expectancy (Phase 4); `otc_research.strategies` implements
H1-H10 against that engine (Phase 5); `backtest/robustness.py` sweeps a
strategy's parameters, expiry, and time windows (Phase 6);
`backtest/walkforward.py` slides train/test fold windows across the whole
history and reports mean/dispersion/worst-fold win rate (Phase 7).
**Phase 8 (Monte Carlo) is not implemented yet**, so nothing in this
repository is entitled to a final NO EDGE / WEAK EDGE / PROMISING /
ROBUST EDGE classification — see "What's implemented" below, and
STRATEGIES.md for what an actual run against real EUR/USD data has shown
so far (short version: nothing robust yet across 20 hand-designed
hypotheses, H1-H20). This document was written before any of that code,
so the rules were fixed before results existed that could tempt bending
them — see git history for the pre-implementation version.

**This methodology is also now reused, unchanged, by a systematic
statistical-discovery layer** (`otc_research/research/`, see
STRATEGIES.md's "Pivot" section): instead of a person hand-picking the
next indicator combination, an FDR-corrected search proposes candidate
conditions, and `research/candidacy.py` puts every one of them through
this exact same engine before it's eligible to be called a candidate —
see the "What's implemented" table's new row below.

## Train / validation / test split

Strict temporal split, never shuffled:

- 60% TRAIN — used to explore and fit any strategy parameters.
- 20% VALIDATION — used to select among candidate strategies/parameter
  sets.
- 20% OUT-OF-SAMPLE TEST — touched exactly once, after every development
  decision (which hypotheses to keep, which parameters to use) is final.

The test split is frozen the moment training begins. If a result on it
looks bad, the answer is to report that, not to go back and re-tune.

## Walk-forward analysis

In addition to the single train/val/test split, strategies that pass it are
re-evaluated with walk-forward analysis: train on a window, test on the
following out-of-sample window, slide forward, repeat. Results are reported
aggregated across all folds (mean, dispersion, worst fold) — a strategy that
only works in one fold is not robust.

## Execution realism

Every backtest is run under three scenarios, not just the best case:

- **Optimistic** — no latency, no slippage, every signal fills.
- **Realistic** — configurable latency/entry delay, occasional missed
  candles/discarded signals, execution errors.
- **Pessimistic** — worse versions of the same.

A strategy that only survives the optimistic scenario is classified
**fragile**, not profitable.

## Payout and expectancy

A plain directional Forex signal has no fixed payout — expectancy for it is
expressed in price terms (expected pips/return per trade), not win-rate vs.
payout. The payout math below only applies if you choose to act on a signal
by manually placing it as a fixed-payout instrument elsewhere (e.g. a binary
option) — when it does apply, payout is never hardcoded, and it changes
what "an edge" even means: **50% win rate is not break-even for a
fixed-payout instrument.** For any payout `p` (e.g. `0.92`), implemented in
`backtest/metrics.py::break_even_win_rate` / `payout_adjusted_expectancy`:

```
break_even_win_rate = 1 / (1 + p)
expectancy_per_trade = win_rate * p - loss_rate
```

(`loss_rate = 1 - win_rate` for a binary win/loss outcome with no ties.) A
typical binary-options payout of 0.80-0.90 puts break-even at roughly
52.6%-55.6%, not 50% — every "does this clear 50%?" read elsewhere in this
codebase is a necessary but not sufficient condition once you're actually
trading a fixed-payout instrument; see STRATEGIES.md's H16-H20 for why this
matters in practice.
Expectancy is computed per strategy, per asset, per hour, per session, and
per market regime — a strategy with a high win rate but negative expectancy
after payout is **NOT PROFITABLE**, full stop.

## Monte Carlo

For any strategy that passes walk-forward and out-of-sample testing, the
sequence of trade outcomes is reshuffled (bootstrap/permutation) to
estimate: expected and worst-case drawdown, probability of losing X%,
probability of reaching a given profit target, the distribution of possible
equity curves, and losing-streak length. The original backtest's single
equity curve is never treated as "the" result — it's one draw from a
distribution.

## Robustness / sensitivity testing

Every promising strategy is deliberately stress-tested by varying its
parameters, the assets, the time period, the time of day, the payout, the
expiration, and the sample size. A strategy that only works at one exact
parameter value (e.g. RSI=73) and breaks under small perturbations is
classified **OVERFITTED / FRAGILE**, regardless of how good its original
backtest looked.

## Multiple testing / data-mining control

Every hypothesis considered is registered in the `hypotheses` table *before*
its results are known (see DATA.md, STRATEGIES.md). When many hypotheses are
tested, some will look good purely by chance — reporting only the best one
without accounting for how many were tried overstates significance. This
project's controls:

- All tested hypotheses are logged, win or lose, so the total number tried
  is always visible for an outside audit.
- The out-of-sample test split is used exactly once, after hypothesis
  selection is final — it is not available during the exploration phase, so
  it cannot itself be data-mined.
- Statistical significance claims account for the number of comparisons
  made (e.g. adjusting confidence thresholds when many hypotheses/parameter
  combinations were tried), rather than treating a single p-value from the
  best-performing configuration at face value.

## Edge classification

A strategy is classified based on the combination of: sample size,
confidence interval on win rate, expectancy sign and stability, walk-forward
consistency, out-of-sample performance, Monte Carlo drawdown/ruin risk, and
parameter sensitivity — never on a single metric like "win rate > 55%".

- **NO EDGE** — fails one or more of the above; default classification.
- **WEAK EDGE** — marginal, small sample, or unstable across folds/regimes.
- **PROMISING** — positive and stable in validation, not yet confirmed OOS.
- **ROBUST EDGE** — positive expectancy that survives out-of-sample testing,
  walk-forward, Monte Carlo, and parameter sensitivity checks simultaneously.

Only **ROBUST EDGE** strategies are eligible to feed the signal engine
(Phase 9), and even then only above the configured quality tier (default:
A+ only — see RISK_MANAGEMENT.md).

## What's implemented (Phase 4) vs. still planned

| This document's rule | Implemented in |
|---|---|
| Strict temporal 60/20/20 train/validation/test split, never shuffled | `backtest/splits.py` |
| Test split touched deliberately, not by accident | `backtest/engine.py::run_backtest` requires an explicit `split=` argument and logs a loud warning on `split="test"` |
| Execution realism: optimistic / realistic / pessimistic | `backtest/execution.py` (parameters in `config.yaml: backtest.execution`), applied by `backtest/simulator.py` (entry delay, signal drop probability, slippage) |
| Win rate reported only with sample size + 95% CI, never a bare point estimate | `backtest/metrics.py::wilson_confidence_interval` / `summarize_trades` |
| Expectancy in price terms (a plain directional signal has no fixed payout) | `backtest/metrics.py::TradeStats.expectancy_pct` |
| Every run's inputs auditable | `db.models.BacktestRun`, one row per (strategy, asset, timeframe, split, scenario), including the RNG seed used |
| Walk-forward analysis | `backtest/walkforward.py::generate_folds` + `run_walk_forward` + `summarize_walk_forward` — slides train/test fold windows (default non-overlapping) across the whole ingested history, evaluates each fold's test window independently via `run_backtest(split="walk_forward")`, and reports mean win rate, population stdev across folds, and the worst single fold — never just the best-looking fold. The train window isn't used for fitting yet since no current strategy fits parameters from data; see the module docstring |
| Monte Carlo (bootstrap/permutation of trade sequences) | **Not implemented (Phase 8)** |
| A discovered (not hand-designed) condition's candidacy bar | `research/candidacy.py::evaluate_candidacy` — re-runs a `research.discovery`-found condition through the SAME unmodified `run_backtest`/`evaluate_robustness`/`run_walk_forward` this table already describes (sample size + payout-adjusted margin → robustness sweep → walk-forward → TEST, touched once) — see STRATEGIES.md's "Pivot" section for the first real-data run and why nearly every discovered condition failed at the very first gate |
| Robustness / parameter sensitivity sweeps | `backtest/robustness.py::run_parameter_sweep` + `evaluate_robustness` — varies a strategy's constructor parameters, its expiry, and/or the time window, then checks whether a Wilson-CI edge holds across at least `min_edge_fraction` (default 70%) of the sufficiently-sampled points ("consistent_direction") or only at a lucky few ("fragile"). Deliberately uses different vocabulary from the final edge classification below — see the module docstring |
| Multiple-testing / data-mining control across many hypotheses | Partially: `Hypothesis` table + `BacktestRun` rows make every run auditable, but no automatic adjustment of significance thresholds for the number of comparisons made exists yet |
| Final edge classification (NO EDGE / WEAK EDGE / PROMISING / ROBUST EDGE) | **Not implemented** — requires walk-forward + Monte Carlo + robustness together; walk-forward and robustness (Phases 6-7) are done, Monte Carlo (Phase 8) is not, so nothing produced by the engine today should be read as a final classification |
| The H1-H10 strategies themselves | Implemented (Phase 5) — `otc_research/strategies/` |

Individual simulated trades are not persisted to the database — they are
exactly reproducible from (candles, features, strategy, split, scenario,
rng_seed), all of which a `BacktestRun` row records, so only the
aggregate statistics are stored to keep the database lean without losing
auditability.
