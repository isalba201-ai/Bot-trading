# Backtesting methodology

**Status: this is the planned methodology. The backtesting engine itself
(Phase 4) is not implemented yet — nothing in this repository has run a
backtest.** This document exists now, before any engine code, so the rules
are fixed before results exist that could tempt bending them.

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
by manually placing it as a fixed-payout instrument elsewhere; when it does
apply, payout is never hardcoded. For any payout `p` (e.g. `0.92`):

```
break_even_win_rate = 1 / (1 + p)
expectancy_per_trade = win_rate * p - loss_rate
```

(`loss_rate = 1 - win_rate` for a binary win/loss outcome with no ties.)
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
