# Binary-options backtest of existing strategies (Step 9 addendum)

**Status: complete.** This report answers the practical question the
project's audits (`PREDICTABILITY_AUDIT.md`, `BINARY_OPTIONS_REFRAME_AUDIT.md`,
`EXPIRY_UNIVERSE_AUDIT.md`) built up to but did not themselves answer:
taking every strategy/hypothesis that already exists in this codebase —
the 20 hand-designed H1-H20 strategies and the systematically-discovered
conditions from Step 7/8d — which ones show consistent statistical and
economic evidence of working as a real, fixed-expiry binary CALL/PUT
option? **No new strategy was invented to produce a better result.** The
implementation plan (strategy inventory, protocol, architecture changes)
is recorded in `/root/.claude/plans/sequential-sparking-candle.md`'s
"Addendum 2 (Step 9)" section; this document is the result.

## 1. What was run

| Family | Evaluations | Source |
|---|---:|---|
| H1-H20 (base, delay-only corrected model) | 608 | New run: 18 strategies × 8 datasets × 4 horizons |
| H1-H20 + NO_TRADE (volatility-contraction) filter | 16 | New run: only strategies that cleared gate 2 unfiltered |
| H9 (session bias), restricted discovery + candidacy | 44 | New run: 8 datasets × 4 horizons × 24 hours × 2 directions, FDR-screened |
| H9 + NO_TRADE filter | 23 | New run: only H9 candidates that cleared gate 2 unfiltered |
| Discovered conditions (Step 7 + Step 8d, naive target) | 193 | **Reused, zero new runs** — see §7 |
| Triple-barrier discovered conditions | 10 (footnote only) | **Reused, zero new runs** — see §8 |
| **Total** | **884 (+10 footnote)** | |

Every evaluation used the exact same corrected pipeline already applied
to discovered conditions: `delay_only_scenario(1)` execution (no
slippage — see `BINARY_OPTIONS_REFRAME_AUDIT.md`), the four-gate funnel
(`research.candidacy.evaluate_candidacy`, generalized this step to accept
any `Strategy`, not just a discovered `Condition`), and no a-priori
expiry assumption: `expiry_seconds = h × timeframe_seconds`, h ∈
{1,2,3,5} (`EXPIRY_UNIVERSE_AUDIT.md`) — applied uniformly to H1-H20 too,
overriding each class's own hardcoded 300s default.

**Entry price / WIN-LOSS**: unchanged `backtest.simulator.simulate()` —
entry at the open of the candle `entry_delay_candles` after the signal
candle's close, exit at the close of the candle `expiry_seconds` later,
WIN/LOSS from direction vs. price move. No spot-Forex framing anywhere:
every trade's monetary result is `+payout` (WIN) or `-1` (LOSS) in units
of stake, never the underlying's raw percentage move.

**Payout**: gating decisions used 85%; §9 below sweeps {70,75,80,85,90}%
post-hoc from the stored win rates (WIN/LOSS doesn't depend on payout,
only break-even/EV do).

## 2. Classification breakdown

| Family | Total | ACCEPTED | PROMISING_BUT_UNPROVEN | EXPLORATORY | REJECTED |
|---|---:|---:|---:|---:|---:|
| H1-H20 | 608 | 0 | 0 | 156 | 452 |
| H1-H20 + NO_TRADE | 16 | 0 | 0 | 4 | 12 |
| H9 | 44 | 1 | 2 | 0 | 41 |
| H9 + NO_TRADE | 23 | 0 | 2 | 4 | 17 |
| discovered-condition | 193 | 0 | 5 | 0 | 188 |
| **TOTAL** | **884** | **1** | **9** | **164** | **710** |

Classification is mechanical, derived directly from which gate a
`CandidacyVerdict` cleared/failed (never a per-case judgment call):

- **ACCEPTED**: cleared all four gates (sample/margin → robustness →
  walk-forward → TEST, CI-low > break-even).
- **PROMISING_BUT_UNPROVEN**: reached TEST (cleared gates 1-3) but TEST
  CI-low doesn't clear break-even.
- **EXPLORATORY**: rejected at gate 1 specifically for `train_sample_size
  < 100` (signal too rare to judge, not shown to lack edge), or at gate 3
  for too few sufficiently-sampled walk-forward folds.
- **REJECTED**: every other rejection — adequate sample but no real
  margin, inconsistent under perturbation, or walk-forward fails with
  enough folds sampled to mean something.

**H1-H20 (every hand-designed strategy): 0 ACCEPTED, 0 PROMISING_BUT_UNPROVEN.**
452/608 were cleanly REJECTED with adequate data (436 at the margin gate —
real signal, no real edge at 85% payout; 9 at walk-forward; 7 at the
robustness/perturbation gate). The other 156 were EXPLORATORY purely
because the signal is rare on some (strategy, dataset, h) combinations —
36 of those had fewer than 20 TRAIN trades total, which is not
"almost-accepted," it's "not enough occurrences to judge at all." **No
H1-H20 strategy, under the corrected execution model and any of the four
swept expiries, shows real evidence of a binary-options edge.**

## 3. The one ACCEPTED result — and why it should NOT be treated as validated

```
H9 (session bias): EUR_USD, 15m, hour_utc ∈ (5,6] → CALL, h=5 (expiry 4500s ≈ 75 min)
TRAIN:  n=124, win_rate=67.74%, margin@85%=+13.69pp
Robustness: consistent_direction (3/3 sufficiently-sampled points, 100% w/ edge)
Walk-forward: 4 folds sufficiently sampled, 75% with edge, worst fold 55.6% (> break-even 54.05%)
TEST:   n=40, win_rate=70.0%, CI low=54.57%  (> break-even 54.05% by +0.52pp)
```

This mechanically cleared all four pre-registered gates and is therefore
**ACCEPTED under this project's own established criteria.** It is
reported as such. But per the user's own standing instruction not to
present a manufactured or overstated positive result, three things make
this a fragile finding that should NOT be treated as validated evidence
without further, independent replication:

**(a) Payout sensitivity — the accept is knife-edge and payout-dependent.**
Sweeping the TEST CI-low against all 5 required payouts:

| Payout | Break-even | TEST CI-low margin | Clears? |
|---:|---:|---:|---|
| 70% | 58.82% | −4.25pp | **No** |
| 75% | 57.14% | −2.57pp | **No** |
| 80% | 55.56% | −0.99pp | **No** |
| 85% | 54.05% | +0.52pp | Yes (barely) |
| 90% | 52.63% | +1.94pp | Yes |

The crossover payout — the exact point where TEST CI-low equals
break-even — is **83.25%**. Below that, at three of the five required
sensitivity payouts, **this exact same result does NOT clear the bar.**
Point-estimate EV is positive at every payout (because the point win
rate, 70%, is comfortably above break-even everywhere), but this
project's own discipline has always used the CI-low, not the point
estimate, as the pass bar — and by that bar this "ACCEPTED" candidate is
only accepted in the upper fifth of the required payout range.

**(b) TEST sample size (n=40) is well below this project's own established
floor** (100, used at gate 1) — no separate floor is currently enforced
at gate 4. A Wilson CI at n=40 is wide; the margin over break-even (+0.52pp
at 85% payout) is inside the noise a single additional loss would erase
(41 wins/40 losses out of ~80 more trades would materially move this CI).

**(c) Multiple-testing exposure.** H9's restricted discovery ran 64
independent (asset, timeframe, h, target_col) cells — each with its own
Benjamini-Hochberg correction across 24 hour-bins, never pooled across
the 64 cells (matching this project's established per-run-id FDR
discipline, the same as Step 7/8d — not a new gap introduced here, but
worth stating plainly). 44 candidates reached full candidacy evaluation
across those 64 cells; exactly 1 barely cleared TEST. That is the
signature of a plausible false positive slipping through a multi-stage
funnel, not a distinguishing feature of this one candidate.

**Conclusion for this result**: it is reported honestly as ACCEPTED under
the pre-registered mechanical criteria — the system is not overriding its
own rule to relabel it. But given (a)-(c), it should be treated
functionally like the PROMISING_BUT_UNPROVEN tier (real signal, thin and
payout-fragile evidence) until it survives a **fresh, independently
carved TEST slice it has never touched** (per `EXPIRY_UNIVERSE_AUDIT.md`
§5's protocol for validly re-testing an already-observed result) with a
larger resolved-trade count. **No live signal or real-money use should be
based on this result as-is.**

## 4. The 9 PROMISING_BUT_UNPROVEN results

| Estrategia | Activo/TF/h | TRAIN n / WR | TEST n / WR (CI-low) | Nota |
|---|---|---|---|---|
| H9 hour_utc∈(14,15]→CALL | GBP_USD/5m/h=3 | 132 / 66.7% | 48 / 58.3% (44.3%) | +NOTRADE variant: identical (filter barely triggers this hour) |
| H9 hour_utc∈(14,15]→CALL | GBP_USD/5m/h=5 | 132 / 70.5% | 48 / 56.2% (42.3%) | +NOTRADE variant: identical |
| discovered: `adx_14∈(34.99,98.52] AND cci_20∈(-666.7,-76.22]` | EUR_USD/5m | 146 / 79.5% | not reached (rejected earlier, triple-barrier target mismatch — see below) | reused from corrected rerun |
| discovered: `pct_position_in_range_20∈(-2.13,0.236] AND day_of_week∈(4,6]` | USD_JPY/5m | 110 / 72-75% | not reached | reused |
| discovered: `rsi_14∈(42.4,49.59] AND day_of_week∈(4,6]` | USD_JPY/5m | 205 / 66.8% | not reached | reused |
| discovered: `hour_utc∈(5.75,11] AND trading_session_code∈(-0.001,1]` (session-conditioned) | **EUR_USD/15m, h=5** | 372 / 65.3% | 120 / 60.0% (51.1%) | **This is the already-published, frozen candidate — status unchanged, TEST never re-touched (§7)** |

None of these clear TEST. All are reported at exactly the status they
already had (for the discovered-condition rows, reused verbatim from the
prior corrected rerun — never re-evaluated).

## 5. NO_TRADE (volatility-contraction) filter: does it actually help?

Applied only to the 16 H1-H20 and 27 H9-family candidates that cleared
gate 2 unfiltered (per the "don't evaluate every strategy" instruction) —
each filtered variant is its own fresh hypothesis with its own gates and
(where reached) its own first TEST touch, never a post-hoc comparison.

**H1-H20: no improvement in any of the 16 cases.** Every filtered variant
stayed REJECTED (matching its base) or, in 4 H7/GBP_USD/1h cases, the
filter removed enough trades to drop the sample below the EXPLORATORY
threshold — a regression, not an improvement. The filter never turned a
REJECTED base into anything better.

**H9: no improvement either.** For the two GBP_USD 14:00-15:00 UTC
conditions, the filter removed only ~2 of ~132 TRAIN trades (that hour is
the London/New York overlap — it is very rarely in a volatility-
contraction regime by construction), so the filtered and unfiltered
results are numerically almost identical, both still
PROMISING_BUT_UNPROVEN. **The volatility-contraction NO_TRADE filter, as
tested here, does not improve any existing strategy's binary-options
performance** — consistent with `PREDICTABILITY_AUDIT.md`'s earlier
finding that contracting-volatility regimes' achievable moves are already
the same order of magnitude as this filter would need to exclude to
matter.

## 6. Equity curve / drawdown (gate-2+ survivors)

Computed via the new `research/performance_report.py`, built entirely on
existing `Trade` objects (R-multiple: `+payout` WIN, `-1` LOSS, VOID
excluded). Example, the strongest-by-win-rate H1-H20 gate-2 survivor
(H17, EUR_USD/1h/h=1, still REJECTED at walk-forward):

```
n=209 trades, win_rate=60.3%, profit_factor=1.29
cumulative_return_r=+24.1 R, max_drawdown_r=8.75 R, max_consecutive_losses=6
```

A positive TRAIN-only equity curve with a real drawdown and a walk-forward
failure — exactly the "looked good until you check consistency across
time" pattern this project's gates exist to catch, not evidence it works.

## 7. Discovered conditions — reused, TEST never re-touched

The 193 (naive-target) + 10 (triple-barrier, footnote) discovered-condition
evaluations in this report are **parsed verbatim from the existing,
already-corrected rerun** (`scripts/run_candidacy_corrected_rerun.py`'s
log) — **never re-evaluated**. Re-running them, even with an identical
deterministic seed, would touch the already-frozen EUR/USD 15m
session-conditioned candidate's TEST split a second time, which the
TEST-once discipline forbids as a matter of process regardless of whether
the number comes out the same. Its status is unchanged, exactly as
specified previously: **`TEST 60.0%, n=120, IC inferior 51.1%, break-even
54.05%` → PROMISING_BUT_UNPROVEN (rejected under current criteria)**. Any
modified version of it is a brand-new experiment needing its own fresh
protocol — none was run in this step.

## 8. Triple-barrier — auxiliary footnote only, not in the primary comparison

Per the explicit fallback (and the discovery/candidacy target-definition
mismatch already diagnosed in `EXPIRY_UNIVERSE_AUDIT.md` §1: the FDR
pre-selection screened a variable barrier-touch label, but candidacy's
actual trade simulation always resolves WIN/LOSS via the fixed-expiry
mechanic — so the realized numbers are legitimate fixed-expiry binary
numbers, but were not validly pre-registered for the hypothesis ultimately
tested):

| Activo | TF | Target | Trials | FDR-sig |
|---|---|---|---:|---:|
| EUR_USD | 15m | barrier_call/put_wins | 720 each | 5 / 4 |
| EUR_USD | 1m | barrier_call/put_wins | 720 each | 46 / 162 |
| EUR_USD | 5m | barrier_call/put_wins | 720 each | 123 / 79 |
| GBP_USD | 5m | barrier_call/put_wins | 720 each | 50 / 77 |
| USD_JPY | 5m | barrier_call/put_wins | 720 each | 8 / 10 |

Reported for completeness only — **not compared against H1-H20/H9/naive
discovered conditions as a peer entry**, per the mismatch above.

## 9. Payout sensitivity, EV, and the {70,75,80,85,90}% requirement

Every gate-1+ row's break-even/margin/EV across all 5 required payouts is
in the CSV export (`data/binary_options_backtest_master_table.csv`,
gitignored as a regenerable artifact — regenerate with
`python scripts/build_binary_options_report.py`). §3 above is the
detailed worked example for the one ACCEPTED case, which is also the only
case where the payout choice changes the classification outcome — every
PROMISING_BUT_UNPROVEN/REJECTED case's TEST (or TRAIN, where TEST wasn't
reached) result stays on the same side of break-even across the full
70-90% range, since none of those are anywhere near break-even at the
CI-low.

## 10. Forex backtest vs. Pocket Option OTC — kept strictly separate

Everything in this report is a **Forex backtest result** (Twelve Data
`REAL_FOREX_DATA`, see `research/dataset.py`). It is **not evidence** of
how any strategy performs on Pocket Option's OTC feed — no authorized OTC
data source exists for this project (`PREDICTABILITY_AUDIT.md` §10), and
none has been added. Even the one ACCEPTED result, if it were to survive
independent replication on fresh Forex data, would still need a
completely separate OTC-data validation before it could say anything
about Pocket Option specifically. This report makes no claim either way
about OTC performance.

## 11. Final answer

**ACCEPTED (genuinely validated, no caveats): 0.**
**ACCEPTED (mechanically cleared the pre-registered gates, but fragile —
see §3): 1**, out of 884 evaluations spanning every hand-designed
strategy this project has ever built, every systematically-discovered
condition that survived FDR correction and the corrected execution model,
and a dedicated NO_TRADE-filter comparison. **No strategy — hand-designed
or discovered — shows robust, payout-and-sample-consistent evidence of a
binary-options edge under this project's own established criteria.** This
is not a forced negative any more than the one ACCEPTED case is a forced
positive: it is what 884 pre-registered, gate-disciplined evaluations
against real Forex data actually produced.
