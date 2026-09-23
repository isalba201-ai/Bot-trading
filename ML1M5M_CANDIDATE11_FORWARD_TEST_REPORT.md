# ML_1M5M Candidate #11 — Forward Test Report

**Status: forward test complete. Results are reported exactly as produced —
no threshold, model, feature, or filter was changed after seeing them, and
no unfavorable day/hour/block was removed.**

This report follows directly from `ML1M5M_EXPERIMENT_REPORT.md` (original
candidate discovery), the split-mismatch correction documented there, and
the two-filter controlled comparison in the prior phase. Per the user's
explicit instruction for this phase: *"A partir de ahora NO quiero seguir
tuneando la estrategia. Quiero probarla."* This is a single, blind pass over
genuinely new data — not a new round of tuning, and not a final verdict on
the strategy's viability.

---

## 1. What was frozen (unchanged from all prior phases)

- **Model**: `GradientBoostingClassifier`, `seed=0`, refit deterministically
  on the exact original TRAIN range `2026-07-21 00:41 → 2026-07-29 10:00`
  (11,374 rows). Never retrained on validation, TEST, or forward data.
- **Features**: the same 30 `ML1M5M_FEATURES` (feature set `v4`), unchanged.
- **Target**: `call_wins_5` (5-candle-ahead binary outcome), unchanged.
- **Expiry**: 300 seconds (5 candles), unchanged.
- **Asset/timeframe**: EUR_USD, 1-minute candles, unchanged.
- **WIN/LOSS/VOID definition**: unchanged (`backtest.simulator.simulate`).
- **Delay treatment**: both pre-registered scenarios evaluated, unchanged —
  `delay0` (`entry_delay_candles=0`, near-instant/optimistic execution) and
  `delay1` (`entry_delay_candles=1`, the project's standing realistic
  default — 60 seconds of latency before entry).
- **Three strategies, frozen at these exact thresholds**:
  - **Baseline**: `P(CALL) > 0.50`
  - **B (P>0.65)**: `P(CALL) > 0.65` (same model object, different cutoff)
  - **C (CCI≥-60)**: baseline wrapped in `FilteredStrategy`, blocked when
    `cci_20 < -60`
- No new threshold was tried, no filters were combined, no new features or
  hyperparameters were introduced, and no time period was cherry-picked.

**Reproducibility check**: refitting the model produced VALIDATION
`n=1703, WR=0.5890` — exactly matching every prior phase's refit of this
same model. The frozen model was persisted to
`data/forward_test_models/ml1m5m_candidate11_gradient_boosting_frozen.joblib`
(SHA-256 `eaa78365572b6e16f1f716d217951e064cb4dd5c8746e58cf9222091ec2296c7`).

## 2. Forward data (traceability)

- **Source**: Twelve Data, `EUR/USD`, 1-minute interval.
- **Range obtained**: `2026-09-18 21:02:00` → `2026-09-22 21:02:00` request
  window; actual candles returned span `2026-09-19 09:43:00 →
  2026-09-22 21:02:00` (n = 5,000 candles).
- **Disjointness**: strictly after `LAST_HISTORICAL_CUTOFF =
  2026-09-18 21:01:01`, which is one second after the last candle used by
  *any* prior phase — both the TRAIN/VALIDATION/TEST window
  (`2026-07-21 → 2026-08-04`) and the unrelated, already-spent Step10/11
  Sept 15–18 block. Verified at ingestion time (`candles_skipped = 0`,
  confirming zero overlap with any previously-stored candle) and again at
  runtime via explicit asserts (`fwd_start > LAST_HISTORICAL_CUTOFF`,
  `fwd_start > TEST.end`).
- **TEST split** (`2026-08-01 05:21 → 2026-08-04`): not touched in this
  phase. It was never loaded, never scored, never used as a substitute for
  anything.
- **Execution**: a single blind pass — `simulate()` called exactly once per
  (strategy, delay-scenario) over the *entire* forward block. All
  aggregate/per-day/per-block numbers below are the same trade list, sliced
  by timestamp after the fact — nothing was re-run with a different
  sub-window.

## 3. Aggregate results

Break-even win rate at payout 0.85: **54.05%**.

| Strategy | Delay | n | WIN | LOSS | VOID | WR | 95% CI | Margin | Coverage |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (P>0.50) | delay0 | 2,373 | 1,371 | 1,002 | 3 | 57.77% | (55.78%, 59.75%) | **+3.72pp** | 100.0% |
| P>0.65 | delay0 | 1,461 | 975 | 486 | 0 | 66.74% | (64.28%, 69.10%) | **+12.68pp** | 61.5% |
| CCI≥-60 | delay0 | 1,399 | 953 | 446 | 2 | 68.12% | (65.63%, 70.51%) | **+14.07pp** | 59.0% |
| Baseline (P>0.50) | delay1 | 2,373 | 1,141 | 1,232 | 3 | 48.08% | (46.08%, 50.09%) | **-5.97pp** | 100.0% |
| P>0.65 | delay1 | 1,461 | 644 | 817 | 0 | 44.08% | (41.55%, 46.64%) | **-9.97pp** | 61.5% |
| CCI≥-60 | delay1 | 1,399 | 570 | 829 | 2 | 40.74% | (38.20%, 43.34%) | **-13.31pp** | 59.0% |

The single dominant finding: **every strategy reverses sign between delay0
and delay1.** Under delay0 all three clear break-even with margins that are
in fact *larger* than anything seen historically. Under delay1 — the
project's standing realistic execution assumption — all three fall not just
below break-even but below a 50% coin-flip, and every 95% CI sits entirely
below break-even (the CI upper bound for CCI≥-60 delay1 is 43.34%, nowhere
near the 54.05% bar).

## 4. Per-day breakdown (no days removed)

| Date | Strategy | Delay | WIN | LOSS | n | WR |
|---|---|---|---|---|---|---|
| 09-19 | Baseline | delay0 | 267 | 143 | 410 | 65.12% |
| 09-20 | Baseline | delay0 | 438 | 250 | 688 | 63.66% |
| 09-21 | Baseline | delay0 | 353 | 335 | 688 | 51.31% |
| 09-22 | Baseline | delay0 | 313 | 274 | 587 | 53.32% |
| 09-19 | P>0.65 | delay0 | 257 | 104 | 361 | 71.19% |
| 09-20 | P>0.65 | delay0 | 429 | 186 | 615 | 69.76% |
| 09-21 | P>0.65 | delay0 | 180 | 117 | 297 | 60.61% |
| 09-22 | P>0.65 | delay0 | 109 | 79 | 188 | 57.98% |
| 09-19 | CCI≥-60 | delay0 | 220 | 58 | 278 | 79.14% |
| 09-20 | CCI≥-60 | delay0 | 342 | 127 | 469 | 72.92% |
| 09-21 | CCI≥-60 | delay0 | 209 | 139 | 348 | 60.06% |
| 09-22 | CCI≥-60 | delay0 | 182 | 122 | 304 | 59.87% |
| 09-19 | Baseline | delay1 | 190 | 220 | 410 | 46.34% |
| 09-20 | Baseline | delay1 | 327 | 361 | 688 | 47.53% |
| 09-21 | Baseline | delay1 | 339 | 349 | 688 | 49.27% |
| 09-22 | Baseline | delay1 | 285 | 302 | 587 | 48.55% |
| 09-19 | P>0.65 | delay1 | 150 | 211 | 361 | 41.55% |
| 09-20 | P>0.65 | delay1 | 257 | 358 | 615 | 41.79% |
| 09-21 | P>0.65 | delay1 | 144 | 153 | 297 | 48.48% |
| 09-22 | P>0.65 | delay1 | 93 | 95 | 188 | 49.47% |
| 09-19 | CCI≥-60 | delay1 | 85 | 193 | 278 | 30.58% |
| 09-20 | CCI≥-60 | delay1 | 166 | 303 | 469 | 35.39% |
| 09-21 | CCI≥-60 | delay1 | 169 | 179 | 348 | 48.56% |
| 09-22 | CCI≥-60 | delay1 | 150 | 154 | 304 | 49.34% |

Under delay0, day 09-19 and 09-20 are consistently the strongest for every
strategy (WR 60–79%), while 09-21/09-22 are markedly weaker (51–61%) —
still all above break-even, but a real and visible day-to-day decline
within the 4-day window, not a flat effect.

Under delay1, **every single day for every strategy is below break-even**
(range 30.6%–49.5%). This is not one bad day dragging an otherwise-good
average down — it is uniform across all four days. The CCI≥-60 filter is
the worst performer on the first two days specifically (30.6%, 35.4%)
before drifting back toward ~49% on days 3–4.

## 5. Four sequential temporal blocks (no blocks removed)

| Block | Window | Strategy | Delay | n | WR | Margin | Edge? |
|---|---|---|---|---|---|---|---|
| 0 | 09-19 09:43 → 09-20 06:33 | Baseline | delay0 | 596 | 65.10% | +11.05pp | Yes |
| 1 | 09-20 06:33 → 09-21 03:23 | Baseline | delay0 | 608 | 62.66% | +8.61pp | Yes |
| 2 | 09-21 03:23 → 09-22 00:12 | Baseline | delay0 | 592 | 49.32% | -4.73pp | No |
| 3 | 09-22 00:12 → 09-22 21:02 | Baseline | delay0 | 577 | 53.73% | -0.33pp | No |
| 0 | " | P>0.65 | delay0 | 529 | 70.70% | +16.65pp | Yes |
| 1 | " | P>0.65 | delay0 | 532 | 69.74% | +15.68pp | Yes |
| 2 | " | P>0.65 | delay0 | 214 | 57.01% | +2.96pp | Yes |
| 3 | " | P>0.65 | delay0 | 186 | 58.06% | +4.01pp | Yes |
| 0 | " | CCI≥-60 | delay0 | 401 | 77.06% | +23.00pp | Yes |
| 1 | " | CCI≥-60 | delay0 | 415 | 73.25% | +19.20pp | Yes |
| 2 | " | CCI≥-60 | delay0 | 279 | 56.63% | +2.58pp | Yes |
| 3 | " | CCI≥-60 | delay0 | 304 | 59.87% | +5.81pp | Yes |
| 0 | " | Baseline | delay1 | 596 | 47.32% | -6.74pp | No |
| 1 | " | Baseline | delay1 | 608 | 47.53% | -6.52pp | No |
| 2 | " | Baseline | delay1 | 592 | 48.82% | -5.24pp | No |
| 3 | " | Baseline | delay1 | 577 | 48.70% | -5.35pp | No |
| 0 | " | P>0.65 | delay1 | 529 | 42.53% | -11.52pp | No |
| 1 | " | P>0.65 | delay1 | 532 | 41.17% | -12.89pp | No |
| 2 | " | P>0.65 | delay1 | 214 | 50.47% | -3.59pp | No |
| 3 | " | P>0.65 | delay1 | 186 | 49.46% | -4.59pp | No |
| 0 | " | CCI≥-60 | delay1 | 401 | 33.17% | -20.89pp | No |
| 1 | " | CCI≥-60 | delay1 | 415 | 34.22% | -19.84pp | No |
| 2 | " | CCI≥-60 | delay1 | 279 | 51.97% | -2.08pp | No |
| 3 | " | CCI≥-60 | delay1 | 304 | 49.34% | -4.71pp | No |

Under delay1, **all 12 block-level cells across all three strategies show
`edge=False`** — the negative result is uniform across every time slice of
the forward window, not concentrated in any one block. Under delay0, the
baseline loses its edge in the second half of the window (blocks 2–3),
while the two filters retain a small positive margin throughout, including
in the weaker blocks 2–3 — the one place the filters' historical
Fold-3-stabilization hypothesis shows any support at all, and only under
the less realistic delay0 assumption.

## 6. Descriptive comparison against the historical baseline (non-chasing)

| Metric | Historical (TRAIN / walk-forward) | Forward (new data) |
|---|---|---|
| TRAIN WR, delay0 | 58.78% (n=5,612) | — |
| TRAIN WR, delay1 | 57.05% (n=5,612) | — |
| VALIDATION WR | 58.90% (n=1,703) | — |
| Walk-forward fold WR, delay0 (4 folds) | 57.97%, 56.31%, 57.07%, **51.49%** (weakest: fold 3) | — |
| Forward WR, delay0 (baseline) | — | 57.77% (n=2,373) |
| Forward WR, delay1 (baseline) | — | **48.08%** (n=2,373) |

Historically, **both** delay scenarios showed a modest but consistent edge
over break-even (delay0 margin ≈+4.7pp on TRAIN, delay1 margin ≈+3.0pp on
TRAIN) — delay sensitivity was present but not dramatic. On the new forward
data, delay0 not only preserved an edge but showed a *larger* one than
history for the two filtered variants (up to +14.07pp vs. history's best
fold-level number of ~+3.9pp). Delay1, by contrast, **did not just weaken —
it fully inverted**, landing 9–13 percentage points below break-even where
history showed a small positive margin. This specific pattern — a wide,
newly-emergent gap between delay0 and delay1 that did not exist at this
scale in the historical data — is the forward test's central, surprising
finding.

## 7. Answers to the six required questions

**A — What happened to the Baseline (P>0.50)?** Under delay0 it kept a
statistically real edge (WR 57.77%, CI entirely above break-even, n=2,373).
Under delay1 it inverted to WR 48.08%, with a 95% CI (46.08%, 50.09%) that
sits entirely below break-even and mostly below 50%. The edge exists at
delay0 and evaporates — reverses — at delay1.

**B — What happened to P>0.65?** Same pattern, more pronounced in both
directions: delay0 WR 66.74% (its best historical showing by a wide
margin), delay1 WR 44.08%, CI (41.55%, 46.64%) — clearly below break-even.
The higher-confidence filter did not protect against the delay1 collapse;
if anything its delay1 margin (-9.97pp) is worse than the unfiltered
baseline's (-5.97pp).

**C — What happened to CCI≥-60?** The most extreme case in both
directions: delay0 WR 68.12% (highest of the three), delay1 WR 40.74%
(lowest of the three), margin -13.31pp. The filter that showed the
strongest historical hypothesis for stabilizing weak folds shows the
largest reversal on new data under the realistic delay assumption.

**D — Are the results consistent with history?** Only partially, and only
for delay0. The direction of the historical edge (positive, filters ranked
above baseline) is reproduced under delay0 with even larger margins. Under
delay1, the historical result (small positive margin, ~+3pp) is **not**
reproduced — the new data shows the opposite sign, roughly 3–4x larger in
magnitude, and uniform across all 4 days and all 4 time blocks. This is not
a case of "results are noisier but pointing the same way" — the delay1
result is a clean, consistent reversal.

**E — What is the uncertainty level given the sample size?** Sample sizes
are large enough that this is not a small-sample artifact: n ranges from
1,399 to 2,373 per cell, and every delay1 CI's *upper* bound (e.g. 50.09%
for baseline, 46.64% for P>0.65, 43.34% for CCI≥-60) sits at or below
break-even. The per-day and per-block breakdowns show the same sign in
every one of the 4 days and all 12 block-level delay1 cells — this is a
precise, internally consistent negative result under delay1, not a wide,
uncertain one. The delay0 result is similarly precise and internally
consistent in the positive direction. The genuine uncertainty is not
statistical (within this 4-day window) but about *why* delay0 and delay1
diverge so much more sharply here than in the historical window, and
whether that divergence itself is stable across further out-of-sample
periods — a single ~4-day forward block cannot answer that.

**F — Is there sufficient evidence to justify further validation?** There
is enough evidence to say the delay1 (realistic) result on this new block
is decisively negative and not explained by sample-size noise or a single
bad day/period. There is not enough evidence — a single ~2,373-signal, 4-day
window — to conclude the historical delay0 edge is durable, nor to
conclude the delay1 collapse is a permanent property of the model rather
than a feature of this particular 4-day stretch of the market. Further
validation on additional, independently-sourced forward blocks would be
needed before drawing either conclusion with confidence — but that
validation should be planned as a continuation of blind forward testing
(same protocol, new data), not as a re-tuning exercise on this block.

## 8. What this report does not do

- It does not declare a winner among Baseline / P>0.65 / CCI≥-60.
- It does not propose a parameter, threshold, or filter change in response
  to the delay1 result.
- It does not exclude, adjust, or explain away any day or block.
- The delay1 result is reported as it came out: **decisively below both
  break-even and 50%, consistently across every day and every time block
  of the forward window.**

## 9. Files changed in this phase

- `scripts/run_ml1m5m_candidate11_forward_test.py` (new) — the forward-test
  driver (model freezing, blind single-pass simulation, aggregate/per-day/
  per-block reporting).
- `tests/test_ml1m5m_candidate11_forward_test.py` (new) — 6 safety tests
  (frozen thresholds, cutoff correctness, no TRAIN/VALIDATION/TEST overlap
  with the forward block, model-fit-once guarantee, no combined filters).
- `data/forward_test_models/ml1m5m_candidate11_gradient_boosting_frozen.joblib`
  (new) — the persisted, hash-verified frozen model.
- `data/raw/eur_usd_1m_twelvedata_ml1m5m_forward_2026-09-19_to_2026-09-22.csv`
  (new, gitignored) — the raw forward-test data as fetched.
- `ML1M5M_CANDIDATE11_FORWARD_TEST_REPORT.md` (new, this file).

## 10. Tests

Full suite: **392 passed, 1 skipped** (392/392 executable tests passed).
The one skip (`test_freeze_model_calls_fit_and_evaluate_exactly_once`) is a
documented guard for a test that needs the real ingested EUR_USD/1m
historical window, unavailable in the synthetic per-test DB fixture; the
"model fit exactly once" guarantee it checks was instead verified directly
from the forward-test script's own run log, which shows exactly one
`"Refit VALIDATION check"` line.

## 11. Git

- Commit `5e7bcc6` — forward-test script + safety tests.
- Commit `197b7d5` — frozen model artifact.
- This report is committed separately below.
- No files unrelated to this experiment were modified.
