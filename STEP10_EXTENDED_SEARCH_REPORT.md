# Step 10: extended search — 3-way interactions + ML models, actually run and gated

**Status: complete.** This closes the last unexecuted piece of the
original statistical-discovery methodology: Step 7's pipeline
deliberately only ran 2-way feature interactions ("3-way is the natural
next step ... not run automatically here") and only ever logged an ML
model's VALIDATION read without ever pushing it through a real TEST
touch. This step runs both, across all 8 real ingested datasets (adding
the three 1h pairs that had never been systematically searched before),
through the exact same four-gate candidacy discipline as everything else
in this project.

## 1. What was run

- **3-way interactions**: `combo_sizes=(2, 3)` (own `run_id` prefix, own
  independent Benjamini-Hochberg correction per (asset, timeframe,
  target_col) — 64 independent FDR-corrected searches, 8,400 conditions
  each, 537,600 trials total across the 8 datasets.
- **ML model sweep**: logistic regression, random forest, gradient
  boosting, fit independently for every (asset, timeframe, horizon,
  direction) — not gated behind discovery finding anything first. Only
  the single best-performing family per target (when its VALIDATION
  CI-low clears break-even with n≥30) is escalated to the full candidacy
  funnel via the new `research.model_strategy.ModelStrategy`, which was
  previously missing: a promising model used to stay a VALIDATION-only
  number, never receiving a genuine TEST touch.
- Same 10-feature `CORE_DISCOVERY_FEATURES` as Step 7 (not the full v4
  set — a further, separate extension). Same corrected delay-only
  execution model as every other candidacy call in this project.

Two performance fixes were needed mid-run (see `git log`): a one-dataset
timing test initially took 9m16s, with ML candidacy evaluation alone
taking 6.3 of those minutes (`ModelStrategy.decide()` calls
`predict_proba` once per candle, and a candidacy call re-walks the full
candle series many times). Reducing `RandomForestClassifier`'s
`n_estimators` (200→50), the threshold-perturbation grid (5→3 points),
and escalating only the best family per target (not all 3 independently)
brought the same dataset down to 3m21s — methodologically neutral
changes (they change how much is checked, not what "passing" means).
Full 8-dataset run: **193 evaluations in ~24 minutes.**

## 2. Classification breakdown

| Track | Evaluations | ACCEPTED | Reached TEST (rejected) | Walk-forward rejected | Robustness rejected | Sample/margin rejected |
|---|---:|---:|---:|---:|---:|---:|
| Discovery (3-way) | 164 | 0 | 10 | 34 | 26 | 94 |
| ML (3 families, best-per-target escalated) | 29 | 1 | 3 | 4 | 0 | 21 |
| **TOTAL** | **193** | **1** | **13** | **38** | **26** | **115** |

**3-way interactions found dramatically MORE raw FDR-significant hits**
than 2-way alone (some target_cols went from ~10-20 significant
conditions under 2-way to 300+ once 3-way was added — e.g. USD_JPY/1h
`call_wins_5` alone had 330 FDR-significant winning-direction 3-way
conditions). **This did not translate into more real edge** — of the 10
(top-3-per-target, capped) discovery candidates that reached TEST, every
single one failed clearly (TEST CI-low ranging 0.27-0.48, all well below
the 0.5405 break-even) — no near-misses, no ambiguity. Adding
combinatorial complexity mainly manufactured more candidates that looked
significant in TRAIN, not more that survived out-of-sample.

## 3. The one ACCEPTED result (again — read this with the same scrutiny as Step 9's)

```
Random forest model, USD_JPY, 1h, target=call_wins_3, h=3 (expiry 3h) -> CALL
TRAIN:  n=2,134, win_rate=62.93%
TEST:   n=730,   win_rate=57.67%, CI low=54.06%  (> break-even 54.05% by +0.0001)
```

This mechanically cleared all four gates and is honestly reported as
ACCEPTED. As with Step 9's H9 finding, this should NOT be treated as
validated:

**(a) The margin is not just thin — it's effectively zero.** CI-low
clears break-even by 0.00002 (two hundred-thousandths). This is the
smallest possible margin above the threshold this project's own bar would
still call a pass; one different random split of the exact same window,
or one different candle at the boundary, could flip it.

**(b) Payout sensitivity — fails at every realistic payout except 85-90%:**

| Payout | Break-even | TEST CI-low margin | Clears? |
|---:|---:|---:|---|
| 70% | 58.82% | −4.77pp | **No** |
| 75% | 57.14% | −3.09pp | **No** |
| 80% | 55.56% | −1.50pp | **No** |
| 85% | 54.05% | +0.002pp | Yes (by a hair) |
| 90% | 52.63% | +1.42pp | Yes |

Crossover payout: **85.0%** — even higher than Step 9's H9 finding
(83.25%). Despite an 18x larger TEST sample (730 vs 40), this result is
if anything MORE payout-fragile, not less — the larger n makes the CI
narrower, but the point estimate itself sits closer to break-even here.

**(c) The strongest reason to distrust this specific case: it did not
arise in isolation.** USD_JPY/1h is the single dataset responsible for
**11 of this run's 13 TEST arrivals** across BOTH tracks (9 from 3-way
discovery, 2 from ML) — nearly every one built from an overlapping set of
the same few features (`bb_pct_b_20`, `adx_14`, `rsi_14`, in various
2-3-way combinations, plus this one ML model trained on the same 10
features). **10 of those 11 failed TEST clearly.** One passing by a
hair, out of eleven highly-correlated attempts drawn from the same
window, is the textbook signature of a multiple-comparisons false
positive slipping through per-search FDR control (which bounds the false
discovery rate WITHIN each of the 64 independent searches run here, not
across all 64 jointly) — not evidence that USD_JPY's 1h chart in this
period has a real, exploitable Q1-Q3 2026 regime.

**Conclusion for this result, same as Step 9's**: reported honestly as
ACCEPTED under the pre-registered mechanical bar, but functionally no
more trustworthy than a PROMISING_BUT_UNPROVEN result until it survives
an independently-carved, never-before-touched TEST slice. No live signal
or real-money use should be based on it.

## 4. Where this leaves the project's overall search

Across every systematic and hand-designed search this project has run
(H1-H20, H9, H21, 2-way discovery, 3-way discovery, 3 ML model families,
a NO_TRADE filter) — **well over 1,000 evaluations in total** — exactly
**two** results have ever mechanically cleared every pre-registered gate:
Step 9's H9 session-bias condition and this step's random-forest model.
Both share the same fingerprint: a TEST CI-low margin over break-even of
about half a percentage point or less, a payout crossover in the
low-to-mid 80s%, and provenance from a large multi-candidate search where
close siblings failed. Neither is treated as validated. **Zero results
across this entire project constitute unqualified, robust evidence of a
binary-options edge.**

This does not mean no strategy can ever work — see the earlier discussion
of remaining paths (real Pocket Option OTC data, live forward/paper
testing of a candidate, or reframing the goal as decision support rather
than full automation). It does mean that neither adding 3-way feature
interactions nor escalating ML models changed the project's core finding:
this real Forex data, at these timeframes, does not contain an edge large
and stable enough to survive this project's own (appropriately strict)
validation bar.
