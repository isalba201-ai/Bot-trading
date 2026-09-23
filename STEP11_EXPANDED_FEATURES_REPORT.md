# Step 11: expanded-feature discovery search — the ~40 v4 features never yet tried

**Status: complete.** This is the first of the two bounded, disciplined
deliverables approved after the user asked to "keep searching until
something profitable is found." Rather than searching indefinitely over
the same data already used by Steps 7-10 (which cannot manufacture more
real evidence, only more chance false positives), this step runs one
more, pre-registered, FDR-corrected discovery search — restricted to the
subset of `feature_set_version = "v4"` features that had genuinely never
been fed into `research/discovery.py` before. The second approved
deliverable, forward/paper-testing of the two existing fragile candidates
against brand-new data, is documented separately (see
`src/otc_research/research/forward_test.py` and
`scripts/run_forward_test_poll.py`).

## 1. What was run

- **21 features never used by Steps 7-10's `CORE_DISCOVERY_FEATURES`**:
  `ema_slope_12_3`, `roc_10`, `return_2`, `return_3`, `return_10`,
  `return_15`, `return_30`, `cumulative_return_10`, `return_acceleration`,
  `macd_cross_signal`, `same_color_streak`, `range_ratio_20`,
  `rolling_std_return_20`, `upper_wick_ratio`, `lower_wick_ratio`,
  `body_ratio`, `engulfing_signal`, `inside_bar_breakout_signal`,
  `dist_to_high_atr_20`, `dist_to_low_atr_20`, `structure_bias`.
  Deliberately excludes every raw, absolute-price-scale feature (`ema_12`,
  `ema_26`, Bollinger bands, Donchian channel, `atr_14`, MACD line/signal/
  histogram) — a quantile-binned condition on an absolute price level
  isn't a meaningful, portable rule (same concern raised in
  `EXPIRY_UNIVERSE_AUDIT.md` about OTC portability).
- **2-way interactions only** (`combo_sizes=(2,)`) — with 21 features,
  3-way combinations (~1,330 per dataset/target vs. 210 for 2-way) were
  judged disproportionate to run across all 8 datasets for a bounded,
  single search; this is a deliberate scope choice, not a finding.
- Same 8 real datasets as every prior step (EUR_USD 1m/5m/15m/1h, GBP_USD
  5m/1h, USD_JPY 5m/1h), same 4 horizons (h ∈ {1,2,3,5}), same 2 target
  columns (`call_wins_h`, `put_wins_h`), same `n_bins=4`,
  `min_sample_size=50`, own independent Benjamini-Hochberg correction
  (`fdr_q=0.05`) per (asset, timeframe, target_col) — 64 independent
  FDR-corrected searches, never pooled with Step 7/8d/10's runs for
  significance purposes. Same corrected delay-only execution model as
  every other candidacy call in this project.
- Only the top 3 FDR-significant, winning-direction survivors per
  (asset, timeframe, target_col) were escalated to the full candidacy
  funnel (`evaluate_condition_candidacy`), same cap Step 10 used.

Full 8-dataset run: **173 evaluations in ~14 minutes.**

## 2. Classification breakdown

| Evaluations | ACCEPTED | Reached TEST (rejected) | Walk-forward rejected | Robustness rejected | Sample/margin rejected |
|---:|---:|---:|---:|---:|---:|
| 173 | **0** | 13 | 14 | 16 | 130 |

**Zero results cleared all four gates.** Every one of the 13 candidates
that reached TEST was rejected there — no accepted result, no borderline
pass, unlike Steps 9 and 10.

## 3. The 13 TEST-reaching candidates

```
GBP_USD/1h CALL h=5  return_3 AND dist_to_low_atr_20                    n=94  wr=0.4574 CI_low=0.3604
USD_JPY/1h CALL h=1  return_3 AND macd_cross_signal                     n=302 wr=0.5497 CI_low=0.4933
USD_JPY/1h CALL h=1  return_3 AND inside_bar_breakout_signal            n=303 wr=0.5479 CI_low=0.4916
USD_JPY/1h CALL h=1  return_3 AND engulfing_signal                      n=297 wr=0.5556 CI_low=0.4987
USD_JPY/1h CALL h=2  same_color_streak AND engulfing_signal             n=445 wr=0.5506 CI_low=0.5041
USD_JPY/1h CALL h=2  same_color_streak AND inside_bar_breakout_signal   n=445 wr=0.5506 CI_low=0.5041
USD_JPY/1h CALL h=2  macd_cross_signal AND same_color_streak            n=440 wr=0.5523 CI_low=0.5056
USD_JPY/1h CALL h=3  return_3 AND engulfing_signal                      n=297 wr=0.5522 CI_low=0.4953
USD_JPY/1h CALL h=3  same_color_streak AND engulfing_signal             n=444 wr=0.5518 CI_low=0.5053
USD_JPY/1h CALL h=3  same_color_streak AND inside_bar_breakout_signal   n=444 wr=0.5518 CI_low=0.5053
USD_JPY/1h CALL h=5  macd_cross_signal AND same_color_streak            n=439 wr=0.5672 CI_low=0.5205
USD_JPY/1h CALL h=5  same_color_streak AND engulfing_signal             n=444 wr=0.5676 CI_low=0.5211
USD_JPY/1h CALL h=5  same_color_streak AND inside_bar_breakout_signal   n=444 wr=0.5676 CI_low=0.5211
```

Break-even at 85% payout is **0.5405**. Every single row's CI-low sits
below that line — **all 13 are correctly REJECTED**, with no ambiguity
about the mechanical verdict.

**Worth reporting honestly, without overstating it**: the h=5 USD_JPY/1h
rows (CI-low 0.5205-0.5211) sit closer to break-even than the clean,
unambiguous rejections seen elsewhere in this project (e.g. Step 10's
discovery-track TEST failures ranged CI-low 0.27-0.48). This is not a
near-miss finding — a gap of roughly 2 percentage points on CI-low is
still a real, decisive rejection, not a rounding-error margin like the
two already-ACCEPTED fragile candidates' ~0.0002-0.005 pp margins. It is
noted here only because the same honesty-first standard applied to every
prior report requires flagging where a cluster of rejections lands,
rather than silently reporting the 0/173 topline and stopping.

**Provenance caveat — these 13 rows are NOT 13 independent pieces of
evidence.** Every single one is on USD_JPY/1h except the one GBP_USD/1h
row. Within USD_JPY/1h, the 12 rows cluster on only 5 distinct features
(`return_3`, `same_color_streak`, `engulfing_signal`,
`inside_bar_breakout_signal`, `macd_cross_signal`) recombined across
horizons — `engulfing_signal` and `inside_bar_breakout_signal` in
particular produce near-identical numbers when paired with
`same_color_streak` (both n=445, wr=0.5506 at h=2; both n=444, wr=0.5518
at h=3; both n=444, wr=0.5676 at h=5), which is expected since both
features are themselves derived from very similar candle-shape logic and
are highly correlated on this dataset. This is the same
correlated-multiple-comparisons pattern Step 10's report flagged for
USD_JPY/1h (11 of that step's 13 TEST arrivals landed on the same
dataset) — USD_JPY/1h is, across this entire project, disproportionately
likely to produce TEST-reaching candidates relative to other datasets,
which is itself a signal that per-search FDR control (valid *within*
each of the 64 independent searches here) does not fully protect against
one dataset generating many correlated near-threshold trials across
searches. None of this changes the verdict — all 13 remain REJECTED —
but it means these clustered near-misses should not be read as 5-12
separate corroborating signals.

## 4. Comparison to prior 2-way/3-way results

Steps 7, 8d, and 10 already searched exhaustively over the original
10-feature `CORE_DISCOVERY_FEATURES` set (2-way and 3-way). This step's
21 genuinely different features found a **qualitatively similar outcome**:
0 ACCEPTED, a similar-shaped funnel (most trials rejected at
sample-size/margin, a much smaller tail reaching TEST, all TEST arrivals
failing), and the same dataset-concentration pattern (USD_JPY/1h
producing most of the TEST-reaching candidates). The specific features
involved are different — candle-shape/momentum features
(`same_color_streak`, `engulfing_signal`, `return_3`) here, versus
oscillator/volatility features (`bb_pct_b_20`, `adx_14`, `rsi_14`) in
Step 10 — but expanding the feature universe did not surface a
structurally different result. This is additional evidence (not proof)
that the absence of a robust edge in this data is not an artifact of
which 10 features were originally chosen for `CORE_DISCOVERY_FEATURES`.

## 5. Where this leaves the project's overall search

Adding this step's 173 evaluations to the project's running total (well
over 1,100 evaluations now, across H1-H20, H9, H21, 2-way discovery,
3-way discovery, 3 ML model families, a NO_TRADE filter, and now this
21-feature expanded search), the count of results that ever mechanically
cleared every pre-registered gate **remains exactly two**: Step 9's H9
session-bias condition and Step 10's random-forest model — both already
flagged fragile and now under forward/paper test against genuinely new
data (see below). This step adds a third, disciplined line of evidence
(alongside 2-way and 3-way search over the original feature set) that
this real Forex data, at these timeframes, does not contain a second,
independent edge waiting to be found simply by trying more features —
without foreclosing the two live paths already agreed with the user:
forward-testing the two existing candidates, or a future run using
genuinely new (not yet ingested) data, sources, or targets.

No result from this step is treated as a candidate, forwarded to
forward-test, or used to justify any live signal. Zero forced
conclusions; the topline is reported as it mechanically came out.
