# Predictability audit + deep research (Step 8, audit+research addendum)

**Status: complete.** This document is the deliverable of a second research
phase the user explicitly requested after Step 7 (see `STRATEGIES.md`'s
"Pivot" section) found conclusion **B — promising but insufficient**: no
condition survived realistic-execution re-evaluation. The user's stated
goal: don't force Step 7's negative result into a positive one, but don't
stop investigating just because the first 28,800-condition run found
nothing either. This document audits exactly where Step 7's edge
disappeared, runs two literature-grounded new angles under the same
FDR/candidacy discipline (never relaxed), and closes with a hypothesis
menu for anything not yet tried. Full design:
`/root/.claude/plans/sequential-sparking-candle.md`'s "Addendum: Step 8".

**The "BUSCAR SEÑAL" UI is still not built, on purpose** — this document's
own final answer (Section 14) explains why.

**Important correction, made after this document was written — read
[BINARY_OPTIONS_REFRAME_AUDIT.md](BINARY_OPTIONS_REFRAME_AUDIT.md)
alongside this one.** Section 3's "slippage is the dominant destroyer"
finding was found to be an artifact of applying a spot-Forex-broker
bid/ask-crossing cost model (`slippage_pct` in `backtest/simulator.py`)
to a target that doesn't have that cost structure — a binary option's
payoff has no fill-price/spread concept. That specific attribution is
retracted; the delay-only findings and everything else in this document
(the audit numbers, the CALL/PUT symmetry, the TEST-gate case study, the
literature review, the OTC investigation) stand. **The correction has
since been implemented and re-run** (`research/candidacy.py` now
defaults to a delay-only scenario): far more conditions now clear the
early gates, and five reached TEST with a genuinely validated
walk-forward (vs. one before) — including the strongest result this
project has produced (EUR_USD/15m, session-conditioned, rejected at TEST
only on CI margin, not reversal). Still no accepted candidate. See the
linked document's "Corrected re-run results" section for the full
numbers.

---

## 1. Audit of Step 7: what exactly was tested

**Data-hygiene note, reported first because it changes every number
below:** the `condition_trials` table currently holds 34,560 rows, but
8,000 of those are a duplicate `dryrun-test:GBP_USD:5m:*` smoke test run
before the real Step 7 run. **Every number in this section uses the
filtered, official 28,800**, `run_id LIKE 'step7-20260918T161113:%'`.

| Question | Answer |
|---|---|
| What do the 28,800 conditions represent? | Every 2-way combination of 2 features (out of a 10-feature core subset) × 4 quantile bins each, tried independently for each of 8 target columns (`call_wins_h`/`put_wins_h` for h∈{1,2,3,5}) across 5 datasets — `C(10,2) × 4×4 × 8 × 5 = 45 × 16 × 8 × 5 = 28,800`. |
| Feature families tested | 10 of ~40 available `v4` features (a deliberately restricted, documented subset — see `scripts/run_research_pipeline.py`): momentum (`return_1`, `return_5`), trend (`rsi_14`, `adx_14`), mean-reversion (`cci_20`, `rci_9`, `bb_pct_b_20`), volatility (`atr_expansion_ratio`, `move_size_atr`), position-in-range (`pct_position_in_range_20`). **Not tested in Step 7**: session/time features (`hour_utc`, `trading_session_code`, `day_of_week`) — see Section 4/8d below, this gap is exactly what the session-conditioned re-run fills. |
| Combinations | 2-way only (`combo_sizes=(2,)`); 3-way was never run — a runtime-budget choice, not a methodological one. |
| Horizons | 1, 2, 3, 5 candles (both CALL and PUT labeled separately). |
| Assets | EUR_USD, GBP_USD, USD_JPY. |
| Timeframes | EUR_USD: 1m, 5m, 15m. GBP_USD, USD_JPY: 5m. (1h was skipped — already extensively covered by H1-H20's own methodology.) |
| Observations per asset | ~5,000 raw candles each; 4,966-4,967 dataset rows after dropping incomplete-feature warm-up rows; TRAIN≈2,980 / VALIDATION≈993 / TEST≈993 (60/20/20). |
| Effective sample size of the best conditions | FDR-significant trials: n from 50 (the run's own floor) to 656, mean 237. |
| Best RAW result (naive label, zero execution model) | Best single trial: USD_JPY/5m `put_wins_3`, win rate 78.9% (n=213). Several others in the mid-70s-to-high-70s range — see the full breakdown in `condition_trials`. |
| Best result AFTER DELAY (isolated) | **New computation this step** — see Section 3's table. Delay alone is not what kills most of these; see below. |
| Best result AFTER SLIPPAGE (isolated) | **New computation this step** — see Section 3. Slippage alone is nearly as destructive as the full realistic scenario. |
| Best result OUT OF SAMPLE (TEST) | **None from Step 7 itself** — 0/120 candidates ever reached TEST (see the funnel below). Step 8d's session-conditioned run DID reach TEST once — see Section 2's C-vs-overfitting case study. |
| Best result WALK-FORWARD | **None from Step 7 itself** — 0/120 reached walk-forward either. Step 8d reached it twice (Section 2). |
| Best result AFTER PAYOUT | Every number above already reported alongside `margin_over_break_even`/`expectancy` at payout=0.85 (break-even 54.05%) — see Section 3's table for the ladder view. |

**Candidacy funnel (already reported in `STRATEGIES.md`, restated here for
completeness):** 120 top FDR-significant candidates (up to 3 per
asset/timeframe/horizon/direction, by p-value) were re-evaluated through
`research/candidacy.py`'s unchanged 4-gate bar: **119/120 rejected at gate 1**
(TRAIN sample size + margin, median simulated win rate 20.7%), **1/120
reached and failed gate 2** ("fragile" under parameter-perturbation
robustness), **0/120 reached walk-forward or TEST.** The "walk-forward"
and "out-of-sample" cells above are genuinely empty for Step 7 — not
approximated, not estimated.

---

## 2. Three separate problems, kept separate

The user asked to distinguish: **(A)** no real predictability, **(B)**
real but too small vs. payout, **(C)** real but destroyed by
latency/execution — and never blend them.

A new tool, `research/decomposition.py` (+ `scripts/run_execution_decomposition.py`),
mechanically classifies each candidate by running it through a LADDER of
execution scenarios (raw → optimistic → delay-only at 1/2/3/5 candles →
slippage-only → realistic → pessimistic) and comparing margin-over-
break-even at each rung. Run against the 113 Step 7 conditions that were
FDR-significant, n≥100, **in their genuinely winning direction** (a
condition significant for a LOW win rate on `put_wins_h`, say, is the
same underlying pattern as a high win rate on the matching `call_wins_h`
row — see Section 6 — so only the winning-direction row is a meaningful,
non-duplicated decomposition candidate):

| Classification | Count | % |
|---|---|---|
| **C — execution destroys it** | 112 | 99.1% |
| **survives the ladder** (not a failure — still needs candidacy's full bar) | 1 | 0.9% |
| **A — no real predictability** | 0 | 0% |
| **B — real but too small for payout** | 0 | 0% |

**This is an unambiguous answer: among Step 7's significant conditions,
the dominant failure mode is C, not A or B.** Every one of them showed a
genuine, non-trivial directional skew at the raw label stage (none
classified A) — the skew is real in the naive sense — but 112/113 don't
survive contact with the real execution simulator. Zero classified B
means: nothing here is "a real, execution-surviving edge that's just too
small" — either it survives (1 case) or it's destroyed by execution (112
cases), no middle ground at this sample.

**The one "survives_ladder" case is the same condition that reached
candidacy gate 2 in Step 7** (USD_JPY/5m, CALL h=5,
`return_5∈(-0.762,-0.0309] AND pct_position_in_range_20∈(0.476,0.749]`,
raw win rate 75.2%) — it clears the execution ladder (realistic win rate
65.8%, margin +11.7pp) but was rejected at candidacy's **robustness**
gate as "fragile": small perturbations of its bin edges don't hold up.
So even the one condition that survives execution fails a different,
independent check — not a contradiction, a second, distinct failure mode
this document didn't need to re-classify.

### A case study in why the TEST gate exists (from Step 8d, Section 4)

One condition from the new session-conditioned re-run — USD_JPY/5m PUT
h=5, `bb_pct_b_20∈(0.232,0.501] AND day_of_week∈(2,3]` (Thursday) — is
the **first candidate in this entire investigation (H1-H20, Step 7, and
Step 8d combined) to reach the out-of-sample TEST split.** Its TRAIN win
rate was 63.5% (n=156, margin +9.4pp over break-even) and it **passed**
gates 1, 2, and 3:

- Gate 1 (sample size/margin): passed, n=156, margin +9.4pp.
- Gate 2 (robustness): passed, "consistent_direction".
- Gate 3 (walk-forward): technically passed — `fraction_folds_with_edge
  = 1/1 = 100%` — but **only 1 of 5 walk-forward folds had ≥20 matched
  samples at all** (the other 4 had n=0; a single weekday-conditioned bin
  is sparse). This is a real gap in the candidacy bar worth flagging: it
  requires a *fraction* of sufficiently-sampled folds to show an edge,
  but has no floor on *how many* folds must be sufficiently sampled in
  the first place — a condition that only ever matches in one walk-
  forward window can trivially clear "100% of 1 fold." Recommended
  follow-up (not yet built): add a minimum-folds-sampled requirement to
  `CandidacyThresholds`.
- **Gate 4 (TEST, touched once): REJECTED.** Test win rate **35.5%**
  (n=93, CI [26.5%, 45.6%]) — a complete reversal from the 63.5% TRAIN
  read.

This is exactly the failure mode BACKTESTING.md's TEST-once discipline
exists to catch, and it caught it on the very first candidate to reach
that gate. It is strong, concrete evidence that the candidacy bar's
earlier gates (sample size, robustness, and especially the sparse-fold
walk-forward gate above) are not sufficient on their own — TEST remains
load-bearing, not a formality.

---

## 3. Delay sensitivity table

**Data constraint, resolved with the user before this run:** this
project has no sub-minute Forex data (Twelve Data's REST API tops out at
1-minute candles), so the user chose a **1-minute-candle proxy** for the
requested 0/1/2/3/5/10/15/30-second sweep: 0/1/2/3/5 whole candles ≈
0/60/120/180/300 seconds. This is a coarse approximation, not an exact
sub-minute measurement — stated once here, not repeated per number below.

Aggregated across the same 113 winning-direction Step 7 candidates
(fraction still clearing `margin_over_break_even >= 0`, at payout=0.85):

| Rung | Meaning | Still clearing margin≥0 |
|---|---|---|
| `raw` | naive label, zero execution model | 113/113 (100.0%) |
| `optimistic` | real simulator, zero friction | 59/113 (52.2%) |
| `delay_only_1` | +1 candle delay only, no slippage | 40/113 (35.4%) |
| `delay_only_2` | +2 candles delay only | 37/113 (32.7%) |
| `delay_only_3` | +3 candles delay only | 53/113 (46.9%) |
| `delay_only_5` | +5 candles delay only | 27/113 (23.9%) |
| `slippage_only` | 0.01% slippage only, no delay | **1/113 (0.9%)** |
| `realistic` | delay(1) + slippage + 2% drop bundled | 1/113 (0.9%) |
| `pessimistic` | delay(2) + more slippage/drop | 0/113 (0.0%) |

**Two distinct, separable findings here, not one:**

1. **The edge does not disappear "immediately" in a clean, monotonic
   way with delay alone.** `delay_only` rungs fluctuate (32.7% → 46.9% →
   23.9%) rather than decaying smoothly — delay sensitivity is
   condition/horizon-specific, not a single universal decay curve. Some
   individual conditions collapse the instant ANY delay (even 1 candle)
   is introduced (example: EUR_USD/1m CALL h=1,
   `return_1∈(0.00433,0.0643] AND move_size_atr∈(0.705,5.028]`: raw
   70.2% → optimistic 65.5% → delay_only_1 48.6%, already below
   break-even at the very first delay increment). Others tolerate delay
   much better (the `survives_ladder` case above stays at 75.2% through
   `delay_only_1`, only eroding from `delay_only_2` onward).
2. **Slippage, not delay, is the dominant destroyer.** `slippage_only`
   (zero delay, just the 0.01% round-trip cost) collapses survival to
   0.9% — nearly as low as the fully bundled `realistic` scenario (also
   0.9%) — while every `delay_only` rung still leaves 24-47% of
   candidates clearing margin even at 5 candles of pure delay. **This
   means the core problem for a manual-signal system is not primarily
   "can a person react fast enough" — even a person reacting instantly
   would still lose almost everything here to the round-trip cost
   itself.** The predicted moves are, for the overwhelming majority of
   these conditions, smaller than the transaction cost.
3. **Halfway between these**, the `raw → optimistic` drop (100% → 52.2%)
   happens with ZERO friction and ZERO delay — purely from the real
   simulator's own entry-next-open/exit-at-close mechanics differing
   from the naive dataset label's same-candle-window close-to-close
   comparison. Roughly half of what looked significant at the naive
   label level was never really the same trade the simulator executes,
   independent of any cost at all.

---

## 4. Targets: not just "next candle"

Two new target functions, `research/targets.py` (additive, `call_wins_h`/
`put_wins_h` untouched):

- **`triple_barrier_labels`**: ATR-scaled upper/lower barriers, labels
  which is touched first within a fixed candle horizon (López de Prado's
  triple-barrier method — see Section 12's literature review, verdict
  STRONG). Covers the user's "movimiento mínimo favorable antes de
  adverso" and "umbral antes que otro" requests — they're the same
  target shape.
- **`mfe_mae_labels`**: Maximum Favorable/Adverse Excursion within a
  horizon window, both directions — a richer, continuous per-row signal
  used by the no-trade-filter analysis (Section 7).
- **`future_return_h`** (simple continuous return) is **not built** —
  flagged as future work in Section 13; it needs different statistical
  machinery (regression/t-test, not this pipeline's binomial/Wilson-CI
  core) and was out of this step's bounded scope.

**Result of re-running discovery + candidacy with the triple-barrier
target** (`upper_mult=1.0`, `lower_mult=1.0`, `max_horizon=10` candles —
fixed before running, never tuned after seeing a result), same 5
datasets, same 10 core features: 7,200 condition trials, 1,128 (15.7%)
FDR-significant, 28 candidacy evaluations (winning direction, top 3 per
target), **0 accepted.** Rejection pattern differs qualitatively from
Step 7's naive-target run, though: candidacy-stage TRAIN win rates
clustered much closer to 50% (0.31-0.58, one reaching gate 2 at 58.2%)
rather than the naive target's severe 0.15-0.28 reversal pattern. **This
is partial, not full, support for the literature's hypothesis**: a
volatility-scaled, achievable-outcome target does appear to reduce the
severity of the naive target's execution-mismatch collapse, but it
still did not produce anything clearing candidacy's bar in this bounded
run (10 features, 2-way only, one barrier configuration).

---

## 5. Context, not just indicators

Every category the user asked to investigate already exists in `v4`
(see `FEATURES.md`): regime (`research/regimes.py`'s trend×volatility
classification), volatility (`atr_expansion_ratio`, `rolling_std_return_20`),
structure (`donchian_high/low_20`, `structure_bias`), momentum
(`return_{1..30}`, `roc_10`), mean-reversion (`rsi_14`, `cci_20`, `rci_9`,
`bb_pct_b_20`), expansion/compression (`atr_expansion_ratio`), return/
candle sequences (`same_color_streak`, `cumulative_return_10`,
`return_acceleration`), position-in-range (`pct_position_in_range_20`),
acceleration (`return_acceleration`), and post-extreme-move behavior
(`move_size_atr` + the new `mfe_mae_labels`). **No new indicators were
added per the user's explicit instruction not to add features just to
add them** — the one genuine, literature-motivated gap found (session/
time-of-day features excluded from Step 7's search) was closed in
Section 4's session-conditioned run instead of inventing anything new.

---

## 6. CALL/PUT asymmetry

Computed directly from existing Step 7 data (no new run needed): joining
every FDR-significant trial to its mirror-image trial on the opposite
target column (same asset/timeframe/condition/horizon), across **1,451
matched pairs, `call_win_rate + put_win_rate = 1.000` in literally every
single case** (min = median = max = 1.0).

**This is an important, precise finding**: it proves the "both CALL and
PUT show a low win rate" pattern seen after execution (Section 2/3) is
**not** a property of the naive labels — at the naive-label level, CALL
and PUT are perfect directional complements, as they must be by
construction (no meaningful void/tie leakage). The asymmetry that
matters only appears once the realistic execution simulator is applied —
i.e., **the interesting CALL/PUT split is an execution-stage
phenomenon, not a labeling-stage one.**

---

## 7. No-trade filter

Two complementary reads, `scripts/run_no_trade_filter_analysis.py`
(reuses `research/baseline.conditional_by_category` — no new core code):

1. **Void/tie-mass read** (a regime where both `call_wins_h` and
   `put_wins_h` are simultaneously below 50% with n≥100): **found
   nothing**, at any asset/timeframe/horizon. Consistent with Section 6 —
   the naive labels have essentially no void/tie mass anywhere, in any
   regime.
2. **Direction-independent achievable-excursion read** (mean 10-candle
   MFE/|MAE| per regime, from `mfe_mae_labels`): **`*_contracting_vol`
   regimes consistently show the smallest achievable moves of any regime,
   at every asset/timeframe tested** — e.g. EUR_USD/5m
   `strong_trend_contracting_vol`: mean MFE 0.0109%, mean |MAE| 0.0159%,
   vs. `strong_trend_expanding_vol`'s 0.0427%/0.0339% in the same
   dataset. **These contracting-volatility numbers are the same order of
   magnitude as the 0.01% round-trip slippage cost that Section 3 showed
   destroys nearly everything** — meaning in a volatility-contraction
   regime, the achievable move is barely larger than the transaction
   cost itself, independent of which direction you'd trade. This is a
   genuine, quantitatively-grounded, literature-consistent (regime-
   conditional volatility literature, Section 12) candidate for a
   no-trade filter, even though it isn't a directional strategy by
   itself.

---

## 8. Conditional edge, not single-indicator

`research/discovery.py`'s combinatorial search has never searched single
indicators in isolation — every trial in both Step 7 and Step 8d is
already a 2-way (feature, bin) × (feature, bin) combination, exactly the
"state of the market + context, not just one indicator" framing the user
asked for. No new machinery was needed for this section; it's a
statement about what the existing pipeline already does.

---

## 9. Multiple-testing control — unchanged

Every run in this document (Step 7's original 28,800, the session-
conditioned run's 48,384, the triple-barrier run's 7,200 — **84,384 total
systematically-tested conditions across the whole pivot**, on top of
H1-H20's 20 hand-designed hypotheses before it) got its own `run_id`,
its own `ConditionTrial` log, and its own independent Benjamini-Hochberg
correction — **never pooled across runs for significance purposes.** No
FDR threshold, candidacy margin, or robustness bar was relaxed anywhere
in this step. The one candidate that reached TEST (Section 2) was
rejected there, and is reported as rejected — not reframed, not
excluded, not re-tested with looser parameters.

---

## 10. Pocket Option OTC data — what we know, what we don't

**Qué sabemos:**
- Pocket Option's own OTC knowledge-base materials describe the feed as
  broker-generated: synthetic prices produced by their own pricing
  models, explicitly used to keep instruments tradeable during periods
  (e.g. weekends) when the real underlying market is closed.
- No official Pocket Option API exists. Every "Pocket Option API"
  findable on GitHub (multiple independent repositories) explicitly
  self-describes as unofficial and not affiliated with Pocket Option —
  reverse-engineered from the broker's own web client.
- No commercial, licensed market-data vendor (Twelve Data, OANDA,
  Bloomberg, Refinitiv/LSEG, or any other checked) lists or distributes
  a Pocket Option OTC feed.

**Qué NO sabemos (con confianza de fuente primaria):**
- The exact algorithm/methodology Pocket Option uses to generate OTC
  prices — their own materials don't disclose it.
- Whether any private licensing arrangement exists between Pocket
  Option and a data vendor that simply isn't publicly listed.
- This finding is **reasonably confident from multiple consistent
  secondary sources, but not confirmed by one authoritative primary
  document** (no Pocket Option ToS clause or regulator filing was
  found stating this definitively). If this ever matters for a
  compliance/legal decision, it should be verified against Pocket
  Option's actual legal documentation, not this research.

**Qué datos necesitaríamos:** either (a) a licensing/data-sharing
agreement directly with Pocket Option for their OTC feed (no evidence
such a thing is offered to anyone), or (b) Pocket Option publishing an
official, documented API — neither exists today.

**Qué fuentes autorizadas existen:** **none found.** Stated plainly, per
the user's explicit instruction to say so clearly if this is the case.

**Diferencias que podrían romper la transferibilidad Forex → OTC** (even
if an authorized source appeared tomorrow): OTC is a broker-generated
synthetic series, not a real market with real liquidity/order flow — the
execution-cost/microstructure story that Section 2/3 shows dominates
Step 7's results (bid-ask-bounce-like slippage destroying most of the
edge) may not even apply the same way to a synthetic feed with different
(unknown) generation dynamics; a real-Forex-derived edge or its absence
is not evidence about OTC's own behavior one way or the other.

---

## 11. Forex → OTC conceptual transferability

**Not a claim that anything transfers — a discussion of what property,
if any, WOULD be transferable if a real source ever existed.** Every
feature in `features/indicators.py` is already expressed in
scale-invariant terms: percent returns, ATR-normalized distances,
position-in-range percentages — never a raw price level. If a future
condition survived this project's full candidacy bar (which nothing has
yet), the fact that it's expressed this way means it is *conceptually*
re-testable against another feed's candles without assumptions about
absolute price level baked in. This is a property of the feature
definitions, not evidence the strategy performs well on OTC — it would
still need to go through the exact same discovery→candidacy pipeline
against real, authorized OTC data (which, per Section 10, does not
currently exist) before meaning anything.

---

## 12. Literature review

Full findings gathered via a dedicated web-research pass, organized by
the user's 8 requested topics, each tagged by evidence tier per their
explicit instruction to never blend academic / practitioner-backtest /
marketed-strategy / anecdotal evidence.

| Topic | Verdict | Key finding |
|---|---|---|
| 1. Short-horizon (sub-hourly) return predictability | WEAK-MIXED | Real predictability exists at true tick/millisecond horizons (Aït-Sahalia et al. 2022/2025) but requires HFT infrastructure; at half-hour equity horizons the best evidence (Gao et al. 2018) is R²≈1.6-2.6% — real but tiny. Nothing found supports genuine, cost-net predictability at 1-15min retail candle horizons. |
| 2. Intraday momentum | WEAK-MIXED, strongest positive finding | Gao/Han/Li/Zhou 2018 (*J. Financial Economics*): first-half-hour-of-day predicts last-half-hour-of-day, replicated in FX (Elaut et al. 2018, RUB-USD) and crypto. Small, time-of-day-specific, no source claims it survives 1-candle delay + slippage at FX granularity. Directly motivated Section 4/8d's session-conditioned run. |
| 3. Short-term mean reversion | WEAK-MIXED | Largely mechanical bid-ask-bounce (Roll 1984); order-flow shows momentum 10s-1h while price shows reversal at similar scales — a genuinely nuanced, partially contradictory literature. Reversal timescales cluster ~30min to sub-minute depending on market speed. |
| 4. Volatility-regime-conditional predictability | WEAK-MIXED | Momentum/reversal dominance genuinely differs by volatility regime in equity cross-sectional literature; no direct FX-intraday evidence found. Plausible, unproven mechanism — matches Section 7's own quantitative (not yet causal) finding that contracting-vol regimes show the smallest achievable moves. |
| 5. Microstructure/execution-cost timescales | WEAK-MIXED, directionally consistent with our own result | Bid-ask spread has a documented intraday U-shape; a practitioner source (not independently verified) reports FX majors break even around ~3bp/trade. Qualitatively consistent with Section 3's finding that slippage alone destroys nearly everything. |
| 6. Serial dependence / autocorrelation | **DEBUNKED as an exploitable retail edge** | Literature is fairly consistent: measurable short-horizon autocorrelation is largely the SAME phenomenon as bid-ask-bounce transaction cost, not a separate signal net of it — directly supports Section 2/3's finding. |
| 7. Candlestick/price-pattern predictability | WEAK-MIXED leaning DEBUNKED | Mixed raw hit-rates across studies; Aronson's data-mining-bias-corrected methodology (the closest academic precedent to this project's own FDR approach) found no TA rules survive proper multiple-comparisons correction — validates this project's own methodology as sound, not overly conservative. |
| 8. Triple-barrier / meta-labeling | **STRONG, directly actionable** | Fixed-horizon labels misrepresent achievable trade outcomes under time-varying volatility; a volatility-scaled barrier label reflects what a real position would actually do. Directly implemented in Section 4/8d. |

---

## 13. Hypothesis menu

Each entry: fundamento, mecanismo, features, target, horizonte, razón de
supervivencia al payout, riesgo de overfitting. The first two were
**actually run** in this step (8d); the rest are candidates for a future,
separately-scoped run — none of them run indiscriminately, per the
user's explicit instruction.

**1. Session-conditioned momentum (RUN — Section 4/8d, 0 accepted).**
*Fundamento*: Gao et al. 2018, peer-reviewed, replicated in FX.
*Mecanismo*: liquidity-provider/informed-trading asymmetry concentrated
in specific intraday windows. *Features*: `hour_utc`,
`trading_session_code`, `day_of_week` + the 10 core features.
*Target*: `call_wins_h`/`put_wins_h`. *Horizonte*: 1-5 candles.
*Supervivencia al payout*: unproven at FX candle granularity per the
literature itself — this run tested it directly. *Riesgo de
overfitting*: HIGH for day-of-week-specific bins (small, sparse
subgroups — see Section 2's TEST-gate case study, drawn from exactly
this run).

**2. Triple-barrier / ATR-scaled target (RUN — Section 4, 0 accepted).**
*Fundamento*: triple-barrier/meta-labeling literature (STRONG verdict).
*Mecanismo*: fixed-horizon labels don't reflect achievable trade
outcomes under time-varying volatility. *Features*: the 10 core
features. *Target*: `triple_barrier_labels` (1.0×ATR barriers, 10-candle
horizon). *Horizonte*: up to 10 candles. *Supervivencia al payout*:
partial support found (less severe reversal than the naive target) but
nothing cleared candidacy. *Riesgo de overfitting*: MEDIUM — barrier
parameters were fixed before running, but only one configuration was
tried; a parameter sweep (still pre-registered, never post-hoc) is the
natural next step.

**3. Volatility-contraction no-trade filter (NOT RUN as a strategy —
Section 7 found the underlying signal, not yet turned into a gate).**
*Fundamento*: this step's own MFE/MAE-by-regime finding (Section 7) plus
volatility-regime literature (Section 12, topic 4). *Mecanismo*:
achievable move size shrinks to the same order of magnitude as
round-trip cost in `*_contracting_vol` regimes. *Features*: `regime`
(already built). *Target*: none directly — this is a FILTER, not a
directional target; would be tested as "does excluding
`*_contracting_vol` periods improve any OTHER candidate's realized
margin," not as a standalone win/loss search. *Horizonte*: n/a.
*Supervivencia al payout*: this is about avoiding low-edge periods, not
generating edge, so the question doesn't directly apply. *Riesgo de
overfitting*: LOW — it's a coarse, already-existing regime label, not a
new fitted threshold.

**4. 3-way interactions on the core feature set (NOT RUN).**
*Fundamento*: Step 7/8d only ever tested 2-way; the user's own spec
explicitly asked about "2-3 way." *Mecanismo*: none specific — a
combinatorics extension of the existing method, not a new hypothesis
about market behavior. *Features*: the same 10 core features.
*Target*: `call_wins_h`/`put_wins_h` or the triple-barrier target.
*Horizonte*: same as existing runs. *Supervivencia al payout*: unknown —
no prior evidence either way. *Riesgo de overfitting*: **HIGH** — the
combinatorics grow to `C(10,3) × 4³ = 120 × 64 = 7,680` conditions per
target per dataset, ~40× Step 7's per-target trial count; needs a
correspondingly larger FDR-corrected sample and more runtime budget
before it's worth doing.

**5. Barrier-parameter sweep on the triple-barrier target (NOT RUN).**
*Fundamento*: hypothesis 2 above showed a real but insufficient signal
at one barrier configuration. *Mecanismo*: same as hypothesis 2, testing
whether a different, still-pre-registered volatility multiplier/horizon
combination fares better. *Features*: the 10 core features.
*Target*: `triple_barrier_labels` at 2-3 additional, FIXED-IN-ADVANCE
`(upper_mult, lower_mult, max_horizon)` triples. *Horizonte*: varies by
configuration. *Supervivencia al payout*: unknown. *Riesgo de
overfitting*: MEDIUM-HIGH if not disciplined — every additional barrier
configuration tried is itself a multiple-comparisons dimension and must
be logged/FDR-corrected across configurations, not just within one.

**6. Continuous-return target with regression-based significance (NOT
RUN — explicitly deferred, Section 4).** *Fundamento*: the user's
"retorno futuro" request. *Mecanismo*: none specific — a different
statistical lens (magnitude, not just direction) on the same features.
*Features*: the 10 core features. *Target*: `future_return_h` (simple %
return). *Horizonte*: 1-5 candles. *Supervivencia al payout*: would need
a different, not-yet-built translation from a continuous prediction to a
binary-options trade decision. *Riesgo de overfitting*: unknown — needs
new statistical machinery (t-test/regression significance, not this
pipeline's binomial/Wilson-CI core) before it can even be assessed the
same way.

---

## 14. Final synthesis

**1. Qué aprendimos del Paso 7**: the naive, frictionless discovery
target finds real, FDR-significant, non-random directional structure
(2,902/28,800 trials, well above chance) — but at the naive-label level
only. Re-evaluated through the exact same realistic-execution engine
every hand-designed H1-H20 strategy was also judged by, essentially none
of it survives.

**2. Dónde exactamente desaparece el edge**: **primarily execution
(Cause C), not absence of predictability (A) and not "real but too
small" (B)** — 112/113 winning-direction candidates classified C, 0
classified A or B. Within C, **slippage/cost is the dominant mechanism,
not delay**: `slippage_only` alone collapses survival to 0.9%, while
pure delay (even 5 candles) still leaves 24-47% of candidates clearing
margin. A further ~48% of the naive "edge" evaporates from the
label-vs-simulator timing mismatch alone, before any friction at all.

**3. Qué hipótesis nuevas tienen fundamento**: session/time-of-day
conditioning (Gao et al. 2018, peer-reviewed) and the triple-barrier
target (broad practitioner/quant-ML consensus) — both run this step,
both found real but non-candidate-clearing structure. A
volatility-contraction no-trade filter has genuine quantitative support
from this step's own data plus regime-conditional literature.

**4. Qué datos adicionales necesitaríamos**: sub-minute/tick Forex data
would sharpen Section 3's delay analysis beyond the 1-minute-candle
proxy (a real, not yet pursued, option per the user's own choice this
session). For OTC specifically: no authorized source exists at all
(Section 10) — this is not a "need more data" gap, it's a "the data
doesn't exist to be gotten" finding.

**5. Qué hipótesis vale la pena probar** (a future, bounded run): the
barrier-parameter sweep (hypothesis 5) and formally testing the
volatility-contraction filter against an existing candidate (hypothesis
3) are the two with the clearest mechanism and the most direct support
from this step's own data.

**6. Qué hipótesis NO vale la pena probar** (yet): the 3-way interaction
sweep (hypothesis 4) — not because the idea is bad, but because its
combinatorics (~40× the per-target trial count) need a deliberate budget
decision first, not a default "why not."

**7. Qué necesitamos para probar específicamente OTC**: an authorized
data source, which per Section 10 does not currently exist. Until one
does, no OTC-specific validation is possible under this project's own
rules (never use Pocket Option's unofficial WebSocket, never present
real Forex data as OTC).

**8. ¿Existe actualmente alguna estrategia candidata suficientemente
robusta?** **No.** Nothing in Step 7 or Step 8d passed all four
candidacy gates including TEST. One condition (Section 2) reached TEST
for the first time in this project's history and was rejected there —
genuine progress in rigor, not in finding an edge. This remains an
honest, unforced negative result, exactly as the user asked for.
