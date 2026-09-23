# ML_1M5M experiment: 1-minute analysis → 5-minute expiry (independent, experimental)

**Status: complete. 0 candidates accepted under either pre-registered
execution scenario; none reached the TEST gate at all.** This is a
genuinely independent, from-scratch experiment requested by the user —
NOT a retimeframing of `ML10_FWD` (USD/JPY 1h → 3h expiry). It does not
touch `H9_FWD`, `ML10_FWD`, or `FORWARD_TEST_CANDIDATES`; nothing here
was wired into the live forward-test poller.

> **Correction (post-publication)**: a follow-up investigation into
> candidate #11 (`gradient_boosting`) found that `evaluate_candidacy`'s
> gates 1/2/4 silently ignored this experiment's intended window and ran
> against the FULL EUR_USD/1m candle history instead — gate 1's real
> TRAIN was 2026-07-21→07-31 (not 07-21→07-29 as stated below), and its
> real TEST would have silently aliased to the already-used 09-15→09-18
> block had any candidate reached gate 4 (none did). Walk-forward
> (Section 4/6 below) was unaffected — it already used explicit
> start/end per fold — except Fold 3 overflowed ~2h into the intended
> TEST window due to a calendar-proportion estimate rather than the true
> row-based boundary. Both bugs are now fixed
> (`backtest.engine.compute_split_windows`,
> `backtest.walkforward.compute_walk_forward_folds`,
> `evaluate_candidacy`'s new `start`/`end`/`allow_test` params — see
> `scripts/rerun_ml1m5m_candidate11_baseline.py`). The corrected baseline
> for candidate #11 reaches the **same qualitative verdict** (rejected at
> walk-forward, worst fold below break-even) with a smaller, correctly-
> scoped TRAIN (n=5,612, not 6,980) — reported in full in the chat
> transcript of that investigation. The 0-accepted headline for the other
> 10 candidates in this report is unaffected in substance (none reached
> gate 3 at all, so the gate-1/2 window bug could only have made their
> TRAIN numbers ~2x too large, not changed a rejection into an
> acceptance) but their exact TRAIN sample sizes/dates above are
> likewise stated for the uncorrected window and should not be quoted
> as precise.

## 1. What was run

- **Data**: a genuinely new EUR/USD 1-minute block, fetched specifically
  for this experiment: **2026-07-21 → 2026-08-04** (20,000 candles), with
  more than 6 weeks of separation from the EUR/USD 1m block already spent
  on Step 10/11 discovery (2026-09-15 → 2026-09-18) — confirmed via a
  clean ingest (20,000/20,000 inserted, 0 duplicates against existing
  data).
- **Target**: `call_wins_5`/`put_wins_5` — the standard binary "does a
  CALL/PUT opened now win 5 candles (minutes) from now" label, same
  binomial framework as every other candidate in this project.
- **Features (30)**: momentum/multi-lag returns, volatility, oscillators
  (RSI/ADX/CCI/RCI/Bollinger %b/MACD-cross), candle shape, structure, and
  — new for this experiment, per the technical design's recommendation —
  time-of-day/session features (`hour_utc`, `day_of_week`,
  `trading_session_code`), since intraday liquidity effects were expected
  to matter more at 1-minute granularity than they did in prior 1h/15m
  work.
- **Split (60/20/20, standard convention)**:
  - TRAIN: 2026-07-21 01:15 → 2026-07-29 10:08 (n=11,374)
  - VALIDATION: 2026-07-29 10:09 → 2026-08-01 02:57 (n=3,791)
  - TEST: 2026-08-01 02:58 → 2026-08-04 00:00 (n=3,791)
- **Discovery**: 2-way, own independent Benjamini-Hochberg correction
  (own `run_id` prefix `ml1m5m-...`, never pooled with any prior run),
  n_bins=4, min_sample_size=50. Top 5 FDR-significant, winning-direction
  conditions per target column escalated to candidacy.
- **ML**: logistic regression, random forest, gradient boosting fit for
  both targets; only VALIDATION-promising fits (CI-low > break-even,
  n≥30) escalated.
- **Two pre-registered execution scenarios, evaluated for every single
  candidate, decided before any result was seen:**
  - **Scenario "delay0"** (the user's exact request): signal at candle
    close → entry at the very next candle's open (1 min later) → exit 5
    candles after entry (6 minutes total, signal to exit). Zero
    look-ahead, near-instant execution assumed.
  - **Scenario "delay1"** (the project's standing convention, same as
    H9_FWD/ML10_FWD/every discovery run so far): one extra candle of
    execution latency. Entry 2 min after signal, exit 7 min total.

## 2. Headline result

**11 candidates (5 discovered CALL conditions + 5 discovered PUT
conditions + 1 escalated ML model), each evaluated under both scenarios
= 22 scenario-evaluations. Zero accepted. Zero reached TEST.**

| Scenario | sample_size_and_margin | walk_forward | robustness | TEST-reaching | ACCEPTED |
|---|---:|---:|---:|---:|---:|
| delay0 (user's exact scheme) | 8 | 2 | 1 | 0 | 0 |
| delay1 (project convention) | 11 | 0 | 0 | 0 | 0 |

This is a **cleaner, more decisive rejection** than every prior discovery
step in this project (Step 7, 8d, 10, 11) — those runs typically had a
double-digit number of candidates reach the TEST gate, several landing
close to break-even. Here, nothing got past gate 1 or 2 under either
scenario. There is no ambiguity to report: this specific data window,
feature set, and target found nothing worth a TEST touch.

## 3. Discovery found thousands of "significant" conditions — that number was misleading on its own

Discovery flagged **1,697 FDR-significant winning-direction conditions**
for `call_wins_5` and **1,846** for `put_wins_5` out of 5,735 trials each
— a dramatically higher raw hit rate than any prior discovery run in this
project (Step 11's 21-feature search found roughly a dozen per
target_col on 1h/15m/5m data). This is exactly the effect flagged as a
risk in the technical design before this experiment ran: **5-minute
return autocorrelation at 1-minute granularity is, per the literature
review already in `PREDICTABILITY_AUDIT.md`, largely indistinguishable
from bid-ask-bounce microstructure noise — statistically "significant"
at the naive label level in large volume, but not something a real
execution model, even a fast one, can capture net of even a single
minute of delay.** The candidacy funnel's own numbers confirm this
directly (next section) — the raw discovery count was never a reliable
signal of anything tradeable, only of how much naive-label noise exists
at this horizon.

## 4. The delay=0 → delay=1 comparison is itself the most useful finding

This is the first place in the project where the same, identical
candidate has been evaluated under two execution-timing assumptions
side by side, and the effect is large and consistent — not subtle:

| Candidate | TRAIN win rate, delay0 | TRAIN win rate, delay1 | Drop |
|---|---:|---:|---:|
| `return_1∈(-0.189,-0.0035] AND return_2∈(-0.0044,0]` CALL (n=852) | 58.33% | 48.12% | **−10.2pp** |
| `return_1∈(-0.189,-0.0035] AND return_3∈(-0.00528,0]` CALL (n=1006) | 53.98% | 46.62% | **−7.4pp** |
| `return_1∈(-0.0035,0.000875] AND return_3∈(-0.00528,0]` CALL (n=1431) | 63.73% | 45.84% | **−17.9pp** |
| `return_1∈(-0.189,-0.0035] AND return_5∈(-0.00701,0]` CALL (n=921) | 52.55% | 47.67% | −4.9pp |
| gradient_boosting model (n=6980) | 57.66% | 55.79% | −1.9pp |

The strongest raw candidate (63.7% TRAIN win rate under delay0, reaching
the walk-forward gate) loses **almost 18 percentage points** — falling
below break-even entirely — from adding a single extra minute of
execution delay. This confirms, with an actual measurement rather than a
prior expectation, the concern raised in the technical design: at a
5-minute trade duration, one extra minute of delay is a much larger
fraction of the total trade window than it is at 1h/15m/3h horizons, and
it visibly destroys most of whatever the naive label appeared to show.
The gradient-boosting model (a smoother, less bin-boundary-sensitive
signal) was the most delay-resistant candidate here, but even it lost
1.9pp — enough on its own to drop its margin under delay1 below the
required 3pp-over-break-even bar.

**Caveat on "delay0" itself**: even under the more favorable delay0
scenario, only 3 of 11 candidates got past gate 1 at all, and all 3
still failed at gate 2 (robustness) or gate 3 (walk-forward) — so this is
not "delay0 passes, delay1 fails," it is "neither scenario found
anything real"; the comparison above documents *how much* of the raw
signal delay accounts for, not that delay0 found a working strategy.

## 5. Conclusion

No forced result: **this specific experiment — EUR/USD, 1-minute candles,
5-minute binary-option expiry, on this 14-day window — found no
candidate, discovered condition or fitted model, that survives even the
sample-size-and-margin gate under the project's standard execution
delay, and only 3 of 11 survive it under the most optimistic
(zero-latency) assumption, none of which survive the next gate either.**
This is consistent with, and adds a directly-measured confirmation of,
the project's standing concern that very short binary-option horizons
sit in territory where return predictability and bid-ask/microstructure
noise are difficult to tell apart, and where execution delay — even a
single minute — consumes a disproportionate share of the trade's total
duration.

This result does not prove no edge could ever exist at 1m→5m (a
different data window, a longer discovery run, or additional feature
families could still be tried, each as its own bounded, pre-registered
step) — but it does not support pursuing this specific configuration
further without a new reason to expect a different outcome.

## 6. What was NOT done (explicitly out of scope for this step)

- No model was pickled/frozen — nothing cleared candidacy, so there is
  nothing to forward-test.
- `H9_FWD`, `ML10_FWD`, and `FORWARD_TEST_CANDIDATES` are untouched.
- The hourly forward-test Routine continues polling only the two
  existing candidates; nothing from this experiment was added to it.
