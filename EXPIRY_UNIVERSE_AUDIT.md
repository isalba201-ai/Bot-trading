# Expiry universe audit — no expiry duration assumed a priori

**Status: written correction, no new runs, no data expansion.** The
user flagged that `BINARY_OPTIONS_REFRAME_AUDIT.md` Section 6 had
started prioritizing short (1-5 minute) expiries as the primary research
axis — a constraint the user never set. This document retracts that
recommendation, confirms exactly how the code maps timeframe+horizon to
real expiry duration, and lays out the full matrix of what has actually
been investigated so far, so the evidence — not an assumed target
duration — decides which combinations deserve further study.

---

## 1. Cómo funciona `h` exactamente — confirmado en código

`research/dataset.py::build_dataset()` computes, for each horizon `h` in
the `horizons` tuple passed in:

```python
future_close = close.shift(-h)          # h CANDLES ahead, not h minutes
call_wins = 1 if future_close > close else 0   # (ties -> NaN)
put_wins  = 1 if future_close < close else 0
```

`h` is always in **candles of the dataset's own timeframe** — never
minutes directly. Real elapsed expiry time is:

```
expiry_seconds = h * timeframe_to_seconds(timeframe)
```

(`otc_research.utils.timeframes.timeframe_to_seconds`, e.g. `"1m"→60`,
`"5m"→300`, `"15m"→900`). Every script that runs a discovered condition
through `research.candidacy.evaluate_candidacy` computes
`expiry_seconds` this exact way (see e.g.
`scripts/run_candidacy_corrected_rerun.py`). So, confirmed exactly as
the user stated:

| Timeframe | h=1 | h=2 | h=3 | h=5 |
|---|---|---|---|---|
| 1m | 1 min | 2 min | 3 min | 5 min |
| 5m | 5 min | 10 min | 15 min | 25 min |
| 15m | 15 min | 30 min | 45 min | 75 min |

**Two distinct quantities, never to be conflated** (already flagged in
`BINARY_OPTIONS_REFRAME_AUDIT.md` Section 6, restated here because it's
load-bearing for reading the matrix below):

- **Signal cadence** = the timeframe itself — how often a *fresh* signal
  can even be generated (every 1/5/15 minutes).
- **Expiry duration** = `h × timeframe_seconds` — how long the option
  runs once entered.

`1m/h=5`, `5m/h=1`, and `15m/h=?` can all land near "5 minutes" of
expiry, but they are different hypotheses: different signal cadence,
different feature-computation window, different amount of history
"baked into" each decision. The matrix in Section 6 keeps them as
separate rows for exactly this reason.

**One caveat on the triple-barrier target specifically**: its "expiry"
is NOT fixed the way the naive target's is. `research.targets.
triple_barrier_labels(..., max_horizon=10)` labels a row based on
whichever barrier (upper or lower) is touched FIRST, at a variable
candle offset ≤ `max_horizon` — the label's own effective "expiry" is a
random stopping time, not a fixed `h`. When Step 8d's triple-barrier
discovery run fed a discovered condition into `evaluate_candidacy`, it
used `expiry_seconds = max_horizon × timeframe_seconds` as a FIXED
expiry (the same fixed-time mechanic every other candidacy evaluation
uses) — meaning **candidacy re-evaluated these conditions against a
different target than the one discovery originally screened them for**
(a fixed h=10 CALL/PUT outcome, not "which barrier gets touched first").
This is a real, worth-flagging inconsistency for that specific set of
results, not a fabricated one — the triple-barrier section of the
matrix below is labeled accordingly.

---

## 2. No se impone ningún vencimiento a priori — corrección

`BINARY_OPTIONS_REFRAME_AUDIT.md` Section 6 previously recommended
"prioritizing 1-minute-timeframe / short-`h` combinations... as the
primary research axis" and treating 5m/15m as "secondary, exploratory
context." **That recommendation has been retracted in that document**
(cross-referenced to this one). It was an assumption this project
introduced on its own, not a constraint the user ever stated, and the
matrix in Section 6 below shows it would have actively steered the
research away from some of the furthest-reaching results found so far
(several 15-minute and 25+-minute-expiry cells reached walk-forward or
TEST; several 1-minute cells never got past gate 1). Going forward:
**every (asset, timeframe, horizon) combination this project has data
for is an equally legitimate hypothesis until evidence says otherwise.**

---

## 3. El objetivo sigue siendo opciones binarias

Unchanged, restated for completeness: the fixed problem this project
solves is *"dado un precio de entrada y un vencimiento determinado,
¿cuál es la probabilidad de que una opción CALL o PUT termine WIN?"* —
`research/dataset.py`'s `call_wins_h`/`put_wins_h` (Section 1 above) is
already exactly that question, `research/candidacy.py`'s four-gate bar
(corrected per `BINARY_OPTIONS_REFRAME_AUDIT.md`) is already exactly the
validation discipline the user listed in their point 3 (WIN/LOSS
probability, CI, FDR, robustness, walk-forward, TEST-once, payout,
break-even, EV). Nothing here changes that — only which expiries get
studied, and in what order.

---

## 4. EUR/USD 15m h=5 — estado y relevancia metodológica

**Estado, exactamente como el usuario lo pidió, sin más ni menos:**

> Candidato de investigación rechazado en TEST por insuficiente
> evidencia estadística: TEST 60.0%, n=120, IC inferior 51.1%,
> break-even 54.05%.

Not a strategy. Not discarded either — it stays logged, in the matrix
below, as a real result.

**Is it methodologically interesting within the full investigated
universe?** Yes, for a specific, narrow reason — not because of its
75-minute expiry (Section 2 above), but because of the *shape* of its
result relative to every other candidate this project has ever produced:

- It is the only condition, across H1-H20, Step 7, and Step 8d
  (corrected model included), whose TEST result was **directionally
  consistent** with TRAIN and with every one of a genuinely-validated
  (4-of-4-folds-sampled) walk-forward's folds — 65.3% TRAIN, 59-75%
  across all four walk-forward folds, 60.0% TEST. Every other
  TEST-reaching candidate (the original Step 8d day-of-week case, the
  triple-barrier EUR_USD/5m case, Section 6's other `test`-gate cells)
  showed a clean reversal or a structural zero-sample TEST failure
  instead.
- It failed specifically on **confidence-interval width relative to
  break-even**, not on sign — a fundamentally different failure mode
  from "overfit and reversed."
- This distinguishes it as the one result in the entire investigation
  that looks like a genuine, small, boundary-case candidate rather than
  either noise (Cause A/reversal) or a data-mining artifact (sparse
  folds, structural TEST-emptiness). That is what makes it
  *methodologically* interesting — not its expiry length, which per
  Section 2 was never a criterion to begin with.
- It is **one data point** in a still-small, bounded search (10-13
  core features, 2-way interactions only, 5 asset/timeframe datasets,
  horizons {1,2,3,5}). It says nothing about whether 75-minute expiries
  specifically are more promising than any other duration — the matrix
  in Section 6 shows walk-forward and TEST arrivals scattered across
  1-minute through 75-minute expiries, with no visible expiry-length
  pattern in what reaches the later gates.

---

## 5. ¿Sería válido ampliar datos de EUR/USD 15m para este candidato? — NO, y por qué

**Answer: not for the purpose of re-testing this specific, already-
observed candidate. No data was fetched or expanded for this audit.**
Here is the reasoning, worked through explicitly per the user's request.

`backtest.splits.compute_temporal_split` computes TRAIN/VALIDATION/TEST
as **fractions of the whole candle count**, recomputed fresh every time
it's called. This is the mechanism that breaks a naive "just add more
data and re-run":

- **Appending newer candles at the end** (if any existed — they don't;
  see below) and re-running the SAME 60/20/20 split over the enlarged
  series shifts the TRAIN/VALIDATION/TEST cut points later in calendar
  time. The rows that were the ORIGINAL TEST split would fall inside the
  NEW TRAIN or VALIDATION region — meaning a condition already SELECTED
  and SCORED using knowledge of the old TEST outcome would now be
  "trained" partly on that same data. That is textbook look-ahead/
  selection bias: the condition was not chosen blind to that region.
- **Prepending older candles at the start** has the identical problem in
  the other direction: worked through arithmetically (adding ~2,000
  older 15-minute rows to the existing ~4,967), the new TEST window
  (last 20% of the enlarged series) partially overlaps what used to be
  the tail of VALIDATION — which the walk-forward gate already used to
  judge this same condition. Any new TEST computed this way is
  contaminated by data the condition's own evaluation has already seen.
- **There is no genuinely new data to fetch anyway.** The most recent
  EUR/USD 15m candle already in this project's database is
  `2026-09-18 23:15:00` — effectively "now" (today's date in this
  environment is 2026-09-18). There is no future data sitting unfetched
  that could serve as a clean, never-seen prospective check; the only
  data available to add is OLDER history, which Section 5's
  prepending analysis above already rules out as a way to re-validate
  THIS specific already-scored condition.
- Even setting the mechanics aside: re-running the exact same,
  already-selected condition against a recomputed TEST split that
  overlaps anything already used is, in substance, **a second look at a
  result already observed** — precisely what BACKTESTING.md's
  "touched exactly once" rule exists to prevent, whether or not the
  underlying row indices technically differ.

**What WOULD be valid, if pursued later (not done here, not
recommended as urgent)**: treat additional older EUR/USD 15m history as
a **separate, independent study**, not a re-test of this candidate —
1. Fetch the older candles and add them ONLY to a new "extended
   development set" = old_data + new_older_data, explicitly EXCLUDING
   the original TEST region (the exact rows already spent).
2. Run discovery + candidacy completely fresh on that extended
   development set, own `run_id`, own FDR correction, no reference to
   the fact that the previous, narrower search found this specific
   condition — it may rediscover it, find something else, or find
   nothing.
3. Whatever survives gates 1-3, its TEST check must come from a
   held-out slice **carved out of the newly added data itself before
   any discovery runs**, or from genuinely future data collected after
   this point — never a recomputed split over the combined old+new
   series, and never the already-spent original TEST rows.

This is a legitimate protocol for a *future, separate* piece of
research — not a way to give the EUR/USD 15m `hour_utc`/session
condition a second chance at the same question. Per the user's explicit
instruction, none of this was executed.

---

## 6. Matriz completa del universo investigado

Every (dataset, timeframe, h, direction) cell this project has actually
run discovery against, generated by `scripts/build_expiry_matrix.py`
directly from `condition_trials` (FDR counts) and the corrected
candidacy re-run's own log (furthest gate reached) — no value below is
estimated or invented. "FDR-sig" counts only winning-direction
(`win_rate > 0.5`) significant trials, consistent with every other
report in this project (see `PREDICTABILITY_AUDIT.md` Section 6 for why).
"Furthest gate" is the deepest candidacy stage ANY selected condition
from that cell reached under the corrected (delay-only) execution model
— `NO EVALUADO en candidacy corregida` means FDR-significant conditions
existed there but none were among the top-3-by-p-value selected into the
corrected re-run (a selection-bound gap, not evidence against the cell).

### Naive target (`call_wins_h`/`put_wins_h`) — Step 7 (10 core features) and Step 8d-session (core + `hour_utc`/`trading_session_code`/`day_of_week`)

| Dataset | Timeframe | h | Vencimiento | Dirección | Trials (S7) | FDR-sig (S7) | Trials (S8d-session) | FDR-sig (S8d-session) | Furthest gate (candidacy corregida) |
|---|---|---:|---:|---|---:|---:|---:|---:|---|
| EUR_USD | 15m | 1 | 15 min | CALL | 720 | 4 | 1200 | 7 | walk_forward |
| EUR_USD | 15m | 2 | 30 min | CALL | 720 | 41 | 1200 | 52 | sample_size_and_margin |
| EUR_USD | 15m | 3 | 45 min | CALL | 720 | 68 | 1200 | 104 | sample_size_and_margin |
| EUR_USD | 15m | 5 | 75 min | CALL | 720 | 43 | 1200 | 61 | **test** |
| EUR_USD | 15m | 1 | 15 min | PUT | 720 | 14 | 1200 | 25 | sample_size_and_margin |
| EUR_USD | 15m | 2 | 30 min | PUT | 720 | 23 | 1200 | 29 | sample_size_and_margin |
| EUR_USD | 15m | 3 | 45 min | PUT | 720 | 27 | 1200 | 39 | robustness |
| EUR_USD | 15m | 5 | 75 min | PUT | 720 | 8 | 1200 | 14 | robustness |
| EUR_USD | 1m | 1 | 1 min | CALL | 720 | 107 | 1152 | 138 | sample_size_and_margin |
| EUR_USD | 1m | 2 | 2 min | CALL | 720 | 30 | 1152 | 43 | sample_size_and_margin |
| EUR_USD | 1m | 3 | 3 min | CALL | 720 | 85 | 1152 | 128 | sample_size_and_margin |
| EUR_USD | 1m | 5 | 5 min | CALL | 720 | 44 | 1152 | 64 | walk_forward |
| EUR_USD | 1m | 1 | 1 min | PUT | 720 | 26 | 1152 | 32 | sample_size_and_margin |
| EUR_USD | 1m | 2 | 2 min | PUT | 720 | 0 | 1152 | 2 | sample_size_and_margin |
| EUR_USD | 1m | 3 | 3 min | PUT | 720 | 53 | 1152 | 86 | sample_size_and_margin |
| EUR_USD | 1m | 5 | 5 min | PUT | 720 | 53 | 1152 | 86 | walk_forward |
| EUR_USD | 5m | 1 | 5 min | CALL | 720 | 11 | 1200 | 11 | walk_forward |
| EUR_USD | 5m | 2 | 10 min | CALL | 720 | 41 | 1200 | 54 | walk_forward |
| EUR_USD | 5m | 3 | 15 min | CALL | 720 | 72 | 1200 | 92 | sample_size_and_margin |
| EUR_USD | 5m | 5 | 25 min | CALL | 720 | 54 | 1200 | 89 | walk_forward |
| EUR_USD | 5m | 1 | 5 min | PUT | 720 | 12 | 1200 | 14 | sample_size_and_margin |
| EUR_USD | 5m | 2 | 10 min | PUT | 720 | 77 | 1200 | 104 | robustness |
| EUR_USD | 5m | 3 | 15 min | PUT | 720 | 72 | 1200 | 110 | robustness |
| EUR_USD | 5m | 5 | 25 min | PUT | 720 | 58 | 1200 | 91 | robustness |
| GBP_USD | 5m | 1 | 5 min | CALL | 720 | 2 | 1248 | 5 | sample_size_and_margin |
| GBP_USD | 5m | 2 | 10 min | CALL | 720 | 24 | 1248 | 24 | sample_size_and_margin |
| GBP_USD | 5m | 3 | 15 min | CALL | 720 | 16 | 1248 | 19 | sample_size_and_margin |
| GBP_USD | 5m | 5 | 25 min | CALL | 720 | 4 | 1248 | 11 | walk_forward |
| GBP_USD | 5m | 1 | 5 min | PUT | 720 | 6 | 1248 | 8 | sample_size_and_margin |
| GBP_USD | 5m | 2 | 10 min | PUT | 720 | 43 | 1248 | 52 | sample_size_and_margin |
| GBP_USD | 5m | 3 | 15 min | PUT | 720 | 77 | 1248 | 101 | sample_size_and_margin |
| GBP_USD | 5m | 5 | 25 min | PUT | 720 | 58 | 1248 | 102 | walk_forward |
| USD_JPY | 5m | 1 | 5 min | CALL | 720 | 15 | 1248 | 26 | **test** |
| USD_JPY | 5m | 2 | 10 min | CALL | 720 | 33 | 1248 | 44 | sample_size_and_margin |
| USD_JPY | 5m | 3 | 15 min | CALL | 720 | 48 | 1248 | 72 | walk_forward |
| USD_JPY | 5m | 5 | 25 min | CALL | 720 | 13 | 1248 | 38 | **test** |
| USD_JPY | 5m | 1 | 5 min | PUT | 720 | 1 | 1248 | 6 | sample_size_and_margin |
| USD_JPY | 5m | 2 | 10 min | PUT | 720 | 34 | 1248 | 49 | sample_size_and_margin |
| USD_JPY | 5m | 3 | 15 min | PUT | 720 | 52 | 1248 | 75 | sample_size_and_margin |
| USD_JPY | 5m | 5 | 25 min | PUT | 720 | 2 | 1248 | 22 | walk_forward |

*(USD_JPY/5m CALL h=1 and h=5's "test" cells are the three day-of-week-
conditioned candidates from `PREDICTABILITY_AUDIT.md`/
`BINARY_OPTIONS_REFRAME_AUDIT.md` whose TEST split had zero matching
rows — rejected on a structural sparse-data problem, not on evidence
against the pattern; see `BINARY_OPTIONS_REFRAME_AUDIT.md`'s "Corrected
re-run results" section.)*

### Triple-barrier target (auxiliary — see Section 1's caveat on its expiry semantics)

| Dataset | Timeframe | h (`max_horizon`) | Vencimiento equivalente | Dirección | Trials | FDR-sig | Furthest gate |
|---|---|---:|---:|---|---:|---:|---|
| EUR_USD | 15m | 10 | 150 min (variable, barrier-stop) | CALL | 720 | 5 | walk_forward |
| EUR_USD | 15m | 10 | 150 min (variable, barrier-stop) | PUT | 720 | 4 | sample_size_and_margin |
| EUR_USD | 1m | 10 | 10 min (variable, barrier-stop) | CALL | 720 | 46 | sample_size_and_margin |
| EUR_USD | 1m | 10 | 10 min (variable, barrier-stop) | PUT | 720 | 162 | walk_forward |
| EUR_USD | 5m | 10 | 50 min (variable, barrier-stop) | CALL | 720 | 123 | **test** |
| EUR_USD | 5m | 10 | 50 min (variable, barrier-stop) | PUT | 720 | 79 | robustness |
| GBP_USD | 5m | 10 | 50 min (variable, barrier-stop) | CALL | 720 | 50 | walk_forward |
| GBP_USD | 5m | 10 | 50 min (variable, barrier-stop) | PUT | 720 | 77 | robustness |
| USD_JPY | 5m | 10 | 50 min (variable, barrier-stop) | CALL | 720 | 8 | walk_forward |
| USD_JPY | 5m | 10 | 50 min (variable, barrier-stop) | PUT | 720 | 10 | sample_size_and_margin |

### Combinaciones que existen en el universo del proyecto pero NUNCA han sido evaluadas por este pipeline

- **Timeframe 1h**: candles ingested for all three pairs (H1-H20's own
  hand-designed hypotheses were tested there), but the systematic
  discovery/candidacy pipeline (Step 7 onward) never included it — a
  scope choice at the time, not a finding against it.
- **GBP_USD and USD_JPY at 1m or 15m timeframes**: never ingested at
  all — only EUR_USD has 1m/15m coverage; GBP_USD/USD_JPY only have 5m
  and 1h.
- **Horizons beyond {1,2,3,5} candles on the naive target** (e.g.
  h=10, 15, 20, 30): never tried on `call_wins_h`/`put_wins_h` — the
  triple-barrier run's `max_horizon=10` is a different target shape
  (Section 1's caveat), not a substitute for this.
- **Timeframes 30m, 4h, 1d, 1week**: no candles ingested at all for any
  pair.
- **3-way feature interactions**: at any timeframe/horizon — Step 7 and
  Step 8d only ever ran 2-way (`PREDICTABILITY_AUDIT.md` Section 13,
  hypothesis 4 — flagged there as high-overfitting-risk without a larger
  FDR-corrected budget, not evaluated here either).

---

## 7. Principio fundamental — reafirmado

No expiry duration is chosen in advance, and none is optimized for. The
question this project answers, per cell, stays exactly:

> *Predecir correctamente WIN/LOSS de una opción binaria CALL/PUT al
> vencimiento elegido, con evidencia estadística robusta y fuera de
> muestra.*

Forex remains the research dataset/proxy only, used when no authorized
OTC source exists (`BINARY_OPTIONS_REFRAME_AUDIT.md` Section 11 /
`PREDICTABILITY_AUDIT.md` Section 10 — unchanged by this correction).
The matrix in Section 6 is the current, complete, unfiltered state of
what evidence exists across every (asset, timeframe, horizon) this
project has actually investigated — nothing in it is deprioritized by
expiry length. Future research effort should be allocated by what that
evidence shows (which cells reached the furthest gates, and why), not by
an assumed target duration.
