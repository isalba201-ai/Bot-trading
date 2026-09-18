# Binary-options reframing audit

**Status: this document is a written compatibility audit, not a new
research run.** It responds to an explicit correction from the user:
this project's real objective has never been a profitable Forex trading
strategy — it is a research engine for **manual CALL/PUT binary-options
signals**, ultimately for Pocket Option OTC, with real Forex serving only
as a research laboratory when a real, authorized dataset is needed. No
pipeline was re-run, no new candidates were evaluated, and no number
from Step 7/8 was silently reinterpreted as if it meant something else —
per the user's explicit instruction, anything not actually recomputed
under a corrected methodology is marked `NO EVALUADO` in Section 16's
table, not estimated or reused across methodologies as if equivalent.

**One correction accepted up front, stated plainly because the user
asked to be told if their own framing had an error too**: their
framing is *not* the source of the problem. The problem identified below
is in this project's own code — specifically, `backtest/simulator.py`
imported a cost assumption appropriate for a spot-Forex broker
(bid/ask-crossing slippage) into an evaluation that was, from Step 7
onward, silently being read as if it applied to a binary option. That
mismatch, not a flaw in the user's request, is the main technical finding
of this audit.

---

## 1. Qué partes de Step 7 siguen siendo válidas

- **The naive discovery target itself** (`research/dataset.py`'s
  `call_wins_h`/`put_wins_h` = `close[t+h] > close[t]` for CALL,
  `<` for PUT) is, on inspection, **already a structurally correct
  binary-option WIN/LOSS definition** — see Section 4. This was not
  understood as such at the time it was built (it was framed as "does
  price go up," a Forex-strategy question), but the formula is right.
- **The FDR-corrected combinatorial search itself**
  (`research/discovery.py`): fully target-agnostic, needs no changes.
- **The CALL/PUT symmetry finding** (Section 6 of
  `PREDICTABILITY_AUDIT.md`, `call_win_rate + put_win_rate = 1.000` across
  1,451 matched pairs): unaffected by anything below — it's a property of
  the naive label, computed with no execution model involved at all.
- **The literature review** (Step 8, Section 12): unaffected. Its
  findings about short-horizon predictability, autocorrelation, and the
  triple-barrier critique don't depend on which broker's execution
  mechanics get simulated afterward.
- **`payout=0.85 → break_even≈54.05%`, `payout_adjusted_expectancy`**
  (`backtest/metrics.py`): exactly the formulas the user re-derived in
  their message — no changes needed, see Section 7.

## 2. Qué partes de Step 8 siguen siendo válidas

- **`research/decomposition.py`'s delay-only rungs**
  (`delay_only_1..5`, zero slippage, zero drop): directly reusable —
  see Section 8. These measure a genuine, binary-options-relevant
  quantity (how much does a delayed entry cost you), with no borrowed
  Forex-broker assumption baked in.
- **The mechanical A/B/C classification structure** (no predictability /
  too small for payout / execution destroys it): the *structure* is
  sound and still exactly what's needed. What needs revisiting is
  *which* rungs feed the "C" classification — see Section 3.
- **The session-conditioned run's TEST-gate case study**: still valid
  evidence of overfitting, for a specific, important reason explained in
  Section 3 — TEST-once discipline's value here is independent of the
  execution-cost question.
- **The literature-grounded triple-barrier/MFE-MAE tooling**
  (`research/targets.py`): still useful code — see Section 9, just not
  as a *primary* target.
- **The OTC data-sourcing finding** (Section 10 of
  `PREDICTABILITY_AUDIT.md`): unaffected, restated in Section 11 below.

## 3. Qué partes están enfocadas incorrectamente hacia Forex/trading convencional

**The central finding of this audit.** `backtest/simulator.py`
(Phase 4, built for evaluating a hand-designed strategy manually placed
in a real spot-Forex broker) applies two assumptions that do not
represent a Pocket Option binary option:

```python
def _apply_slippage(price, direction, slippage_pct):
    adjustment = slippage_pct / 100.0
    if direction == "CALL":
        return price * (1 + adjustment)   # worse (higher) entry for a buyer
    return price * (1 - adjustment)        # worse (lower) entry for a seller
```

This applies a **fixed, deterministic, always-adverse** price shift to
*every single trade's entry price*, modeling the cost of crossing a
bid/ask spread when a market order fills in a real spot-FX broker. **A
binary option has no bid/ask spread to cross.** You click CALL or PUT
against a *quoted* price and the platform compares that quote to another
quote at expiry — there is no "fill price" concept, no market-impact
cost, nothing analogous to paying the spread. The option's entire
economics are `(WIN or LOSS) × payout`, not `(exit_price − entry_price)
× position_size` the way a spot trade's P&L is computed — and
`_pnl_pct`, further down the same function, computes exactly that
continuous spot-style P&L and only reads its *sign* to decide WIN/LOSS:

```python
pnl_pct = _pnl_pct(entry_price, exit_price, direction)
result = "WIN" if pnl_pct > 0 else "LOSS"
```

Because `slippage_pct` shifts `entry_price` before this sign check, it
can — and, per Step 8's own numbers, overwhelmingly does — **flip the
WIN/LOSS determination for any trade whose true price move was smaller
than the slippage adjustment**, even though nothing resembling that
adjustment happens on a real binary-options platform.

**Consequence for Step 8's own conclusions**: the finding "slippage
alone (0.01%, zero delay) collapses survival to 0.9%, nearly as
destructive as the full realistic scenario" (`PREDICTABILITY_AUDIT.md`
Section 3) is **not valid evidence about binary options** — it is an
artifact of testing a Forex-broker fill-cost model that shouldn't have
been applied to this target at all. This needs to be explicitly
retracted, not just re-labeled: **Cause C's dominant driver, as
currently reported, is wrong.** The `delay_only` rungs (no slippage
applied) remain valid — see Section 8 — and they showed a *much* less
destructive picture (24-47% of candidates still clearing margin even at
5 minutes of pure delay) than the slippage-contaminated "realistic"
number implied.

**The `raw → optimistic` gap (100% → 52.2% clearing margin, zero
friction, zero delay)** is a separate, still-real issue: it comes from
`entry_idx = i + 1 + scenario.entry_delay_candles` — even at
`entry_delay_candles=0`, the simulator enters at the **open of the
candle *after* the signal candle**, one full candle later than the naive
label's own `close[t] → close[t+h]` window. This is not a borrowed
Forex assumption — it's the earliest entry achievable in a candle-based
backtest, and arguably still a reasonable proxy for "fastest possible
manual entry" given this project has no sub-minute data. But it is a
**real, currently-undocumented definitional misalignment**, addressed in
Section 4/8 below, not eliminated by removing slippage.

**`research/targets.py`'s triple-barrier method** is not "Forex
thinking" in the same contaminating sense — see Section 9 — but it does
model a stop/take-profit-style payoff shape that a fixed-expiry binary
option doesn't have, so it was mis-scoped as a *candidate primary
target* in Step 8 rather than what it actually is: a useful auxiliary
diagnostic.

## 4. Cómo debe redefinirse formalmente el target binario

**The naive label formula does not need to change.** For a CALL entered
at row `t` with expiry `h` candles later:

```
entry_price_proxy = close[t]
expiry_price      = close[t+h]
WIN  if expiry_price >  entry_price_proxy
LOSS if expiry_price <= entry_price_proxy   (tie -> LOSS per the user's
                                              own spec; see note below)
```

— exactly `research/dataset.py`'s existing `call_wins_h`/`put_wins_h`,
mirrored for PUT. **What needs explicit, permanent documentation** (not
a code change) is the one approximation this makes and its bias:

- **`close[t]` is a point-in-time-safe proxy for "price at the moment
  the signal fires," not a claim about the exact price a real click on
  Pocket Option would lock in.** The real entry happens some seconds
  after `close[t]` (time to read a notification, decide, click), during
  which the quote can move. This project has no sub-minute data to
  measure that gap directly — the *closest* honest measurement of its
  effect is `research/decomposition.py`'s `delay_only_N` rungs (Section
  8), which should continue to be read as "how much does N whole
  minutes of reaction time cost," not as a precise sub-minute number.
- **Tie handling**: `call_wins_h`/`put_wins_h` currently treat
  `close[t+h] == close[t]` as `NaN` (excluded from `n`, from both
  directions) — a "void, stake returned" convention. The user's own
  formula above specifies `LOSS` for a CALL tie. **This is a genuine,
  small discrepancy worth resolving deliberately, not silently**: real
  binary-options brokers differ on tie handling (some void/refund, some
  count as loss), and this project does not have Pocket Option's exact
  contract rules for this case. Recommendation: **keep the existing
  void/NaN convention** (it's the more conservative, no-fabrication
  choice already consistent with this codebase's "never guess an
  outcome" rule — see `DATA.md`), but state explicitly in
  `research/dataset.py`'s docstring that this is a modeling choice, not
  a confirmed Pocket Option rule, and that it should be revisited if the
  exact contract terms are ever obtained. In practice this is
  measure-zero for continuous FX prices and immaterial to any result so
  far.
- **What should change**: `research/candidacy.py`'s TRAIN-gate
  re-evaluation should stop calling the bundled "realistic" scenario
  (which still includes `slippage_pct`) the binary-option-relevant
  number. See Section 8 for the corrected execution model.

## 5. Cómo debe evaluarse CALL vs PUT

No change needed to the *search* (already symmetric — CALL and PUT are
searched as separate target columns, never assumed to be mirror images
until actually checked; Section 6 of `PREDICTABILITY_AUDIT.md` confirmed
they empirically are, at the naive-label level). What changes is
**interpretation discipline going forward**: a condition significant for
a *low* win rate on one direction's target column is the same underlying
pattern as its complementary direction's *high* win rate — Step 8's
`scripts/run_execution_decomposition.py` already filters to
winning-direction rows only (`ConditionTrial.win_rate > 0.5`) to avoid
double-counting one pattern as two findings; this convention should
carry forward into any future run without needing to be rediscovered.

## 6. Cómo deben tratarse los diferentes vencimientos

Two genuinely different quantities were being conflated:

- **Signal cadence** = the timeframe candles are computed on (how often
  a *fresh* signal can even be generated).
- **Expiry duration** = `horizon_candles × timeframe_seconds` (how long
  the option itself runs once entered).

`1m timeframe, h=5` and `5m timeframe, h=1` both currently get loosely
called "5-minute expiry" work, but they are not comparable for a manual
system: the first can produce a fresh signal every minute; the second
only every 5 minutes, meaning a person could be looking at up-to-5-
minute-stale context before acting. **Recommendation**: report both
quantities explicitly, side by side, for every hypothesis from here on
(the table in Section 16 does this) — and prioritize the
**1-minute-timeframe / short-`h`** combinations (h=1,3,5 → 1/3/5-minute
expiries) as the primary research axis, since they most directly match
common Pocket Option contract durations and give the freshest signal
cadence; treat 5m/15m timeframe runs as secondary, exploratory context
(regime/session structure), not as alternate ways to reach the same
expiry duration.

Respecting TRAIN/VALIDATION/TEST, FDR, walk-forward, TEST-once, and
minimum-sample discipline **across every horizon tried, without
exception**, exactly as the user requires, is already how
`research/discovery.py`/`research/candidacy.py` are built — no
structural change needed, just continued discipline (already the rule,
not a new one).

## 7. Cómo deben incorporarse payout, break-even y EV

No change. `backtest/metrics.py::break_even_win_rate`/
`payout_adjusted_expectancy` already implement exactly the user's own
formulas verbatim:

```
break_even_win_rate = 1 / (1 + payout)
EV_per_1_dollar      = P * payout - (1 - P)
```

This is, and remains, a first-class economic evaluation of a binary
option — never a "Forex metric" in the sense the user was worried about.
What Section 3 retracts is a *different, mis-scoped* cost (slippage)
that was being computed *before* this stage and contaminating the win
rate `P` that gets fed into it — not the break-even/EV math itself.

## 8. Cómo debe modelarse el delay sin convertir el problema en uno de ejecución Forex

**Keep exactly one friction dimension: delay.** Drop `slippage_pct` and
`signal_drop_probability` from the binary-option-relevant evaluation
entirely (see Section 3's reasoning — neither has a real binary-options
analog; `signal_drop_probability` additionally has near-zero effect on
`win_rate` itself since VOID trades are already excluded from
`TradeStats`, so removing it loses nothing).

**Corrected execution model for binary-options candidacy**: entry at
`open[t+1+delay_candles]` (unavoidable minimum given candle-granular
data — document this explicitly as "fastest achievable entry," not
silently), exit/expiry resolved at `close[t+1+delay_candles+h]`, **no
price adjustment at all** — WIN/LOSS determined purely by
`exit_price` vs. `entry_price`, exactly the option's real payoff
structure. `research/decomposition.py`'s existing `delay_only_1..5`
rungs already implement precisely this (zero slippage, zero drop,
delay only) — **they need no code changes**, only to be recognized as
*the* corrected model going forward, replacing "realistic"/"pessimistic"
as the binary-options-relevant read. `research/candidacy.py`'s gate 1
should be re-pointed at a `delay_only`-style scenario (a specific,
pre-registered delay value representing realistic manual reaction time —
e.g. 1-2 minutes — proposed as an open decision for the next step, not
decided unilaterally here) instead of the bundled "realistic" scenario.

**A related, separate gap already flagged in `PREDICTABILITY_AUDIT.md`
Section 2, restated here because it's relevant to correcting
`candidacy.py`**: the walk-forward gate currently only requires a
*fraction* of sufficiently-sampled folds to show an edge, with no floor
on how many folds must be sufficiently sampled at all — the one
candidate that reached TEST exploited exactly this (1 of 5 folds
sampled, "100%" of 1 fold). **Recommended fix, bundled with the delay
correction since both touch `CandidacyThresholds`**: add a minimum
folds-sampled requirement (e.g. `min_folds_sampled=3`) before gate 3 can
be marked passed.

## 9. Qué hacer con triple-barrier y MFE/MAE

**DESCARTAR as the primary discovery/candidacy target. MANTENER as
auxiliary features/diagnostics.** A binary option has no stop-loss or
take-profit — its only resolution is price at a fixed expiry vs. entry,
exactly Section 4's target. Triple-barrier and MFE/MAE model a
fundamentally different payoff shape (whichever threshold gets touched
first, or the best/worst excursion within a window) that doesn't
describe what a Pocket Option contract actually pays out on.

They remain useful for two things that are NOT "redefining the target":

1. **As descriptive features** for the discovery search itself — e.g.
   "did this setup historically show strong, early directional
   persistence" is a legitimate INPUT signal, computed from
   `mfe_mae_labels`/`triple_barrier_labels`, fed into the SAME
   fixed-horizon binary candidacy funnel as any other feature. This is
   different from what Step 8 did (using them AS the target).
2. **As diagnostics** — Section 10 below reuses `mfe_mae_labels`
   directly for exactly this, unchanged.

**Was Step 8's triple-barrier discovery RUN wasted, then?** Not framed
correctly, but not worthless either: it's informative as a check that
the naive target's severe post-execution reversal pattern is somewhat
(not fully) sensitive to label definition — worth keeping as a footnote,
not as evidence about binary-options predictability specifically, since
it was never evaluating the actual binary payoff to begin with.

## 10. Qué hacer con la hipótesis de volatility-contraction NO_TRADE

**ADAPTAR — the empirical finding survives, its economic justification
needs correcting.** `PREDICTABILITY_AUDIT.md` Section 7 justified this
filter by comparing MFE/MAE magnitude to the (now-retracted) slippage
cost. Under the corrected binary-options framing, the *right*
justification is different but still valid: for a fixed-payout binary
option, what matters isn't "is the move bigger than a transaction cost"
(there isn't one) — it's **"is `P(WIN)` sufficiently above break-even."**
In a volatility-contraction regime, price direction over a short, fixed
horizon is plausibly closer to a random walk (smaller moves are harder
to call directionally with any real skill), which is a mechanism for
`P(WIN)` sitting closer to 50% — closer to (and likely below) the
54.05%-at-0.85-payout break-even — independent of any cost model. **This
hypothesis is not yet validated as a filter** (Section 7's analysis
measured excursion magnitude, not `P(WIN)` directly, in each regime) —
recommended as a candidate for the next bounded run: measure
`win_rate`/CI directly, per regime, on the corrected fixed-horizon
binary target, rather than inferring it from excursion size.

## 11. Qué datos adicionales necesitamos para validar realmente OTC

**Unchanged from `PREDICTABILITY_AUDIT.md` Section 10 — restated here
because the user asked for it in this document too, not re-derived.**
No authorized/licensed source of Pocket Option's OTC feed was found
(reasonably confident from consistent secondary sources — Pocket
Option's own materials describing it as broker-generated, every
"Pocket Option API" self-described as unofficial, no commercial vendor
listing it — but not confirmed by one primary document). This finding
is orthogonal to everything else in this audit: fixing the execution
model changes how we'd evaluate a Forex-derived hypothesis, it does not
create an OTC data source. **Forex real remains a research laboratory
only** — any condition that eventually clears the full corrected funnel
on Forex data should be reported exactly as: *"esta hipótesis merece ser
validada posteriormente sobre datos OTC,"* never as validated for OTC
itself, per the user's own Section 3.

## 12. Qué experimentos ya realizados podemos reutilizar

- The naive-label `ConditionTrial` rows from Step 7 and Step 8d's
  session-conditioned run — **the label itself is correct** (Section 4);
  what's reusable is the `raw` win rate, sample size, and FDR
  significance flag for each condition. Re-deriving these from scratch
  would just reproduce the same numbers.
- `research/decomposition.py`'s `delay_only_1..5` rungs, already
  computed for 113 Step 7 candidates — directly usable as the corrected
  execution read (Section 8), no re-run needed for those specific 113.
- The literature review (Step 8, Section 12) and the OTC data-sourcing
  investigation (Section 10) — unaffected by any of this, fully reusable.
- The session-conditioned and triple-barrier discovery infrastructure
  (`scripts/run_step8d_research.py`) — reusable as-is for future bounded
  re-runs; only what target/scenario they feed into `candidacy.py`
  changes.

## 13. Qué experimentos debemos repetir

- **`research/candidacy.py`'s gate 1 (and downstream gates 2-4) for
  every condition that was significant at the naive-label level**, using
  a `delay_only`-style scenario instead of the slippage-contaminated
  "realistic" one. This is expected to change which conditions pass gate
  1 — likely more than Step 7's 1/120, since Section 3 showed
  `slippage_only` alone was responsible for most of the collapse. **This
  does not mean more will survive gates 2-4** — robustness, walk-forward,
  and TEST are unaffected by this correction and remain the real bar
  (see Section 14's honest calibration of expectations below).
- The one candidate that reached TEST in Step 8d should be **left as
  rejected** — its TRAIN→TEST reversal is a real, execution-model-
  independent overfitting signal (both splits used the same, even if
  imperfect, execution model) — but is worth re-confirming under the
  corrected model as a sanity check, not because the rejection is in
  doubt.

## 14. Qué experimentos nuevos son necesarios

- The volatility-contraction filter, measured directly as `P(WIN)` by
  regime on the corrected binary target (Section 10 above), not
  inferred from excursion size.
- The minimum-folds-sampled fix to `CandidacyThresholds` (Section 8)
  should be in place before any new candidacy run, so a future TEST-
  reaching candidate isn't again a sparse-fold artifact.
- **Calibration expectation, stated honestly**: the literature review
  (Step 8, Section 12, still valid — Section 2 above) found short-
  horizon return predictability, where it exists at all, on the order of
  R²≈1-3% even at favorable (half-hour equity) horizons, and found
  short-horizon autocorrelation to be largely the SAME phenomenon as
  bid-ask-bounce microstructure noise, not a separable signal. Removing
  the mis-scoped slippage cost will very likely let MORE conditions pass
  gate 1 than Step 7's 1/120 — but this is a *correction of an
  overcounted failure mode*, not new evidence of a real edge. The FDR
  and full 4-gate discipline, especially TEST-once, remain essential,
  and a negative overall result is still a live, honest possible
  outcome — not something this correction is expected to overturn on
  its own.

## 15. Qué debe considerarse "evidencia suficiente" antes de pasar a señales

Unchanged from the existing, already-correct `research/candidacy.py`
bar, with the two Section 8 amendments (delay-only execution model,
minimum-folds-sampled) layered in: `sample_size >= 100`,
`payout_adjusted_expectancy > 0`, `margin_over_break_even >= 0.03`,
`evaluate_robustness(...) == "consistent_direction"`,
`fraction_folds_with_edge >= 0.7` **with at least `min_folds_sampled`
folds actually sufficiently sampled**, worst-fold win rate still above
break-even, and a positive, CI-clearing out-of-sample TEST result,
touched exactly once. **Nothing here is being loosened — the amendments
tighten the walk-forward gate and correct which execution model feeds
gate 1, they don't lower any bar.** Only a condition that clears every
gate, on the corrected model, earns any of the words "candidate,"
"edge," or "señal" — per the user's own Section 11 vocabulary discipline.

## 16. Arquitectura metodológica corregida para los próximos Steps

```
DATA (real Forex, laboratory only — never presented as OTC)
  ↓
FEATURES (v4, unchanged — descriptive, target-agnostic)
  ↓
BINARY TARGET  <- Section 4: close[t] vs close[t+h], documented proxy,
                   tie=void (unchanged formula, now explicitly justified)
  ↓
DISCOVERY (FDR-corrected combinatorial search — unchanged)
  ↓
CANDIDACY, corrected:
  gate 1: sample size + margin, using a DELAY-ONLY execution scenario
          (no slippage, no signal-drop) — Section 8
  gate 2: parameter-perturbation robustness — unchanged
  gate 3: walk-forward, WITH a minimum-folds-sampled floor — Section 8
  gate 4: TEST, touched once — unchanged
  ↓
PAYOUT / BREAK-EVEN / EV  <- Section 7, unchanged, always reported
  ↓
SIGNAL CANDIDATE (only past this point may the word "candidate" be used)
  ↓
MANUAL VALIDATION  <- signals/ + notifications/ (Phase 9), unchanged,
                       still not pointed at a live feed
```

Auxiliary, non-target uses layered alongside (never substituting the
binary target above): triple-barrier/MFE-MAE as descriptive features
(Section 9), volatility-regime `P(WIN)` as a candidate no-trade filter
(Section 10), session/time-of-day features (already tested, Step 8d).

---

## Tabla resumen

Per the user's explicit instruction: no value below is invented or
reused across methodologies as if equivalent. Anything requiring the
Section 8 execution-model correction, and not yet actually recomputed
under it, is `NO EVALUADO` — including for the one candidate that
reached TEST, since even its earlier gates were passed under the
slippage-contaminated model.

| Hipótesis | Dataset | Target binario | Vencimiento | n | WIN% | CI | Payout | Break-even | EV | FDR | Robustez | Walk-forward | TEST | Estado |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| USD_JPY/5m CALL h=5, `return_5∈(-0.762,-0.0309] AND pct_position_in_range_20∈(0.476,0.749]` (Step 7's only gate-2 arrival) | USD_JPY 5m (Forex, laboratory) | naive `call_wins_5` (formula = Section 4's binary target) | 5 velas × 5m = 25 min (señal cada 5 min) | 113 (TRAIN, naive) | 75.2% (raw) | NO EVALUADO (raw CI not computed) | 0.85 | 54.05% | NO EVALUADO bajo modelo corregido | Significativo (run step7-...) | Fragile (bajo el modelo con slippage) | NO EVALUADO bajo modelo corregido | NO EVALUADO | Necesita re-evaluación bajo Sección 8 |
| USD_JPY/5m PUT h=5, `bb_pct_b_20∈(0.232,0.501] AND day_of_week∈(2,3]` (Step 8d, único que llegó a TEST) | USD_JPY 5m (Forex, laboratory) | naive `put_wins_5` | 5 velas × 5m = 25 min (señal cada 5 min) | 156 (TRAIN) | 63.5% (TRAIN, bajo modelo con slippage) | [55.7%, 70.6%] | 0.85 | 54.05% | NO EVALUADO bajo modelo corregido | Significativo (run step8d-session-...) | consistent_direction | 1/1 folds suficientes (hueco de metodología, Sección 8) | **Rechazado**: 35.5% (n=93, CI [26.5%,45.6%]) | Rechazado en TEST — válido independientemente de la corrección de slippage |
| Volatility-contraction no-trade filter | Los 5 datasets (Forex, laboratory) | n/a (filtro, no target direccional) | n/a | NO EVALUADO como P(WIN) directo | NO EVALUADO | NO EVALUADO | n/a | n/a | n/a | No aplica (no es una condición de discovery) | NO EVALUADO | NO EVALUADO | NO EVALUADO | Hipótesis, no evaluada como P(WIN) — solo excursión medida |
| Cualquier condición bajo modelo de ejecución corregido (delay-only, sin slippage) | Los 5 datasets (Forex, laboratory) | binario corregido (Sección 4) | Por definir (recomendado: 1m/h=1,3,5) | NO EVALUADO | NO EVALUADO | NO EVALUADO | 0.85 (propuesto) | 54.05% | NO EVALUADO | NO EVALUADO (requiere nueva corrida) | NO EVALUADO | NO EVALUADO | NO EVALUADO | Pendiente de aprobación para ejecutar |
| Cualquier condición sobre datos OTC Pocket Option | N/A — sin fuente autorizada (Sección 11) | N/A | N/A | NO EVALUADO | NO EVALUADO | NO EVALUADO | N/A | N/A | N/A | N/A | N/A | N/A | N/A | Bloqueado — no existe fuente de datos |

---

## Resumen de una línea

El objetivo del proyecto queda formalmente re-anclado a: *¿podemos
identificar, con evidencia estadística robusta y fuera de muestra,
condiciones donde una opción binaria CALL o PUT tenga `P(WIN)`
suficientemente por encima del break-even del payout disponible? —
usando Forex real solo como laboratorio, nunca como sustituto de OTC.*
El error técnico encontrado (costo de slippage estilo Forex aplicado a
un payoff que no lo tiene) se corrige sin descartar la mayoría del
trabajo ya hecho — la disciplina FDR/candidacy/TEST-once permanece
intacta y sigue siendo la barra real. No se ha ejecutado ningún nuevo
cálculo bajo el modelo corregido; eso queda pendiente de tu aprobación
para el próximo Step.
