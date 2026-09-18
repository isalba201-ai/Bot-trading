# Signal engine (Phase 9)

**Status: the signal data model and lifecycle backend are implemented
and tested (`otc_research/signals/`, `otc_research/notifications/` —
see the pivot plan's sub-phase B,
`/root/.claude/plans/sequential-sparking-candle.md`); the "BUSCAR SEÑAL"
UI and any live data poller are still NOT built.** What exists today:
`signals.service.create_signal`/`send_notification`/
`refresh_expired_signals` (the PENDING/EXPIRED/DECIDED lifecycle),
`notifications.console_provider.ConsoleNotificationProvider` (the only
channel wired up so far, per this document's own "one non-negotiable
rule" below — no channel is added until it's actually wanted),
`signals.decisions.record_decision` (the manual TOOK_TRADE/DID_NOT_TAKE/
ARRIVED_LATE journal), and `signals.performance.theoretical_performance`/
`executable_performance` (the two read side-by-side, per this document's
"Signal journal" section). All of it has been verified only against a
**historical replay** (`scripts/replay_signals_historical.py`) fed
already-ingested real candles in timestamp order — explicitly not a live
run. It is not pointed at a live feed for a structural reason, not just
a sequencing one: Step 7's research run (STRATEGIES.md) found no
condition that survived realistic-execution re-validation, so nothing in
this codebase is currently entitled to a **ROBUST EDGE** classification,
and per this document's own rule, nothing below that bar may produce a
live signal. The "BUSCAR SEÑAL" UI, the live OANDA/Twelve Data polling
loop, and the pair/timeframe selection flow described below remain
unbuilt targets, not because the plumbing isn't ready, but because there
is nothing yet worth pointing it at.

## What triggers analysis

The user selects a **pair** and a **timeframe** and presses **"BUSCAR
SEÑAL"**. There is no continuous/background scanning and no automatic
execution of anything — this is an on-demand, human-in-the-loop tool.

Supported pairs (extensible): EUR/USD, GBP/USD, USD/JPY, USD/CHF, AUD/USD,
USD/CAD, NZD/USD.

Supported timeframes (extensible, 1 minute minimum — see DATA.md for why
there is no 30-second option): 1m, 5m, 15m, and others added later.

## What happens on "BUSCAR SEÑAL"

1. Fetch current real OANDA data for the selected pair/timeframe (recent
   closed candles — never the still-forming one).
2. Compute the same point-in-time features used in backtesting (Phase 3),
   using only data available as of the last closed candle.
3. Evaluate those features against whichever strategies have been
   classified **ROBUST EDGE** in BACKTESTING.md's methodology — nothing
   below that bar is eligible to produce a signal at all.
4. If a valid setup exists at or above the configured minimum quality tier
   (default: A+ only, see RISK_MANAGEMENT.md), show a signal. Otherwise show
   **"NO HAY SEÑAL"**. There is no fallback/lower-quality default signal.

## Signal format

```
PAR: EUR/USD
TEMPORALIDAD: 5 minutos
DIRECCIÓN: CALL / PUT
MOMENTO DE ENTRADA: 2026-03-04 14:32:00 UTC
VENCIMIENTO/TIEMPO OBJETIVO: 15 minutos (según estrategia)
NIVEL DE CONFIANZA ESTIMADO: 63% (IC 95%: 59%-67%, n=1,842 señales históricas equivalentes)
CONDICIONES QUE RESPALDAN LA SEÑAL:
  ✓ tendencia alcista (EMA slope > umbral)
  ✓ momentum confirmatorio
  ✓ estructura de mercado (higher-low reciente)
  ✓ sin expansión de volatilidad anómala
ESTADO: PROMISING / ROBUST EDGE (ver BACKTESTING.md)
```

## The one non-negotiable rule: what a confidence number is allowed to mean

**"NIVEL DE CONFIANZA ESTIMADO" is never invented.** It is the historical win
rate of statistically equivalent past setups (same strategy, same
conditions, ideally same regime), reported together with:

- **sample size** (how many equivalent historical setups this is based on),
- **confidence interval** (e.g. Wilson or Clopper-Pearson at 95%, not just a
  point estimate), and
- whether that estimate came from in-sample, out-of-sample, or walk-forward
  testing.

If the sample size is too small or the confidence interval too wide to say
anything meaningful, the system does not show a confidence number — it
shows **NO HAY SEÑAL** instead. A number with no sample size, CI, or
methodology behind it must never be displayed, and no wording like
"probabilidad histórica estimada" may ever be replaced with "certeza" or
similar — a historical win rate is a description of the past, not a
prediction guaranteed for the next trade. See BACKTESTING.md's edge
classification and RISK_MANAGEMENT.md's discipline rule.

## Signal journal

Every signal generated (shown to the user, at any quality tier, including
ones discarded before display) is written to the `signals` table
(`src/otc_research/db/models.py`) at generation time — write-once for the
inputs that produced it:

- pair, direction, timeframe, generated_at, target expiry/horizon
- strategy_code, model_version, score, quality_tier
- historical_sample_size, historical_win_rate, win_rate_ci_low/high, expectancy
- features_snapshot, conditions_met, conditions_summary
- market_regime, session, day_of_week, hour
- **(Phase 9)** `signal_ref` (human-readable id), `data_received_at`/
  `sent_at`/`valid_from`/`valid_until` (the four timestamps latency is
  computed from — `signals.latency`), `status`
  (PENDING/EXPIRED/DECIDED), `break_even_win_rate`/
  `margin_over_break_even` (payout-aware, from `backtest.metrics`),
  `payout_is_estimated` (true whenever the real broker payout at signal
  time isn't known), `confidence_label` (ALTA/MEDIA/BAJA — the fixed rule
  in `signals.confidence`, never hand-waved per signal), and
  `reasons_rejected` (the conditions that did NOT hold, alongside
  `conditions_met`).

Once the target horizon has elapsed, the actual outcome is fetched from real
market data and written back (`result`, `pnl`, `expiry_price`) — never
inferred, never estimated. This is what makes it possible to later compare
"predicted historical probability" against "what actually happened" and
measure, with real evidence, whether the system's edge claims hold up. No
row is ever edited to make a past signal look better after the fact.

## Manual decision journal and theoretical vs. executable performance

A signal is a proposal, not an order — **this system has no automated
execution path and never will** (see README.md). What a person actually
does about a signal is recorded separately, in its own
`signal_decisions` table (`signals.decisions.record_decision`, the one
write path — a CLI today, `scripts/record_signal_decision.py`, since
there is no UI yet): `TOOK_TRADE`, `DID_NOT_TAKE`, or `ARRIVED_LATE`
(the last one assigned automatically whenever a `TOOK_TRADE` claim is
recorded after the signal's own `valid_until` has already passed — an
explicit `DID_NOT_TAKE` is never overridden this way).

`signals.performance` reports two aggregates over the same signal
history, always side by side, never blended into one number:

- **Theoretical** — every signal with a resolved outcome, as if filled
  exactly at `entry_price`/`generated_at` (what the backtest engine
  already computes for a strategy).
- **Executable** — filtered to `TOOK_TRADE` decisions only, using the
  decision's own recorded result/pnl when available, falling back to the
  theoretical read (and flagging that it did) only when the person never
  recorded their own outcome.

The gap between the two is the honest answer to "does this survive
becoming a real, manually-executed habit" — it only accumulates meaning
once the system has been running live for a while, which it is not yet.

## What this system will never show or say

- A fixed confidence number attached to a strategy that hasn't been
  registered as a hypothesis and tested per BACKTESTING.md.
- Words like "infalible", "garantizado", "bot ganador", "ganancia segura".
- "NO HAY SEÑAL" replaced by a lower-quality signal just to always have
  something to show — the default behavior is to show few, high-quality
  signals, not to always be talking.
