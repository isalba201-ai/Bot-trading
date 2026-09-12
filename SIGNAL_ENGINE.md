# Signal engine (planned — Phase 9, not built yet)

**Status: this document specifies the target design. None of the UI, the
"BUSCAR SEÑAL" flow, or signal generation exists in code yet.** It is
written now, before that code, so the rules — especially around what a
confidence number is allowed to mean — are fixed in advance.

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

Once the target horizon has elapsed, the actual outcome is fetched from real
market data and written back (`result`, `pnl`, `expiry_price`) — never
inferred, never estimated. This is what makes it possible to later compare
"predicted historical probability" against "what actually happened" and
measure, with real evidence, whether the system's edge claims hold up. No
row is ever edited to make a past signal look better after the fact.

## What this system will never show or say

- A fixed confidence number attached to a strategy that hasn't been
  registered as a hypothesis and tested per BACKTESTING.md.
- Words like "infalible", "garantizado", "bot ganador", "ganancia segura".
- "NO HAY SEÑAL" replaced by a lower-quality signal just to always have
  something to show — the default behavior is to show few, high-quality
  signals, not to always be talking.
