# Manual-live signal test — ML_1M5M candidate #11

**This is a signal generator + result logger for manual/paper trading. It
never places an order. It never trains or modifies the strategy. A
person always decides and executes manually.**

This document covers the manual-live phase that follows
`ML1M5M_CANDIDATE11_FORWARD_TEST_REPORT.md`. That report's central finding
was a sharp reversal between the `delay0` and `delay1` scenarios on new
data — this phase exists to see whether that pattern (or any other)
continues to hold as further new market data arrives, using the exact
same frozen model and rules, evaluated live instead of after the fact.

## 1. What is frozen (do not change while an observation period is running)

| | |
|---|---|
| Model | `GradientBoostingClassifier`, the exact object frozen for the forward test |
| Model file | `data/forward_test_models/ml1m5m_candidate11_gradient_boosting_frozen.joblib` |
| Model SHA-256 | `eaa78365572b6e16f1f716d217951e064cb4dd5c8746e58cf9222091ec2296c7` |
| Strategy | Baseline only — `P(CALL) > 0.50` (no P>0.65, no CCI filter in this phase) |
| Asset / timeframe | EUR_USD, 1-minute candles |
| Expiry | 300 seconds (5 candles) |
| Delay | `entry_delay_candles = 1` — the project's standing "delay1" scenario: signal candle closes, entry is the candle **after next** (see §4) |
| Direction | CALL only (this model never proposes PUT) |
| Filters | None — every `P(CALL) > 0.50` candle produces a signal, no post-hoc filtering |

All of this lives in `src/otc_research/live/candidate.py`. Loading the
model re-verifies its SHA-256 against the recorded constant every time
(`load_frozen_model_payload`) and refuses to run if the file on disk has
changed at all.

## 2. What "delay=1" means here (read before interpreting a signal)

The evaluation happens once the **signal candle** closes. With
`entry_delay_candles = 1` (this project's standard, reproduced exactly —
see `backtest/execution.py` / `backtest/simulator.py`), the entry is not
the very next candle but the one after that. Concretely, for a signal
candle that closes at `10:36:00`:

```
Signal candle: 10:36   (closes at 10:36:00 -- this is when the decision is made)
Entry candle:  10:37   (its OPEN is the entry price)
Expiry: entry + 5 minutes (unaffected by delay)
```

Delay never changes the expiry duration — only how many candles pass
between the signal and the entry.

## 3. Starting the live monitor

### Prerequisite

A Twelve Data API key (free, signup-only, no broker account needed):
https://twelvedata.com — then set it as an environment variable:

```bash
export TWELVEDATA_API_KEY="your-key-here"
```

Optional, only if you want Telegram alerts:

```bash
export TELEGRAM_BOT_TOKEN="..."   # from @BotFather
export TELEGRAM_CHAT_ID="..."     # your chat id (message the bot once, then
                                   # GET https://api.telegram.org/bot<token>/getUpdates)
```

No file needs editing for normal use — everything is environment
variables and command-line flags. `config/config.yaml` only needs
touching if you want to point at a different database file
(`database.url`, default `sqlite:///data/otc_research.db`).

### Commands

Register the hypothesis code once (idempotent, already done in this repo's DB, needed again only on a fresh database):

```bash
python scripts/seed_hypotheses.py
```

**Before ever pointing this at the live market**, verify the infrastructure reproduces the published forward-test result exactly:

```bash
python scripts/replay_live_candidate11.py
```

This replays the already-ingested, already-analyzed forward-test block
(2026-09-19→22) through the SAME evaluation code the live monitor uses,
and refuses to proceed silently if the aggregate WIN/LOSS/VOID numbers
don't match `ML1M5M_CANDIDATE11_FORWARD_TEST_REPORT.md` exactly — this
is what proves "live infra == backtest engine" rather than just asserting it.

Run one poll cycle (fetch, evaluate, notify, resolve) and exit:

```bash
python scripts/run_live_signal_monitor.py --once
```

Run continuously (Ctrl+C to stop):

```bash
python scripts/run_live_signal_monitor.py --loop --poll-interval-seconds 20
```

With extra alert channels:

```bash
python scripts/run_live_signal_monitor.py --loop --notify console,sound,telegram
```

### Stopping

`--loop` mode: Ctrl+C. The current poll cycle finishes, then the process
exits — nothing is left running in the background, and no state is lost
(everything already committed to the database stays there; the next run
resumes from the last evaluated candle automatically, see §7).

### A note on "real-time" in this environment

`run_live_signal_monitor.py` is a self-contained script — it needs only a
Twelve Data API key, no connection to any Claude session. Run it on any
machine that can stay on (your own computer, a small VPS, etc.) for genuinely
continuous monitoring. It is not something this cloud coding session keeps
running for you across turns, since a sub-hourly always-on process isn't
something this sandboxed environment can guarantee.

## 4. What a signal looks like

```
━━━━━━━━━━━━━━━━━━━━━━
🚨 SEÑAL CALL (SIGNAL-2026-000123)

EUR/USD
Timeframe: 1 minuto

Señal generada: 2026-09-24 10:36:00 UTC
Entrada prevista: 2026-09-24 10:37:00 UTC
Expiry: 5 minutos

Probabilidad CALL: 67.2%

Modelo: ML_1M5M #11
Threshold: >50%
Delay: 1 candle

Signal candle: 10:36 (close 10:36)
Entry candle: 10:37

━━━━━━━━━━━━━━━━━━━━━━
ESTO NO ES UNA ORDEN AUTOMÁTICA.
El modelo está generando una señal CALL y debes decidir/hacer
manualmente la entrada.
Todos los timestamps en UTC.
━━━━━━━━━━━━━━━━━━━━━━
```

Every timestamp is UTC. `--notify console` (the default) prints this to
the terminal; `sound` additionally rings the terminal bell; `desktop`
sends an OS notification (Linux `notify-send` / macOS `osascript`);
`telegram` sends the same text via a Telegram bot. Multiple channels can
run together (`--notify console,sound,telegram`) — at least one must
succeed for the signal to be marked delivered.

## 5. Where everything is stored

SQLite database at `data/otc_research.db` (or wherever
`config/config.yaml`'s `database.url` points):

- **`live_evaluations`** — one row per closed candle evaluated for this
  candidate, whether it fired or not (`signal` = `CALL` / `NO_TRADE` /
  `DATA_ERROR`). This is the full per-minute decision log — table 6 of the
  original spec.
- **`signals`** — one row per CALL only, with `strategy_code =
  'ML1M5M_11_LIVE'`. `result` (WIN/LOSS/VOID, filled in once the entry/exit
  candles arrive) is the **theoretical result** — it is never touched by a
  manual decision.
- **`signal_decisions`** — your manual record: did you take the trade, and
  what actually happened. Kept entirely separate from `signals.result`.

Query them directly with any SQLite tool, or use the report script (§8).

## 6. Marking a manual entry

After you decide whether you took a signal (and, once it resolves,
what happened):

```bash
python scripts/record_signal_decision.py \
    --signal-id 123 \
    --decision TOOK_TRADE \
    --entry-price-actual 1.08421 \
    --payout-observed 0.82
```

`--decision` is one of `TOOK_TRADE`, `DID_NOT_TAKE`, `ARRIVED_LATE`. A
`TOOK_TRADE` claimed after the signal's entry window already closed is
automatically reclassified to `ARRIVED_LATE`. This never changes the
signal's own theoretical `result`/`pnl` — see the safety test
`test_manual_decision_never_mutates_theoretical_result`.

To later fill in your own WIN/LOSS/VOID and PnL for a trade you took, add
a note about why you skipped one, etc.: the underlying
`otc_research.signals.decisions.record_decision` accepts those directly;
extend `record_signal_decision.py`'s CLI flags if you want that from the
command line too — the DB column (`SignalDecision.result`, `.pnl`) is
already there and honored by the report.

## 7. Duplicate protection / resuming after a restart

Every evaluation is keyed uniquely by `(strategy_code, asset, timeframe,
candle_timestamp)` — the database enforces this, so a candle can never
be evaluated twice for this candidate, and a CALL candle can never
produce two `Signal` rows. On startup, the monitor resumes from
`MAX(candle_timestamp)` already in `live_evaluations` — it never
re-processes anything, and it never reaches back before
`live.candidate.LIVE_TEST_START_AFTER` (2026-09-22 21:02:01, one second
after the forward-test block already reported on).

## 8. Checking results / exporting history

```bash
python scripts/report_live_candidate11.py
```

Prints total evaluations, CALL/NO_TRADE/DATA_ERROR counts, the
**theoretical** result (every CALL the model generated: WIN/LOSS/VOID,
win rate, 95% CI, margin vs. the 54.05% break-even), and the **manual**
result (only what you actually executed: WIN/LOSS/skipped, win rate) —
kept side by side, never blended.

To export the full row-by-row history to CSV:

```bash
python scripts/report_live_candidate11.py --export live_candidate11_history.csv
```

Columns: `candle_timestamp, signal_time, probability_call, signal,
entry_time, expiry_time, signal_ref, entry_price, expiry_price,
theoretical_result, manual_decision, manual_result, notes`.

## 9. What this phase deliberately does NOT do

- No automatic order placement, ever — no broker/execution API is
  connected anywhere in this code.
- No re-tuning: the threshold, model, features, delay, and expiry are
  fixed for the whole observation period. If you want to test P>0.65 or
  the CCI filter live, that is a **separate, later** candidate — not a
  change to this one mid-observation.
- No retraining on new data: the model is loaded once from the frozen
  `.joblib` file and never refit (`otc_research.live.evaluation` never
  calls `.fit()` — see `test_model_is_never_fit_during_evaluation`).
- No silent data substitution: if candles can't be fetched or a gap is
  detected immediately before the candle being evaluated, the result is
  logged as `DATA_ERROR`, never a fabricated CALL/NO_TRADE.

## 10. Safety tests

`tests/test_live_candidate11.py` (16 tests) covers the user's full
pre-flight checklist: incomplete-candle handling, one-evaluation-per-candle
and no-duplicate-signal dedup, frozen-model hash verification, no training
during evaluation, threshold boundary behavior (`P>0.50` → CALL, `P<=0.50`
→ NO_TRADE), delay=1 and expiry arithmetic, theoretical-result parity with
`backtest.simulator.simulate`, manual decisions never mutating the
theoretical result, and historical (forward-test/TEST) data exclusion via
`LIVE_TEST_START_AFTER`. Run with:

```bash
python -m pytest tests/test_live_candidate11.py -v
```
