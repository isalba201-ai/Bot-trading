# Architecture

## Design principle

Each layer only depends on the layer(s) below it, and each layer's output is
persisted and auditable. This is deliberate: a later phase (e.g. the signal
engine) must never be able to quietly reach back and change what an earlier
phase (e.g. raw candles) recorded. It also makes it possible to test each
layer in isolation.

```
┌───────────────────────────────────────────────────────────────┐
│ 12. Machine learning (gated: only escalated to if statistical │
│     discovery finds an FDR-significant condition)              │
├───────────────────────────────────────────────────────────────┤
│ 11. Dashboard (read-only view over the DB) -- not built        │
├───────────────────────────────────────────────────────────────┤
│ 10. Notifications (console today; other channels on request)   │
├───────────────────────────────────────────────────────────────┤
│  9. Signal lifecycle + manual decision journal + theoretical   │
│     vs. executable performance -- "BUSCAR SEÑAL" UI not built  │
├───────────────────────────────────────────────────────────────┤
│  Statistical discovery pivot: baseline stats -> regime         │
│  analysis -> FDR-corrected interaction search -> candidacy     │
│  (4-gate accept/reject bar, reuses layers 4-7 unchanged)        │
├───────────────────────────────────────────────────────────────┤
│  8. Monte Carlo -- not built                                    │
├───────────────────────────────────────────────────────────────┤
│  7. Walk-forward analysis                                       │
├───────────────────────────────────────────────────────────────┤
│  6. Robustness / parameter-sensitivity sweeps                   │
├───────────────────────────────────────────────────────────────┤
│  5. Baseline strategies (H1-H20)                                 │
├───────────────────────────────────────────────────────────────┤
│  4. Backtesting engine                                          │
├───────────────────────────────────────────────────────────────┤
│  3. Feature engineering (point-in-time only)                   │
├───────────────────────────────────────────────────────────────┤
│  2. Data validation                                            │
├───────────────────────────────────────────────────────────────┤
│  1. Data storage + ingestion                                   │
└───────────────────────────────────────────────────────────────┘
```

The statistical-discovery pivot sits beside layers 5-7, not above them:
it is a second, systematic way of proposing a condition (instead of a
hand-designed H1-H20 strategy), but every condition it proposes is
judged by the SAME layer-4/6/7 machinery, unchanged — see
`research/candidacy.py` below.

## Repository layout

```
src/otc_research/
  config.py              # loads config/config.yaml, validates it
  db/
    models.py            # SQLAlchemy models: Candle, DataQualityIssue,
                          # Feature, Hypothesis, BacktestRun, Signal,
                          # SignalDecision, ConditionTrial
    session.py           # engine/session factory, init_db()
  data/
    sources/
      base.py            # DataSource ABC + RawCandle
      csv_source.py       # user-provided CSV (any historical export)
      twelvedata_source.py  # real Forex data via Twelve Data (default; API-key signup only)
      oanda_source.py     # real Forex market data via OANDA's v20 API (needs an approved broker account)
      factory.py          # builds the configured live source from MarketConfig
    validation.py         # pure functions, one per data-quality rule
    ingestion.py          # orchestrates source -> validation -> storage
  features/
    indicators.py         # pure, point-in-time-safe indicator functions
    engine.py              # compute_features(df) -> versioned feature set
    pipeline.py            # orchestrates candles -> engine -> Feature rows
  backtest/
    strategy.py            # Strategy protocol every Phase 5 hypothesis implements
    splits.py               # strict temporal train/validation/test split
    execution.py            # optimistic/realistic/pessimistic execution scenarios
    simulator.py            # walks candles+features -> simulated trades
    metrics.py               # Wilson CI win rate, expectancy — never a bare point estimate
    engine.py                # orchestrates candles+features -> split -> simulate -> BacktestRun
    robustness.py             # parameter/expiry/time-window sweeps + fragility verdict
    walkforward.py             # sliding train/test folds + mean/dispersion/worst-fold summary
  strategies/
    h1_streak.py .. h20_donchian_atr.py  # one Strategy implementation per hypothesis (STRATEGIES.md)
    __init__.py               # BASELINE_STRATEGIES registry (H9 excluded, needs explicit params)
  research/                 # statistical-discovery pivot (see STRATEGIES.md's
                             # "Pivot" section) -- reuses backtest/* unchanged
    dataset.py               # point-in-time feature+target table builder (per pair/timeframe/horizon)
    regimes.py                # trend-strength x volatility-state regime classification
    baseline.py                # unconditional/conditional/regime win-rate stats, Wilson CI
    discovery.py                # systematic 2-3-way condition search + Benjamini-Hochberg FDR correction
    candidacy.py                 # the fixed 4-gate accept/reject bar (sample size/margin ->
                                  # robustness -> walk-forward -> TEST, touched once)
    condition_strategy.py         # adapts a discovered Condition into the Strategy protocol
    models.py                     # logistic regression -> random forest -> gradient boosting,
                                   # gated behind discovery finding something significant (no neural nets)
  signals/                   # Phase 9: manual-review signal lifecycle -- never places a trade
    service.py                 # create_signal / send_notification / refresh_expired_signals
    confidence.py                # fixed ALTA/MEDIA/BAJA rule
    decisions.py                  # record_decision (TOOK_TRADE/DID_NOT_TAKE/ARRIVED_LATE)
    performance.py                 # theoretical vs. executable performance comparison
    formatting.py                   # format_signal_message() (one template, every provider)
    latency.py                       # latency_to_generate / latency_to_notify (on demand, not stored)
  notifications/              # Phase 9: delivery channels
    base.py                     # NotificationProvider ABC
    console_provider.py          # ConsoleNotificationProvider (the only channel wired up so far)
  utils/
    logging.py            # shared logger (stderr + logs/otc_research.log)
    timeframes.py          # timeframe string <-> seconds (dependency-free, avoids import cycles)
scripts/
  init_db.py             # create schema
  import_csv.py          # CLI to ingest a CSV file
  fetch_market_data.py    # CLI to fetch/backfill real candles from the configured provider
  compute_features.py     # CLI to compute point-in-time features from stored candles
  run_backtest.py         # CLI to run one strategy through the Phase 4 backtesting engine
  run_baseline_backtests.py # CLI to run every Phase 5 baseline strategy at once (train/validation only)
  run_robustness_sweep.py  # CLI for Phase 6 parameter/expiry/time-window sweeps
  run_walk_forward.py      # CLI for Phase 7 walk-forward analysis
  run_research_pipeline.py  # CLI: the full statistical-discovery pipeline against real data,
                             # reports the A/B/C/D scientific conclusion (see STRATEGIES.md)
  replay_signals_historical.py # CLI: proves the Phase 9 signal lifecycle against already-ingested
                                # candles in timestamp order -- explicitly not a live run
  record_signal_decision.py  # CLI: the one write path for a manual TOOK_TRADE/DID_NOT_TAKE/ARRIVED_LATE decision
  seed_hypotheses.py      # registers the hypotheses in STRATEGIES.md
config/
  config.yaml            # risk limits, signal thresholds, pairs, DB url
tests/                    # one test module per src module
```

Everything under `src/otc_research/` is a plain Python package (`pip install
-e .`), so later phases (features, backtesting, ...) are added as sibling
subpackages (`otc_research/features/`, `otc_research/backtest/`, ...) without
restructuring what already exists.

## Data flow (current phases)

1. Real candles come from `TwelveDataSource` (default — Twelve Data API,
   key-only signup, token in `TWELVEDATA_API_KEY`), `OandaDataSource`
   (OANDA v20 API, requires an approved broker demo/live account, token in
   `OANDA_API_TOKEN`), or `CsvDataSource` (any historical export you
   provide) — `data.sources.factory.build_live_data_source()` picks
   between the two live sources based on `config.yaml`'s
   `market.data_provider`. All yield the same `RawCandle` objects, sorted
   ascending, always timezone-aware UTC, and neither live source ever
   returns a candle that hasn't actually closed yet.
2. `ingestion.ingest()` runs every validator in `validation.py` against the
   full batch, logs every finding as a `DataQualityIssue` row, and inserts
   only candles that are not duplicates, not out-of-order, and not
   internally impossible. Gaps and suspicious moves are logged but do not
   block insertion of the surrounding valid candles.
3. Candles are stored once, keyed by `(asset, timeframe, timestamp)`, with a
   `source` field recording provenance and an `is_synthetic_test_data` flag
   that must be `False` for anything used in real research.
4. `features.pipeline.compute_and_store()` reads `Candle` rows for an
   asset/timeframe (never a live source directly), runs
   `features.engine.compute_features()`, and writes one `Feature` row per
   `(asset, timeframe, timestamp, feature_set_version, name)` — skipping
   rows that already exist, and never storing a value for a timestamp that
   doesn't yet have enough history for that indicator's lookback. See
   FEATURES.md for the exact feature set and its point-in-time guarantee.
5. `backtest.engine.run_backtest()` reads `Candle` + `Feature` rows for an
   asset/timeframe (never recomputing indicators itself), applies the
   strict temporal split (`backtest.splits`), and runs a `Strategy`
   (`backtest.strategy`) through the simulator (`backtest.simulator`)
   under all three execution-realism scenarios (`backtest.execution`).
   Results are summarized with a 95% Wilson confidence interval
   (`backtest.metrics`) and persisted as one `BacktestRun` row per
   scenario — see BACKTESTING.md for the full methodology this
   implements, and what's still Phase 6-8.
6. `otc_research.strategies` implements H1-H10 (STRATEGIES.md) against
   this interface — each one a pure function of a point-in-time feature
   dict (plus the current candle's raw OHLC) to `"CALL"`/`"PUT"`/`None`,
   with every threshold a constructor parameter so Phase 6's sensitivity
   sweeps can vary it. `scripts/run_baseline_backtests.py` runs all of
   them (except H9, which needs explicit parameters) against train/
   validation data and advances each `Hypothesis.status` from
   `registered` to `tested` — never a claim of edge, just that it has
   actually been run through the engine.
7. `backtest.robustness.run_parameter_sweep()` runs the same strategy
   shape across a grid of constructor parameters and, optionally, several
   time windows (`run_backtest`'s `start`/`end`, which restrict the
   candle universe before the train/validation/test split is computed).
   `evaluate_robustness()` then checks whether a Wilson-CI edge holds up
   across most of the sufficiently-sampled points or only a lucky few —
   `scripts/run_robustness_sweep.py` is the CLI. This already found, on
   real EUR/USD data, that H4's apparent edge does not survive a
   time-period sweep — see STRATEGIES.md.
8. `backtest.walkforward.generate_folds()` slides a (train_span,
   test_span) window pair across a strategy's whole ingested history;
   `run_walk_forward()` evaluates each fold's test window independently
   via `run_backtest(split="walk_forward")` — a mode that skips the
   internal 60/20/20 split and evaluates the given window as one block,
   persisting each fold's `BacktestRun` row with its `fold_index`.
   `summarize_walk_forward()` reports mean, dispersion, and worst-fold
   win rate across folds, never just the best-looking one.
   `scripts/run_walk_forward.py` is the CLI. Run against H4 on real
   EUR/USD 1h data (12 folds, 14 days each): mean win rate 56.9%
   optimistic / 47.2% realistic, and under the realistic scenario 0 of 12
   folds showed a statistically credible edge — see STRATEGIES.md.

9. **Statistical discovery pivot** (`otc_research.research`, see
   STRATEGIES.md's "Pivot" section): after H1-H20 (hand-designed
   hypotheses) showed no credible edge at any timeframe,
   `research.dataset.build_dataset()` builds a point-in-time
   feature+target table per pair/timeframe from the same `Candle`/
   `Feature` rows (never recomputed), `research.baseline` computes
   unconditional/conditional/regime win-rate statistics, and
   `research.discovery.run_discovery()` runs a systematic 2-3-way
   condition search over TRAIN only — every combination tried is logged
   to `ConditionTrial` before Benjamini-Hochberg FDR correction decides
   which ones are even eligible to be considered. A condition that
   survives correction is wrapped by `research.condition_strategy.
   ConditionStrategy` into the SAME `Strategy` protocol H1-H20 used, so
   `research.candidacy.evaluate_candidacy()` can re-judge it through the
   unmodified layer-4/6/7 machinery (sample size/margin → robustness
   sweep → walk-forward → TEST, touched at most once) — a discovered
   condition gets no lighter-weight path than a hand-designed one.
   `research.models` (logistic regression → random forest → gradient
   boosting; no neural networks) is only invoked if discovery finds
   something FDR-significant. `scripts/run_research_pipeline.py` runs
   the whole thing and reports one of four scientific conclusions
   (A: robust edge / B: promising but insufficient / C: edge vanishes
   out-of-sample / D: no evidence) — the first real run found **B**, see
   STRATEGIES.md for the full result and why the naive discovery target
   and the realistic-execution simulator disagreed so sharply.
10. **Signal lifecycle** (`otc_research.signals`, `otc_research.
    notifications` — Phase 9, SIGNAL_ENGINE.md): a strategy or model's
    decision becomes a persisted `Signal` row
    (`signals.service.create_signal`, which computes break-even win
    rate, margin, expectancy, and a fixed ALTA/MEDIA/BAJA confidence
    label once, never left for a template to re-derive), delivered
    through a `NotificationProvider` (only `ConsoleNotificationProvider`
    exists today) via `signals.service.send_notification` — which
    refuses to send past the signal's own entry window rather than
    notifying late. A person's real, manual decision
    (`signals.decisions.record_decision`) is recorded separately from
    the signal itself, and `signals.performance` reports theoretical
    (every signal) vs. executable (`TOOK_TRADE` only) performance side
    by side. **This layer never places a trade and has no live-execution
    path** — it has only been run against a historical replay
    (`scripts/replay_signals_historical.py`), never a live feed, both
    because the live poller doesn't exist yet and because Step 7 found
    no condition worth watching live in the first place.

Everything past this point (Monte Carlo, the dashboard, a live data
poller, and the "BUSCAR SEÑAL" UI itself) does not exist yet and must be
built strictly on top of validated `Feature` rows and this engine — never
by recomputing indicators ad hoc or reading candles directly, so that
every strategy sees the same, auditable numbers and validation can't be
silently bypassed. The eventual "BUSCAR SEÑAL" UI is a thin layer that
triggers this same fetch → validate → store → feature → strategy-
evaluation → signal-lifecycle path on demand, then either shows a signal
or "NO HAY SEÑAL" — see SIGNAL_ENGINE.md. It never places an order.

## Why not Pocket Option OTC

Pocket Option does not publish an official API for OTC candle data, and
automating anything against it — even just reading your own logged-in
session — risks account detection and closure. This project analyzes real
Forex market data instead (see DATA.md), which does carry one honest
caveat: an edge found on real Forex data is not automatically valid on
Pocket Option's OTC synthetic instrument, since the two are different
data-generating processes.
