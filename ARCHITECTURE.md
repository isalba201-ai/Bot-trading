# Architecture

## Design principle

Each layer only depends on the layer(s) below it, and each layer's output is
persisted and auditable. This is deliberate: a later phase (e.g. the signal
engine) must never be able to quietly reach back and change what an earlier
phase (e.g. raw candles) recorded. It also makes it possible to test each
layer in isolation.

```
┌───────────────────────────────────────────────────────────────┐
│ 12. Machine learning (optional, only if it beats simple      │
│     baselines out-of-sample)                                  │
├───────────────────────────────────────────────────────────────┤
│ 11. Dashboard (read-only view over the DB)                    │
├───────────────────────────────────────────────────────────────┤
│ 10. Signal journal reconciliation (predicted vs. actual result)│
├───────────────────────────────────────────────────────────────┤
│  9. Signal engine ("BUSCAR SEÑAL" UI, A+/A/B/NO_TRADE tiers)   │
├───────────────────────────────────────────────────────────────┤
│  6-8. Robustness, walk-forward, Monte Carlo                   │
├───────────────────────────────────────────────────────────────┤
│  5. Baseline strategies (H1-H10, this phase)                    │
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

## Repository layout

```
src/otc_research/
  config.py              # loads config/config.yaml, validates it
  db/
    models.py            # SQLAlchemy models: Candle, DataQualityIssue,
                          # Feature, Hypothesis, Signal
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
  strategies/
    h1_streak.py .. h10_combined.py  # one Strategy implementation per hypothesis (STRATEGIES.md)
    __init__.py               # BASELINE_STRATEGIES registry (H9 excluded, needs explicit params)
  utils/
    logging.py            # shared logger (stderr + logs/otc_research.log)
    timeframes.py          # timeframe string <-> seconds (dependency-free, avoids import cycles)
scripts/
  init_db.py             # create schema
  import_csv.py          # CLI to ingest a CSV file
  fetch_market_data.py    # CLI to fetch/backfill real candles from the configured provider
  compute_features.py     # CLI to compute Phase 3 features from stored candles
  run_backtest.py         # CLI to run one strategy through the Phase 4 backtesting engine
  run_baseline_backtests.py # CLI to run every Phase 5 baseline strategy at once (train/validation only)
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

Everything past this point (robustness/walk-forward/Monte Carlo testing,
signals) does not exist yet and must be built strictly on top of
validated `Feature` rows and this engine — never by recomputing
indicators ad hoc or reading candles directly, so that every strategy
sees the same, auditable numbers and validation can't be silently
bypassed. The eventual "BUSCAR SEÑAL" UI (Phase 9) is a thin layer that
triggers this same fetch → validate → store → feature →
strategy-evaluation path on demand, then either shows a signal or "NO HAY
SEÑAL" — see SIGNAL_ENGINE.md. It never places an order.

## Why not Pocket Option OTC

Pocket Option does not publish an official API for OTC candle data, and
automating anything against it — even just reading your own logged-in
session — risks account detection and closure. This project analyzes real
Forex market data instead (see DATA.md), which does carry one honest
caveat: an edge found on real Forex data is not automatically valid on
Pocket Option's OTC synthetic instrument, since the two are different
data-generating processes.
