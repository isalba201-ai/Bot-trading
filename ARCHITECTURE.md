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
│ 10. Paper trading (simulated fills, no real money)             │
├───────────────────────────────────────────────────────────────┤
│  9. Signal engine (A+/A/B/NO_TRADE tiers, quality filter)      │
├───────────────────────────────────────────────────────────────┤
│  6-8. Robustness, walk-forward, Monte Carlo                   │
├───────────────────────────────────────────────────────────────┤
│  4-5. Backtesting engine + baseline strategies                │
├───────────────────────────────────────────────────────────────┤
│  3. Feature engineering (point-in-time only)                  │
├───────────────────────────────────────────────────────────────┤
│  2. Data validation (this phase)                               │
├───────────────────────────────────────────────────────────────┤
│  1. Data storage + ingestion (this phase)                      │
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
      csv_source.py       # the only implemented source: user-provided CSV
    validation.py         # pure functions, one per data-quality rule
    ingestion.py          # orchestrates source -> validation -> storage
  utils/
    logging.py            # shared logger (stderr + logs/otc_research.log)
scripts/
  init_db.py             # create schema
  import_csv.py          # CLI to ingest a CSV file
config/
  config.yaml            # risk limits, signal thresholds, DB url
tests/                    # one test module per src module
```

Everything under `src/otc_research/` is a plain Python package (`pip install
-e .`), so later phases (features, backtesting, ...) are added as sibling
subpackages (`otc_research/features/`, `otc_research/backtest/`, ...) without
restructuring what already exists.

## Data flow (current phases)

1. You obtain a CSV file of real candle data yourself (see DATA.md for why
   this is the only supported path today).
2. `CsvDataSource` parses it into `RawCandle` objects, sorted ascending,
   always timezone-aware UTC.
3. `ingestion.ingest()` runs every validator in `validation.py` against the
   full batch, logs every finding as a `DataQualityIssue` row, and inserts
   only candles that are not duplicates, not out-of-order, and not
   internally impossible. Gaps and suspicious moves are logged but do not
   block insertion of the surrounding valid candles.
4. Candles are stored once, keyed by `(asset, timeframe, timestamp)`, with a
   `source` field recording provenance and an `is_synthetic_test_data` flag
   that must be `False` for anything used in real research.

Everything past this point (features, backtesting, signals) does not exist
yet and must be built strictly on top of validated `Candle` rows — never by
reading files directly, so that validation can't be silently bypassed.

## Why no live Pocket Option connector

Pocket Option does not publish an official API for OTC candle data. Building
a scraper or automating login against the platform would mean evading its
own controls, which this project will not do (also see project brief,
section 22, and the top-level README). The `DataSource` abstraction exists
so that if you obtain data through a legitimate, authorized channel later,
it plugs in without changing anything downstream.
