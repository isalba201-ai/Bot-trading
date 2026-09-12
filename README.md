# OTC Binary Research System

A research, backtesting and (eventually) signal-generation system for OTC
binary options. **This is a research project, not a profit machine.** Every
strategy is treated as an unproven hypothesis until it survives out-of-sample
testing, walk-forward analysis, Monte Carlo stress testing, and parameter
robustness checks. If it doesn't survive that, the answer is **NO EDGE** and
the system will not emit signals — see [STRATEGIES.md](STRATEGIES.md) and
[BACKTESTING.md](BACKTESTING.md).

## What this system explicitly does NOT do

- Does not assume any strategy is profitable.
- Does not use martingale or any loss-triggered position-size increase.
- Does not fabricate, interpolate, or "fill in" market data.
- Does not declare a strategy profitable because a single backtest looked good.
- Does not scrape, automate login to, or otherwise interact with Pocket
  Option's platform. There is no official OTC data API, so this project only
  ingests data you provide yourself (e.g. a CSV export) — see [DATA.md](DATA.md).
- Does not execute real trades. Automated live execution is not implemented.

## Project status

Building in phases, each gated by tests before moving to the next (see
project brief, section 31). **Currently complete: Phase 1 (architecture +
storage) and Phase 2 (data validation).** Everything else — feature
engineering, backtesting engine, walk-forward, Monte Carlo, signal engine,
paper trading, dashboard, ML — is **not built yet**.

| Phase | Scope | Status |
|---|---|---|
| 1 | Architecture + data storage | Done |
| 2 | Data validation | Done |
| 3 | Feature engineering | Not started |
| 4 | Backtesting engine | Not started |
| 5 | Baseline strategies | Not started |
| 6 | Robustness testing | Not started |
| 7 | Walk-forward analysis | Not started |
| 8 | Monte Carlo | Not started |
| 9 | Signal engine | Not started |
| 10 | Paper trading | Not started |
| 11 | Dashboard | Not started |
| 12 | Machine learning (only if it adds value) | Not started |

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Initialize the database

```bash
python scripts/init_db.py
```

This creates `data/otc_research.db` (SQLite) with the schema described in
[DATA.md](DATA.md). It never drops or alters existing tables.

## Import real candle data

There is no automated data feed. You provide a CSV file (columns:
`timestamp, open, high, low, close`) yourself:

```bash
python scripts/import_csv.py --asset EURUSD_OTC --timeframe 30s \
    --csv data/raw/eurusd_otc_30s.csv
```

The importer validates the data (duplicate/out-of-order timestamps, gaps,
impossible OHLC values, suspicious moves, source changes) and logs every
finding to the `data_quality_issues` table. It never invents a candle to
fill a gap. See [DATA.md](DATA.md) for details.

## Run tests

```bash
pip install -e ".[dev]"
pytest
```

## Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) — system layers and data flow
- [DATA.md](DATA.md) — schema, data acquisition constraints, validation rules
- [BACKTESTING.md](BACKTESTING.md) — methodology (train/val/test, walk-forward,
  Monte Carlo, multiple-testing controls) — implemented starting Phase 4
- [STRATEGIES.md](STRATEGIES.md) — hypotheses under investigation and their status
- [RISK_MANAGEMENT.md](RISK_MANAGEMENT.md) — position sizing and stop rules,
  no martingale
