#!/usr/bin/env python
"""Create the database schema (idempotent: never drops or alters existing tables)."""

from __future__ import annotations

import argparse

from otc_research.config import load_config
from otc_research.db.session import get_engine, init_db


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    engine = get_engine(config.database_url)
    init_db(engine)
    print(f"Database ready at {config.database_url}")


if __name__ == "__main__":
    main()
