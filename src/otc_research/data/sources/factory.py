"""Builds the configured live DataSource.

Centralized so every entry point (fetch scripts today, the "BUSCAR SEÑAL"
flow later) picks a provider the same way, instead of each duplicating the
if/elif and each source's own environment-variable conventions.
"""

from __future__ import annotations

from otc_research.config import MarketConfig
from otc_research.data.sources.base import DataSource
from otc_research.data.sources.oanda_source import OandaDataSource
from otc_research.data.sources.twelvedata_source import TwelveDataSource


def build_live_data_source(market_config: MarketConfig) -> DataSource:
    if market_config.data_provider == "twelvedata":
        return TwelveDataSource()
    if market_config.data_provider == "oanda":
        return OandaDataSource(environment=market_config.oanda_environment)
    raise ValueError(
        f"{market_config.data_provider!r} is not a live data source "
        f"(use CsvDataSource directly for 'csv')"
    )
