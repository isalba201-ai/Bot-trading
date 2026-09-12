import pytest

from otc_research.config import MarketConfig
from otc_research.data.sources.factory import build_live_data_source
from otc_research.data.sources.oanda_source import OandaDataSource
from otc_research.data.sources.twelvedata_source import TwelveDataSource


def _market_config(provider: str) -> MarketConfig:
    return MarketConfig(
        data_provider=provider,
        oanda_environment="practice",
        pairs=["EUR_USD"],
        timeframes=["1m"],
    )


def test_builds_twelvedata_source(monkeypatch):
    monkeypatch.setenv("TWELVEDATA_API_KEY", "fake-key")
    source = build_live_data_source(_market_config("twelvedata"))
    assert isinstance(source, TwelveDataSource)


def test_builds_oanda_source(monkeypatch):
    monkeypatch.setenv("OANDA_API_TOKEN", "fake-token")
    source = build_live_data_source(_market_config("oanda"))
    assert isinstance(source, OandaDataSource)


def test_csv_is_not_a_live_source():
    with pytest.raises(ValueError, match="not a live data source"):
        build_live_data_source(_market_config("csv"))


def test_unknown_provider_rejected_by_market_config():
    with pytest.raises(ValueError, match="Unknown market.data_provider"):
        _market_config("something_else")
