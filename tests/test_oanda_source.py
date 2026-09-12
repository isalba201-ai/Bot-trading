"""OandaDataSource tests mock the HTTP layer entirely — no real network
call is made and no real OANDA account/token is needed to run these."""

from unittest.mock import MagicMock, patch

import pytest

from otc_research.data.sources.oanda_source import OandaDataSource


def _fake_response(candles):
    resp = MagicMock()
    resp.json.return_value = {"candles": candles}
    resp.raise_for_status.return_value = None
    return resp


def test_missing_token_raises(monkeypatch):
    monkeypatch.delenv("OANDA_API_TOKEN", raising=False)
    with pytest.raises(ValueError, match="API token required"):
        OandaDataSource()


def test_unknown_environment_raises():
    with pytest.raises(ValueError, match="Unknown OANDA environment"):
        OandaDataSource(api_token="fake-token", environment="bogus")


def test_unsupported_timeframe_raises():
    source = OandaDataSource(api_token="fake-token")
    with pytest.raises(ValueError, match="does not support timeframe"):
        list(source.fetch("EUR_USD", "30s"))


def test_fetch_skips_incomplete_candle():
    candles_payload = [
        {
            "complete": True,
            "time": "2026-01-01T00:00:00.000000000Z",
            "mid": {"o": "1.1000", "h": "1.1010", "l": "1.0990", "c": "1.1005"},
        },
        {
            "complete": False,  # still forming — must never be returned
            "time": "2026-01-01T00:01:00.000000000Z",
            "mid": {"o": "1.1005", "h": "1.1006", "l": "1.1004", "c": "1.1005"},
        },
    ]
    source = OandaDataSource(api_token="fake-token")

    with patch("otc_research.data.sources.oanda_source.requests.get") as mock_get:
        mock_get.return_value = _fake_response(candles_payload)
        result = list(source.fetch("EUR_USD", "1m", count=10))

    assert len(result) == 1
    assert result[0].close == pytest.approx(1.1005)
    assert result[0].asset == "EUR_USD"
    assert result[0].timeframe == "1m"


def test_fetch_uses_practice_host_and_bearer_token():
    source = OandaDataSource(api_token="secret-token", environment="practice")

    with patch("otc_research.data.sources.oanda_source.requests.get") as mock_get:
        mock_get.return_value = _fake_response([])
        list(source.fetch("EUR_USD", "5m", count=100))

    called_url = mock_get.call_args.args[0]
    called_headers = mock_get.call_args.kwargs["headers"]
    called_params = mock_get.call_args.kwargs["params"]

    assert called_url == "https://api-fxpractice.oanda.com/v3/instruments/EUR_USD/candles"
    assert called_headers["Authorization"] == "Bearer secret-token"
    assert called_params["granularity"] == "M5"
    assert called_params["count"] == 100


def test_fetch_with_start_end_range_sets_from_to():
    import datetime as dt

    source = OandaDataSource(api_token="secret-token")
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)

    with patch("otc_research.data.sources.oanda_source.requests.get") as mock_get:
        mock_get.return_value = _fake_response([])
        list(source.fetch("EUR_USD", "1m", start=start, end=end))

    called_params = mock_get.call_args.kwargs["params"]
    assert "from" in called_params and "to" in called_params
    assert "count" not in called_params


def test_count_is_capped_at_max_per_request():
    source = OandaDataSource(api_token="secret-token")

    with patch("otc_research.data.sources.oanda_source.requests.get") as mock_get:
        mock_get.return_value = _fake_response([])
        list(source.fetch("EUR_USD", "1m", count=999_999))

    called_params = mock_get.call_args.kwargs["params"]
    assert called_params["count"] == 5000
