"""TwelveDataSource tests mock the HTTP layer entirely — no real network
call is made and no real Twelve Data API key is needed to run these."""

import datetime as dt
from unittest.mock import MagicMock, patch

import pytest

from otc_research.data.sources.twelvedata_source import TwelveDataSource


def _fake_response(values, status="ok"):
    resp = MagicMock()
    resp.json.return_value = {"status": status, "values": values}
    resp.raise_for_status.return_value = None
    return resp


def test_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("TWELVEDATA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key required"):
        TwelveDataSource()


def test_unsupported_timeframe_raises():
    source = TwelveDataSource(api_key="fake-key")
    with pytest.raises(ValueError, match="does not support timeframe"):
        list(source.fetch("EUR_USD", "30s"))


def test_underscore_asset_converted_to_slash_symbol():
    source = TwelveDataSource(api_key="fake-key")
    with patch("otc_research.data.sources.twelvedata_source.requests.get") as mock_get:
        mock_get.return_value = _fake_response([])
        list(source.fetch("EUR_USD", "1m", count=10))

    called_params = mock_get.call_args.kwargs["params"]
    assert called_params["symbol"] == "EUR/USD"
    assert called_params["interval"] == "1min"
    assert called_params["apikey"] == "fake-key"
    assert called_params["outputsize"] == 10


def test_api_error_status_raises_runtime_error():
    source = TwelveDataSource(api_key="fake-key")
    with patch("otc_research.data.sources.twelvedata_source.requests.get") as mock_get:
        mock_get.return_value = _fake_response(None, status="error")
        mock_get.return_value.json.return_value = {
            "status": "error",
            "message": "invalid symbol",
        }
        with pytest.raises(RuntimeError, match="invalid symbol"):
            list(source.fetch("EUR_USD", "1m"))


def test_still_forming_bar_is_dropped():
    now = dt.datetime.now(dt.timezone.utc)
    closed_bar_open = (now - dt.timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
    forming_bar_open = now.strftime("%Y-%m-%d %H:%M:%S")  # opened "now" -> not closed yet

    values = [
        {"datetime": closed_bar_open, "open": "1.10", "high": "1.11", "low": "1.09", "close": "1.105"},
        {"datetime": forming_bar_open, "open": "1.105", "high": "1.106", "low": "1.104", "close": "1.1055"},
    ]

    source = TwelveDataSource(api_key="fake-key")
    with patch("otc_research.data.sources.twelvedata_source.requests.get") as mock_get:
        mock_get.return_value = _fake_response(values)
        result = list(source.fetch("EUR_USD", "1m"))

    assert len(result) == 1
    assert result[0].close == pytest.approx(1.105)


def test_start_end_range_sets_start_date_end_date():
    source = TwelveDataSource(api_key="fake-key")
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    end = dt.datetime(2026, 1, 2, tzinfo=dt.timezone.utc)

    with patch("otc_research.data.sources.twelvedata_source.requests.get") as mock_get:
        mock_get.return_value = _fake_response([])
        list(source.fetch("EUR_USD", "5m", start=start, end=end))

    called_params = mock_get.call_args.kwargs["params"]
    assert called_params["start_date"] == "2026-01-01 00:00:00"
    assert called_params["end_date"] == "2026-01-02 00:00:00"
    assert "outputsize" not in called_params
