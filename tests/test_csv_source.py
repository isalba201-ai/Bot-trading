import pytest

from otc_research.data.sources.csv_source import CsvDataSource


def test_parses_valid_csv(tmp_path):
    csv_path = tmp_path / "candles.csv"
    csv_path.write_text(
        "timestamp,open,high,low,close\n"
        "2026-01-01T00:00:00Z,1.1000,1.1005,1.0998,1.1002\n"
        "2026-01-01T00:00:30Z,1.1002,1.1010,1.1001,1.1008\n"
    )

    source = CsvDataSource(csv_path)
    candles = list(source.fetch("EUR_USD", "1m"))

    assert len(candles) == 2
    assert candles[0].asset == "EUR_USD"
    assert candles[0].timeframe == "1m"
    assert candles[0].close == pytest.approx(1.1002)
    # strictly ascending order
    assert candles[0].timestamp < candles[1].timestamp
    assert candles[0].timestamp.tzinfo is not None


def test_missing_column_raises(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("timestamp,open,high,low\n2026-01-01T00:00:00Z,1,1,1\n")

    source = CsvDataSource(csv_path)
    with pytest.raises(ValueError, match="missing required column"):
        list(source.fetch("EUR_USD", "1m"))
