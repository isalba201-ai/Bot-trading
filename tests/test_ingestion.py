"""Ingestion tests use small hand-built CSVs of synthetic numbers, purely to
exercise the pipeline's logic (reject vs. accept, issue logging, no
fabrication of missing bars). This is not market data.
"""

from otc_research.data.ingestion import ingest
from otc_research.data.sources.csv_source import CsvDataSource
from otc_research.db.models import Candle, DataQualityIssue


def _write_csv(tmp_path, rows: str):
    path = tmp_path / "candles.csv"
    path.write_text("timestamp,open,high,low,close\n" + rows)
    return path


def test_clean_ingestion_inserts_all_candles(session, tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "2026-01-01T00:00:00Z,1.0,1.01,0.99,1.0\n"
        "2026-01-01T00:00:30Z,1.0,1.02,0.99,1.01\n",
    )
    report = ingest(session, CsvDataSource(csv_path), "TEST_FX", "1m", is_synthetic_test_data=True)

    assert report.candles_inserted == 2
    assert report.candles_skipped == 0
    assert session.query(Candle).count() == 2


def test_gap_is_logged_but_does_not_block_valid_candles(session, tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "2026-01-01T00:00:00Z,1.0,1.01,0.99,1.0\n"
        "2026-01-01T00:01:30Z,1.0,1.02,0.99,1.01\n",  # 90s later on a 1m tf -> gap
    )
    report = ingest(session, CsvDataSource(csv_path), "TEST_FX", "1m", is_synthetic_test_data=True)

    assert report.candles_inserted == 2  # both real candles are still stored
    assert session.query(Candle).count() == 2
    gap_issues = session.query(DataQualityIssue).filter_by(issue_type="gap").all()
    assert len(gap_issues) == 1
    # the pipeline must NOT have inserted a synthetic filler candle for the gap
    assert session.query(Candle).count() == 2


def test_impossible_candle_is_rejected_not_inserted(session, tmp_path):
    csv_path = _write_csv(
        tmp_path,
        "2026-01-01T00:00:00Z,1.0,1.01,0.99,1.0\n"
        "2026-01-01T00:00:30Z,1.0,0.5,0.9,1.0\n",  # high < low
    )
    report = ingest(session, CsvDataSource(csv_path), "TEST_FX", "1m", is_synthetic_test_data=True)

    assert report.candles_inserted == 1
    assert report.candles_skipped == 1
    assert session.query(Candle).count() == 1
    assert session.query(DataQualityIssue).filter_by(issue_type="impossible_value").count() == 1


def test_reingesting_same_data_does_not_duplicate(session, tmp_path):
    csv_path = _write_csv(tmp_path, "2026-01-01T00:00:00Z,1.0,1.01,0.99,1.0\n")
    source = CsvDataSource(csv_path)

    first = ingest(session, source, "TEST_FX", "1m", is_synthetic_test_data=True)
    second = ingest(session, source, "TEST_FX", "1m", is_synthetic_test_data=True)

    assert first.candles_inserted == 1
    assert second.candles_inserted == 0
    assert second.candles_skipped == 1
    assert session.query(Candle).count() == 1


def test_source_change_is_logged(session, tmp_path):
    csv_a = _write_csv(tmp_path, "2026-01-01T00:00:00Z,1.0,1.01,0.99,1.0\n")
    csv_b_path = tmp_path / "other_candles.csv"
    csv_b_path.write_text(
        "timestamp,open,high,low,close\n2026-01-01T00:00:30Z,1.0,1.01,0.99,1.0\n"
    )

    ingest(session, CsvDataSource(csv_a), "TEST_FX", "1m", is_synthetic_test_data=True)
    ingest(session, CsvDataSource(csv_b_path), "TEST_FX", "1m", is_synthetic_test_data=True)

    assert session.query(DataQualityIssue).filter_by(issue_type="source_change").count() == 1
