import pytest

from otc_research.db.session import get_engine, get_session_factory, init_db


@pytest.fixture()
def session():
    """In-memory SQLite session with schema created, fresh per test."""
    engine = get_engine("sqlite:///:memory:")
    init_db(engine)
    factory = get_session_factory(engine)
    s = factory()
    try:
        yield s
    finally:
        s.close()
