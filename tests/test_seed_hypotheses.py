from otc_research.db.models import Hypothesis


def test_seeding_is_idempotent(session, monkeypatch):
    from scripts import seed_hypotheses

    # Reuse the test session's engine/session instead of the real config file.
    monkeypatch.setattr(
        seed_hypotheses, "get_session_factory", lambda engine: (lambda: session)
    )
    monkeypatch.setattr(seed_hypotheses, "init_db", lambda engine: None)
    monkeypatch.setattr(seed_hypotheses, "get_engine", lambda url: None)

    seed_hypotheses.main()
    count_after_first_run = session.query(Hypothesis).count()
    assert count_after_first_run == len(seed_hypotheses.INITIAL_HYPOTHESES)

    seed_hypotheses.main()
    count_after_second_run = session.query(Hypothesis).count()
    assert count_after_second_run == count_after_first_run
