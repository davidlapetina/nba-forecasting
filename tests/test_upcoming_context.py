from nba_predictor.analytics.upcoming_context import latest_injury_report_url
from nba_predictor.jobs import operator_actions


def test_latest_injury_report_url_uses_last_pdf_link() -> None:
    html = """
    <a href="https://example.com/unrelated.pdf">unrelated</a>
    <a href="https://example.com/referee/injury/Injury-Report_2026-05-17_07_15AM.pdf">older</a>
    <a href="https://example.com/referee/injury/Injury-Report_2026-05-17_07_30AM.pdf">latest</a>
    <a href="https://example.com/another-unrelated.pdf">another unrelated</a>
    """

    assert latest_injury_report_url(html) == "https://example.com/referee/injury/Injury-Report_2026-05-17_07_30AM.pdf"


def test_run_upcoming_context_keeps_news_when_injury_refresh_fails(monkeypatch) -> None:
    monkeypatch.setattr(
        operator_actions,
        "refresh_upcoming_injury_context",
        lambda target_date: (_ for _ in ()).throw(RuntimeError("timeout")),
    )
    monkeypatch.setattr(operator_actions, "refresh_upcoming_news_context", lambda target_date: 3)

    assert operator_actions.run_upcoming_context(None) == {
        "injury_context_summaries": 0,
        "news_context_summaries": 3,
    }
