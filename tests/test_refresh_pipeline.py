from __future__ import annotations

from datetime import date, datetime

from nba_predictor.jobs.refresh_pipeline import season_for_date, seasons_touched_since


def test_season_for_date_uses_july_boundary() -> None:
    assert season_for_date(date(2026, 5, 16)) == "2025-26"
    assert season_for_date(date(2026, 10, 1)) == "2026-27"


def test_seasons_touched_since_catches_up_across_boundary() -> None:
    seasons = seasons_touched_since(datetime(2026, 5, 16), date(2026, 10, 2))
    assert seasons == ["2025-26", "2026-27"]
