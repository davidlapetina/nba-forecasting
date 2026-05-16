from __future__ import annotations

import pandas as pd

from nba_predictor.ingest.ingest_schedule import normalize_schedule_rows


def test_schedule_rows_keep_future_games_without_scores() -> None:
    frame = pd.DataFrame(
        [
            {
                "gameId": "0022500001",
                "gameDate": "2026-01-15",
                "homeTeam_teamId": 1,
                "awayTeam_teamId": 2,
                "homeTeam_score": None,
                "awayTeam_score": None,
                "gameSubtype": "Regular Season",
            }
        ]
    )
    row = normalize_schedule_rows(frame, "2025-26")[0]
    assert row["game_id"] == "0022500001"
    assert row["home_score"] is None
    assert row["away_score"] is None
    assert row["home_team_win"] is None

