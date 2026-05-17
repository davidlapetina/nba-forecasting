from __future__ import annotations

from datetime import date

import pandas as pd

from nba_predictor.ingest.ingest_play_by_play import ingest_play_by_play_history, normalize_play_by_play_rows


def test_normalize_play_by_play_rows_supports_v3_headers() -> None:
    frame = pd.DataFrame(
        [
            {
                "gameId": "0022500001",
                "actionNumber": 7,
                "clock": "PT11M42.00S",
                "period": 1,
                "teamId": 1610612738,
                "teamTricode": "BOS",
                "personId": 1,
                "playerName": "Example Player",
                "playerNameI": "E. Player",
                "xLegacy": 8,
                "yLegacy": 14,
                "shotDistance": 12,
                "shotResult": "Made",
                "isFieldGoal": 1,
                "scoreHome": 2,
                "scoreAway": 0,
                "pointsTotal": 2,
                "location": "h",
                "description": "Example Player 12' Jump Shot",
                "actionType": "2pt",
                "subType": "Jump Shot",
                "videoAvailable": 0,
                "actionId": 1007,
            }
        ]
    )

    rows = normalize_play_by_play_rows(frame, "0022500001", "2025-26", date(2025, 10, 21))

    assert rows == [
        {
            "game_id": "0022500001",
            "season": "2025-26",
            "game_date": date(2025, 10, 21),
            "action_number": 7,
            "action_id": 1007,
            "period": 1,
            "clock": "PT11M42.00S",
            "team_id": 1610612738,
            "team_tricode": "BOS",
            "person_id": 1,
            "player_name": "Example Player",
            "player_name_i": "E. Player",
            "x_legacy": 8.0,
            "y_legacy": 14.0,
            "shot_distance": 12.0,
            "shot_result": "Made",
            "is_field_goal": True,
            "score_home": 2,
            "score_away": 0,
            "points_total": 2,
            "location": "h",
            "description": "Example Player 12' Jump Shot",
            "action_type": "2pt",
            "sub_type": "Jump Shot",
            "video_available": False,
        }
    ]


def test_normalize_play_by_play_rows_deduplicates_action_numbers() -> None:
    frame = pd.DataFrame(
        [
            {"actionNumber": 7, "description": "first"},
            {"actionNumber": 7, "description": "latest"},
        ]
    )

    rows = normalize_play_by_play_rows(frame, "g1", "2025-26", date(2025, 10, 21))

    assert len(rows) == 1
    assert rows[0]["description"] == "latest"


def test_normalize_play_by_play_rows_treats_blank_scores_as_missing() -> None:
    frame = pd.DataFrame([{"actionNumber": 7, "scoreHome": "", "scoreAway": ""}])

    row = normalize_play_by_play_rows(frame, "g1", "2025-26", date(2025, 10, 21))[0]

    assert row["score_home"] is None
    assert row["score_away"] is None


def test_play_by_play_history_starts_at_available_era(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "nba_predictor.ingest.ingest_play_by_play.ingest_play_by_play",
        lambda season, include_existing=False: calls.append(season) or 1,
    )

    summary = ingest_play_by_play_history("1946-47", "1997-98")

    assert calls == ["1996-97", "1997-98"]
    assert summary == {"1996-97": 1, "1997-98": 1}
