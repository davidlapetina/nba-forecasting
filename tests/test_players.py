from __future__ import annotations

import pandas as pd

from nba_predictor.ingest.ingest_players import normalize_player_game_rows
from nba_predictor.ingest.ingest_games import normalize_team_identity_rows, normalize_team_rows
from nba_predictor.ingest.ingest_rosters import normalize_coach_rows, normalize_roster_rows


def test_normalize_player_game_rows_supports_player_mode_headers() -> None:
    frame = pd.DataFrame(
        [
            {
                "PLAYER_ID": 1,
                "TEAM_ID": 10,
                "GAME_ID": "g1",
                "GAME_DATE": "2025-10-21",
                "MATCHUP": "BOS vs. NYK",
                "WL": "W",
                "MIN": 32,
                "PTS": 24,
                "REB": 8,
                "AST": 6,
                "PLUS_MINUS": 7,
            }
        ]
    )

    rows = normalize_player_game_rows(frame, "2025-26", "Regular Season")

    assert rows == [
        {
            "game_id": "g1",
            "player_id": 1,
            "team_id": 10,
            "game_date": pd.Timestamp("2025-10-21").date(),
            "season": "2025-26",
            "season_type": "Regular Season",
            "matchup": "BOS vs. NYK",
            "is_home": True,
            "won": True,
            "minutes": 32,
            "points": 24,
            "rebounds": 8,
            "assists": 6,
            "steals": None,
            "blocks": None,
            "turnovers": None,
            "field_goal_pct": None,
            "three_point_pct": None,
            "free_throw_pct": None,
            "plus_minus": 7.0,
        }
    ]


def test_normalize_player_game_rows_deduplicates_game_player_pairs() -> None:
    frame = pd.DataFrame(
        [
            {"PLAYER_ID": 1, "TEAM_ID": 10, "GAME_ID": "g1", "GAME_DATE": "2025-10-21", "PTS": 10},
            {"PLAYER_ID": 1, "TEAM_ID": 10, "GAME_ID": "g1", "GAME_DATE": "2025-10-21", "PTS": 12},
        ]
    )

    rows = normalize_player_game_rows(frame, "2025-26", "Regular Season")

    assert len(rows) == 1
    assert rows[0]["points"] == 12


def test_normalize_team_rows_learns_historical_teams_from_logs() -> None:
    frame = pd.DataFrame(
        [
            {"TEAM_ID": 1610610035, "TEAM_ABBREVIATION": "HUS", "TEAM_NAME": "Toronto Huskies"},
            {"TEAM_ID": 1610612752, "TEAM_ABBREVIATION": "NYK", "TEAM_NAME": "New York Knicks"},
        ]
    )

    rows = normalize_team_rows(frame)

    assert rows[0]["abbreviation"] == "HUS"
    assert rows[0]["full_name"] == "Toronto Huskies"


def test_normalize_team_identity_rows_preserves_season_label() -> None:
    frame = pd.DataFrame(
        [
            {"TEAM_ID": 1610612760, "TEAM_ABBREVIATION": "SEA", "TEAM_NAME": "Seattle SuperSonics"},
        ]
    )

    rows = normalize_team_identity_rows(frame, "1994-95")

    assert rows == [
        {
            "team_id": 1610612760,
            "season": "1994-95",
            "abbreviation": "SEA",
            "full_name": "Seattle SuperSonics",
        }
    ]


def test_normalize_roster_and_coach_rows() -> None:
    roster_frame = pd.DataFrame(
        [
            {
                "PLAYER_ID": 1,
                "NUM": "7",
                "POSITION": "G",
                "HEIGHT": "6-4",
                "WEIGHT": "210",
                "BIRTH_DATE": "2000-01-01",
                "AGE": 25,
                "EXP": "3",
                "SCHOOL": "Example",
            }
        ]
    )
    coach_frame = pd.DataFrame(
        [
            {
                "COACH_ID": 99,
                "FIRST_NAME": "Pat",
                "LAST_NAME": "Coach",
                "COACH_NAME": "Pat Coach",
                "IS_ASSISTANT": 0,
                "COACH_TYPE": "Head Coach",
                "SORT_SEQUENCE": 1,
            }
        ]
    )

    roster_rows = normalize_roster_rows(roster_frame, 10, "2025-26")
    coach_rows, team_coach_rows = normalize_coach_rows(coach_frame, 10, "2025-26")

    assert roster_rows[0]["player_id"] == 1
    assert roster_rows[0]["birth_date"] == pd.Timestamp("2000-01-01").date()
    assert coach_rows[0]["coach_name"] == "Pat Coach"
    assert team_coach_rows[0]["is_assistant"] is False
