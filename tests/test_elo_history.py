from __future__ import annotations

import pandas as pd

from nba_predictor.features.build_team_elo_history import compute_team_elo_history_rows
from nba_predictor.features.elo import INITIAL_ELO


def test_elo_history_starts_new_franchise_at_initial_rating() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_date": "2026-01-01",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 101,
                "away_score": 95,
                "home_team_win": True,
            },
            {
                "game_id": "g2",
                "game_date": "2026-01-02",
                "home_team_id": 3,
                "away_team_id": 1,
                "home_score": 99,
                "away_score": 100,
                "home_team_win": False,
            },
        ]
    )

    rows = compute_team_elo_history_rows(games)
    first_rows = [row for row in rows if row["game_id"] == "g1"]
    newcomer = next(row for row in rows if row["game_id"] == "g2" and row["team_id"] == 3)

    assert all(row["pregame_elo"] == INITIAL_ELO for row in first_rows)
    assert newcomer["pregame_elo"] == INITIAL_ELO


def test_elo_history_carries_forward_previous_postgame_rating() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_date": "2026-01-01",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 101,
                "away_score": 95,
                "home_team_win": True,
            },
            {
                "game_id": "g2",
                "game_date": "2026-01-02",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 94,
                "away_score": 99,
                "home_team_win": False,
            },
        ]
    )

    rows = compute_team_elo_history_rows(games)
    team_one_game_one = next(row for row in rows if row["game_id"] == "g1" and row["team_id"] == 1)
    team_one_game_two = next(row for row in rows if row["game_id"] == "g2" and row["team_id"] == 1)

    assert team_one_game_two["pregame_elo"] == team_one_game_one["postgame_elo"]


def test_elo_history_is_zero_sum_per_game() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "g1",
                "game_date": "2026-01-01",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 101,
                "away_score": 95,
                "home_team_win": True,
            }
        ]
    )

    rows = compute_team_elo_history_rows(games)
    deltas = [row["postgame_elo"] - row["pregame_elo"] for row in rows]
    assert len(rows) == 2
    assert round(sum(deltas), 10) == 0.0


def test_elo_history_skips_placeholder_games() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "placeholder",
                "game_date": "2026-06-19",
                "home_team_id": 0,
                "away_team_id": 0,
                "home_score": 0,
                "away_score": 0,
                "home_team_win": False,
            }
        ]
    )

    assert compute_team_elo_history_rows(games) == []
