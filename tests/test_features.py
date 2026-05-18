from __future__ import annotations

import pandas as pd

from nba_predictor.features.build_game_features import compute_game_feature_rows
from nba_predictor.features.build_team_daily_features import compute_team_daily_feature_rows
from nba_predictor.features.matchup_context import compute_matchup_context_rows


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.DataFrame(
        [
            {"game_id": "g1", "game_date": "2025-01-01", "home_team_id": 1, "away_team_id": 2, "home_score": 100, "away_score": 90, "home_team_win": True},
            {"game_id": "g2", "game_date": "2025-01-03", "home_team_id": 2, "away_team_id": 1, "home_score": 500, "away_score": 95, "home_team_win": False},
        ]
    )
    stats = pd.DataFrame(
        [
            {"game_id": "g1", "team_id": 1, "game_date": "2025-01-01", "points": 100, "rebounds": 40, "assists": 20, "turnovers": 10, "offensive_rating": 110, "defensive_rating": 100, "pace": 98},
            {"game_id": "g1", "team_id": 2, "game_date": "2025-01-01", "points": 90, "rebounds": 38, "assists": 18, "turnovers": 12, "offensive_rating": 100, "defensive_rating": 110, "pace": 98},
            {"game_id": "g2", "team_id": 2, "game_date": "2025-01-03", "points": 500, "rebounds": 39, "assists": 19, "turnovers": 11, "offensive_rating": 101, "defensive_rating": 109, "pace": 99},
            {"game_id": "g2", "team_id": 1, "game_date": "2025-01-03", "points": 95, "rebounds": 41, "assists": 21, "turnovers": 9, "offensive_rating": 111, "defensive_rating": 99, "pace": 99},
        ]
    )
    return stats, games


def test_daily_features_do_not_use_same_game_or_future_game() -> None:
    stats, games = _frames()
    rows = compute_team_daily_feature_rows(stats, games)
    team_two_g2 = next(row for row in rows if row["team_id"] == 2 and str(row["feature_date"]) == "2025-01-03")
    assert team_two_g2["avg_points_last_10"] == 90.0
    assert team_two_g2["games_played"] == 1


def test_daily_features_emit_one_row_for_same_day_doubleheader() -> None:
    stats, games = _frames()
    games = pd.concat(
        [
            games,
            pd.DataFrame(
                [
                    {
                        "game_id": "g3",
                        "game_date": "2025-01-03",
                        "home_team_id": 1,
                        "away_team_id": 2,
                        "home_score": 99,
                        "away_score": 98,
                        "home_team_win": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    stats = pd.concat(
        [
            stats,
            pd.DataFrame(
                [
                    {"game_id": "g3", "team_id": 1, "game_date": "2025-01-03", "points": 99, "rebounds": 40, "assists": 20, "turnovers": 10, "offensive_rating": 110, "defensive_rating": 100, "pace": 98},
                    {"game_id": "g3", "team_id": 2, "game_date": "2025-01-03", "points": 98, "rebounds": 39, "assists": 19, "turnovers": 11, "offensive_rating": 109, "defensive_rating": 101, "pace": 98},
                ]
            ),
        ],
        ignore_index=True,
    )

    rows = compute_team_daily_feature_rows(stats, games)
    team_one_g2_rows = [row for row in rows if row["team_id"] == 1 and str(row["feature_date"]) == "2025-01-03"]
    assert len(team_one_g2_rows) == 1
    assert team_one_g2_rows[0]["games_played"] == 1


def test_daily_features_use_prior_playoff_results() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "regular",
                "game_date": "2026-04-10",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 100,
                "away_score": 90,
                "home_team_win": True,
                "season_type": "Regular Season",
            },
            {
                "game_id": "playoff_1",
                "game_date": "2026-04-20",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_score": 102,
                "away_score": 98,
                "home_team_win": True,
                "season_type": "Playoffs",
            },
            {
                "game_id": "playoff_2",
                "game_date": "2026-04-22",
                "home_team_id": 2,
                "away_team_id": 1,
                "home_score": 99,
                "away_score": 101,
                "home_team_win": False,
                "season_type": "Playoffs",
            },
        ]
    )
    common = {
        "rebounds": 40,
        "assists": 20,
        "turnovers": 10,
        "offensive_rating": 110,
        "defensive_rating": 100,
        "pace": 98,
    }
    stats = pd.DataFrame(
        [
            {"game_id": "regular", "team_id": 1, "game_date": "2026-04-10", "points": 100, **common},
            {"game_id": "regular", "team_id": 2, "game_date": "2026-04-10", "points": 90, **common},
            {"game_id": "playoff_1", "team_id": 1, "game_date": "2026-04-20", "points": 102, **common},
            {"game_id": "playoff_1", "team_id": 2, "game_date": "2026-04-20", "points": 98, **common},
            {"game_id": "playoff_2", "team_id": 2, "game_date": "2026-04-22", "points": 99, **common},
            {"game_id": "playoff_2", "team_id": 1, "game_date": "2026-04-22", "points": 101, **common},
        ]
    )

    rows = compute_team_daily_feature_rows(stats, games)
    team_one_second_playoff = next(
        row for row in rows if row["team_id"] == 1 and str(row["feature_date"]) == "2026-04-22"
    )

    assert team_one_second_playoff["games_played"] == 2
    assert team_one_second_playoff["last_10_win_pct"] == 1.0
    assert team_one_second_playoff["avg_points_last_10"] == 101.0


def test_game_features_include_prior_zero_minute_rate() -> None:
    games = pd.DataFrame(
        [
            {"game_id": "g1", "game_date": "2025-01-01", "season": "2024-25", "home_team_id": 1, "away_team_id": 2, "home_team_win": True},
            {"game_id": "g2", "game_date": "2025-01-03", "season": "2024-25", "home_team_id": 1, "away_team_id": 2, "home_team_win": False},
        ]
    )
    daily = pd.DataFrame(
        [
            {"team_id": 1, "feature_date": "2025-01-01", "win_pct": 0.5, "last_10_win_pct": 0.5, "avg_points_last_10": 100, "avg_off_rating_last_10": 110, "avg_def_rating_last_10": 100, "elo_rating": 2500},
            {"team_id": 2, "feature_date": "2025-01-01", "win_pct": 0.5, "last_10_win_pct": 0.5, "avg_points_last_10": 100, "avg_off_rating_last_10": 110, "avg_def_rating_last_10": 100, "elo_rating": 2500},
            {"team_id": 1, "feature_date": "2025-01-03", "win_pct": 1.0, "last_10_win_pct": 1.0, "avg_points_last_10": 101, "avg_off_rating_last_10": 111, "avg_def_rating_last_10": 99, "elo_rating": 2510},
            {"team_id": 2, "feature_date": "2025-01-03", "win_pct": 0.0, "last_10_win_pct": 0.0, "avg_points_last_10": 99, "avg_off_rating_last_10": 109, "avg_def_rating_last_10": 101, "elo_rating": 2490},
        ]
    )
    team_stats = pd.DataFrame(
        [
            {"game_id": "g1", "team_id": 1, "game_date": "2025-01-01"},
            {"game_id": "g1", "team_id": 2, "game_date": "2025-01-01"},
        ]
    )
    player_stats = pd.DataFrame(
        [
            {"game_id": "g1", "team_id": 1, "player_id": 1, "game_date": "2025-01-01", "minutes": 0},
            {"game_id": "g1", "team_id": 1, "player_id": 2, "game_date": "2025-01-01", "minutes": 20},
            {"game_id": "g1", "team_id": 2, "player_id": 3, "game_date": "2025-01-01", "minutes": 20},
            {"game_id": "g1", "team_id": 2, "player_id": 4, "game_date": "2025-01-01", "minutes": 20},
        ]
    )

    rows = compute_game_feature_rows(games, daily, team_stats, player_stats)
    second_game = next(row for row in rows if row["game_id"] == "g2")

    assert second_game["home_recent_zero_minute_rate"] == 0.5
    assert second_game["away_recent_zero_minute_rate"] == 0.0
    assert second_game["recent_zero_minute_rate_diff"] == 0.5


def test_matchup_context_uses_prior_regular_and_playoff_games() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "regular_1",
                "season": "2025-26",
                "game_date": "2026-01-01",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_win": True,
                "season_type": "Regular Season",
            },
            {
                "game_id": "playoff_1",
                "season": "2025-26",
                "game_date": "2026-04-20",
                "home_team_id": 2,
                "away_team_id": 1,
                "home_team_win": False,
                "season_type": "Playoffs",
            },
            {
                "game_id": "future",
                "season": "2025-26",
                "game_date": "2026-05-01",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_win": None,
                "season_type": "Playoffs",
            },
        ]
    )

    rows = compute_matchup_context_rows(games)
    future = next(row for row in rows if row["game_id"] == "future")

    assert future["h2h_games"] == 2
    assert future["h2h_recent_games"] == 2
    assert future["h2h_regular_games"] == 1
    assert future["h2h_playoff_games"] == 1
    assert future["same_season_h2h_games"] == 2
    assert future["same_season_regular_games"] == 1
    assert future["same_season_playoff_games"] == 1
    assert future["playoff_series_game_number"] == 2
    assert future["playoff_series_home_wins"] == 1
    assert future["playoff_series_away_wins"] == 0
    assert future["h2h_history_home_win_probability"] > 0.5


def test_matchup_context_resets_same_season_history_between_seasons() -> None:
    games = pd.DataFrame(
        [
            {
                "game_id": "old_regular",
                "season": "2024-25",
                "game_date": "2025-01-01",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_win": True,
                "season_type": "Regular Season",
            },
            {
                "game_id": "new_regular",
                "season": "2025-26",
                "game_date": "2026-01-01",
                "home_team_id": 1,
                "away_team_id": 2,
                "home_team_win": None,
                "season_type": "Regular Season",
            },
        ]
    )

    rows = compute_matchup_context_rows(games)
    current = next(row for row in rows if row["game_id"] == "new_regular")

    assert current["h2h_games"] == 1
    assert current["same_season_h2h_games"] == 0
    assert current["playoff_series_games"] == 0
