from __future__ import annotations

from collections import defaultdict, deque
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import and_, or_, select

from nba_predictor.db import games

REGULAR_SEASON = "Regular Season"
PLAYOFFS = "Playoffs"

MATCHUP_CONTEXT_COLUMNS = [
    "h2h_games",
    "h2h_home_win_pct",
    "h2h_recent_games",
    "h2h_recent_home_win_pct",
    "h2h_regular_games",
    "h2h_regular_home_win_pct",
    "h2h_playoff_games",
    "h2h_playoff_home_win_pct",
    "same_season_h2h_games",
    "same_season_h2h_home_win_pct",
    "same_season_regular_games",
    "same_season_regular_home_win_pct",
    "same_season_playoff_games",
    "same_season_playoff_home_win_pct",
    "playoff_series_games",
    "playoff_series_game_number",
    "playoff_series_home_wins",
    "playoff_series_away_wins",
    "playoff_series_home_win_pct",
    "h2h_history_home_win_probability",
]


def _smoothed_win_pct(wins: int, games_played: int, prior_games: int) -> float:
    return (wins + 0.5 * prior_games) / (games_played + prior_games)


def season_for_game_date(game_date: date) -> str:
    start_year = game_date.year if game_date.month >= 7 else game_date.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def _context_from_history(
    history: list[tuple[int, str | None, str | None]],
    home_team_id: int,
    current_season: str,
) -> dict[str, float | int]:
    all_games = len(history)
    all_wins = sum(int(winner == home_team_id) for winner, _, _ in history)
    recent = history[-10:]
    recent_games = len(recent)
    recent_wins = sum(int(winner == home_team_id) for winner, _, _ in recent)
    regular = [winner for winner, _, season_type in history if season_type == REGULAR_SEASON]
    playoff = [winner for winner, _, season_type in history if season_type == PLAYOFFS]
    same_season = [item for item in history if item[1] == current_season]
    same_season_regular = [winner for winner, _, season_type in same_season if season_type == REGULAR_SEASON]
    same_season_playoff = [winner for winner, _, season_type in same_season if season_type == PLAYOFFS]
    regular_games = len(regular)
    regular_wins = sum(int(winner == home_team_id) for winner in regular)
    playoff_games = len(playoff)
    playoff_wins = sum(int(winner == home_team_id) for winner in playoff)
    same_season_games = len(same_season)
    same_season_wins = sum(int(winner == home_team_id) for winner, _, _ in same_season)
    same_season_regular_games = len(same_season_regular)
    same_season_regular_wins = sum(int(winner == home_team_id) for winner in same_season_regular)
    same_season_playoff_games = len(same_season_playoff)
    same_season_playoff_wins = sum(int(winner == home_team_id) for winner in same_season_playoff)
    playoff_series_games = same_season_playoff_games
    playoff_series_home_wins = same_season_playoff_wins
    playoff_series_away_wins = playoff_series_games - playoff_series_home_wins

    all_pct = _smoothed_win_pct(all_wins, all_games, prior_games=8)
    recent_pct = _smoothed_win_pct(recent_wins, recent_games, prior_games=6)
    regular_pct = _smoothed_win_pct(regular_wins, regular_games, prior_games=8)
    playoff_pct = _smoothed_win_pct(playoff_wins, playoff_games, prior_games=6)
    same_season_pct = _smoothed_win_pct(same_season_wins, same_season_games, prior_games=4)
    same_season_regular_pct = _smoothed_win_pct(
        same_season_regular_wins,
        same_season_regular_games,
        prior_games=4,
    )
    same_season_playoff_pct = _smoothed_win_pct(
        same_season_playoff_wins,
        same_season_playoff_games,
        prior_games=3,
    )
    playoff_series_pct = _smoothed_win_pct(
        playoff_series_home_wins,
        playoff_series_games,
        prior_games=3,
    )
    history_probability = (
        (0.25 * recent_pct)
        + (0.20 * regular_pct)
        + (0.15 * playoff_pct)
        + (0.25 * same_season_pct)
        + (0.15 * playoff_series_pct)
    )
    return {
        "h2h_games": all_games,
        "h2h_home_win_pct": all_pct,
        "h2h_recent_games": recent_games,
        "h2h_recent_home_win_pct": recent_pct,
        "h2h_regular_games": regular_games,
        "h2h_regular_home_win_pct": regular_pct,
        "h2h_playoff_games": playoff_games,
        "h2h_playoff_home_win_pct": playoff_pct,
        "same_season_h2h_games": same_season_games,
        "same_season_h2h_home_win_pct": same_season_pct,
        "same_season_regular_games": same_season_regular_games,
        "same_season_regular_home_win_pct": same_season_regular_pct,
        "same_season_playoff_games": same_season_playoff_games,
        "same_season_playoff_home_win_pct": same_season_playoff_pct,
        "playoff_series_games": playoff_series_games,
        "playoff_series_game_number": playoff_series_games + 1,
        "playoff_series_home_wins": playoff_series_home_wins,
        "playoff_series_away_wins": playoff_series_away_wins,
        "playoff_series_home_win_pct": playoff_series_pct,
        "h2h_history_home_win_probability": history_probability,
    }


def compute_matchup_context_rows(games_df: pd.DataFrame) -> list[dict[str, Any]]:
    if games_df.empty:
        return []
    games_copy = games_df.copy()
    games_copy["game_date"] = pd.to_datetime(games_copy["game_date"])
    history_by_pair: dict[tuple[int, int], deque[tuple[int, str | None, str | None]]] = defaultdict(deque)
    rows: list[dict[str, Any]] = []
    for game in games_copy.sort_values(["game_date", "game_id"]).itertuples(index=False):
        home_team_id = int(game.home_team_id)
        away_team_id = int(game.away_team_id)
        pair = tuple(sorted((home_team_id, away_team_id)))
        history = list(history_by_pair[pair])
        current_season = str(
            getattr(
                game,
                "season",
                season_for_game_date(pd.Timestamp(game.game_date).date()),
            )
        )
        rows.append(
            {
                "game_id": str(game.game_id),
                **_context_from_history(history, home_team_id, current_season),
            }
        )
        if pd.isna(game.home_team_win):
            continue
        winner = home_team_id if bool(game.home_team_win) else away_team_id
        history_by_pair[pair].append((winner, current_season, getattr(game, "season_type", None)))
    return rows


def matchup_context_for_teams(
    conn: Any,
    home_team_id: int,
    away_team_id: int,
    game_date: date,
) -> dict[str, float | int]:
    stmt = (
        select(
            games.c.home_team_id,
            games.c.away_team_id,
            games.c.home_team_win,
            games.c.season,
            games.c.season_type,
        )
        .where(
            and_(
                games.c.game_date < game_date,
                games.c.home_team_win.is_not(None),
                or_(
                    and_(games.c.home_team_id == home_team_id, games.c.away_team_id == away_team_id),
                    and_(games.c.home_team_id == away_team_id, games.c.away_team_id == home_team_id),
                ),
            )
        )
        .order_by(games.c.game_date, games.c.game_id)
    )
    history: list[tuple[int, str | None, str | None]] = []
    for row in conn.execute(stmt):
        winner = int(row.home_team_id) if bool(row.home_team_win) else int(row.away_team_id)
        history.append((winner, row.season, row.season_type))
    return _context_from_history(history, home_team_id, season_for_game_date(game_date))
