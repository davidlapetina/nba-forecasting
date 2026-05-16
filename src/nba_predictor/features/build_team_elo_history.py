from __future__ import annotations

from typing import Any

import pandas as pd
from sqlalchemy import delete, select

from nba_predictor.db import games, get_engine, team_elo_history, upsert_rows
from nba_predictor.features.elo import INITIAL_ELO, update_elo


def compute_team_elo_history_rows(games_df: pd.DataFrame) -> list[dict[str, Any]]:
    if games_df.empty:
        return []
    played = games_df.dropna(subset=["home_team_win"]).copy()
    if {"home_score", "away_score"}.issubset(played.columns):
        played = played[(played["home_score"] > 0) & (played["away_score"] > 0)]
    played = played[
        (played["home_team_id"].astype(int) > 0)
        & (played["away_team_id"].astype(int) > 0)
        & (played["home_team_id"] != played["away_team_id"])
    ]
    if played.empty:
        return []
    played["game_date"] = pd.to_datetime(played["game_date"])
    ratings: dict[int, float] = {}
    rows: list[dict[str, Any]] = []
    for row in played.sort_values(["game_date", "game_id"]).itertuples(index=False):
        home_team_id = int(row.home_team_id)
        away_team_id = int(row.away_team_id)
        home_elo = ratings.setdefault(home_team_id, INITIAL_ELO)
        away_elo = ratings.setdefault(away_team_id, INITIAL_ELO)
        new_home_elo, new_away_elo = update_elo(home_elo, away_elo, bool(row.home_team_win))
        game_date = pd.Timestamp(row.game_date).date()
        rows.extend(
            [
                {
                    "game_id": row.game_id,
                    "team_id": home_team_id,
                    "opponent_team_id": away_team_id,
                    "game_date": game_date,
                    "is_home": True,
                    "won": bool(row.home_team_win),
                    "pregame_elo": home_elo,
                    "postgame_elo": new_home_elo,
                },
                {
                    "game_id": row.game_id,
                    "team_id": away_team_id,
                    "opponent_team_id": home_team_id,
                    "game_date": game_date,
                    "is_home": False,
                    "won": not bool(row.home_team_win),
                    "pregame_elo": away_elo,
                    "postgame_elo": new_away_elo,
                },
            ]
        )
        ratings[home_team_id] = new_home_elo
        ratings[away_team_id] = new_away_elo
    return rows


def build_team_elo_history() -> int:
    engine = get_engine()
    with engine.connect() as conn:
        games_df = pd.read_sql(select(games), conn)
    rows = compute_team_elo_history_rows(games_df)
    with engine.begin() as conn:
        conn.execute(delete(team_elo_history))
    return upsert_rows(
        engine,
        team_elo_history,
        rows,
        ["game_id", "team_id"],
        [
            column.name
            for column in team_elo_history.columns
            if column.name not in {"id", "game_id", "team_id", "created_at"}
        ],
    )


def main() -> None:
    count = build_team_elo_history()
    print(f"upserted {count} team elo history rows")


if __name__ == "__main__":
    main()
