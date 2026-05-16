from __future__ import annotations

import argparse
from datetime import date
from typing import Any

import pandas as pd

from nba_predictor.db import games, get_engine, teams, upsert_rows
from nba_predictor.ingest.nba_client import NBAClient


def _is_home(matchup: str) -> bool:
    return " vs. " in matchup


def normalize_team_rows(log_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows_by_id: dict[int, dict[str, Any]] = {}
    for _, row in log_df.iterrows():
        team_id = row.get("TEAM_ID")
        abbreviation = row.get("TEAM_ABBREVIATION")
        if pd.isna(team_id) or pd.isna(abbreviation):
            continue
        full_name = row.get("TEAM_NAME")
        rows_by_id[int(team_id)] = {
            "team_id": int(team_id),
            "abbreviation": str(abbreviation),
            "full_name": None if pd.isna(full_name) else str(full_name),
            "city": None,
            "nickname": None,
        }
    return list(rows_by_id.values())


def normalize_game_rows(log_df: pd.DataFrame, season: str, season_type: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for game_id, game_rows in log_df.groupby("GAME_ID"):
        if len(game_rows) != 2:
            continue
        home_row = game_rows[game_rows["MATCHUP"].map(_is_home)]
        away_row = game_rows[~game_rows["MATCHUP"].map(_is_home)]
        if home_row.empty or away_row.empty:
            continue
        home = home_row.iloc[0]
        away = away_row.iloc[0]
        home_score = int(home["PTS"]) if pd.notna(home["PTS"]) else None
        away_score = int(away["PTS"]) if pd.notna(away["PTS"]) else None
        rows.append(
            {
                "game_id": str(game_id),
                "season": season,
                "game_date": pd.Timestamp(home["GAME_DATE"]).date(),
                "home_team_id": int(home["TEAM_ID"]),
                "away_team_id": int(away["TEAM_ID"]),
                "home_score": home_score,
                "away_score": away_score,
                "home_team_win": None if home_score is None or away_score is None else home_score > away_score,
                "season_type": season_type,
            }
        )
    return rows


def ingest_games(season: str, season_type: str = "Regular Season") -> int:
    client = NBAClient()
    engine = get_engine()
    upsert_rows(engine, teams, client.team_directory(), ["team_id"], ["abbreviation", "full_name", "city", "nickname"])
    log_df = client.fetch_league_game_log(season, season_type)
    upsert_rows(engine, teams, normalize_team_rows(log_df), ["team_id"], ["abbreviation", "full_name", "city", "nickname"])
    rows = normalize_game_rows(log_df, season, season_type)
    return upsert_rows(
        engine,
        games,
        rows,
        ["game_id"],
        [
            "season",
            "game_date",
            "home_team_id",
            "away_team_id",
            "home_score",
            "away_score",
            "home_team_win",
            "season_type",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA games")
    parser.add_argument("--season", default=f"{date.today().year - 1}-{str(date.today().year)[-2:]}")
    parser.add_argument("--season-type", default="Regular Season")
    args = parser.parse_args()
    count = ingest_games(args.season, args.season_type)
    print(f"upserted {count} games for {args.season}")


if __name__ == "__main__":
    main()
