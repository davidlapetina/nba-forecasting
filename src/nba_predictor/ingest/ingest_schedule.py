from __future__ import annotations

import argparse
from datetime import date
from typing import Any

import pandas as pd

from nba_predictor.db import games, get_engine, teams, upsert_rows
from nba_predictor.ingest.nba_client import NBAClient


def _nullable_int(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def normalize_schedule_rows(schedule_df: pd.DataFrame, season: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in schedule_df.itertuples(index=False):
        home_score = _nullable_int(getattr(row, "homeTeam_score", None))
        away_score = _nullable_int(getattr(row, "awayTeam_score", None))
        if home_score == 0 and away_score == 0:
            home_score = None
            away_score = None
        game_subtype = getattr(row, "gameSubtype", None)
        rows.append(
            {
                "game_id": str(row.gameId),
                "season": season,
                "game_date": pd.Timestamp(row.gameDate).date(),
                "home_team_id": int(row.homeTeam_teamId),
                "away_team_id": int(row.awayTeam_teamId),
                "home_score": home_score,
                "away_score": away_score,
                "home_team_win": None if home_score is None or away_score is None else home_score > away_score,
                "season_type": str(game_subtype) if game_subtype else "Regular Season",
            }
        )
    return rows


def ingest_schedule(season: str) -> int:
    client = NBAClient()
    engine = get_engine()
    upsert_rows(engine, teams, client.team_directory(), ["team_id"], ["abbreviation", "full_name", "city", "nickname"])
    rows = normalize_schedule_rows(client.fetch_season_schedule(season), season)
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
    parser = argparse.ArgumentParser(description="Ingest NBA season schedule")
    parser.add_argument("--season", default=f"{date.today().year - 1}-{str(date.today().year)[-2:]}")
    args = parser.parse_args()
    count = ingest_schedule(args.season)
    print(f"upserted {count} scheduled games for {args.season}")


if __name__ == "__main__":
    main()
