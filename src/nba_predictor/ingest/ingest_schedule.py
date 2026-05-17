from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Any

import pandas as pd

from nba_predictor.db import games, get_engine, teams, upsert_rows
from nba_predictor.ingest.nba_client import NBAClient


def _nullable_int(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _season_for_date(target_date: date) -> str:
    start_year = target_date.year if target_date.month >= 7 else target_date.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


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


def normalize_scoreboard_team_rows(line_score_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows_by_id: dict[int, dict[str, Any]] = {}
    for row in line_score_df.itertuples(index=False):
        team_id = _nullable_int(getattr(row, "teamId", None))
        if team_id is None:
            continue
        city = getattr(row, "teamCity", None)
        nickname = getattr(row, "teamName", None)
        full_name = " ".join(str(part) for part in [city, nickname] if part)
        rows_by_id[team_id] = {
            "team_id": team_id,
            "abbreviation": str(getattr(row, "teamTricode", "")),
            "full_name": full_name or None,
            "city": None if city is None else str(city),
            "nickname": None if nickname is None else str(nickname),
        }
    return list(rows_by_id.values())


def normalize_scoreboard_rows(
    game_header_df: pd.DataFrame,
    line_score_df: pd.DataFrame,
    game_date: date,
    season: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    line_scores_by_game = {
        str(game_id): frame.reset_index(drop=True)
        for game_id, frame in line_score_df.groupby("gameId", sort=False)
    }
    for row in game_header_df.itertuples(index=False):
        game_id = str(row.gameId)
        line_scores = line_scores_by_game.get(game_id)
        if line_scores is None or len(line_scores) < 2:
            continue
        home = line_scores.iloc[0]
        away = line_scores.iloc[1]
        home_score = _nullable_int(home.get("score"))
        away_score = _nullable_int(away.get("score"))
        if home_score == 0 and away_score == 0:
            home_score = None
            away_score = None
        game_subtype = getattr(row, "gameSubtype", None)
        rows.append(
            {
                "game_id": game_id,
                "season": season,
                "game_date": game_date,
                "home_team_id": int(home["teamId"]),
                "away_team_id": int(away["teamId"]),
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
        ],
    )


def ingest_upcoming_schedule(start_date: date, days: int = 14) -> int:
    client = NBAClient()
    engine = get_engine()
    rows: list[dict[str, Any]] = []
    team_rows_by_id: dict[int, dict[str, Any]] = {}
    for offset in range(days):
        game_date = start_date + timedelta(days=offset)
        game_headers, line_scores = client.fetch_daily_scoreboard(game_date.isoformat())
        for team_row in normalize_scoreboard_team_rows(line_scores):
            team_rows_by_id[int(team_row["team_id"])] = team_row
        rows.extend(
            normalize_scoreboard_rows(
                game_headers,
                line_scores,
                game_date,
                _season_for_date(game_date),
            )
        )
    upsert_rows(engine, teams, team_rows_by_id.values(), ["team_id"], ["abbreviation", "full_name", "city", "nickname"])
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
