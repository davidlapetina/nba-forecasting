from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
from sqlalchemy import select, update

from nba_predictor.db import game_officials, games, get_engine, referees, team_game_stats, upsert_rows
from nba_predictor.ingest.nba_client import NBAClient


def _maybe_float(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def ingest_box_scores(season: str | None = None) -> int:
    if season is None:
        raise ValueError("season is required for bulk advanced team log ingestion")
    client = NBAClient()
    engine = get_engine()
    advanced = client.fetch_advanced_team_game_logs(season)
    updates: list[dict[str, Any]] = []
    for _, row in advanced.iterrows():
        updates.append(
            {
                "game_id": str(row["GAME_ID"]),
                "team_id": int(row["TEAM_ID"]),
                "offensive_rating": _maybe_float(row.get("OFF_RATING")),
                "defensive_rating": _maybe_float(row.get("DEF_RATING")),
                "pace": _maybe_float(row.get("PACE")),
            }
        )
    with engine.begin() as conn:
        for row in updates:
            conn.execute(
                update(team_game_stats)
                .where(
                    team_game_stats.c.game_id == row["game_id"],
                    team_game_stats.c.team_id == row["team_id"],
                )
                .values(
                    offensive_rating=row["offensive_rating"],
                    defensive_rating=row["defensive_rating"],
                    pace=row["pace"],
                )
            )
    return len(updates)


def ingest_officials(season: str | None = None) -> int:
    client = NBAClient()
    engine = get_engine()
    stmt = select(games.c.game_id)
    if season:
        stmt = stmt.where(games.c.season == season)
    with engine.connect() as conn:
        game_ids = [row.game_id for row in conn.execute(stmt)]

    referee_rows: list[dict[str, Any]] = []
    official_rows: list[dict[str, Any]] = []
    for game_id in game_ids:
        try:
            officials = client.fetch_game_officials(game_id)
        except Exception as exc:  # pragma: no cover - upstream service availability
            print(f"skipping officials for {game_id}: {exc}")
            continue
        for _, row in officials.iterrows():
            referee_id = int(row["OFFICIAL_ID"])
            referee_rows.append(
                {
                    "referee_id": referee_id,
                    "first_name": row.get("FIRST_NAME"),
                    "last_name": row.get("LAST_NAME"),
                    "jersey_num": None if pd.isna(row.get("JERSEY_NUM")) else str(row.get("JERSEY_NUM")),
                }
            )
            official_rows.append({"game_id": game_id, "referee_id": referee_id})

    upsert_rows(
        engine,
        referees,
        referee_rows,
        ["referee_id"],
        ["first_name", "last_name", "jersey_num"],
    )
    upsert_rows(engine, game_officials, official_rows, ["game_id", "referee_id"])
    return len(official_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA box score advanced metrics")
    parser.add_argument("--season", required=True)
    parser.add_argument("--include-officials", action="store_true")
    args = parser.parse_args()
    count = ingest_box_scores(args.season)
    print(f"upserted advanced metrics for {count} team rows")
    if args.include_officials:
        official_count = ingest_officials(args.season)
        print(f"upserted {official_count} game official rows")


if __name__ == "__main__":
    main()
