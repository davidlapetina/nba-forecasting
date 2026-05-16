from __future__ import annotations

import argparse
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import select

from nba_predictor.db import coaches, get_engine, player_game_stats, players, team_coaches, team_rosters, teams, upsert_rows
from nba_predictor.ingest.nba_client import NBAClient


def _maybe_date(value: Any) -> date | None:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).date()


def _maybe_float(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _maybe_int(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def normalize_roster_rows(roster_df: pd.DataFrame, team_id: int, season: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in roster_df.iterrows():
        rows.append(
            {
                "team_id": team_id,
                "season": season,
                "player_id": int(row["PLAYER_ID"]),
                "jersey_number": None if pd.isna(row.get("NUM")) else str(row.get("NUM")),
                "position": row.get("POSITION"),
                "height": row.get("HEIGHT"),
                "weight": None if pd.isna(row.get("WEIGHT")) else str(row.get("WEIGHT")),
                "birth_date": _maybe_date(row.get("BIRTH_DATE")),
                "age": _maybe_float(row.get("AGE")),
                "experience": None if pd.isna(row.get("EXP")) else str(row.get("EXP")),
                "school": row.get("SCHOOL"),
            }
        )
    return rows


def normalize_roster_players(roster_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, row in roster_df.iterrows():
        full_name = str(row["PLAYER"])
        first_name, _, last_name = full_name.partition(" ")
        rows.append(
            {
                "player_id": int(row["PLAYER_ID"]),
                "full_name": full_name,
                "first_name": first_name or None,
                "last_name": last_name or None,
                "is_active": None,
            }
        )
    return rows


def normalize_coach_rows(coach_df: pd.DataFrame, team_id: int, season: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    coach_rows: list[dict[str, Any]] = []
    team_rows: list[dict[str, Any]] = []
    for _, row in coach_df.iterrows():
        coach_id = _maybe_int(row.get("COACH_ID"))
        if coach_id is None:
            continue
        coach_rows.append(
            {
                "coach_id": coach_id,
                "first_name": row.get("FIRST_NAME"),
                "last_name": row.get("LAST_NAME"),
                "coach_name": row.get("COACH_NAME") or f"{row.get('FIRST_NAME', '')} {row.get('LAST_NAME', '')}".strip(),
            }
        )
        team_rows.append(
            {
                "team_id": team_id,
                "season": season,
                "coach_id": coach_id,
                "is_assistant": None if pd.isna(row.get("IS_ASSISTANT")) else bool(row.get("IS_ASSISTANT")),
                "coach_type": row.get("COACH_TYPE"),
                "sort_sequence": _maybe_int(row.get("SORT_SEQUENCE")),
            }
        )
    return coach_rows, team_rows


def ingest_rosters(season: str) -> dict[str, int]:
    client = NBAClient()
    engine = get_engine()
    with engine.connect() as conn:
        team_ids = [
            int(row.team_id)
            for row in conn.execute(
                select(player_game_stats.c.team_id)
                .where(player_game_stats.c.season == season)
                .distinct()
            )
        ]
        if not team_ids:
            team_ids = [int(row.team_id) for row in conn.execute(select(teams.c.team_id))]

    roster_rows: list[dict[str, Any]] = []
    player_rows: list[dict[str, Any]] = []
    coach_rows: list[dict[str, Any]] = []
    team_coach_rows: list[dict[str, Any]] = []
    for team_id in team_ids:
        try:
            roster_df, coach_df = client.fetch_team_roster(team_id, season)
        except Exception as exc:  # pragma: no cover - upstream service availability
            print(f"skipping roster for {team_id} in {season}: {exc}")
            continue
        roster_rows.extend(normalize_roster_rows(roster_df, team_id, season))
        player_rows.extend(normalize_roster_players(roster_df))
        normalized_coaches, normalized_team_coaches = normalize_coach_rows(coach_df, team_id, season)
        coach_rows.extend(normalized_coaches)
        team_coach_rows.extend(normalized_team_coaches)

    upsert_rows(
        engine,
        players,
        player_rows,
        ["player_id"],
        ["full_name", "first_name", "last_name", "is_active"],
    )
    upsert_rows(
        engine,
        team_rosters,
        roster_rows,
        ["team_id", "season", "player_id"],
        ["jersey_number", "position", "height", "weight", "birth_date", "age", "experience", "school"],
    )
    upsert_rows(
        engine,
        coaches,
        coach_rows,
        ["coach_id"],
        ["first_name", "last_name", "coach_name"],
    )
    upsert_rows(
        engine,
        team_coaches,
        team_coach_rows,
        ["team_id", "season", "coach_id"],
        ["is_assistant", "coach_type", "sort_sequence"],
    )
    return {"rosters": len(roster_rows), "coaches": len(team_coach_rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest team rosters and coaches for a season")
    parser.add_argument("--season", default=f"{date.today().year - 1}-{str(date.today().year)[-2:]}")
    args = parser.parse_args()
    counts = ingest_rosters(args.season)
    print(f"upserted {counts['rosters']} roster rows and {counts['coaches']} coach rows for {args.season}")


if __name__ == "__main__":
    main()
