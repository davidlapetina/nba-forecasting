from __future__ import annotations

import argparse
from datetime import date
from typing import Any

import pandas as pd

from nba_predictor.db import get_engine, team_game_stats, upsert_rows
from nba_predictor.ingest.ingest_games import _is_home
from nba_predictor.ingest.nba_client import NBAClient


def normalize_team_stat_rows(log_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, game_rows in log_df.groupby("GAME_ID"):
        if len(game_rows) != 2:
            continue
        by_team_id = {int(row["TEAM_ID"]): row for _, row in game_rows.iterrows()}
        team_ids = list(by_team_id)
        for team_id, row in by_team_id.items():
            opponent_team_id = next(other for other in team_ids if other != team_id)
            rows.append(
                {
                    "game_id": str(row["GAME_ID"]),
                    "team_id": team_id,
                    "opponent_team_id": opponent_team_id,
                    "game_date": pd.Timestamp(row["GAME_DATE"]).date(),
                    "is_home": _is_home(row["MATCHUP"]),
                    "points": _maybe_int(row.get("PTS")),
                    "rebounds": _maybe_int(row.get("REB")),
                    "assists": _maybe_int(row.get("AST")),
                    "steals": _maybe_int(row.get("STL")),
                    "blocks": _maybe_int(row.get("BLK")),
                    "turnovers": _maybe_int(row.get("TOV")),
                    "field_goal_pct": _maybe_float(row.get("FG_PCT")),
                    "three_point_pct": _maybe_float(row.get("FG3_PCT")),
                    "free_throw_pct": _maybe_float(row.get("FT_PCT")),
                }
            )
    return rows


def _maybe_int(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _maybe_float(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def ingest_team_logs(season: str, season_type: str = "Regular Season") -> int:
    client = NBAClient()
    engine = get_engine()
    log_df = client.fetch_league_game_log(season, season_type)
    rows = normalize_team_stat_rows(log_df)
    return upsert_rows(
        engine,
        team_game_stats,
        rows,
        ["game_id", "team_id"],
        [
            "opponent_team_id",
            "game_date",
            "is_home",
            "points",
            "rebounds",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "field_goal_pct",
            "three_point_pct",
            "free_throw_pct",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA team game logs")
    parser.add_argument("--season", default=f"{date.today().year - 1}-{str(date.today().year)[-2:]}")
    parser.add_argument("--season-type", default="Regular Season")
    args = parser.parse_args()
    count = ingest_team_logs(args.season, args.season_type)
    print(f"upserted {count} team game stat rows for {args.season}")


if __name__ == "__main__":
    main()

