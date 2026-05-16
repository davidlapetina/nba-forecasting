from __future__ import annotations

import argparse
from datetime import date
from typing import Any

import pandas as pd

from nba_predictor.db import get_engine, player_game_stats, players, teams, upsert_rows
from nba_predictor.ingest.ingest_games import _is_home, normalize_team_rows
from nba_predictor.ingest.nba_client import NBAClient


def _get(row: pd.Series, *keys: str) -> Any:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return None


def _maybe_int(value: Any) -> int | None:
    return None if value is None or pd.isna(value) else int(value)


def _maybe_float(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def normalize_player_game_rows(log_df: pd.DataFrame, season: str, season_type: str) -> list[dict[str, Any]]:
    rows_by_key: dict[tuple[str, int], dict[str, Any]] = {}
    for _, row in log_df.iterrows():
        player_id = _get(row, "PLAYER_ID", "Player_ID")
        team_id = _get(row, "TEAM_ID", "Team_ID")
        game_id = _get(row, "GAME_ID", "Game_ID")
        game_date = _get(row, "GAME_DATE")
        if player_id is None or team_id is None or game_id is None or game_date is None:
            continue
        matchup = _get(row, "MATCHUP")
        win_loss = _get(row, "WL")
        normalized = {
                "game_id": str(game_id),
                "player_id": int(player_id),
                "team_id": int(team_id),
                "game_date": pd.Timestamp(game_date).date(),
                "season": season,
                "season_type": season_type,
                "matchup": None if matchup is None else str(matchup),
                "is_home": None if matchup is None else _is_home(str(matchup)),
                "won": None if win_loss is None else str(win_loss).upper() == "W",
                "minutes": _maybe_int(_get(row, "MIN")),
                "points": _maybe_int(_get(row, "PTS")),
                "rebounds": _maybe_int(_get(row, "REB")),
                "assists": _maybe_int(_get(row, "AST")),
                "steals": _maybe_int(_get(row, "STL")),
                "blocks": _maybe_int(_get(row, "BLK")),
                "turnovers": _maybe_int(_get(row, "TOV")),
                "field_goal_pct": _maybe_float(_get(row, "FG_PCT")),
                "three_point_pct": _maybe_float(_get(row, "FG3_PCT")),
                "free_throw_pct": _maybe_float(_get(row, "FT_PCT")),
                "plus_minus": _maybe_float(_get(row, "PLUS_MINUS")),
        }
        rows_by_key[(normalized["game_id"], normalized["player_id"])] = normalized
    return list(rows_by_key.values())


def ingest_players(season: str, season_type: str = "Regular Season") -> int:
    client = NBAClient()
    engine = get_engine()
    upsert_rows(
        engine,
        players,
        client.player_directory(),
        ["player_id"],
        ["full_name", "first_name", "last_name", "is_active"],
    )
    log_df = client.fetch_player_game_logs(season, season_type)
    upsert_rows(engine, teams, normalize_team_rows(log_df), ["team_id"], ["abbreviation", "full_name", "city", "nickname"])
    rows = normalize_player_game_rows(log_df, season, season_type)
    return upsert_rows(
        engine,
        player_game_stats,
        rows,
        ["game_id", "player_id"],
        [
            "team_id",
            "game_date",
            "season",
            "season_type",
            "matchup",
            "is_home",
            "won",
            "minutes",
            "points",
            "rebounds",
            "assists",
            "steals",
            "blocks",
            "turnovers",
            "field_goal_pct",
            "three_point_pct",
            "free_throw_pct",
            "plus_minus",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA player game logs")
    parser.add_argument("--season", default=f"{date.today().year - 1}-{str(date.today().year)[-2:]}")
    parser.add_argument("--season-type", default="Regular Season")
    args = parser.parse_args()
    count = ingest_players(args.season, args.season_type)
    print(f"upserted {count} player game stat rows for {args.season}")


if __name__ == "__main__":
    main()
