from __future__ import annotations

import argparse
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy import select

from nba_predictor.db import (
    games,
    get_engine,
    player_availability_sync_state,
    player_game_stats,
    players,
    teams,
    upsert_rows,
)
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


def normalize_player_box_score_rows(
    box_score_df: pd.DataFrame,
    game_context: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in box_score_df.itertuples(index=False):
        player_id = _maybe_int(getattr(row, "personId", None))
        game_id = getattr(row, "gameId", None)
        if player_id is None or game_id is None:
            continue
        availability_comment = getattr(row, "comment", None) or None
        normalized = {
            "game_id": str(game_id),
            "player_id": player_id,
            "availability_comment": availability_comment,
        }
        if game_context is not None:
            team_id = _maybe_int(getattr(row, "teamId", None))
            is_home = team_id == game_context["home_team_id"]
            normalized.update(
                {
                    "team_id": team_id,
                    "game_date": game_context["game_date"],
                    "season": game_context["season"],
                    "season_type": game_context["season_type"],
                    "matchup": None,
                    "is_home": is_home,
                    "won": None
                    if game_context["home_team_win"] is None
                    else bool(game_context["home_team_win"]) == is_home,
                    "minutes": 0 if availability_comment else None,
                    "points": _maybe_int(getattr(row, "points", None)),
                    "rebounds": _maybe_int(getattr(row, "reboundsTotal", None)),
                    "assists": _maybe_int(getattr(row, "assists", None)),
                    "steals": _maybe_int(getattr(row, "steals", None)),
                    "blocks": _maybe_int(getattr(row, "blocks", None)),
                    "turnovers": _maybe_int(getattr(row, "turnovers", None)),
                    "field_goal_pct": _maybe_float(getattr(row, "fieldGoalsPercentage", None)),
                    "three_point_pct": _maybe_float(getattr(row, "threePointersPercentage", None)),
                    "free_throw_pct": _maybe_float(getattr(row, "freeThrowsPercentage", None)),
                    "plus_minus": _maybe_float(getattr(row, "plusMinusPoints", None)),
                }
            )
        rows.append(normalized)
    return rows


def ingest_player_availability_comments(season: str) -> int:
    client = NBAClient()
    engine = get_engine()
    with engine.connect() as conn:
        game_rows = [
            dict(row._mapping)
            for row in conn.execute(
                select(
                    games.c.game_id,
                    games.c.game_date,
                    games.c.season,
                    games.c.season_type,
                    games.c.home_team_id,
                    games.c.away_team_id,
                    games.c.home_team_win,
                )
                .join(player_game_stats, player_game_stats.c.game_id == games.c.game_id)
                .outerjoin(
                    player_availability_sync_state,
                    player_availability_sync_state.c.game_id == games.c.game_id,
                )
                .where(
                    games.c.season == season,
                    games.c.home_score.is_not(None),
                    games.c.away_score.is_not(None),
                    player_availability_sync_state.c.game_id.is_(None),
                )
                .group_by(games.c.game_id)
            )
        ]
    count = 0
    for game_row in game_rows:
        game_id = str(game_row["game_id"])
        try:
            rows = normalize_player_box_score_rows(client.fetch_player_box_score(game_id), game_row)
        except Exception as exc:  # pragma: no cover - upstream service availability
            print(f"skipping player availability comments for {game_id}: {exc}")
            continue
        comment_rows = [row for row in rows if row["availability_comment"]]
        count += upsert_rows(
            engine,
            player_game_stats,
            comment_rows,
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
                "availability_comment",
            ],
        )
        upsert_rows(
            engine,
            player_availability_sync_state,
            [{"game_id": game_id}],
            ["game_id"],
            [],
        )
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA player game logs")
    parser.add_argument("--season", default=f"{date.today().year - 1}-{str(date.today().year)[-2:]}")
    parser.add_argument("--season-type", default="Regular Season")
    args = parser.parse_args()
    count = ingest_players(args.season, args.season_type)
    print(f"upserted {count} player game stat rows for {args.season}")


if __name__ == "__main__":
    main()
