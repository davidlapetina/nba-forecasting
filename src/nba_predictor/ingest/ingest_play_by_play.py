from __future__ import annotations

import argparse
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd
from sqlalchemy import and_, select

from nba_predictor.db import games, get_engine, play_by_play_events, play_by_play_sync_state, upsert_rows
from nba_predictor.ingest.ingest_history import season_range
from nba_predictor.ingest.nba_client import NBAClient

PLAY_BY_PLAY_START_SEASON = "1996-97"


def _is_missing(value: Any) -> bool:
    return value is None or value == "" or pd.isna(value)


def _maybe_int(value: Any) -> int | None:
    return None if _is_missing(value) else int(value)


def _maybe_float(value: Any) -> float | None:
    return None if _is_missing(value) else float(value)


def _maybe_bool(value: Any) -> bool | None:
    return None if _is_missing(value) else bool(value)


def normalize_play_by_play_rows(
    play_by_play_df: pd.DataFrame,
    game_id: str,
    season: str,
    game_date: date,
) -> list[dict[str, Any]]:
    rows_by_action: dict[int, dict[str, Any]] = {}
    for row in play_by_play_df.itertuples(index=False):
        action_number = _maybe_int(getattr(row, "actionNumber", None))
        if action_number is None:
            continue
        rows_by_action[action_number] = {
            "game_id": game_id,
            "season": season,
            "game_date": game_date,
            "action_number": action_number,
            "action_id": _maybe_int(getattr(row, "actionId", None)),
            "period": _maybe_int(getattr(row, "period", None)),
            "clock": getattr(row, "clock", None),
            "team_id": _maybe_int(getattr(row, "teamId", None)),
            "team_tricode": getattr(row, "teamTricode", None),
            "person_id": _maybe_int(getattr(row, "personId", None)),
            "player_name": getattr(row, "playerName", None),
            "player_name_i": getattr(row, "playerNameI", None),
            "x_legacy": _maybe_float(getattr(row, "xLegacy", None)),
            "y_legacy": _maybe_float(getattr(row, "yLegacy", None)),
            "shot_distance": _maybe_float(getattr(row, "shotDistance", None)),
            "shot_result": getattr(row, "shotResult", None),
            "is_field_goal": _maybe_bool(getattr(row, "isFieldGoal", None)),
            "score_home": _maybe_int(getattr(row, "scoreHome", None)),
            "score_away": _maybe_int(getattr(row, "scoreAway", None)),
            "points_total": _maybe_int(getattr(row, "pointsTotal", None)),
            "location": getattr(row, "location", None),
            "description": getattr(row, "description", None),
            "action_type": getattr(row, "actionType", None),
            "sub_type": getattr(row, "subType", None),
            "video_available": _maybe_bool(getattr(row, "videoAvailable", None)),
        }
    return list(rows_by_action.values())


def _completed_games_for_season(season: str, include_existing: bool) -> list[dict[str, Any]]:
    stmt = select(games.c.game_id, games.c.game_date).where(
        and_(
            games.c.season == season,
            games.c.home_score.is_not(None),
            games.c.away_score.is_not(None),
        )
    )
    if not include_existing:
        stmt = stmt.where(
            ~games.c.game_id.in_(select(play_by_play_events.c.game_id).distinct()),
            ~games.c.game_id.in_(select(play_by_play_sync_state.c.game_id)),
        )
    with get_engine().connect() as conn:
        return [dict(row._mapping) for row in conn.execute(stmt.order_by(games.c.game_date, games.c.game_id))]


def ingest_play_by_play(season: str, include_existing: bool = False) -> int:
    client = NBAClient()
    engine = get_engine()
    inserted = 0
    for game in _completed_games_for_season(season, include_existing):
        frame = client.fetch_play_by_play(str(game["game_id"]))
        rows = normalize_play_by_play_rows(frame, str(game["game_id"]), season, game["game_date"])
        inserted += upsert_rows(
            engine,
            play_by_play_events,
            rows,
            ["game_id", "action_number"],
            [
                column.name
                for column in play_by_play_events.columns
                if column.name not in {"id", "game_id", "action_number", "created_at"}
            ],
        )
        upsert_rows(
            engine,
            play_by_play_sync_state,
            [
                {
                    "game_id": str(game["game_id"]),
                    "fetched_at": datetime.now(UTC).replace(tzinfo=None),
                    "event_count": len(rows),
                }
            ],
            ["game_id"],
            ["fetched_at", "event_count"],
        )
    return inserted


def ingest_play_by_play_history(start_season: str, end_season: str, include_existing: bool = False) -> dict[str, int]:
    start_season = max(start_season, PLAY_BY_PLAY_START_SEASON)
    return {
        season: ingest_play_by_play(season, include_existing=include_existing)
        for season in season_range(start_season, end_season)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest NBA play-by-play events")
    parser.add_argument("--season", default=f"{date.today().year - 1}-{str(date.today().year)[-2:]}")
    parser.add_argument("--start-season")
    parser.add_argument("--end-season")
    parser.add_argument("--include-existing", action="store_true")
    args = parser.parse_args()
    if args.start_season or args.end_season:
        if not args.start_season or not args.end_season:
            raise ValueError("--start-season and --end-season must be provided together")
        print(ingest_play_by_play_history(args.start_season, args.end_season, include_existing=args.include_existing))
        return
    count = ingest_play_by_play(args.season, include_existing=args.include_existing)
    print(f"upserted {count} play-by-play events for {args.season}")


if __name__ == "__main__":
    main()
