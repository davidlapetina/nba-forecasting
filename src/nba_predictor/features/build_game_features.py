from __future__ import annotations

from collections import deque
from typing import Any

import pandas as pd
from sqlalchemy import select

from nba_predictor.db import (
    game_features,
    games,
    get_engine,
    player_game_stats,
    team_daily_features,
    team_game_stats,
    upsert_rows,
)


def _rest_days_by_team(stats_df: pd.DataFrame) -> dict[tuple[int, pd.Timestamp], int]:
    rest: dict[tuple[int, pd.Timestamp], int] = {}
    stats = stats_df.copy()
    stats["game_date"] = pd.to_datetime(stats["game_date"])
    for team_id, team_rows in stats.sort_values(["game_date", "game_id"]).groupby("team_id"):
        previous_date: pd.Timestamp | None = None
        for row in team_rows.itertuples():
            game_date = pd.Timestamp(row.game_date)
            rest.setdefault((int(team_id), game_date), 7 if previous_date is None else int((game_date - previous_date).days))
            previous_date = game_date
    return rest


def _recent_zero_minute_rate_by_team(
    player_stats_df: pd.DataFrame,
    games_df: pd.DataFrame,
) -> dict[tuple[int, pd.Timestamp], float]:
    rates: dict[tuple[int, pd.Timestamp], float] = {}
    if player_stats_df.empty:
        return rates
    stats = player_stats_df.copy()
    stats["game_date"] = pd.to_datetime(stats["game_date"])
    requested_dates: dict[int, list[pd.Timestamp]] = {}
    for row in games_df.itertuples(index=False):
        game_date = pd.Timestamp(row.game_date)
        requested_dates.setdefault(int(row.home_team_id), []).append(game_date)
        requested_dates.setdefault(int(row.away_team_id), []).append(game_date)
    grouped = {
        int(team_id): team_rows.sort_values(["game_date", "game_id", "player_id"])
        for team_id, team_rows in stats.groupby("team_id")
    }
    for team_id, dates in requested_dates.items():
        team_rows = grouped.get(team_id)
        if team_rows is None:
            continue
        rows = list(team_rows.itertuples(index=False))
        cursor = 0
        recent: deque[bool] = deque()
        zero_count = 0
        for game_date in sorted(set(dates)):
            while cursor < len(rows) and pd.Timestamp(rows[cursor].game_date) < game_date:
                is_zero = rows[cursor].minutes == 0
                recent.append(is_zero)
                zero_count += int(is_zero)
                if len(recent) > 100:
                    zero_count -= int(recent.popleft())
                cursor += 1
            rates[(team_id, game_date)] = 0.0 if not recent else zero_count / len(recent)
    return rates


def compute_game_feature_rows(
    games_df: pd.DataFrame,
    daily_features_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    player_stats_df: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if games_df.empty:
        return []
    games_copy = games_df.copy()
    games_copy["game_date"] = pd.to_datetime(games_copy["game_date"])
    daily = daily_features_df.copy()
    daily["feature_date"] = pd.to_datetime(daily["feature_date"])
    feature_map = {
        (int(row.team_id), pd.Timestamp(row.feature_date)): row
        for row in daily.itertuples(index=False)
    }
    rest_map = _rest_days_by_team(stats_df)
    zero_minute_rate_map = _recent_zero_minute_rate_by_team(
        player_stats_df if player_stats_df is not None else pd.DataFrame(),
        games_copy,
    )
    rows: list[dict[str, Any]] = []
    for game in games_copy.sort_values(["game_date", "game_id"]).itertuples(index=False):
        game_date = pd.Timestamp(game.game_date)
        home = feature_map.get((int(game.home_team_id), game_date))
        away = feature_map.get((int(game.away_team_id), game_date))
        if home is None or away is None:
            continue
        rest_home = rest_map.get((int(game.home_team_id), game_date), 7)
        rest_away = rest_map.get((int(game.away_team_id), game_date), 7)
        home_zero_minute_rate = zero_minute_rate_map.get((int(game.home_team_id), game_date), 0.0)
        away_zero_minute_rate = zero_minute_rate_map.get((int(game.away_team_id), game_date), 0.0)
        rows.append(
            {
                "game_id": str(game.game_id),
                "game_date": game_date.date(),
                "season": game.season,
                "home_team_id": int(game.home_team_id),
                "away_team_id": int(game.away_team_id),
                "home_win_pct": home.win_pct,
                "away_win_pct": away.win_pct,
                "win_pct_diff": home.win_pct - away.win_pct,
                "home_last_10_win_pct": home.last_10_win_pct,
                "away_last_10_win_pct": away.last_10_win_pct,
                "last_10_win_pct_diff": home.last_10_win_pct - away.last_10_win_pct,
                "home_avg_points_last_10": home.avg_points_last_10,
                "away_avg_points_last_10": away.avg_points_last_10,
                "avg_points_diff": home.avg_points_last_10 - away.avg_points_last_10,
                "home_avg_off_rating_last_10": home.avg_off_rating_last_10,
                "away_avg_off_rating_last_10": away.avg_off_rating_last_10,
                "off_rating_diff": home.avg_off_rating_last_10 - away.avg_off_rating_last_10,
                "home_avg_def_rating_last_10": home.avg_def_rating_last_10,
                "away_avg_def_rating_last_10": away.avg_def_rating_last_10,
                "def_rating_diff": home.avg_def_rating_last_10 - away.avg_def_rating_last_10,
                "home_recent_zero_minute_rate": home_zero_minute_rate,
                "away_recent_zero_minute_rate": away_zero_minute_rate,
                "recent_zero_minute_rate_diff": home_zero_minute_rate - away_zero_minute_rate,
                "home_elo": home.elo_rating,
                "away_elo": away.elo_rating,
                "elo_diff": home.elo_rating - away.elo_rating,
                "rest_days_home": rest_home,
                "rest_days_away": rest_away,
                "rest_days_diff": rest_home - rest_away,
                "home_back_to_back": rest_home == 1,
                "away_back_to_back": rest_away == 1,
                "home_team_win": None if pd.isna(game.home_team_win) else bool(game.home_team_win),
            }
        )
    return rows


def build_game_features() -> int:
    engine = get_engine()
    with engine.connect() as conn:
        games_df = pd.read_sql(select(games), conn)
        daily_features_df = pd.read_sql(select(team_daily_features), conn)
        stats_df = pd.read_sql(select(team_game_stats), conn)
        player_stats_df = pd.read_sql(select(player_game_stats), conn)
    rows = compute_game_feature_rows(games_df, daily_features_df, stats_df, player_stats_df)
    return upsert_rows(
        engine,
        game_features,
        rows,
        ["game_id"],
        [column.name for column in game_features.columns if column.name not in {"game_id", "created_at"}],
    )


def main() -> None:
    count = build_game_features()
    print(f"upserted {count} game feature rows")


if __name__ == "__main__":
    main()
