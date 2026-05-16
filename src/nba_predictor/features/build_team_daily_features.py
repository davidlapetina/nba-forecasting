from __future__ import annotations

from collections import defaultdict
from typing import Any

import pandas as pd
from sqlalchemy import select

from nba_predictor.db import games, get_engine, team_daily_features, team_game_stats, upsert_rows
from nba_predictor.features.elo import INITIAL_ELO, update_elo


ROLLING_METRICS = {
    "points": ("avg_points_last_5", 5),
    "points_10": ("avg_points_last_10", 10),
    "rebounds": ("avg_rebounds_last_10", 10),
    "assists": ("avg_assists_last_10", 10),
    "turnovers": ("avg_turnovers_last_10", 10),
    "offensive_rating": ("avg_off_rating_last_10", 10),
    "defensive_rating": ("avg_def_rating_last_10", 10),
    "pace": ("avg_pace_last_10", 10),
}


def _mean_or_zero(series: pd.Series) -> float:
    return 0.0 if series.dropna().empty else float(series.dropna().mean())


def _win_history(stats: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    outcomes = games_df[["game_id", "home_team_id", "away_team_id", "home_team_win"]].copy()
    merged = stats.merge(outcomes, on="game_id", how="left")
    merged["team_win"] = merged.apply(
        lambda row: bool(row["home_team_win"]) if row["team_id"] == row["home_team_id"] else not bool(row["home_team_win"]),
        axis=1,
    )
    return merged


def compute_pregame_elo(games_df: pd.DataFrame) -> dict[tuple[int, pd.Timestamp], float]:
    ratings: defaultdict[int, float] = defaultdict(lambda: INITIAL_ELO)
    pregame: dict[tuple[int, pd.Timestamp], float] = {}
    played = games_df.dropna(subset=["home_team_win"]).copy()
    if {"home_score", "away_score"}.issubset(played.columns):
        played = played[(played["home_score"] > 0) & (played["away_score"] > 0)]
    played = played[
        (played["home_team_id"].astype(int) > 0)
        & (played["away_team_id"].astype(int) > 0)
        & (played["home_team_id"] != played["away_team_id"])
    ].sort_values(["game_date", "game_id"])
    for row in played.itertuples():
        game_date = pd.Timestamp(row.game_date)
        home_elo = ratings[int(row.home_team_id)]
        away_elo = ratings[int(row.away_team_id)]
        pregame.setdefault((int(row.home_team_id), game_date), home_elo)
        pregame.setdefault((int(row.away_team_id), game_date), away_elo)
        ratings[int(row.home_team_id)], ratings[int(row.away_team_id)] = update_elo(
            home_elo,
            away_elo,
            bool(row.home_team_win),
        )
    return pregame


def compute_team_daily_feature_rows(stats_df: pd.DataFrame, games_df: pd.DataFrame) -> list[dict[str, Any]]:
    if stats_df.empty:
        return []
    stats = stats_df.copy()
    stats["game_date"] = pd.to_datetime(stats["game_date"])
    games_copy = games_df.copy()
    games_copy["game_date"] = pd.to_datetime(games_copy["game_date"])
    enriched = _win_history(stats, games_copy)
    elo_by_team_date = compute_pregame_elo(games_copy)
    rows: list[dict[str, Any]] = []
    for team_id, team_rows in enriched.sort_values(["game_date", "game_id"]).groupby("team_id"):
        prior = pd.DataFrame(columns=team_rows.columns)
        for feature_date, day_rows in team_rows.groupby("game_date", sort=True):
            feature_date = pd.Timestamp(feature_date)
            wins = prior["team_win"].astype(float) if not prior.empty else pd.Series(dtype=float)
            rows.append(
                {
                    "team_id": int(team_id),
                    "feature_date": feature_date.date(),
                    "games_played": int(len(prior)),
                    "win_pct": 0.5 if wins.empty else float(wins.mean()),
                    "last_5_win_pct": 0.5 if wins.empty else float(wins.tail(5).mean()),
                    "last_10_win_pct": 0.5 if wins.empty else float(wins.tail(10).mean()),
                    "avg_points_last_5": _mean_or_zero(prior["points"].tail(5)) if not prior.empty else 0.0,
                    "avg_points_last_10": _mean_or_zero(prior["points"].tail(10)) if not prior.empty else 0.0,
                    "avg_rebounds_last_10": _mean_or_zero(prior["rebounds"].tail(10)) if not prior.empty else 0.0,
                    "avg_assists_last_10": _mean_or_zero(prior["assists"].tail(10)) if not prior.empty else 0.0,
                    "avg_turnovers_last_10": _mean_or_zero(prior["turnovers"].tail(10)) if not prior.empty else 0.0,
                    "avg_off_rating_last_10": _mean_or_zero(prior["offensive_rating"].tail(10)) if not prior.empty else 0.0,
                    "avg_def_rating_last_10": _mean_or_zero(prior["defensive_rating"].tail(10)) if not prior.empty else 0.0,
                    "avg_pace_last_10": _mean_or_zero(prior["pace"].tail(10)) if not prior.empty else 0.0,
                    "elo_rating": elo_by_team_date.get((int(team_id), feature_date), INITIAL_ELO),
                }
            )
            prior = pd.concat([prior, day_rows], ignore_index=True)
    return rows


def build_team_daily_features() -> int:
    engine = get_engine()
    with engine.connect() as conn:
        stats_df = pd.read_sql(select(team_game_stats), conn)
        games_df = pd.read_sql(select(games), conn)
    rows = compute_team_daily_feature_rows(stats_df, games_df)
    return upsert_rows(
        engine,
        team_daily_features,
        rows,
        ["team_id", "feature_date"],
        [column.name for column in team_daily_features.columns if column.name not in {"id", "team_id", "feature_date", "created_at"}],
    )


def main() -> None:
    count = build_team_daily_features()
    print(f"upserted {count} team daily feature rows")


if __name__ == "__main__":
    main()
