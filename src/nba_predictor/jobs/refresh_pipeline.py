from __future__ import annotations

import argparse
import os
import time
from datetime import UTC, date, datetime

from sqlalchemy import select

from nba_predictor.db import get_engine, scheduler_sync_state, upsert_rows
from nba_predictor.features.build_game_features import build_game_features
from nba_predictor.features.build_team_elo_history import build_team_elo_history
from nba_predictor.features.build_team_daily_features import build_team_daily_features
from nba_predictor.forecast.forecast_team_metrics import forecast_team_metrics
from nba_predictor.ingest.ingest_box_scores import ingest_box_scores
from nba_predictor.ingest.ingest_games import ingest_games
from nba_predictor.ingest.ingest_players import ingest_players
from nba_predictor.ingest.ingest_rosters import ingest_rosters
from nba_predictor.ingest.ingest_schedule import ingest_schedule
from nba_predictor.ingest.ingest_team_logs import ingest_team_logs
from nba_predictor.predict.predict_games import predict_games_for_date
from nba_predictor.train.evaluate_model import evaluate_saved_model
from nba_predictor.train.train_classifier import train_classifier


def season_for_date(target_date: date) -> str:
    start_year = target_date.year if target_date.month >= 7 else target_date.year - 1
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def seasons_touched_since(last_sync: datetime | None, target_date: date) -> list[str]:
    if last_sync is None:
        return [season_for_date(target_date)]
    cursor = last_sync.date()
    seasons: list[str] = []
    while cursor <= target_date:
        season = season_for_date(cursor)
        if season not in seasons:
            seasons.append(season)
        if cursor.month >= 7:
            cursor = date(cursor.year + 1, 1, 1)
        else:
            cursor = date(cursor.year, 7, 1)
    current_season = season_for_date(target_date)
    if current_season not in seasons:
        seasons.append(current_season)
    return seasons


def _get_last_successful_sync(job_name: str) -> datetime | None:
    engine = get_engine()
    with engine.connect() as conn:
        return conn.scalar(
            select(scheduler_sync_state.c.last_successful_sync).where(scheduler_sync_state.c.job_name == job_name)
        )


def _mark_successful_sync(job_name: str) -> None:
    now = datetime.now(UTC).replace(tzinfo=None)
    upsert_rows(
        get_engine(),
        scheduler_sync_state,
        [{"job_name": job_name, "last_successful_sync": now, "updated_at": now}],
        ["job_name"],
        ["last_successful_sync", "updated_at"],
    )


def _sum_counts(target: dict[str, int], addition: dict[str, int]) -> None:
    for key, value in addition.items():
        target[key] = target.get(key, 0) + value


def run_data_refresh(seasons: list[str], predict_date: date, job_name: str = "daily_refresh") -> dict[str, int]:
    counts: dict[str, int] = {}
    for season in seasons:
        _sum_counts(
            counts,
            {
                "schedule": ingest_schedule(season),
                "games": ingest_games(season),
                "team_logs": ingest_team_logs(season),
                "box_scores": ingest_box_scores(season),
                "player_logs": ingest_players(season),
            },
        )
        _sum_counts(counts, ingest_rosters(season))
    counts.update(
        {
        "team_elo_history": build_team_elo_history(),
        "team_features": build_team_daily_features(),
        "game_features": build_game_features(),
        "forecasts": forecast_team_metrics(),
        }
    )
    counts["predictions"] = predict_games_for_date(predict_date)
    _mark_successful_sync(job_name)
    return counts


def run_full_refresh(seasons: list[str], predict_date: date) -> dict[str, int]:
    counts = run_data_refresh(seasons, predict_date, job_name="full_refresh")
    train_classifier()
    evaluate_saved_model()
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or schedule the NBA refresh pipeline")
    parser.add_argument("--season")
    parser.add_argument("--predict-date", default=date.today().isoformat())
    parser.add_argument("--interval-hours", type=int, default=int(os.getenv("REFRESH_INTERVAL_HOURS", "24")))
    parser.add_argument("--retrain", action="store_true", help="Also retrain and evaluate the classifier after refreshing data")
    parser.add_argument("--run-once", action="store_true")
    args = parser.parse_args()

    predict_date = date.fromisoformat(args.predict_date)
    while True:
        job_name = "full_refresh" if args.retrain else "daily_refresh"
        seasons = [args.season] if args.season else seasons_touched_since(_get_last_successful_sync(job_name), predict_date)
        counts = run_full_refresh(seasons, predict_date) if args.retrain else run_data_refresh(seasons, predict_date)
        counts["seasons_refreshed"] = len(seasons)
        print(counts)
        if args.run_once:
            return
        time.sleep(max(1, args.interval_hours) * 60 * 60)


if __name__ == "__main__":
    main()
