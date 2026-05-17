from __future__ import annotations

from datetime import date
from typing import Any

from nba_predictor.features.build_game_features import build_game_features
from nba_predictor.features.build_team_daily_features import build_team_daily_features
from nba_predictor.features.build_team_elo_history import build_team_elo_history
from nba_predictor.features.build_team_season_identities import build_team_season_identities
from nba_predictor.forecast.forecast_team_metrics import forecast_team_metrics
from nba_predictor.ingest.ingest_box_scores import ingest_box_scores
from nba_predictor.ingest.ingest_games import ingest_games
from nba_predictor.ingest.ingest_players import ingest_players
from nba_predictor.ingest.ingest_play_by_play import ingest_play_by_play
from nba_predictor.ingest.ingest_rosters import ingest_rosters
from nba_predictor.ingest.ingest_schedule import ingest_schedule
from nba_predictor.ingest.ingest_team_logs import ingest_team_logs
from nba_predictor.jobs.refresh_pipeline import run_data_refresh, run_full_refresh
from nba_predictor.predict.predict_games import predict_games_for_date
from nba_predictor.train.evaluate_model import evaluate_saved_model
from nba_predictor.train.train_classifier import train_classifier


def run_ingest(season: str) -> dict[str, int]:
    counts = {
        "schedule": ingest_schedule(season),
        "games": ingest_games(season),
        "team_logs": ingest_team_logs(season),
        "box_scores": ingest_box_scores(season),
        "player_logs": ingest_players(season),
        "play_by_play_events": ingest_play_by_play(season),
    }
    counts.update(ingest_rosters(season))
    return counts


def run_features() -> dict[str, int]:
    return {
        "team_season_identities": build_team_season_identities(),
        "team_elo_history": build_team_elo_history(),
        "team_features": build_team_daily_features(),
        "game_features": build_game_features(),
    }


def run_forecast() -> dict[str, int]:
    return {"forecasts": forecast_team_metrics()}


def run_train() -> dict[str, Any]:
    metadata = train_classifier()
    return {
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        **metadata["metrics"],
    }


def run_evaluate() -> dict[str, float]:
    return evaluate_saved_model()


def run_predict(target_date: date) -> dict[str, int]:
    return {"predictions": predict_games_for_date(target_date)}


def run_refresh(season: str, target_date: date) -> dict[str, int]:
    return run_data_refresh([season], target_date)


def run_refresh_full(season: str, target_date: date) -> dict[str, int]:
    return run_full_refresh([season], target_date)


def run_full_pipeline(season: str, target_date: date) -> dict[str, Any]:
    results: dict[str, Any] = {}
    results.update(run_ingest(season))
    results.update(run_features())
    results.update(run_forecast())
    results.update(run_train())
    results.update(run_evaluate())
    results.update(run_predict(target_date))
    return results
