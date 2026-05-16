from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime
from typing import Any

import joblib
import pandas as pd
from sqlalchemy import and_, select

from nba_predictor.config import settings
from nba_predictor.db import (
    game_predictions,
    games,
    get_engine,
    team_daily_features,
    team_game_stats,
    team_metric_forecasts,
    teams,
    upsert_rows,
)
from nba_predictor.train.train_classifier import FEATURE_COLUMNS


def normalize_probability(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def load_model_bundle() -> tuple[Any, dict[str, Any]]:
    metadata_path = settings.model_dir / "classifier" / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = settings.model_dir / "classifier" / f"game_winner_{metadata['model_name']}.joblib"
    return joblib.load(model_path), metadata


def _latest_team_feature(engine: Any, team_id: int, game_date: date) -> dict[str, Any]:
    stmt = (
        select(team_daily_features)
        .where(
            and_(
                team_daily_features.c.team_id == team_id,
                team_daily_features.c.feature_date < game_date,
            )
        )
        .order_by(team_daily_features.c.feature_date.desc())
        .limit(1)
    )
    with engine.connect() as conn:
        row = conn.execute(stmt).mappings().first()
    if row is None:
        raise ValueError(f"no historical features available for team {team_id}")
    return dict(row)


def _rest_days(engine: Any, team_id: int, game_date: date) -> int:
    stmt = (
        select(team_game_stats.c.game_date)
        .where(
            and_(
                team_game_stats.c.team_id == team_id,
                team_game_stats.c.game_date < game_date,
            )
        )
        .order_by(team_game_stats.c.game_date.desc())
        .limit(1)
    )
    with engine.connect() as conn:
        previous = conn.execute(stmt).scalar_one_or_none()
    return 7 if previous is None else max(0, (game_date - previous).days)


def _metric_forecast(engine: Any, team_id: int, game_date: date, metric_name: str, fallback: float) -> float:
    stmt = (
        select(team_metric_forecasts.c.forecast_value)
        .where(
            and_(
                team_metric_forecasts.c.team_id == team_id,
                team_metric_forecasts.c.forecast_date == game_date,
                team_metric_forecasts.c.metric_name == metric_name,
            )
        )
        .order_by(team_metric_forecasts.c.created_at.desc())
        .limit(1)
    )
    with engine.connect() as conn:
        value = conn.execute(stmt).scalar_one_or_none()
    return fallback if value is None else float(value)


def build_matchup_feature_row(engine: Any, home_team_id: int, away_team_id: int, game_date: date) -> dict[str, Any]:
    home = _latest_team_feature(engine, home_team_id, game_date)
    away = _latest_team_feature(engine, away_team_id, game_date)
    rest_home = _rest_days(engine, home_team_id, game_date)
    rest_away = _rest_days(engine, away_team_id, game_date)
    home_points = _metric_forecast(engine, home_team_id, game_date, "points", home["avg_points_last_10"])
    away_points = _metric_forecast(engine, away_team_id, game_date, "points", away["avg_points_last_10"])
    home_off = _metric_forecast(engine, home_team_id, game_date, "offensive_rating", home["avg_off_rating_last_10"])
    away_off = _metric_forecast(engine, away_team_id, game_date, "offensive_rating", away["avg_off_rating_last_10"])
    home_def = _metric_forecast(engine, home_team_id, game_date, "defensive_rating", home["avg_def_rating_last_10"])
    away_def = _metric_forecast(engine, away_team_id, game_date, "defensive_rating", away["avg_def_rating_last_10"])
    return {
        "home_win_pct": home["win_pct"],
        "away_win_pct": away["win_pct"],
        "win_pct_diff": home["win_pct"] - away["win_pct"],
        "home_last_10_win_pct": home["last_10_win_pct"],
        "away_last_10_win_pct": away["last_10_win_pct"],
        "last_10_win_pct_diff": home["last_10_win_pct"] - away["last_10_win_pct"],
        "home_avg_points_last_10": home["avg_points_last_10"],
        "away_avg_points_last_10": away["avg_points_last_10"],
        "avg_points_diff": home["avg_points_last_10"] - away["avg_points_last_10"],
        "home_avg_off_rating_last_10": home["avg_off_rating_last_10"],
        "away_avg_off_rating_last_10": away["avg_off_rating_last_10"],
        "off_rating_diff": home["avg_off_rating_last_10"] - away["avg_off_rating_last_10"],
        "home_avg_def_rating_last_10": home["avg_def_rating_last_10"],
        "away_avg_def_rating_last_10": away["avg_def_rating_last_10"],
        "def_rating_diff": home["avg_def_rating_last_10"] - away["avg_def_rating_last_10"],
        "home_elo": home["elo_rating"],
        "away_elo": away["elo_rating"],
        "elo_diff": home["elo_rating"] - away["elo_rating"],
        "rest_days_home": rest_home,
        "rest_days_away": rest_away,
        "rest_days_diff": rest_home - rest_away,
        "home_back_to_back": int(rest_home == 1),
        "away_back_to_back": int(rest_away == 1),
        "forecasted_home_points": home_points,
        "forecasted_away_points": away_points,
        "forecasted_points_diff": home_points - away_points,
        "forecasted_home_off_rating": home_off,
        "forecasted_away_off_rating": away_off,
        "forecasted_off_rating_diff": home_off - away_off,
        "forecasted_home_def_rating": home_def,
        "forecasted_away_def_rating": away_def,
        "forecasted_def_rating_diff": home_def - away_def,
    }


def predict_matchup(home_team_id: int, away_team_id: int, game_date: date) -> dict[str, Any]:
    engine = get_engine()
    model, metadata = load_model_bundle()
    feature_row = build_matchup_feature_row(engine, home_team_id, away_team_id, game_date)
    probability = normalize_probability(model.predict_proba(pd.DataFrame([feature_row])[FEATURE_COLUMNS])[:, 1][0])
    return {
        "home_win_probability": probability,
        "away_win_probability": 1.0 - probability,
        "predicted_winner_team_id": home_team_id if probability >= 0.5 else away_team_id,
        "forecasted_home_points": feature_row["forecasted_home_points"],
        "forecasted_away_points": feature_row["forecasted_away_points"],
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
    }


def predict_games_for_date(target_date: date) -> int:
    engine = get_engine()
    stmt = select(games).where(games.c.game_date == target_date)
    with engine.connect() as conn:
        scheduled = [dict(row._mapping) for row in conn.execute(stmt)]
    rows = []
    for game in scheduled:
        result = predict_matchup(game["home_team_id"], game["away_team_id"], target_date)
        rows.append(
            {
                "game_id": game["game_id"],
                "prediction_date": datetime.now(UTC),
                "home_team_id": game["home_team_id"],
                "away_team_id": game["away_team_id"],
                **result,
            }
        )
    return upsert_rows(
        engine,
        game_predictions,
        rows,
        ["game_id"],
        [
            "prediction_date",
            "home_team_id",
            "away_team_id",
            "home_win_probability",
            "away_win_probability",
            "predicted_winner_team_id",
            "forecasted_home_points",
            "forecasted_away_points",
            "model_name",
            "model_version",
        ],
    )


def team_id_for_abbreviation(abbreviation: str) -> int:
    engine = get_engine()
    with engine.connect() as conn:
        team_id = conn.execute(
            select(teams.c.team_id).where(teams.c.abbreviation == abbreviation.upper())
        ).scalar_one_or_none()
    if team_id is None:
        raise ValueError(f"unknown team abbreviation: {abbreviation}")
    return int(team_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict scheduled games")
    parser.add_argument("--date", required=True)
    args = parser.parse_args()
    count = predict_games_for_date(date.fromisoformat(args.date))
    print(f"saved {count} predictions for {args.date}")


if __name__ == "__main__":
    main()
