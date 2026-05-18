from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.calibration import calibration_curve
from sqlalchemy import select

from nba_predictor.config import settings
from nba_predictor.db import game_features, games, get_engine, team_metric_forecasts
from nba_predictor.features.elo import expected_home_win_probability
from nba_predictor.features.matchup_context import MATCHUP_CONTEXT_COLUMNS, compute_matchup_context_rows

BASE_FEATURES = [
    "home_win_pct",
    "away_win_pct",
    "win_pct_diff",
    "home_last_10_win_pct",
    "away_last_10_win_pct",
    "last_10_win_pct_diff",
    "home_avg_points_last_10",
    "away_avg_points_last_10",
    "avg_points_diff",
    "home_avg_off_rating_last_10",
    "away_avg_off_rating_last_10",
    "off_rating_diff",
    "home_avg_def_rating_last_10",
    "away_avg_def_rating_last_10",
    "def_rating_diff",
    "home_elo",
    "away_elo",
    "elo_diff",
    "rest_days_home",
    "rest_days_away",
    "rest_days_diff",
    "home_back_to_back",
    "away_back_to_back",
    *MATCHUP_CONTEXT_COLUMNS,
]

FORECAST_FEATURES = [
    "forecasted_home_points",
    "forecasted_away_points",
    "forecasted_points_diff",
    "forecasted_home_off_rating",
    "forecasted_away_off_rating",
    "forecasted_off_rating_diff",
    "forecasted_home_def_rating",
    "forecasted_away_def_rating",
    "forecasted_def_rating_diff",
]

FEATURE_COLUMNS = BASE_FEATURES + FORECAST_FEATURES


def _forecast_pivot(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    if forecasts_df.empty:
        return pd.DataFrame(columns=["team_id", "forecast_date"])
    pivot = forecasts_df.pivot_table(
        index=["team_id", "forecast_date"],
        columns="metric_name",
        values="forecast_value",
        aggfunc="last",
    ).reset_index()
    return pivot


def prepare_training_frame(features_df: pd.DataFrame, forecasts_df: pd.DataFrame) -> pd.DataFrame:
    frame = features_df.copy()
    frame.columns = frame.columns.map(str)
    frame["game_date"] = pd.to_datetime(frame["game_date"])
    forecasts_copy = forecasts_df.copy()
    forecasts_copy.columns = forecasts_copy.columns.map(str)
    forecasts = _forecast_pivot(forecasts_copy)
    if not forecasts.empty:
        forecasts["forecast_date"] = pd.to_datetime(forecasts["forecast_date"])
        home = forecasts.rename(
            columns={
                "team_id": "home_team_id",
                "forecast_date": "game_date",
                "points": "forecasted_home_points",
                "offensive_rating": "forecasted_home_off_rating",
                "defensive_rating": "forecasted_home_def_rating",
            }
        )
        away = forecasts.rename(
            columns={
                "team_id": "away_team_id",
                "forecast_date": "game_date",
                "points": "forecasted_away_points",
                "offensive_rating": "forecasted_away_off_rating",
                "defensive_rating": "forecasted_away_def_rating",
            }
        )
        frame = frame.merge(
            home[["home_team_id", "game_date", "forecasted_home_points", "forecasted_home_off_rating", "forecasted_home_def_rating"]],
            on=["home_team_id", "game_date"],
            how="left",
        )
        frame = frame.merge(
            away[["away_team_id", "game_date", "forecasted_away_points", "forecasted_away_off_rating", "forecasted_away_def_rating"]],
            on=["away_team_id", "game_date"],
            how="left",
        )
    fallbacks = {
        "forecasted_home_points": "home_avg_points_last_10",
        "forecasted_away_points": "away_avg_points_last_10",
        "forecasted_home_off_rating": "home_avg_off_rating_last_10",
        "forecasted_away_off_rating": "away_avg_off_rating_last_10",
        "forecasted_home_def_rating": "home_avg_def_rating_last_10",
        "forecasted_away_def_rating": "away_avg_def_rating_last_10",
    }
    for target, source in fallbacks.items():
        if target not in frame:
            frame[target] = frame[source]
        else:
            frame[target] = frame[target].fillna(frame[source])
    frame["forecasted_points_diff"] = frame["forecasted_home_points"] - frame["forecasted_away_points"]
    frame["forecasted_off_rating_diff"] = frame["forecasted_home_off_rating"] - frame["forecasted_away_off_rating"]
    frame["forecasted_def_rating_diff"] = frame["forecasted_home_def_rating"] - frame["forecasted_away_def_rating"]
    frame[["home_back_to_back", "away_back_to_back"]] = frame[
        ["home_back_to_back", "away_back_to_back"]
    ].astype(int)
    return frame.dropna(subset=["home_team_win"]).sort_values(["season", "game_date", "game_id"])


def add_matchup_context(features_df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    frame = features_df.copy()
    context = pd.DataFrame(compute_matchup_context_rows(games_df))
    if context.empty:
        for column in MATCHUP_CONTEXT_COLUMNS:
            frame[column] = 0.5 if column.endswith("_pct") or column.endswith("_probability") else 0
        return frame
    return frame.merge(context, on="game_id", how="left")


def time_based_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    seasons = sorted(frame["season"].dropna().unique())
    if len(seasons) >= 2:
        latest = seasons[-1]
        return frame[frame["season"] != latest], frame[frame["season"] == latest]
    cutoff = max(1, int(len(frame) * 0.8))
    return frame.iloc[:cutoff], frame.iloc[cutoff:]


def _build_model(model_name: str) -> tuple[Any, str]:
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return (
            XGBClassifier(
                n_estimators=250,
                learning_rate=0.03,
                max_depth=4,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="logloss",
                random_state=42,
            ),
            "xgboost",
        )
    try:
        from lightgbm import LGBMClassifier

        return (
            LGBMClassifier(
                n_estimators=250,
                learning_rate=0.03,
                num_leaves=31,
                random_state=42,
            ),
            "lightgbm",
        )
    except (ImportError, OSError) as exc:
        from sklearn.ensemble import HistGradientBoostingClassifier

        print(f"lightgbm unavailable, using sklearn fallback: {exc}")
        return (
            HistGradientBoostingClassifier(
                learning_rate=0.03,
                max_iter=250,
                random_state=42,
            ),
            "sklearn_hist_gradient_boosting",
        )


def evaluate_predictions(y_true: pd.Series, probabilities: pd.Series) -> dict[str, float]:
    labels = probabilities >= 0.5
    return {
        "accuracy": float(accuracy_score(y_true, labels)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
    }


def elo_probabilities(frame: pd.DataFrame) -> pd.Series:
    return frame.apply(
        lambda row: expected_home_win_probability(row["home_elo"], row["away_elo"]),
        axis=1,
    ).astype(float)


def history_probabilities(frame: pd.DataFrame) -> pd.Series:
    return frame["h2h_history_home_win_probability"].astype(float)


def blended_probabilities(
    classifier_probabilities: pd.Series,
    elo_probability: pd.Series,
    history_probability: pd.Series,
    weights: dict[str, float],
) -> pd.Series:
    return (
        classifier_probabilities.astype(float) * weights["classifier"]
        + elo_probability.astype(float) * weights["elo"]
        + history_probability.astype(float) * weights["history"]
    )


def _weight_grid(step: int = 20) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = []
    for classifier_units in range(step + 1):
        for elo_units in range(step + 1 - classifier_units):
            history_units = step - classifier_units - elo_units
            candidates.append(
                {
                    "classifier": classifier_units / step,
                    "elo": elo_units / step,
                    "history": history_units / step,
                }
            )
    return candidates


def choose_blend_weights(
    y_true: pd.Series,
    classifier_probability: pd.Series,
    elo_probability: pd.Series,
    history_probability: pd.Series,
) -> dict[str, float]:
    best_weights = {"classifier": 0.6, "elo": 0.3, "history": 0.1}
    best_loss = float("inf")
    for weights in _weight_grid():
        probabilities = blended_probabilities(
            classifier_probability,
            elo_probability,
            history_probability,
            weights,
        )
        candidate_loss = log_loss(y_true, probabilities)
        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_weights = weights
    return best_weights


def calibration_curve_data(y_true: pd.Series, probabilities: pd.Series) -> dict[str, list[float]]:
    prob_true, prob_pred = calibration_curve(y_true, probabilities, n_bins=10, strategy="uniform")
    return {
        "prob_true": [float(value) for value in prob_true],
        "prob_pred": [float(value) for value in prob_pred],
    }


def load_training_data() -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        features_df = pd.read_sql(select(game_features), conn)
        forecasts_df = pd.read_sql(select(team_metric_forecasts), conn)
        games_df = pd.read_sql(select(games), conn)
    return prepare_training_frame(add_matchup_context(features_df, games_df), forecasts_df)


def train_classifier(model_name: str | None = None) -> dict[str, Any]:
    frame = load_training_data()
    if frame.empty:
        raise ValueError("no completed game features available for training")
    train_df, valid_df = time_based_split(frame)
    if train_df.empty or valid_df.empty:
        raise ValueError("time-based split requires both train and validation rows")
    requested_model = model_name or settings.classifier_model
    calibration_train_df, calibration_df = time_based_split(train_df)
    calibration_model, _ = _build_model(requested_model)
    if calibration_train_df.empty or calibration_df.empty:
        blend_weights = {"classifier": 0.6, "elo": 0.3, "history": 0.1}
    else:
        calibration_model.fit(calibration_train_df[FEATURE_COLUMNS], calibration_train_df["home_team_win"].astype(int))
        calibration_classifier = pd.Series(
            calibration_model.predict_proba(calibration_df[FEATURE_COLUMNS])[:, 1],
            index=calibration_df.index,
        )
        blend_weights = choose_blend_weights(
            calibration_df["home_team_win"].astype(int),
            calibration_classifier,
            elo_probabilities(calibration_df),
            history_probabilities(calibration_df),
        )

    model, selected_model = _build_model(requested_model)
    model.fit(train_df[FEATURE_COLUMNS], train_df["home_team_win"].astype(int))
    probabilities = pd.Series(model.predict_proba(valid_df[FEATURE_COLUMNS])[:, 1], index=valid_df.index)
    y_valid = valid_df["home_team_win"].astype(int)
    elo_probability = elo_probabilities(valid_df)
    history_probability = history_probabilities(valid_df)
    blend_probability = blended_probabilities(probabilities, elo_probability, history_probability, blend_weights)
    metrics = evaluate_predictions(y_valid, probabilities)
    elo_metrics = evaluate_predictions(y_valid, elo_probability)
    history_metrics = evaluate_predictions(y_valid, history_probability)
    blend_metrics = evaluate_predictions(y_valid, blend_probability)
    classifier_dir = settings.model_dir / "classifier"
    classifier_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = classifier_dir / f"game_winner_{selected_model}.joblib"
    joblib.dump(model, artifact_path)
    metadata = {
        "model_name": selected_model,
        "requested_model_name": requested_model,
        "model_version": "v1",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "features": FEATURE_COLUMNS,
        "metrics": metrics,
        "elo_baseline_metrics": elo_metrics,
        "history_baseline_metrics": history_metrics,
        "blend_metrics": blend_metrics,
        "blend_weights": blend_weights,
        "calibration_curve": calibration_curve_data(y_valid, probabilities),
    }
    (classifier_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return metadata


def main() -> None:
    train_classifier()


if __name__ == "__main__":
    main()
