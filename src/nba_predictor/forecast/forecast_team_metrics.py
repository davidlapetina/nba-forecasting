from __future__ import annotations

import argparse
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select

from nba_predictor.db import get_engine, team_game_stats, team_metric_forecasts, upsert_rows
from nba_predictor.forecast.timesfm_loader import load_timesfm

METRICS = [
    "points",
    "offensive_rating",
    "defensive_rating",
    "pace",
    "rebounds",
    "assists",
    "turnovers",
    "field_goal_pct",
    "three_point_pct",
]


def _fallback_forecast(series: pd.Series, horizon: int) -> tuple[np.ndarray, None]:
    value = 0.0 if series.dropna().empty else float(series.dropna().tail(10).mean())
    return np.repeat(value, horizon), None


def forecast_metric_series(series: pd.Series, horizon: int, model: Any | None) -> tuple[np.ndarray, np.ndarray | None]:
    clean = series.dropna().astype(float)
    if clean.empty or model is None:
        return _fallback_forecast(clean, horizon)
    try:
        point, quantiles = model.forecast(horizon=horizon, inputs=[clean.to_numpy()])
        return np.asarray(point[0]), np.asarray(quantiles[0]) if quantiles is not None else None
    except Exception:
        return _fallback_forecast(clean, horizon)


def build_forecast_rows(stats_df: pd.DataFrame, horizon: int, model: Any | None) -> list[dict[str, Any]]:
    if stats_df.empty:
        return []
    rows: list[dict[str, Any]] = []
    stats = stats_df.copy()
    stats["game_date"] = pd.to_datetime(stats["game_date"])
    latest_observed = stats["game_date"].max()
    first_forecast_date = max(date.today(), latest_observed.date() + timedelta(days=1))
    for team_id, team_rows in stats.sort_values("game_date").groupby("team_id"):
        for metric in METRICS:
            point, quantiles = forecast_metric_series(team_rows[metric], horizon, model)
            for offset in range(horizon):
                forecast_date = first_forecast_date + timedelta(days=offset)
                p10 = p50 = p90 = None
                if quantiles is not None and quantiles.ndim == 2 and quantiles.shape[1] >= 10:
                    p10 = float(quantiles[offset, 1])
                    p50 = float(quantiles[offset, 5])
                    p90 = float(quantiles[offset, 9])
                rows.append(
                    {
                        "team_id": int(team_id),
                        "forecast_date": forecast_date,
                        "metric_name": metric,
                        "forecast_value": float(point[offset]),
                        "forecast_p10": p10,
                        "forecast_p50": p50,
                        "forecast_p90": p90,
                        "model_name": "timesfm" if model is not None else "rolling_average_fallback",
                    }
                )
    return rows


def forecast_team_metrics(horizon: int = 7) -> int:
    engine = get_engine()
    with engine.connect() as conn:
        stats_df = pd.read_sql(select(team_game_stats), conn)
    handle = load_timesfm()
    if handle.error:
        print(handle.error)
    rows = build_forecast_rows(stats_df, horizon, handle.model)
    return upsert_rows(
        engine,
        team_metric_forecasts,
        rows,
        ["team_id", "forecast_date", "metric_name", "model_name"],
        ["forecast_value", "forecast_p10", "forecast_p50", "forecast_p90"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Forecast team metrics")
    parser.add_argument("--horizon", type=int, default=7)
    args = parser.parse_args()
    count = forecast_team_metrics(args.horizon)
    print(f"upserted {count} team metric forecast rows")


if __name__ == "__main__":
    main()
