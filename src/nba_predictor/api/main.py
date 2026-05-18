from __future__ import annotations

import json
from datetime import date
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select

from nba_predictor.config import settings
from nba_predictor.analytics.chatbot import ChatAnswer, answer_question
from nba_predictor.db import game_predictions, games, get_engine, teams
from nba_predictor.predict.predict_games import predict_matchup as predict_matchup_by_id
from nba_predictor.predict.predict_games import team_id_for_abbreviation

app = FastAPI(title="NBA OSS Game Predictor")


class HealthResponse(BaseModel):
    status: str


class PredictionResponse(BaseModel):
    game_id: str | None = None
    home_team: str
    away_team: str
    home_win_probability: float = Field(ge=0.0, le=1.0)
    away_win_probability: float = Field(ge=0.0, le=1.0)
    predicted_winner: str
    classifier_home_win_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    classifier_away_win_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    classifier_predicted_winner: str | None = None
    elo_home_win_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    elo_away_win_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    elo_predicted_winner: str | None = None
    history_home_win_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    history_away_win_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    h2h_games: int | None = None
    h2h_recent_games: int | None = None
    h2h_playoff_games: int | None = None
    forecasted_home_points: float | None = None
    forecasted_away_points: float | None = None


class MatchupRequest(BaseModel):
    home_team: str
    away_team: str
    game_date: date


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    trained_at: str
    metrics: dict[str, float]
    elo_baseline_metrics: dict[str, float] | None = None
    history_baseline_metrics: dict[str, float] | None = None
    blend_metrics: dict[str, float] | None = None
    blend_weights: dict[str, float] | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def load_model_metadata() -> dict[str, Any]:
    path = settings.model_dir / "classifier" / "model_metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _team_abbreviation_map() -> dict[int, str]:
    engine = get_engine()
    with engine.connect() as conn:
        return {int(row.team_id): row.abbreviation for row in conn.execute(select(teams))}


def list_predictions(
    prediction_date: date | None = None,
    team: str | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    engine = get_engine()
    stmt = (
        select(game_predictions)
        .select_from(game_predictions.join(games, game_predictions.c.game_id == games.c.game_id))
        .order_by(game_predictions.c.prediction_date.desc())
        .limit(limit)
    )
    if prediction_date:
        stmt = stmt.where(games.c.game_date == prediction_date)
    abbreviations = _team_abbreviation_map()
    with engine.connect() as conn:
        rows = [dict(row._mapping) for row in conn.execute(stmt)]
    response = []
    for row in rows:
        home = abbreviations.get(int(row["home_team_id"]), str(row["home_team_id"]))
        away = abbreviations.get(int(row["away_team_id"]), str(row["away_team_id"]))
        if team and team.upper() not in {home, away}:
            continue
        response.append(
            {
                "game_id": row["game_id"],
                "home_team": home,
                "away_team": away,
                "home_win_probability": row["home_win_probability"],
                "away_win_probability": row["away_win_probability"],
                "predicted_winner": abbreviations.get(
                    int(row["predicted_winner_team_id"]),
                    str(row["predicted_winner_team_id"]),
                ),
                "forecasted_home_points": row["forecasted_home_points"],
                "forecasted_away_points": row["forecasted_away_points"],
            }
        )
    return response


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/predictions", response_model=list[PredictionResponse])
def predictions(
    date_filter: date | None = Query(default=None, alias="date"),
    team: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[PredictionResponse]:
    return [PredictionResponse(**item) for item in list_predictions(date_filter, team, limit)]


@app.post("/predict/matchup", response_model=PredictionResponse)
def predict_matchup(request: MatchupRequest) -> PredictionResponse:
    try:
        home_team_id = team_id_for_abbreviation(request.home_team)
        away_team_id = team_id_for_abbreviation(request.away_team)
        result = predict_matchup_by_id(home_team_id, away_team_id, request.game_date)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return PredictionResponse(
        home_team=request.home_team.upper(),
        away_team=request.away_team.upper(),
        home_win_probability=result["home_win_probability"],
        away_win_probability=result["away_win_probability"],
        predicted_winner=request.home_team.upper()
        if result["predicted_winner_team_id"] == home_team_id
        else request.away_team.upper(),
        classifier_home_win_probability=result["classifier_home_win_probability"],
        classifier_away_win_probability=result["classifier_away_win_probability"],
        classifier_predicted_winner=request.home_team.upper()
        if result["classifier_predicted_winner_team_id"] == home_team_id
        else request.away_team.upper(),
        elo_home_win_probability=result["elo_home_win_probability"],
        elo_away_win_probability=result["elo_away_win_probability"],
        elo_predicted_winner=request.home_team.upper()
        if result["elo_predicted_winner_team_id"] == home_team_id
        else request.away_team.upper(),
        history_home_win_probability=result["history_home_win_probability"],
        history_away_win_probability=result["history_away_win_probability"],
        h2h_games=result["h2h_games"],
        h2h_recent_games=result["h2h_recent_games"],
        h2h_playoff_games=result["h2h_playoff_games"],
        forecasted_home_points=result["forecasted_home_points"],
        forecasted_away_points=result["forecasted_away_points"],
    )


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        metadata = load_model_metadata()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="model metadata not found") from exc
    return ModelInfoResponse(**metadata)


@app.post("/chat/query", response_model=ChatAnswer)
def chat_query(request: ChatRequest) -> ChatAnswer:
    try:
        return answer_question(request.question)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            detail = (
                f"ollama model '{settings.ollama_model}' is not available; "
                f"run 'ollama pull {settings.ollama_model}' or update OLLAMA_MODEL"
            )
            raise HTTPException(status_code=503, detail=detail) from exc
        raise HTTPException(status_code=503, detail=f"ollama unavailable: {exc}") from exc
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=503, detail=f"ollama unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
