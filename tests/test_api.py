from __future__ import annotations

from fastapi.testclient import TestClient

from nba_predictor.api import main


client = TestClient(main.app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predictions(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "list_predictions",
        lambda prediction_date=None, team=None, limit=100: [
            {
                "game_id": "g1",
                "home_team": "BOS",
                "away_team": "NYK",
                "home_win_probability": 0.64,
                "away_win_probability": 0.36,
                "predicted_winner": "BOS",
                "forecasted_home_points": 116.2,
                "forecasted_away_points": 110.8,
            }
        ],
    )
    response = client.get("/predictions")
    assert response.status_code == 200
    assert response.json()[0]["predicted_winner"] == "BOS"


def test_model_info(monkeypatch) -> None:
    monkeypatch.setattr(
        main,
        "load_model_metadata",
        lambda: {
            "model_name": "lightgbm",
            "model_version": "v1",
            "trained_at": "2026-01-01T00:00:00+00:00",
            "features": [],
            "metrics": {"accuracy": 0.7, "roc_auc": 0.75, "log_loss": 0.6, "brier_score": 0.2},
        },
    )
    response = client.get("/model/info")
    assert response.status_code == 200
    assert response.json()["model_name"] == "lightgbm"

