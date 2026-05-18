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


def test_predict_matchup_includes_elo_baseline(monkeypatch) -> None:
    monkeypatch.setattr(main, "team_id_for_abbreviation", lambda abbreviation: {"BOS": 1, "NYK": 2}[abbreviation])
    monkeypatch.setattr(
        main,
        "predict_matchup_by_id",
        lambda home_team_id, away_team_id, game_date: {
            "home_win_probability": 0.64,
            "away_win_probability": 0.36,
            "predicted_winner_team_id": 1,
            "classifier_home_win_probability": 0.66,
            "classifier_away_win_probability": 0.34,
            "classifier_predicted_winner_team_id": 1,
            "elo_home_win_probability": 0.58,
            "elo_away_win_probability": 0.42,
            "elo_predicted_winner_team_id": 1,
            "history_home_win_probability": 0.54,
            "history_away_win_probability": 0.46,
            "h2h_games": 20,
            "h2h_recent_games": 10,
            "h2h_playoff_games": 6,
            "same_season_h2h_games": 5,
            "same_season_playoff_games": 3,
            "playoff_series_game_number": 4,
            "playoff_series_home_wins": 2,
            "playoff_series_away_wins": 1,
            "forecasted_home_points": 116.2,
            "forecasted_away_points": 110.8,
        },
    )

    response = client.post(
        "/predict/matchup",
        json={"home_team": "BOS", "away_team": "NYK", "game_date": "2026-05-17"},
    )

    assert response.status_code == 200
    assert response.json()["elo_home_win_probability"] == 0.58
    assert response.json()["elo_predicted_winner"] == "BOS"
    assert response.json()["classifier_home_win_probability"] == 0.66
    assert response.json()["h2h_playoff_games"] == 6
    assert response.json()["same_season_h2h_games"] == 5
    assert response.json()["playoff_series_home_wins"] == 2
