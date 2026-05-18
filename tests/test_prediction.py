from datetime import date

import numpy as np

from nba_predictor.predict import predict_games
from nba_predictor.predict.predict_games import normalize_probability


def test_probability_clamps_to_valid_range() -> None:
    assert normalize_probability(-0.2) == 0.0
    assert normalize_probability(0.64) == 0.64
    assert normalize_probability(1.5) == 1.0


def test_matchup_prediction_includes_blend_and_component_signals(monkeypatch) -> None:
    class FakeModel:
        def predict_proba(self, frame):
            assert len(frame) == 1
            return np.array([[0.3, 0.7]])

    feature_row = {column: 0.0 for column in predict_games.FEATURE_COLUMNS}
    feature_row.update(
        {
            "home_elo": 2600.0,
            "away_elo": 2500.0,
            "h2h_history_home_win_probability": 0.6,
            "h2h_games": 12,
            "h2h_recent_games": 10,
            "h2h_playoff_games": 4,
            "same_season_h2h_games": 5,
            "same_season_playoff_games": 2,
            "playoff_series_game_number": 3,
            "playoff_series_home_wins": 2,
            "playoff_series_away_wins": 0,
            "forecasted_home_points": 112.0,
            "forecasted_away_points": 108.0,
        }
    )

    monkeypatch.setattr(predict_games, "get_engine", lambda: object())
    monkeypatch.setattr(
        predict_games,
        "load_model_bundle",
        lambda: (
            FakeModel(),
            {
                "model_name": "fake",
                "model_version": "v1",
                "blend_weights": {"classifier": 0.5, "elo": 0.3, "history": 0.2},
            },
        ),
    )
    monkeypatch.setattr(predict_games, "build_matchup_feature_row", lambda engine, home, away, game_date: feature_row)

    result = predict_games.predict_matchup(1, 2, date(2026, 5, 17))

    assert result["classifier_home_win_probability"] == 0.7
    assert 0.6 < result["home_win_probability"] < 0.7
    assert result["elo_home_win_probability"] > 0.5
    assert result["elo_predicted_winner_team_id"] == 1
    assert result["h2h_playoff_games"] == 4
    assert result["same_season_h2h_games"] == 5
    assert result["playoff_series_home_wins"] == 2


def test_batch_matchup_prediction_reuses_loaded_bundle(monkeypatch) -> None:
    class FakeModel:
        def predict_proba(self, frame):
            return np.array([[0.4, 0.6]])

    feature_row = {column: 0.0 for column in predict_games.FEATURE_COLUMNS}
    feature_row.update(
        {
            "home_elo": 2500.0,
            "away_elo": 2500.0,
            "h2h_history_home_win_probability": 0.5,
            "h2h_games": 0,
            "h2h_recent_games": 0,
            "h2h_playoff_games": 0,
            "same_season_h2h_games": 0,
            "same_season_playoff_games": 0,
            "playoff_series_game_number": 1,
            "playoff_series_home_wins": 0,
            "playoff_series_away_wins": 0,
            "forecasted_home_points": 110.0,
            "forecasted_away_points": 108.0,
        }
    )
    loads = {"count": 0}

    monkeypatch.setattr(predict_games, "get_engine", lambda: object())

    def fake_load_model_bundle():
        loads["count"] += 1
        return FakeModel(), {
            "model_name": "fake",
            "model_version": "v1",
            "blend_weights": {"classifier": 1.0, "elo": 0.0, "history": 0.0},
        }

    monkeypatch.setattr(predict_games, "load_model_bundle", fake_load_model_bundle)
    monkeypatch.setattr(predict_games, "build_matchup_feature_row", lambda engine, home, away, game_date: feature_row)

    results = predict_games.predict_matchups([(1, 2, date(2026, 5, 17)), (3, 4, date(2026, 5, 18))])

    assert loads["count"] == 1
    assert [result["home_win_probability"] for result in results] == [0.6, 0.6]
