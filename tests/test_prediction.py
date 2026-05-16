from nba_predictor.predict.predict_games import normalize_probability


def test_probability_clamps_to_valid_range() -> None:
    assert normalize_probability(-0.2) == 0.0
    assert normalize_probability(0.64) == 0.64
    assert normalize_probability(1.5) == 1.0

