from nba_predictor.features.elo import expected_home_win_probability, update_elo


def test_expected_home_probability_uses_home_court_adjustment() -> None:
    probability = expected_home_win_probability(2500.0, 2500.0)
    assert round(probability, 3) == 0.592


def test_elo_update_is_zero_sum() -> None:
    home, away = update_elo(2500.0, 2500.0, True)
    assert home > 2500.0
    assert away < 2500.0
    assert round(home + away, 8) == 5000.0
