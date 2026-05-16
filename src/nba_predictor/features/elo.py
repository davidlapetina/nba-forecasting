from __future__ import annotations

from dataclasses import dataclass


INITIAL_ELO = 2500.0
K_FACTOR = 20.0
HOME_COURT_ADVANTAGE = 65.0


def expected_home_win_probability(home_elo: float, away_elo: float) -> float:
    return 1.0 / (1.0 + 10 ** ((away_elo - (home_elo + HOME_COURT_ADVANTAGE)) / 400.0))


def update_elo(home_elo: float, away_elo: float, home_won: bool) -> tuple[float, float]:
    expected_home = expected_home_win_probability(home_elo, away_elo)
    actual_home = 1.0 if home_won else 0.0
    delta = K_FACTOR * (actual_home - expected_home)
    return home_elo + delta, away_elo - delta


@dataclass(frozen=True)
class PregameElo:
    home_elo: float
    away_elo: float
