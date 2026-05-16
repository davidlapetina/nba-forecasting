from __future__ import annotations

import pandas as pd

from nba_predictor.features.build_team_season_identities import compute_team_season_identity_rows, historical_display_name


def test_compute_team_season_identity_rows_uses_historical_codes() -> None:
    stats = pd.DataFrame(
        [
            {"team_id": 1610612760, "season": "1994-95", "matchup": "SEA vs. BOS"},
            {"team_id": 1610612760, "season": "1994-95", "matchup": "SEA @ LAL"},
            {"team_id": 1610612760, "season": "2008-09", "matchup": "OKC vs. BOS"},
        ]
    )
    teams = pd.DataFrame(
        [
            {"team_id": 1610612760, "full_name": "Oklahoma City Thunder"},
        ]
    )

    rows = compute_team_season_identity_rows(stats, teams)

    assert rows == [
        {
            "team_id": 1610612760,
            "season": "1994-95",
            "abbreviation": "SEA",
            "full_name": "Seattle SuperSonics",
        },
        {
            "team_id": 1610612760,
            "season": "2008-09",
            "abbreviation": "OKC",
            "full_name": "Oklahoma City Thunder",
        },
    ]


def test_historical_display_name_handles_same_code_renames() -> None:
    assert historical_display_name(1610612764, "WAS", "1996-97", "Washington Wizards") == "Washington Bullets"
    assert historical_display_name(1610612764, "WAS", "1997-98", "Washington Wizards") == "Washington Wizards"
    assert historical_display_name(1610612766, "CHA", "2013-14", "Charlotte Hornets") == "Charlotte Bobcats"
    assert historical_display_name(1610612766, "CHA", "2014-15", "Charlotte Hornets") == "Charlotte Hornets"
