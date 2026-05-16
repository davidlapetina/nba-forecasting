from __future__ import annotations

import pandas as pd
from sqlalchemy import select

from nba_predictor.db import get_engine, player_game_stats, team_season_identities, teams, upsert_rows

LEGACY_DISPLAY_NAMES = {
    "ATL": "Atlanta Hawks",
    "BLT": "Baltimore Bullets",
    "BUF": "Buffalo Braves",
    "CAP": "Capital Bullets",
    "CHH": "Charlotte Hornets",
    "CHP": "Chicago Packers",
    "CHZ": "Chicago Zephyrs",
    "CIN": "Cincinnati Royals",
    "FTW": "Fort Wayne Pistons",
    "GOS": "Golden State Warriors",
    "KCK": "Kansas City Kings",
    "MEM": "Memphis Grizzlies",
    "MIH": "Milwaukee Hawks",
    "MNL": "Minneapolis Lakers",
    "NJN": "New Jersey Nets",
    "NOH": "New Orleans Hornets",
    "NOJ": "New Orleans Jazz",
    "NOK": "New Orleans/Oklahoma City Hornets",
    "NYN": "New York Nets",
    "PHL": "Philadelphia 76ers",
    "PHW": "Philadelphia Warriors",
    "ROC": "Rochester Royals",
    "SAN": "San Antonio Spurs",
    "SDC": "San Diego Clippers",
    "SDR": "San Diego Rockets",
    "SEA": "Seattle SuperSonics",
    "SFW": "San Francisco Warriors",
    "STL": "St. Louis Hawks",
    "SYR": "Syracuse Nationals",
    "TCB": "Tri-Cities Blackhawks",
    "UTH": "Utah Jazz",
    "VAN": "Vancouver Grizzlies",
}

SEASON_DISPLAY_NAME_OVERRIDES = (
    {
        "team_id": 1610612764,
        "abbreviation": "WAS",
        "start_season": "1974-75",
        "end_season": "1996-97",
        "full_name": "Washington Bullets",
    },
    {
        "team_id": 1610612766,
        "abbreviation": "CHA",
        "start_season": "2004-05",
        "end_season": "2013-14",
        "full_name": "Charlotte Bobcats",
    },
)


def _season_start_year(season: str) -> int:
    return int(season.split("-", maxsplit=1)[0])


def historical_display_name(team_id: int, abbreviation: str, season: str, current_name: str | None) -> str | None:
    season_year = _season_start_year(season)
    for override in SEASON_DISPLAY_NAME_OVERRIDES:
        if (
            team_id == override["team_id"]
            and abbreviation == override["abbreviation"]
            and _season_start_year(str(override["start_season"])) <= season_year <= _season_start_year(str(override["end_season"]))
        ):
            return str(override["full_name"])
    return LEGACY_DISPLAY_NAMES.get(abbreviation, current_name)


def compute_team_season_identity_rows(player_stats_df: pd.DataFrame, teams_df: pd.DataFrame) -> list[dict[str, object]]:
    if player_stats_df.empty:
        return []
    frame = player_stats_df[["team_id", "season", "matchup"]].dropna(subset=["team_id", "season", "matchup"]).copy()
    frame["abbreviation"] = frame["matchup"].astype(str).str.extract(r"^([A-Z]+)", expand=False)
    frame = frame.dropna(subset=["abbreviation"])
    if frame.empty:
        return []

    counts = (
        frame.groupby(["team_id", "season", "abbreviation"], as_index=False)
        .size()
        .sort_values(["team_id", "season", "size", "abbreviation"], ascending=[True, True, False, True])
    )
    winners = counts.drop_duplicates(["team_id", "season"])
    current_names = {
        int(row["team_id"]): None if pd.isna(row["full_name"]) else str(row["full_name"])
        for _, row in teams_df.iterrows()
    }
    return [
        {
            "team_id": int(row["team_id"]),
            "season": str(row["season"]),
            "abbreviation": str(row["abbreviation"]),
            "full_name": historical_display_name(
                int(row["team_id"]),
                str(row["abbreviation"]),
                str(row["season"]),
                current_names.get(int(row["team_id"])),
            ),
        }
        for _, row in winners.iterrows()
    ]


def build_team_season_identities() -> int:
    engine = get_engine()
    with engine.connect() as conn:
        player_stats_df = pd.read_sql(
            select(player_game_stats.c.team_id, player_game_stats.c.season, player_game_stats.c.matchup),
            conn,
        )
        teams_df = pd.read_sql(select(teams.c.team_id, teams.c.full_name), conn)
    rows = compute_team_season_identity_rows(player_stats_df, teams_df)
    return upsert_rows(
        engine,
        team_season_identities,
        rows,
        ["team_id", "season"],
        ["abbreviation", "full_name"],
    )


def main() -> None:
    count = build_team_season_identities()
    print(f"upserted {count} team season identity rows")


if __name__ == "__main__":
    main()
