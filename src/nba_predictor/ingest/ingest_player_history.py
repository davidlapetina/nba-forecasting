from __future__ import annotations

import argparse

from nba_predictor.ingest.ingest_history import PLAYOFFS, REGULAR_SEASON, season_range
from nba_predictor.ingest.ingest_players import ingest_players


def ingest_player_history(
    start_season: str,
    end_season: str,
    include_playoffs: bool = False,
    playoffs_only: bool = False,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    season_types = [PLAYOFFS] if playoffs_only else [REGULAR_SEASON, PLAYOFFS] if include_playoffs else [REGULAR_SEASON]
    for season in season_range(start_season, end_season):
        counts = {}
        for season_type in season_types:
            prefix = "playoff_" if season_type == PLAYOFFS else ""
            counts[f"{prefix}player_logs"] = ingest_players(season, season_type)
        summary[season] = counts
        print(season, counts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill player game logs across a season range")
    parser.add_argument("--start-season", default="1946-47")
    parser.add_argument("--end-season", required=True)
    parser.add_argument("--include-playoffs", action="store_true")
    parser.add_argument("--playoffs-only", action="store_true")
    args = parser.parse_args()
    ingest_player_history(
        args.start_season,
        args.end_season,
        include_playoffs=args.include_playoffs,
        playoffs_only=args.playoffs_only,
    )


if __name__ == "__main__":
    main()
