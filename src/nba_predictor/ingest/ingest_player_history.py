from __future__ import annotations

import argparse

from nba_predictor.ingest.ingest_history import season_range
from nba_predictor.ingest.ingest_players import ingest_players


def ingest_player_history(start_season: str, end_season: str) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for season in season_range(start_season, end_season):
        counts = {"player_logs": ingest_players(season)}
        summary[season] = counts
        print(season, counts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill player game logs across a season range")
    parser.add_argument("--start-season", default="1946-47")
    parser.add_argument("--end-season", required=True)
    args = parser.parse_args()
    ingest_player_history(args.start_season, args.end_season)


if __name__ == "__main__":
    main()
