from __future__ import annotations

import argparse

from nba_predictor.ingest.ingest_box_scores import ingest_box_scores, ingest_officials
from nba_predictor.ingest.ingest_games import ingest_games
from nba_predictor.ingest.ingest_players import ingest_players
from nba_predictor.ingest.ingest_rosters import ingest_rosters
from nba_predictor.ingest.ingest_schedule import ingest_schedule
from nba_predictor.ingest.ingest_team_logs import ingest_team_logs

REGULAR_SEASON = "Regular Season"
PLAYOFFS = "Playoffs"


def parse_season_start(season: str) -> int:
    return int(season.split("-", maxsplit=1)[0])


def season_label(start_year: int) -> str:
    return f"{start_year}-{str(start_year + 1)[-2:]}"


def season_range(start_season: str, end_season: str) -> list[str]:
    return [season_label(year) for year in range(parse_season_start(start_season), parse_season_start(end_season) + 1)]


def ingest_history(
    start_season: str,
    end_season: str,
    include_box_scores: bool = True,
    include_officials: bool = False,
    include_players: bool = True,
    include_rosters: bool = False,
    include_playoffs: bool = False,
    playoffs_only: bool = False,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    seasons = season_range(start_season, end_season)
    latest = seasons[-1]
    season_types = [PLAYOFFS] if playoffs_only else [REGULAR_SEASON, PLAYOFFS] if include_playoffs else [REGULAR_SEASON]
    for season in seasons:
        counts: dict[str, int] = {}
        for season_type in season_types:
            prefix = "playoff_" if season_type == PLAYOFFS else ""
            counts[f"{prefix}games"] = ingest_games(season, season_type)
            counts[f"{prefix}team_logs"] = ingest_team_logs(season, season_type)
            if include_players:
                counts[f"{prefix}player_logs"] = ingest_players(season, season_type)
            if include_box_scores:
                counts[f"{prefix}box_scores"] = ingest_box_scores(season, season_type)
        if include_rosters:
            counts.update(ingest_rosters(season))
        if season == latest:
            counts["schedule"] = ingest_schedule(season)
        if include_officials:
            counts["officials"] = ingest_officials(season)
        summary[season] = counts
        print(season, counts)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill NBA history across a season range")
    parser.add_argument("--start-season", default="1946-47")
    parser.add_argument("--end-season", required=True)
    parser.add_argument("--skip-box-scores", action="store_true")
    parser.add_argument("--include-officials", action="store_true")
    parser.add_argument("--skip-players", action="store_true")
    parser.add_argument("--include-rosters", action="store_true")
    parser.add_argument("--include-playoffs", action="store_true")
    parser.add_argument("--playoffs-only", action="store_true")
    args = parser.parse_args()
    ingest_history(
        args.start_season,
        args.end_season,
        include_box_scores=not args.skip_box_scores,
        include_officials=args.include_officials,
        include_players=not args.skip_players,
        include_rosters=args.include_rosters,
        include_playoffs=args.include_playoffs,
        playoffs_only=args.playoffs_only,
    )


if __name__ == "__main__":
    main()
