from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, TypeVar

import pandas as pd
from nba_api.stats.endpoints import (
    boxscoretraditionalv3,
    boxscoresummaryv2,
    commonteamroster,
    leaguegamelog,
    playbyplayv3,
    scheduleleaguev2,
    scoreboardv3,
    teamgamelogs,
)
from nba_api.stats.library.parameters import MeasureTypePlayerGameLogs
from nba_api.stats.static import players as static_players
from nba_api.stats.static import teams as static_teams

T = TypeVar("T")


class NBAClient:
    def __init__(self, retries: int = 3, retry_delay: float = 2.0, rate_limit_seconds: float = 0.7) -> None:
        self.retries = retries
        self.retry_delay = retry_delay
        self.rate_limit_seconds = rate_limit_seconds
        self._last_call_at = 0.0

    def _call(self, fn: Callable[[], T]) -> T:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            elapsed = time.monotonic() - self._last_call_at
            if elapsed < self.rate_limit_seconds:
                time.sleep(self.rate_limit_seconds - elapsed)
            try:
                result = fn()
                self._last_call_at = time.monotonic()
                return result
            except Exception as exc:  # pragma: no cover - depends on upstream service
                last_error = exc
                if attempt == self.retries:
                    raise
                time.sleep(self.retry_delay * attempt)
        raise RuntimeError("unreachable NBA API retry state") from last_error

    @staticmethod
    def team_directory() -> list[dict[str, Any]]:
        return [
            {
                "team_id": team["id"],
                "abbreviation": team["abbreviation"],
                "full_name": team["full_name"],
                "city": team["city"],
                "nickname": team["nickname"],
            }
            for team in static_teams.get_teams()
        ]

    @staticmethod
    def player_directory() -> list[dict[str, Any]]:
        return [
            {
                "player_id": player["id"],
                "full_name": player["full_name"],
                "first_name": player.get("first_name"),
                "last_name": player.get("last_name"),
                "is_active": player.get("is_active"),
            }
            for player in static_players.get_players()
        ]

    def fetch_league_game_log(self, season: str, season_type: str = "Regular Season") -> pd.DataFrame:
        endpoint = self._call(
            lambda: leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="T",
            )
        )
        return endpoint.get_data_frames()[0]

    def fetch_player_game_logs(self, season: str, season_type: str = "Regular Season") -> pd.DataFrame:
        endpoint = self._call(
            lambda: leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
            )
        )
        return endpoint.get_data_frames()[0]

    def fetch_advanced_team_game_logs(self, season: str, season_type: str = "Regular Season") -> pd.DataFrame:
        endpoint = self._call(
            lambda: teamgamelogs.TeamGameLogs(
                season_nullable=season,
                season_type_nullable=season_type,
                measure_type_player_game_logs_nullable=MeasureTypePlayerGameLogs.advanced,
            )
        )
        return endpoint.team_game_logs.get_data_frame()

    def fetch_season_schedule(self, season: str) -> pd.DataFrame:
        endpoint = self._call(lambda: scheduleleaguev2.ScheduleLeagueV2(season=season))
        return endpoint.season_games.get_data_frame()

    def fetch_daily_scoreboard(self, game_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        endpoint = self._call(lambda: scoreboardv3.ScoreboardV3(game_date=game_date))
        return endpoint.game_header.get_data_frame(), endpoint.line_score.get_data_frame()

    def fetch_game_officials(self, game_id: str) -> pd.DataFrame:
        endpoint = self._call(lambda: boxscoresummaryv2.BoxScoreSummaryV2(game_id=game_id))
        return endpoint.officials.get_data_frame()

    def fetch_player_box_score(self, game_id: str) -> pd.DataFrame:
        endpoint = self._call(lambda: boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id))
        return endpoint.player_stats.get_data_frame()

    def fetch_play_by_play(self, game_id: str) -> pd.DataFrame:
        endpoint = self._call(lambda: playbyplayv3.PlayByPlayV3(game_id=game_id))
        return endpoint.play_by_play.get_data_frame()

    def fetch_team_roster(self, team_id: int, season: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        endpoint = self._call(
            lambda: commonteamroster.CommonTeamRoster(
                team_id=team_id,
                season=season,
                league_id_nullable="00",
            )
        )
        return endpoint.common_team_roster.get_data_frame(), endpoint.coaches.get_data_frame()
