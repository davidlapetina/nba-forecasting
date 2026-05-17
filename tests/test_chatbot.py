from __future__ import annotations

import pytest
import pandas as pd
from pandas.errors import DatabaseError

from nba_predictor.analytics import chatbot
from nba_predictor.analytics.chatbot import (
    QueryPlan,
    _fallback_summary,
    _heuristic_plan,
    create_query_plan,
    execute_sql_with_retries,
    validate_read_only_sql,
)


def test_validate_read_only_sql_adds_limit() -> None:
    sql = validate_read_only_sql("select abbreviation from teams")
    assert "LIMIT 100" in sql


def test_validate_read_only_sql_rejects_writes() -> None:
    with pytest.raises(ValueError):
        validate_read_only_sql("delete from teams")


def test_validate_read_only_sql_allows_ctes() -> None:
    sql = validate_read_only_sql(
        "with team_results as (select abbreviation from teams) select abbreviation from team_results"
    )
    assert "WITH team_results AS" in sql


def test_prediction_plan_shape() -> None:
    plan = QueryPlan(
        mode="prediction",
        home_team="BOS",
        away_team="NYK",
        game_date="2026-01-15",
    )
    assert plan.home_team == "BOS"


def test_heuristic_plan_handles_current_elo_query() -> None:
    plan = _heuristic_plan("Show the highest ELO teams right now.")
    assert plan is not None
    assert plan.mode == "sql"
    assert "latest_team_elo" in str(plan.sql)


def test_heuristic_plan_handles_player_ppg_query() -> None:
    plan = _heuristic_plan("Which players scored the most points per game for BOS in 2025-26?")
    assert plan is not None
    assert plan.mode == "sql"
    assert "player_game_stats" in str(plan.sql)
    assert "coalesce(s.minutes, 0) > 0" in str(plan.sql)


def test_heuristic_plan_handles_player_rebound_query() -> None:
    plan = _heuristic_plan("Who is the best rebounder in 2026?")
    assert plan is not None
    assert plan.mode == "sql"
    assert "avg_rebounds" in str(plan.sql)
    assert "2025-26" in str(plan.sql)
    assert "coalesce(s.minutes, 0) > 0" in str(plan.sql)


def test_heuristic_plan_handles_best_team_year_query() -> None:
    plan = _heuristic_plan("Tell me which team had the best results in 1995.")
    assert plan is not None
    assert plan.mode == "sql"
    assert "1994-95" in str(plan.sql)
    assert "home_team_win" in str(plan.sql)
    assert "team_season_identities" in str(plan.sql)


def test_fallback_summary_handles_player_rows() -> None:
    summary = _fallback_summary("sql", [{"full_name": "Jaylen Brown", "games": 71, "avg_points": 28.7}])
    assert summary == "Jaylen Brown leads the result set at 28.7 points per game across 71 games."


def test_fallback_summary_handles_player_rebound_rows() -> None:
    summary = _fallback_summary("sql", [{"full_name": "Nikola Jokic", "games": 70, "avg_rebounds": 12.1}])
    assert summary == "Nikola Jokic leads the result set at 12.1 rebounds per game across 70 games."


def test_fallback_summary_handles_team_record_rows() -> None:
    summary = _fallback_summary(
        "sql",
        [{"abbreviation": "SAS", "season": "1994-95", "wins": 62, "losses": 20, "win_pct": 0.756}],
    )
    assert summary == "SAS had the best regular-season record in 1994-95 at 62-20 (75.6%)."


def test_create_query_plan_retries_invalid_payload() -> None:
    class FakeClient:
        def __init__(self) -> None:
            self.responses = iter(
                [
                    '{"error":"I could not build a query."}',
                    '{"mode":"sql","sql":"select abbreviation from teams","rationale":"fixed"}',
                ]
            )

        def chat(self, messages, response_format=None):  # type: ignore[no-untyped-def]
            return next(self.responses)

    plan = create_query_plan("List teams.", FakeClient())
    assert plan.mode == "sql"
    assert plan.sql == "select abbreviation from teams"


def test_create_query_plan_raises_clean_error_after_invalid_payloads() -> None:
    class FakeClient:
        def chat(self, messages, response_format=None):  # type: ignore[no-untyped-def]
            return '{"error":"I could not build a query."}'

    with pytest.raises(ValueError, match="Could not build a valid query plan"):
        create_query_plan("List teams.", FakeClient())


def test_execute_sql_with_retries_repairs_until_success(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_execute(sql: str) -> pd.DataFrame:
        attempts.append(sql)
        if len(attempts) < 3:
            raise DatabaseError("bad sql")
        return pd.DataFrame([{"ok": 1}])

    class FakeClient:
        def chat(self, messages, response_format=None):  # type: ignore[no-untyped-def]
            return '{"mode":"sql","sql":"select 1 as ok","rationale":"repair"}'

    monkeypatch.setattr(chatbot, "execute_select", fake_execute)
    sql, frame = execute_sql_with_retries("fix it", "select broken from teams", FakeClient(), attempts=3)

    assert sql == "SELECT 1 AS ok LIMIT 100"
    assert len(attempts) == 3
    assert frame.to_dict(orient="records") == [{"ok": 1}]


def test_execute_sql_with_retries_recovers_from_invalid_repair_payload(monkeypatch) -> None:
    attempts: list[str] = []

    def fake_execute(sql: str) -> pd.DataFrame:
        attempts.append(sql)
        if len(attempts) == 1:
            raise DatabaseError("bad sql")
        return pd.DataFrame([{"ok": 1}])

    class FakeClient:
        def __init__(self) -> None:
            self.responses = iter(
                [
                    '{"query":"select 1 as ok"}',
                    '{"mode":"sql","sql":"select 1 as ok","rationale":"repair"}',
                ]
            )

        def chat(self, messages, response_format=None):  # type: ignore[no-untyped-def]
            return next(self.responses)

    monkeypatch.setattr(chatbot, "execute_select", fake_execute)
    sql, frame = execute_sql_with_retries("fix it", "select broken from teams", FakeClient(), attempts=1)

    assert sql == "SELECT 1 AS ok LIMIT 100"
    assert attempts == ["SELECT broken FROM teams LIMIT 100", "SELECT 1 AS ok LIMIT 100"]
    assert frame.to_dict(orient="records") == [{"ok": 1}]
