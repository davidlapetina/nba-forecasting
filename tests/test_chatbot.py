from __future__ import annotations

import pytest

from nba_predictor.analytics.chatbot import QueryPlan, _fallback_summary, _heuristic_plan, validate_read_only_sql


def test_validate_read_only_sql_adds_limit() -> None:
    sql = validate_read_only_sql("select abbreviation from teams")
    assert "LIMIT 100" in sql


def test_validate_read_only_sql_rejects_writes() -> None:
    with pytest.raises(ValueError):
        validate_read_only_sql("delete from teams")


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


def test_fallback_summary_handles_player_rows() -> None:
    summary = _fallback_summary("sql", [{"full_name": "Jaylen Brown", "games": 71, "avg_points": 28.7}])
    assert summary == "Jaylen Brown leads the result set at 28.7 points per game across 71 games."
