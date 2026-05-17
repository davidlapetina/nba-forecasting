from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import httpx
import pandas as pd
from pandas.errors import DatabaseError
from pydantic import BaseModel, ValidationError
from sqlalchemy import text
from sqlglot import exp, parse_one

from nba_predictor.config import settings
from nba_predictor.db import get_engine
from nba_predictor.predict.predict_games import predict_matchup, team_id_for_abbreviation

MAX_SQL_REPAIR_ATTEMPTS = 3
MAX_PLAN_ATTEMPTS = 3

ALLOWED_TABLES = {
    "teams",
    "team_season_identities",
    "players",
    "games",
    "player_game_stats",
    "play_by_play_events",
    "team_rosters",
    "coaches",
    "team_coaches",
    "team_game_stats",
    "team_daily_features",
    "team_elo_history",
    "game_features",
    "team_metric_forecasts",
    "game_predictions",
    "referees",
    "game_officials",
    "latest_team_features",
    "latest_team_elo",
    "latest_player_team",
}

SCHEMA_CONTEXT = """
teams(team_id, abbreviation, full_name, city, nickname)
team_season_identities(team_id, season, abbreviation, full_name)
players(player_id, full_name, first_name, last_name, is_active)
games(game_id, season, game_date, home_team_id, away_team_id, home_score, away_score, home_team_win, season_type)
player_game_stats(game_id, player_id, team_id, game_date, season, season_type, matchup, is_home, won, minutes, points, rebounds, assists, steals, blocks, turnovers, field_goal_pct, three_point_pct, free_throw_pct, plus_minus)
play_by_play_events(game_id, season, game_date, action_number, action_id, period, clock, team_id, team_tricode, person_id, player_name, score_home, score_away, description, action_type, sub_type)
team_rosters(team_id, season, player_id, jersey_number, position, height, weight, birth_date, age, experience, school)
coaches(coach_id, first_name, last_name, coach_name)
team_coaches(team_id, season, coach_id, is_assistant, coach_type, sort_sequence)
team_game_stats(game_id, team_id, opponent_team_id, game_date, is_home, points, rebounds, assists, turnovers, offensive_rating, defensive_rating, pace)
team_daily_features(team_id, feature_date, games_played, win_pct, last_10_win_pct, avg_points_last_10, avg_off_rating_last_10, avg_def_rating_last_10, elo_rating)
team_elo_history(game_id, team_id, opponent_team_id, game_date, is_home, won, pregame_elo, postgame_elo)
game_features(game_id, game_date, season, home_team_id, away_team_id, home_win_pct, away_win_pct, elo_diff, rest_days_home, rest_days_away, home_team_win)
team_metric_forecasts(team_id, forecast_date, metric_name, forecast_value, forecast_p10, forecast_p50, forecast_p90, model_name)
game_predictions(game_id, prediction_date, home_team_id, away_team_id, home_win_probability, away_win_probability, predicted_winner_team_id, forecasted_home_points, forecasted_away_points)
referees(referee_id, first_name, last_name, jersey_num)
game_officials(game_id, referee_id)
latest_team_features(team_id, feature_date, games_played, win_pct, last_10_win_pct, avg_points_last_10, elo_rating)
latest_team_elo(team_id, game_date, elo_rating)
latest_player_team(player_id, team_id, game_date, season)
""".strip()


class QueryPlan(BaseModel):
    mode: Literal["sql", "prediction"]
    sql: str | None = None
    home_team: str | None = None
    away_team: str | None = None
    game_date: date | None = None
    rationale: str = ""


class ChatAnswer(BaseModel):
    mode: Literal["sql", "prediction"]
    summary: str
    columns: list[str]
    rows: list[dict[str, Any]]
    sql: str | None = None


@dataclass(frozen=True)
class OllamaClient:
    base_url: str = settings.ollama_base_url
    model: str = settings.ollama_model
    timeout_seconds: float = 60.0

    def chat(self, messages: list[dict[str, str]], response_format: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if response_format:
            payload["format"] = response_format
        response = httpx.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        return str(response.json()["message"]["content"])


def _plan_prompt(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You translate NBA analytics questions into JSON plans. "
                "Return JSON only. Use mode='prediction' only when the user asks for a future or hypothetical matchup win probability and provides or implies exactly two NBA team abbreviations plus a date. "
                "Otherwise use mode='sql'. SQL must be a single PostgreSQL SELECT query over only the listed schema. "
                "Never use INSERT, UPDATE, DELETE, DDL, comments, or multiple statements. "
                "Prefer concise aggregated result sets and include LIMIT 100 or less for row-level lists.\n\n"
                "For current ELO rankings, use latest_team_elo. For historical ELO trajectories, use team_elo_history.postgame_elo.\n\n"
                f"Schema:\n{SCHEMA_CONTEXT}\n\n"
                "JSON shape:\n"
                '{"mode":"sql","sql":"SELECT ...","rationale":"..."}\n'
                'or {"mode":"prediction","home_team":"BOS","away_team":"NYK","game_date":"2026-01-15","rationale":"..."}'
            ),
        },
        {"role": "user", "content": question},
    ]


def create_query_plan(question: str, client: OllamaClient | None = None) -> QueryPlan:
    heuristic = _heuristic_plan(question)
    if heuristic is not None:
        return heuristic
    ollama = client or OllamaClient()
    messages = _plan_prompt(question)
    last_error: Exception | None = None
    for attempt in range(MAX_PLAN_ATTEMPTS):
        raw = ollama.chat(messages, response_format="json")
        try:
            return _validate_query_plan(raw)
        except (ValidationError, ValueError) as exc:
            last_error = exc
            if attempt == MAX_PLAN_ATTEMPTS - 1:
                break
            messages = _invalid_plan_prompt(question, raw, str(exc))
    raise ValueError("Could not build a valid query plan.") from last_error


def _heuristic_plan(question: str) -> QueryPlan | None:
    normalized = question.lower()
    if "elo" in normalized and any(token in normalized for token in ("highest", "top", "best", "right now", "current")):
        return QueryPlan(
            mode="sql",
            sql=(
                "select t.abbreviation, t.full_name, e.game_date, e.elo_rating "
                "from latest_team_elo e "
                "join teams t on t.team_id = e.team_id "
                "order by e.elo_rating desc "
                "limit 10"
            ),
            rationale="deterministic current ELO ranking query",
        )
    player_ppg_match = re.search(r"\bFOR\s+([A-Z]{3})\s+IN\s+(\d{4}-\d{2})\b", question.upper())
    if (
        player_ppg_match
        and "player" in normalized
        and "points per game" in normalized
    ):
        team_code, season = player_ppg_match.groups()
        return QueryPlan(
            mode="sql",
            sql=(
                "select p.full_name, count(*) as games, round(avg(s.points)::numeric, 1) as avg_points "
                "from player_game_stats s "
                "join players p on p.player_id = s.player_id "
                "join teams t on t.team_id = s.team_id "
                f"where t.abbreviation = '{team_code}' and s.season = '{season}' "
                "group by p.player_id, p.full_name "
                "order by avg_points desc nulls last, games desc "
                "limit 10"
            ),
            rationale="deterministic player scoring query",
        )
    season_year_match = re.search(r"\b(19\d{2}|20\d{2})\b", question)
    season_result_query = any(
        term in normalized
        for term in (
            "best result",
            "best results",
            "best record",
            "best team",
            "top team",
            "most wins",
        )
    )
    if season_year_match and season_result_query:
        season = _season_ending_in(int(season_year_match.group(1)))
        return QueryPlan(
            mode="sql",
            sql=(
                "with team_results as ("
                "select season, home_team_id as team_id, home_team_win as won "
                "from games "
                "where season_type = 'Regular Season' "
                "union all "
                "select season, away_team_id as team_id, not home_team_win as won "
                "from games "
                "where season_type = 'Regular Season'"
                ") "
                "select coalesce(i.abbreviation, t.abbreviation) as abbreviation, "
                "coalesce(i.full_name, t.full_name) as full_name, r.season, "
                "count(*) filter (where r.won) as wins, "
                "count(*) filter (where not r.won) as losses, "
                "count(*) as games, "
                "round(avg(r.won::int)::numeric, 3) as win_pct "
                "from team_results r "
                "join teams t on t.team_id = r.team_id "
                "left join team_season_identities i on i.team_id = r.team_id and i.season = r.season "
                f"where r.season = '{season}' "
                "group by t.team_id, t.abbreviation, t.full_name, i.abbreviation, i.full_name, r.season "
                "order by win_pct desc, wins desc, coalesce(i.full_name, t.full_name) "
                "limit 10"
            ),
            rationale="deterministic season record query",
        )
    abbreviations = re.findall(r"\b[A-Z]{3}\b", question.upper())
    date_match = re.search(r"\b\d{4}-\d{2}-\d{2}\b", question)
    matchup_terms = ("favored", "favorite", "hosts", "host", "matchup", "beat", "win")
    if len(abbreviations) >= 2 and date_match and any(term in normalized for term in matchup_terms):
        return QueryPlan(
            mode="prediction",
            home_team=abbreviations[0],
            away_team=abbreviations[1],
            game_date=date.fromisoformat(date_match.group(0)),
            rationale="deterministic matchup prediction query",
        )
    return None


def _season_ending_in(year: int) -> str:
    return f"{year - 1}-{year % 100:02d}"


def _validate_query_plan(raw: str) -> QueryPlan:
    plan = QueryPlan.model_validate_json(raw)
    if plan.mode == "sql" and not plan.sql:
        raise ValueError("SQL plans require a query.")
    if plan.mode == "prediction" and (plan.home_team is None or plan.away_team is None or plan.game_date is None):
        raise ValueError("Prediction plans require home_team, away_team, and game_date.")
    return plan


def _invalid_plan_prompt(question: str, bad_response: str, error: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return a corrected JSON plan only. "
                "The previous response was invalid. "
                "Use exactly one of these shapes:\n"
                '{"mode":"sql","sql":"SELECT ...","rationale":"..."}\n'
                'or {"mode":"prediction","home_team":"BOS","away_team":"NYK","game_date":"2026-01-15","rationale":"..."}\n'
                "Do not return an error object, explanation, result rows, markdown, or any extra keys outside the plan."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Invalid response: {bad_response}\n"
                f"Validation error: {error}"
            ),
        },
    ]


def validate_read_only_sql(sql: str) -> str:
    expression = parse_one(sql, read="postgres")
    if not isinstance(expression, exp.Select):
        raise ValueError("Only SELECT queries are allowed.")
    cte_names = {cte.alias_or_name for cte in expression.find_all(exp.CTE)}
    referenced_tables = {table.name for table in expression.find_all(exp.Table)} - cte_names
    unknown_tables = referenced_tables - ALLOWED_TABLES
    if unknown_tables:
        raise ValueError(f"Query references unsupported tables: {', '.join(sorted(unknown_tables))}")
    forbidden_nodes = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Alter, exp.Command)
    if any(expression.find(node_type) for node_type in forbidden_nodes):
        raise ValueError("Only read-only SELECT analytics are allowed.")
    if expression.args.get("limit") is None:
        expression = expression.limit(100)
    return expression.sql(dialect="postgres")


def execute_select(sql: str) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn)


def _prediction_rows(plan: QueryPlan) -> list[dict[str, Any]]:
    if plan.home_team is None or plan.away_team is None or plan.game_date is None:
        raise ValueError("Prediction plans require home_team, away_team, and game_date.")
    result = predict_matchup(
        team_id_for_abbreviation(plan.home_team),
        team_id_for_abbreviation(plan.away_team),
        plan.game_date,
    )
    winner = plan.home_team.upper() if result["predicted_winner_team_id"] == team_id_for_abbreviation(plan.home_team) else plan.away_team.upper()
    return [
        {
            "home_team": plan.home_team.upper(),
            "away_team": plan.away_team.upper(),
            "game_date": plan.game_date.isoformat(),
            "home_win_probability": result["home_win_probability"],
            "away_win_probability": result["away_win_probability"],
            "predicted_winner": winner,
            "elo_home_win_probability": result["elo_home_win_probability"],
            "elo_away_win_probability": result["elo_away_win_probability"],
            "elo_predicted_winner": plan.home_team.upper()
            if result["elo_predicted_winner_team_id"] == team_id_for_abbreviation(plan.home_team)
            else plan.away_team.upper(),
            "forecasted_home_points": result["forecasted_home_points"],
            "forecasted_away_points": result["forecasted_away_points"],
        }
    ]


def _summary_prompt(question: str, rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    sample = rows[:20]
    return [
        {
            "role": "system",
            "content": (
                "Summarize NBA analytics results in at most three concise sentences. "
                "State the main answer, mention important qualifiers, and do not invent facts beyond the provided rows."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nRows: {json.dumps(sample, default=str)}",
        },
    ]


def _repair_prompt(question: str, sql: str, error: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Repair the PostgreSQL SELECT query using only the provided schema. "
                "Return JSON only with the same shape as an SQL plan. "
                "Do not change the user's intent. "
                "For current ELO, use latest_team_elo. For historical ELO, use team_elo_history.postgame_elo.\n\n"
                f"Schema:\n{SCHEMA_CONTEXT}"
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\nBroken SQL: {sql}\nDatabase error: {error}",
        },
    ]


def _invalid_repair_prompt(question: str, sql: str, database_error: str, bad_response: str, validation_error: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Return a corrected SQL-plan JSON object only. "
                "Use exactly this shape: "
                '{"mode":"sql","sql":"SELECT ...","rationale":"..."} '
                "The SQL must be a single PostgreSQL SELECT using only the provided schema. "
                "Do not return query results, error objects, markdown, or explanatory text.\n\n"
                f"Schema:\n{SCHEMA_CONTEXT}"
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Broken SQL: {sql}\n"
                f"Database error: {database_error}\n"
                f"Invalid repair response: {bad_response}\n"
                f"Validation error: {validation_error}"
            ),
        },
    ]


def _request_sql_repair(question: str, sql: str, error: str, client: OllamaClient) -> QueryPlan:
    messages = _repair_prompt(question, sql, error)
    last_error: Exception | None = None
    for attempt in range(MAX_PLAN_ATTEMPTS):
        raw = client.chat(messages, response_format="json")
        try:
            plan = _validate_query_plan(raw)
            if plan.mode != "sql" or plan.sql is None:
                raise ValueError("SQL repair must return an SQL plan.")
            return plan
        except (ValidationError, ValueError) as exc:
            last_error = exc
            if attempt == MAX_PLAN_ATTEMPTS - 1:
                break
            messages = _invalid_repair_prompt(question, sql, error, raw, str(exc))
    raise ValueError("Generated SQL repair response was invalid.") from last_error


def execute_sql_with_retries(question: str, sql: str, client: OllamaClient, attempts: int = MAX_SQL_REPAIR_ATTEMPTS) -> tuple[str, pd.DataFrame]:
    current_sql = validate_read_only_sql(sql)
    last_error: DatabaseError | None = None
    for attempt in range(attempts + 1):
        try:
            return current_sql, execute_select(current_sql)
        except DatabaseError as exc:
            last_error = exc
            if attempt == attempts:
                break
            repaired_plan = _request_sql_repair(question, current_sql, str(exc), client)
            current_sql = validate_read_only_sql(repaired_plan.sql)
    raise ValueError("Generated SQL could not be executed after repair retries.") from last_error


def _fallback_summary(mode: str, rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No matching rows were found."
    if mode == "prediction":
        row = rows[0]
        return (
            f"{row['predicted_winner']} is favored with a "
            f"{row['home_win_probability']:.1%} home win probability."
        )
    first = rows[0]
    if {"full_name", "avg_points", "games"} <= set(first):
        return (
            f"{first['full_name']} leads the result set at "
            f"{float(first['avg_points']):.1f} points per game across {int(first['games'])} games."
        )
    if {"abbreviation", "season", "wins", "losses", "win_pct"} <= set(first):
        return (
            f"{first['abbreviation']} had the best regular-season record in {first['season']} at "
            f"{int(first['wins'])}-{int(first['losses'])} ({float(first['win_pct']):.1%})."
        )
    if {"abbreviation", "elo_rating"} <= set(first):
        return f"{first['abbreviation']} leads the result set with an ELO of {float(first['elo_rating']):.1f}."
    return f"Returned {len(rows)} row(s)."


def answer_question(question: str, client: OllamaClient | None = None) -> ChatAnswer:
    ollama = client or OllamaClient()
    plan = create_query_plan(question, ollama)
    sql: str | None = None
    if plan.mode == "prediction":
        rows = _prediction_rows(plan)
    else:
        if plan.sql is None:
            raise ValueError("SQL plans require a query.")
        sql, frame = execute_sql_with_retries(question, plan.sql, ollama)
        rows = json.loads(frame.to_json(orient="records", date_format="iso"))
    if plan.rationale.startswith("deterministic"):
        summary = _fallback_summary(plan.mode, rows)
    else:
        try:
            summary = ollama.chat(_summary_prompt(question, rows))
        except Exception:
            summary = _fallback_summary(plan.mode, rows)
    columns = list(rows[0].keys()) if rows else []
    return ChatAnswer(mode=plan.mode, summary=summary, columns=columns, rows=rows, sql=sql)
