from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    UniqueConstraint,
    create_engine,
    func,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from nba_predictor.config import settings

metadata = MetaData()

teams = Table(
    "teams",
    metadata,
    Column("team_id", BigInteger, primary_key=True),
    Column("abbreviation", String, nullable=False),
    Column("full_name", String),
    Column("city", String),
    Column("nickname", String),
    Column("created_at", DateTime, server_default=func.now()),
)

team_season_identities = Table(
    "team_season_identities",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("team_id", BigInteger, nullable=False),
    Column("season", String, nullable=False),
    Column("abbreviation", String, nullable=False),
    Column("full_name", String),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("team_id", "season", name="uq_team_season_identities_team_season"),
)

players = Table(
    "players",
    metadata,
    Column("player_id", BigInteger, primary_key=True),
    Column("full_name", String, nullable=False),
    Column("first_name", String),
    Column("last_name", String),
    Column("is_active", Boolean),
    Column("created_at", DateTime, server_default=func.now()),
)

games = Table(
    "games",
    metadata,
    Column("game_id", String, primary_key=True),
    Column("season", String, nullable=False),
    Column("game_date", Date, nullable=False),
    Column("home_team_id", BigInteger, nullable=False),
    Column("away_team_id", BigInteger, nullable=False),
    Column("home_score", Integer),
    Column("away_score", Integer),
    Column("home_team_win", Boolean),
    Column("season_type", String),
    Column("created_at", DateTime, server_default=func.now()),
)

player_game_stats = Table(
    "player_game_stats",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("game_id", String, nullable=False),
    Column("player_id", BigInteger, nullable=False),
    Column("team_id", BigInteger, nullable=False),
    Column("game_date", Date, nullable=False),
    Column("season", String, nullable=False),
    Column("season_type", String),
    Column("matchup", String),
    Column("is_home", Boolean),
    Column("won", Boolean),
    Column("minutes", Integer),
    Column("points", Integer),
    Column("rebounds", Integer),
    Column("assists", Integer),
    Column("steals", Integer),
    Column("blocks", Integer),
    Column("turnovers", Integer),
    Column("field_goal_pct", Float),
    Column("three_point_pct", Float),
    Column("free_throw_pct", Float),
    Column("plus_minus", Float),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("game_id", "player_id", name="uq_player_game_stats_game_player"),
)

team_rosters = Table(
    "team_rosters",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("team_id", BigInteger, nullable=False),
    Column("season", String, nullable=False),
    Column("player_id", BigInteger, nullable=False),
    Column("jersey_number", String),
    Column("position", String),
    Column("height", String),
    Column("weight", String),
    Column("birth_date", Date),
    Column("age", Float),
    Column("experience", String),
    Column("school", String),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("team_id", "season", "player_id", name="uq_team_rosters_team_season_player"),
)

coaches = Table(
    "coaches",
    metadata,
    Column("coach_id", BigInteger, primary_key=True),
    Column("first_name", String),
    Column("last_name", String),
    Column("coach_name", String, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
)

team_coaches = Table(
    "team_coaches",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("team_id", BigInteger, nullable=False),
    Column("season", String, nullable=False),
    Column("coach_id", BigInteger, nullable=False),
    Column("is_assistant", Boolean),
    Column("coach_type", String),
    Column("sort_sequence", Integer),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("team_id", "season", "coach_id", name="uq_team_coaches_team_season_coach"),
)

team_game_stats = Table(
    "team_game_stats",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("game_id", String, nullable=False),
    Column("team_id", BigInteger, nullable=False),
    Column("opponent_team_id", BigInteger, nullable=False),
    Column("game_date", Date, nullable=False),
    Column("is_home", Boolean, nullable=False),
    Column("points", Integer),
    Column("rebounds", Integer),
    Column("assists", Integer),
    Column("steals", Integer),
    Column("blocks", Integer),
    Column("turnovers", Integer),
    Column("field_goal_pct", Float),
    Column("three_point_pct", Float),
    Column("free_throw_pct", Float),
    Column("offensive_rating", Float),
    Column("defensive_rating", Float),
    Column("pace", Float),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("game_id", "team_id", name="uq_team_game_stats_game_team"),
)

referees = Table(
    "referees",
    metadata,
    Column("referee_id", BigInteger, primary_key=True),
    Column("first_name", String),
    Column("last_name", String),
    Column("jersey_num", String),
    Column("created_at", DateTime, server_default=func.now()),
)

game_officials = Table(
    "game_officials",
    metadata,
    Column("game_id", String, primary_key=True),
    Column("referee_id", BigInteger, primary_key=True),
    Column("created_at", DateTime, server_default=func.now()),
)

team_daily_features = Table(
    "team_daily_features",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("team_id", BigInteger, nullable=False),
    Column("feature_date", Date, nullable=False),
    Column("games_played", Integer),
    Column("win_pct", Float),
    Column("last_5_win_pct", Float),
    Column("last_10_win_pct", Float),
    Column("avg_points_last_5", Float),
    Column("avg_points_last_10", Float),
    Column("avg_rebounds_last_10", Float),
    Column("avg_assists_last_10", Float),
    Column("avg_turnovers_last_10", Float),
    Column("avg_off_rating_last_10", Float),
    Column("avg_def_rating_last_10", Float),
    Column("avg_pace_last_10", Float),
    Column("elo_rating", Float),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("team_id", "feature_date", name="uq_team_daily_features_team_date"),
)

team_elo_history = Table(
    "team_elo_history",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("game_id", String, nullable=False),
    Column("team_id", BigInteger, nullable=False),
    Column("opponent_team_id", BigInteger, nullable=False),
    Column("game_date", Date, nullable=False),
    Column("is_home", Boolean, nullable=False),
    Column("won", Boolean, nullable=False),
    Column("pregame_elo", Float, nullable=False),
    Column("postgame_elo", Float, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint("game_id", "team_id", name="uq_team_elo_history_game_team"),
)

game_features = Table(
    "game_features",
    metadata,
    Column("game_id", String, primary_key=True),
    Column("game_date", Date, nullable=False),
    Column("season", String, nullable=False),
    Column("home_team_id", BigInteger, nullable=False),
    Column("away_team_id", BigInteger, nullable=False),
    Column("home_win_pct", Float),
    Column("away_win_pct", Float),
    Column("win_pct_diff", Float),
    Column("home_last_10_win_pct", Float),
    Column("away_last_10_win_pct", Float),
    Column("last_10_win_pct_diff", Float),
    Column("home_avg_points_last_10", Float),
    Column("away_avg_points_last_10", Float),
    Column("avg_points_diff", Float),
    Column("home_avg_off_rating_last_10", Float),
    Column("away_avg_off_rating_last_10", Float),
    Column("off_rating_diff", Float),
    Column("home_avg_def_rating_last_10", Float),
    Column("away_avg_def_rating_last_10", Float),
    Column("def_rating_diff", Float),
    Column("home_elo", Float),
    Column("away_elo", Float),
    Column("elo_diff", Float),
    Column("rest_days_home", Integer),
    Column("rest_days_away", Integer),
    Column("rest_days_diff", Integer),
    Column("home_back_to_back", Boolean),
    Column("away_back_to_back", Boolean),
    Column("home_team_win", Boolean),
    Column("created_at", DateTime, server_default=func.now()),
)

team_metric_forecasts = Table(
    "team_metric_forecasts",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("team_id", BigInteger, nullable=False),
    Column("forecast_date", Date, nullable=False),
    Column("metric_name", String, nullable=False),
    Column("forecast_value", Float),
    Column("forecast_p10", Float),
    Column("forecast_p50", Float),
    Column("forecast_p90", Float),
    Column("model_name", String, server_default="timesfm"),
    Column("created_at", DateTime, server_default=func.now()),
    UniqueConstraint(
        "team_id",
        "forecast_date",
        "metric_name",
        "model_name",
        name="uq_team_metric_forecasts_unique",
    ),
)

game_predictions = Table(
    "game_predictions",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("game_id", String, nullable=False, unique=True),
    Column("prediction_date", DateTime, server_default=func.now()),
    Column("home_team_id", BigInteger, nullable=False),
    Column("away_team_id", BigInteger, nullable=False),
    Column("home_win_probability", Float, nullable=False),
    Column("away_win_probability", Float, nullable=False),
    Column("predicted_winner_team_id", BigInteger, nullable=False),
    Column("forecasted_home_points", Float),
    Column("forecasted_away_points", Float),
    Column("model_name", String, nullable=False),
    Column("model_version", String, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
)

scheduler_sync_state = Table(
    "scheduler_sync_state",
    metadata,
    Column("job_name", String, primary_key=True),
    Column("last_successful_sync", DateTime, nullable=False),
    Column("updated_at", DateTime, server_default=func.now()),
)


def get_engine(database_url: str | None = None) -> Engine:
    return create_engine(database_url or settings.database_url, future=True)


def create_all(engine: Engine | None = None) -> None:
    metadata.create_all(engine or get_engine())


def upsert_rows(
    engine: Engine,
    table: Table,
    rows: Iterable[dict[str, Any]],
    index_elements: Sequence[str],
    update_columns: Sequence[str] | None = None,
) -> int:
    materialized = list(rows)
    if not materialized:
        return 0
    update_columns = tuple(update_columns or ())
    insert_factory = pg_insert if engine.dialect.name == "postgresql" else sqlite_insert
    stmt = insert_factory(table).values(materialized)
    if update_columns:
        stmt = stmt.on_conflict_do_update(
            index_elements=list(index_elements),
            set_={column: getattr(stmt.excluded, column) for column in update_columns},
        )
    else:
        stmt = stmt.on_conflict_do_nothing(index_elements=list(index_elements))
    with engine.begin() as conn:
        conn.execute(stmt)
    return len(materialized)


def fetch_all(engine: Engine, table: Table) -> list[dict[str, Any]]:
    with engine.connect() as conn:
        return [dict(row._mapping) for row in conn.execute(select(table))]
