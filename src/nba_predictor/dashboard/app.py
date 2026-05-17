from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from sqlalchemy import func, select, text

from nba_predictor.analytics.chatbot import answer_question
from nba_predictor.api.main import list_predictions
from nba_predictor.config import settings
from nba_predictor.db import (
    game_predictions,
    get_engine,
    player_game_stats,
    team_context_summaries,
    team_elo_history,
    team_game_stats,
    teams,
)
from nba_predictor.jobs.operator_actions import (
    run_evaluate,
    run_features,
    run_forecast,
    run_full_pipeline,
    run_ingest,
    run_predict,
    run_refresh,
    run_refresh_full,
    run_train,
    run_upcoming_context,
)
from nba_predictor.predict.predict_games import predict_matchup, predict_matchups, team_id_for_abbreviation

st.set_option("client.toolbarMode", "minimal")


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background: #f5f7fb;
            color: #152033;
        }
        [data-testid="stHeader"] {
            background: transparent;
        }
        [data-testid="stTabs"] button {
            min-height: 2.75rem;
        }
        [data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            padding: 0.8rem 0.95rem;
        }
        div[data-testid="stDataFrame"] {
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            overflow: hidden;
        }
        .dashboard-title {
            margin-bottom: 0.1rem;
        }
        .dashboard-subtitle {
            color: #56657a;
            margin-top: 0;
            margin-bottom: 1.25rem;
        }
        .chat-note {
            color: #56657a;
            font-size: 0.92rem;
            margin-bottom: 0.5rem;
        }
        .team-card {
            background: #ffffff;
            border: 1px solid #dbe3ef;
            border-radius: 8px;
            padding: 1rem;
            min-height: 9.2rem;
        }
        .team-code {
            font-size: 1.3rem;
            font-weight: 700;
            color: #152033;
            margin-bottom: 0.35rem;
        }
        .team-meta {
            color: #56657a;
            font-size: 0.92rem;
            line-height: 1.45;
        }
        .result-strip {
            display: inline-flex;
            gap: 0.25rem;
            margin-top: 0.5rem;
        }
        .result-pill {
            width: 1.5rem;
            height: 1.5rem;
            border-radius: 999px;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            color: #ffffff;
            font-size: 0.78rem;
            font-weight: 700;
        }
        .result-pill.win {
            background: #157347;
        }
        .result-pill.loss {
            background: #b42318;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_metadata() -> dict[str, object] | None:
    path = settings.model_dir / "classifier" / "model_metadata.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _team_abbreviations() -> list[str]:
    engine = get_engine()
    with engine.connect() as conn:
        return sorted(row.abbreviation for row in conn.execute(select(teams.c.abbreviation)))


def _upcoming_games(limit: int = 20) -> pd.DataFrame:
    return _read_frame(
        """
        select
            g.game_id,
            g.game_date,
            g.home_team_id,
            g.away_team_id,
            home.abbreviation as home_team,
            away.abbreviation as away_team
        from games g
        join teams home on home.team_id = g.home_team_id
        join teams away on away.team_id = g.away_team_id
        where g.game_date >= current_date
          and g.home_score is null
          and g.away_score is null
        order by g.game_date, g.game_id
        limit :limit
        """,
        {"limit": limit},
    )


def _upcoming_games_with_predictions(upcoming: pd.DataFrame) -> pd.DataFrame:
    if upcoming.empty:
        return upcoming
    rows = upcoming.copy()
    try:
        predictions = predict_matchups(
            [
                (
                    int(row.home_team_id),
                    int(row.away_team_id),
                    pd.Timestamp(row.game_date).date(),
                )
                for row in rows.itertuples(index=False)
            ]
        )
    except Exception:
        rows["classifier_home"] = None
        rows["classifier_winner"] = "Unavailable"
        rows["elo_home"] = None
        rows["elo_winner"] = "Unavailable"
        rows["projected_score"] = "Unavailable"
        return rows
    rows["classifier_home"] = [result["home_win_probability"] for result in predictions]
    rows["classifier_winner"] = [
        row.home_team if result["predicted_winner_team_id"] == row.home_team_id else row.away_team
        for row, result in zip(rows.itertuples(index=False), predictions, strict=True)
    ]
    rows["elo_home"] = [result["elo_home_win_probability"] for result in predictions]
    rows["elo_winner"] = [
        row.home_team if result["elo_predicted_winner_team_id"] == row.home_team_id else row.away_team
        for row, result in zip(rows.itertuples(index=False), predictions, strict=True)
    ]
    rows["projected_score"] = [
        f"{result['forecasted_home_points']:.1f} - {result['forecasted_away_points']:.1f}"
        for result in predictions
    ]
    return rows


def _latest_team_context(team_code: str) -> pd.DataFrame:
    return _read_frame(
        """
        select c.summary_date, c.source_kind, c.summary
        from team_context_summaries c
        join teams t on t.team_id = c.team_id
        where t.abbreviation = :team_code
        order by c.summary_date desc, c.created_at desc
        limit 3
        """,
        {"team_code": team_code},
    )


def _chart_path(name: str) -> Path:
    return settings.data_dir / "processed" / "evaluation" / name


def _read_frame(sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
    engine = get_engine()
    with engine.connect() as conn:
        return pd.read_sql(text(sql), conn, params=params)


def _dashboard_stats() -> dict[str, int]:
    engine = get_engine()
    with engine.connect() as conn:
        return {
            "teams": int(conn.scalar(select(func.count()).select_from(teams)) or 0),
            "team_rows": int(conn.scalar(select(func.count()).select_from(team_game_stats)) or 0),
            "predictions": int(conn.scalar(select(func.count()).select_from(game_predictions)) or 0),
        }


def render_overview() -> None:
    metadata = _load_metadata()
    stats = _dashboard_stats()
    cols = st.columns(4)
    cols[0].metric("Teams", f"{stats['teams']:,}")
    cols[1].metric("Team stat rows", f"{stats['team_rows']:,}")
    cols[2].metric("Saved predictions", f"{stats['predictions']:,}")
    cols[3].metric("Classifier", str(metadata.get("model_name", "not trained")) if metadata else "not trained")


def _team_overview() -> pd.DataFrame:
    return _read_frame(
        """
        with ranked as (
            select
                h.*,
                row_number() over (partition by h.team_id order by h.game_date desc, h.game_id desc) as rn
            from team_elo_history h
        ),
        latest as (
            select distinct on (team_id)
                team_id,
                postgame_elo as current_elo,
                game_date as last_game_date
            from team_elo_history
            order by team_id, game_date desc, game_id desc
        ),
        recent as (
            select
                team_id,
                string_agg(case when won then 'W' else 'L' end, '' order by game_date desc, game_id desc) as last_five
            from ranked
            where rn <= 5
            group by team_id
        )
        select
            t.team_id,
            t.abbreviation,
            t.full_name,
            l.current_elo,
            l.last_game_date,
            coalesce(r.last_five, '') as last_five
        from teams t
        join latest l on l.team_id = t.team_id
        left join recent r on r.team_id = t.team_id
        order by t.abbreviation
        """
    )


def _team_recent_games(team_id: int, limit: int = 10) -> pd.DataFrame:
    return _read_frame(
        """
        select
            h.game_date,
            g.season_type,
            case when h.is_home then t2.abbreviation || ' at ' || t1.abbreviation else t1.abbreviation || ' at ' || t2.abbreviation end as matchup,
            case when h.won then 'W' else 'L' end as result,
            round(h.pregame_elo::numeric, 1) as pregame_elo,
            round(h.postgame_elo::numeric, 1) as postgame_elo
        from team_elo_history h
        join games g on g.game_id = h.game_id
        join teams t1 on t1.team_id = h.team_id
        join teams t2 on t2.team_id = h.opponent_team_id
        where h.team_id = :team_id
        order by h.game_date desc, h.game_id desc
        limit :limit
        """,
        {"team_id": team_id, "limit": limit},
    )


def _team_elo_series(team_id: int) -> pd.DataFrame:
    return _read_frame(
        """
        select game_date, postgame_elo
        from team_elo_history
        where team_id = :team_id
        order by game_date, game_id
        """,
        {"team_id": team_id},
    )


def _team_elo_by_season(team_id: int) -> pd.DataFrame:
    return _read_frame(
        """
        select
            g.season,
            min(h.game_date) as first_game_date,
            max(h.game_date) as last_game_date,
            min(h.pregame_elo) filter (where h.game_date = season_first.first_game_date) as opening_elo,
            max(h.postgame_elo) filter (where h.game_date = season_last.last_game_date) as closing_elo,
            min(h.postgame_elo) as low_elo,
            max(h.postgame_elo) as high_elo,
            count(*) as games,
            count(*) filter (where g.season_type = 'Regular Season') as regular_games,
            count(*) filter (where g.season_type = 'Playoffs') as playoff_games
        from team_elo_history h
        join games g on g.game_id = h.game_id
        join (
            select g2.season, min(h2.game_date) as first_game_date
            from team_elo_history h2
            join games g2 on g2.game_id = h2.game_id
            where h2.team_id = :team_id
            group by g2.season
        ) season_first on season_first.season = g.season
        join (
            select g3.season, max(h3.game_date) as last_game_date
            from team_elo_history h3
            join games g3 on g3.game_id = h3.game_id
            where h3.team_id = :team_id
            group by g3.season
        ) season_last on season_last.season = g.season
        where h.team_id = :team_id
        group by g.season, season_first.first_game_date, season_last.last_game_date
        order by g.season
        """,
        {"team_id": team_id},
    )


def _team_elo_series_for_season(team_id: int, season: str) -> pd.DataFrame:
    return _read_frame(
        """
        select h.game_date, h.postgame_elo
        from team_elo_history h
        join games g on g.game_id = h.game_id
        where h.team_id = :team_id and g.season = :season
        order by h.game_date, h.game_id
        """,
        {"team_id": team_id, "season": season},
    )


def _team_saved_predictions(team_code: str) -> pd.DataFrame:
    rows = list_predictions(None, team_code, 10)
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame[
        [
            "game_id",
            "home_team",
            "away_team",
            "home_win_probability",
            "away_win_probability",
            "predicted_winner",
        ]
    ]


def _team_player_seasons(team_id: int) -> list[str]:
    frame = _read_frame(
        """
        select distinct season
        from player_game_stats
        where team_id = :team_id
        order by season desc
        """,
        {"team_id": team_id},
    )
    return frame["season"].tolist()


def _team_player_summary(team_id: int, season: str, season_type: str | None) -> pd.DataFrame:
    return _read_frame(
        """
        select
            p.full_name,
            count(*) as games,
            round(avg(s.points)::numeric, 1) as avg_points,
            round(avg(s.rebounds)::numeric, 1) as avg_rebounds,
            round(avg(s.assists)::numeric, 1) as avg_assists,
            round(avg(s.minutes)::numeric, 1) as avg_minutes
        from player_game_stats s
        join players p on p.player_id = s.player_id
        where s.team_id = :team_id
          and s.season = :season
          and (:season_type is null or s.season_type = :season_type)
          and coalesce(s.minutes, 0) > 0
        group by p.player_id, p.full_name
        order by avg_points desc nulls last, games desc, p.full_name
        """,
        {"team_id": team_id, "season": season, "season_type": season_type},
    )


def _player_directory(search: str | None = None) -> pd.DataFrame:
    sql = """
        select
            p.player_id,
            p.full_name,
            count(s.*) as games,
            max(s.game_date) as last_game_date
        from players p
        join player_game_stats s on s.player_id = p.player_id
        where (:search is null or p.full_name ilike :search_pattern)
        group by p.player_id, p.full_name
        order by last_game_date desc, games desc, p.full_name
        limit 200
    """
    params = {"search": search or None, "search_pattern": None if not search else f"%{search}%"}
    return _read_frame(sql, params)


def _metric_value(value: Any) -> str:
    return "n/a" if pd.isna(value) else f"{float(value):.1f}"


def _player_seasons(player_id: int) -> list[str]:
    frame = _read_frame(
        """
        select distinct season
        from player_game_stats
        where player_id = :player_id
        order by season desc
        """,
        {"player_id": player_id},
    )
    return frame["season"].tolist()


def _player_season_summary(player_id: int, season: str, season_type: str | None) -> pd.DataFrame:
    return _read_frame(
        """
        select
            count(*) as games,
            round(avg(minutes)::numeric, 1) as avg_minutes,
            round(avg(points)::numeric, 1) as avg_points,
            round(avg(rebounds)::numeric, 1) as avg_rebounds,
            round(avg(assists)::numeric, 1) as avg_assists,
            round(avg(plus_minus)::numeric, 1) as avg_plus_minus
        from player_game_stats
        where player_id = :player_id
          and season = :season
          and (:season_type is null or season_type = :season_type)
          and coalesce(minutes, 0) > 0
        """,
        {"player_id": player_id, "season": season, "season_type": season_type},
    )


def _player_recent_games(player_id: int, season: str, season_type: str | None) -> pd.DataFrame:
    return _read_frame(
        """
        select
            game_date,
            season_type,
            matchup,
            case when won then 'W' else 'L' end as result,
            minutes,
            points,
            rebounds,
            assists,
            plus_minus
        from player_game_stats
        where player_id = :player_id
          and season = :season
          and (:season_type is null or season_type = :season_type)
        order by game_date desc, game_id desc
        limit 15
        """,
        {"player_id": player_id, "season": season, "season_type": season_type},
    )


def _player_scoring_series(player_id: int, season: str, season_type: str | None) -> pd.DataFrame:
    return _read_frame(
        """
        select game_date, points, rebounds, assists
        from player_game_stats
        where player_id = :player_id
          and season = :season
          and (:season_type is null or season_type = :season_type)
          and coalesce(minutes, 0) > 0
        order by game_date, game_id
        """,
        {"player_id": player_id, "season": season, "season_type": season_type},
    )


def _team_player_crosscheck(team_id: int, season: str, season_type: str | None) -> pd.DataFrame:
    return _read_frame(
        """
        with player_game_totals as (
            select game_id, team_id, sum(points) as player_points
            from player_game_stats
            where team_id = :team_id
              and season = :season
              and (:season_type is null or season_type = :season_type)
            group by game_id, team_id
        )
        select
            count(*) as games,
            round(avg(t.points)::numeric, 1) as team_avg_points,
            round(avg(p.player_points)::numeric, 1) as player_derived_avg_points,
            round((avg(p.player_points) - avg(t.points))::numeric, 1) as avg_delta
        from player_game_totals p
        join team_game_stats t on t.game_id = p.game_id and t.team_id = p.team_id
        """,
        {"team_id": team_id, "season": season, "season_type": season_type},
    )


def _team_coach_summary(team_id: int, season: str) -> pd.DataFrame:
    return _read_frame(
        """
        select
            c.coach_name,
            tc.coach_type,
            case when tc.is_assistant then 'Assistant' else 'Head / staff' end as role,
            tc.sort_sequence
        from team_coaches tc
        join coaches c on c.coach_id = tc.coach_id
        where tc.team_id = :team_id and tc.season = :season
        order by tc.sort_sequence nulls last, c.coach_name
        """,
        {"team_id": team_id, "season": season},
    )


def _season_type_options() -> dict[str, str | None]:
    return {"All games": None, "Regular season": "Regular Season", "Playoffs": "Playoffs"}


def _render_result_strip(results: str) -> str:
    pills = "".join(
        f'<span class="result-pill {"win" if result == "W" else "loss"}">{result}</span>'
        for result in results
    )
    return f'<div class="result-strip">{pills}</div>'


def render_teams_tab() -> None:
    st.subheader("Teams")
    overview = _team_overview()
    if overview.empty:
        st.info("No ELO history has been built yet.")
        return
    selected_team = st.session_state.get("selected_team")
    if selected_team is None:
        sort_mode = st.radio(
            "Sort teams",
            ["Name", "ELO"],
            horizontal=True,
            key="team_sort_mode",
        )
        overview = overview.sort_values(
            ["current_elo", "full_name", "abbreviation"] if sort_mode == "ELO" else ["full_name", "abbreviation"],
            ascending=[False, True, True] if sort_mode == "ELO" else [True, True],
        )
        cols = st.columns(5)
        for index, row in enumerate(overview.itertuples(index=False)):
            with cols[index % 5]:
                st.markdown(
                    (
                        '<div class="team-card">'
                        f'<div class="team-code">{row.abbreviation}</div>'
                        f'<div class="team-meta">ELO {row.current_elo:.1f}<br>'
                        f'Last game {row.last_game_date}<br>'
                        f'{_render_result_strip(row.last_five)}</div>'
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )
                if st.button("Open team", key=f"open_team_{row.team_id}", use_container_width=True):
                    st.session_state["selected_team"] = row.abbreviation
                    st.rerun()
        return
    render_team_detail(selected_team, overview)


def render_players_tab() -> None:
    st.subheader("Players")
    search = st.text_input("Search players", key="player_search")
    directory = _player_directory(search.strip() or None)
    if directory.empty:
        st.info("No player game logs match the current filter.")
        return
    selected_name = st.selectbox("Player", directory["full_name"].tolist())
    selected = directory.loc[directory["full_name"] == selected_name].iloc[0]
    player_id = int(selected["player_id"])
    seasons = _player_seasons(player_id)
    if not seasons:
        st.info("No player seasons available.")
        return
    season = st.selectbox("Season", seasons, key=f"player_season_detail_{player_id}")
    season_type_label = st.segmented_control(
        "Game type",
        list(_season_type_options()),
        default="All games",
        key=f"player_season_type_{player_id}_{season}",
    )
    season_type = _season_type_options()[season_type_label or "All games"]
    summary = _player_season_summary(player_id, season, season_type).iloc[0]
    st.markdown(f"#### {selected_name}")
    cols = st.columns(6)
    cols[0].metric("Games", int(summary["games"]))
    cols[1].metric("Minutes", _metric_value(summary["avg_minutes"]))
    cols[2].metric("Points", _metric_value(summary["avg_points"]))
    cols[3].metric("Rebounds", _metric_value(summary["avg_rebounds"]))
    cols[4].metric("Assists", _metric_value(summary["avg_assists"]))
    cols[5].metric("Plus/minus", _metric_value(summary["avg_plus_minus"]))
    series_col, games_col = st.columns([1.2, 1])
    with series_col:
        st.markdown("#### Game Trend")
        st.line_chart(
            _player_scoring_series(player_id, season, season_type),
            x="game_date",
            y=["points", "rebounds", "assists"],
            height=280,
        )
    with games_col:
        st.markdown("#### Recent Games")
        st.dataframe(_player_recent_games(player_id, season, season_type), use_container_width=True, hide_index=True)


def render_upcoming_tab() -> None:
    st.subheader("Upcoming Games")
    upcoming = _upcoming_games_with_predictions(_upcoming_games())
    if upcoming.empty:
        st.info("No future games are currently stored in the schedule.")
        return
    display_upcoming = upcoming[
        [
            "game_date",
            "away_team",
            "home_team",
            "classifier_home",
            "classifier_winner",
            "elo_home",
            "elo_winner",
            "projected_score",
        ]
    ].copy()
    display_upcoming["classifier_home"] = display_upcoming["classifier_home"].map(
        lambda value: None if pd.isna(value) else f"{value:.1%}"
    )
    display_upcoming["elo_home"] = display_upcoming["elo_home"].map(
        lambda value: None if pd.isna(value) else f"{value:.1%}"
    )
    display_upcoming = display_upcoming.rename(
        columns={
            "game_date": "Date",
            "away_team": "Away",
            "home_team": "Home",
            "classifier_home": "Classifier Home",
            "classifier_winner": "Classifier Pick",
            "elo_home": "ELO Home",
            "elo_winner": "ELO Pick",
            "projected_score": "Projected Score",
        }
    )
    st.dataframe(display_upcoming, use_container_width=True, hide_index=True)
    labels = [f"{row.game_date} - {row.away_team} at {row.home_team}" for row in upcoming.itertuples(index=False)]
    selected_index = st.selectbox("Game", range(len(labels)), format_func=lambda index: labels[index])
    game = upcoming.iloc[selected_index]
    cols = st.columns(2)
    for column, team_code in zip(cols, [game["home_team"], game["away_team"]], strict=True):
        with column:
            st.markdown(f"#### {team_code}")
            context = _latest_team_context(str(team_code))
            if context.empty:
                st.info("No saved context summary yet.")
            else:
                st.dataframe(context, use_container_width=True, hide_index=True)


def render_team_detail(team_code: str, overview: pd.DataFrame | None = None) -> None:
    overview = overview if overview is not None else _team_overview()
    selected = overview.loc[overview["abbreviation"] == team_code]
    if selected.empty:
        st.warning("Selected team was not found.")
        return
    row = selected.iloc[0]
    if st.button("Back to teams"):
        st.session_state.pop("selected_team", None)
        st.rerun()
    st.subheader(f"{row['abbreviation']} - {row['full_name'] or row['abbreviation']}")
    cols = st.columns(3)
    cols[0].metric("Current ELO", f"{row['current_elo']:.1f}")
    cols[1].metric("Last recorded game", str(row["last_game_date"]))
    cols[2].metric("Last five", str(row["last_five"]))
    st.markdown("#### ELO History")
    elo_all_time_tab, elo_season_tab, elo_table_tab = st.tabs(["All Time", "By Season", "Season Summary"])
    all_time_frame = _team_elo_series(int(row["team_id"]))
    with elo_all_time_tab:
        if not all_time_frame.empty:
            st.line_chart(all_time_frame, x="game_date", y="postgame_elo", height=280)
    season_summary = _team_elo_by_season(int(row["team_id"]))
    with elo_season_tab:
        if season_summary.empty:
            st.info("No seasonal ELO history available.")
        else:
            selected_season = st.selectbox(
                "Season",
                season_summary["season"].tolist()[::-1],
                key=f"elo_season_{row['abbreviation']}",
            )
            season_frame = _team_elo_series_for_season(int(row["team_id"]), selected_season)
            st.line_chart(season_frame, x="game_date", y="postgame_elo", height=280)
    with elo_table_tab:
        if not season_summary.empty:
            display_summary = season_summary.copy()
            for column in ["opening_elo", "closing_elo", "low_elo", "high_elo"]:
                display_summary[column] = display_summary[column].map(lambda value: f"{value:.1f}")
            st.dataframe(display_summary, use_container_width=True, hide_index=True)
    detail_cols = st.columns([1.3, 1])
    with detail_cols[0]:
        st.markdown("#### Recent Games")
        recent = _team_recent_games(int(row["team_id"]))
        st.dataframe(recent, use_container_width=True, hide_index=True)
    with detail_cols[1]:
        st.markdown("#### Saved Predictions")
        predictions = _team_saved_predictions(str(row["abbreviation"]))
        if predictions.empty:
            st.info("No saved predictions for this team.")
        else:
            predictions["home_win_probability"] = predictions["home_win_probability"].map(lambda value: f"{value:.1%}")
            predictions["away_win_probability"] = predictions["away_win_probability"].map(lambda value: f"{value:.1%}")
            st.dataframe(predictions, use_container_width=True, hide_index=True)
    st.markdown("#### Team / Player Cross-check")
    player_seasons = _team_player_seasons(int(row["team_id"]))
    if not player_seasons:
        st.info("No player game logs have been ingested for this team yet.")
    else:
        selected_player_season = st.selectbox(
            "Player season",
            player_seasons,
            key=f"player_season_{row['abbreviation']}",
        )
        team_season_type_label = st.segmented_control(
            "Player game type",
            list(_season_type_options()),
            default="All games",
            key=f"team_player_season_type_{row['abbreviation']}_{selected_player_season}",
        )
        team_season_type = _season_type_options()[team_season_type_label or "All games"]
        crosscheck = _team_player_crosscheck(int(row["team_id"]), selected_player_season, team_season_type)
        if not crosscheck.empty and pd.notna(crosscheck.iloc[0]["team_avg_points"]):
            totals = crosscheck.iloc[0]
            metric_cols = st.columns(4)
            metric_cols[0].metric("Games checked", int(totals["games"]))
            metric_cols[1].metric("Team PPG", f"{totals['team_avg_points']:.1f}")
            metric_cols[2].metric("Player-derived PPG", f"{totals['player_derived_avg_points']:.1f}")
            metric_cols[3].metric("Average delta", f"{totals['avg_delta']:.1f}")
        roster_col, coach_col = st.columns([1.45, 1])
        with roster_col:
            st.dataframe(
                _team_player_summary(int(row["team_id"]), selected_player_season, team_season_type),
                use_container_width=True,
                hide_index=True,
            )
        with coach_col:
            coaches_for_season = _team_coach_summary(int(row["team_id"]), selected_player_season)
            if coaches_for_season.empty:
                st.info("No coach data recorded for this season.")
            else:
                st.dataframe(coaches_for_season, use_container_width=True, hide_index=True)
    st.markdown("#### Ask About This Team")
    render_chat_interface(
        quick_prompts=[
            f"Summarize {row['abbreviation']}'s last 10 games.",
            f"How has {row['abbreviation']}'s ELO changed this season?",
            f"Compare {row['abbreviation']} team scoring with its top players this season.",
        ],
        input_placeholder=f"Ask about {row['abbreviation']}",
        history_key=f"chat_history_{row['abbreviation']}",
    )


def render_matchup_tab() -> None:
    st.subheader("Matchup")
    teams_available = _team_abbreviations()
    if len(teams_available) < 2:
        st.info("Load teams before requesting matchup predictions.")
        return
    with st.form("matchup_form"):
        col1, col2, col3 = st.columns([1, 1, 1])
        home = col1.selectbox("Home team", teams_available, index=0)
        away = col2.selectbox("Away team", teams_available, index=1)
        game_date = col3.date_input("Game date", value=date.today())
        submitted = st.form_submit_button("Predict")
    if not submitted:
        return
    if home == away:
        st.error("Home and away teams must differ.")
        return
    try:
        result = predict_matchup(team_id_for_abbreviation(home), team_id_for_abbreviation(away), game_date)
    except Exception as exc:
        st.error(str(exc))
        return
    winner = home if result["predicted_winner_team_id"] == team_id_for_abbreviation(home) else away
    elo_winner = home if result["elo_predicted_winner_team_id"] == team_id_for_abbreviation(home) else away
    metrics = st.columns(5)
    metrics[0].metric("Classifier home", f"{result['home_win_probability']:.1%}")
    metrics[1].metric("Classifier winner", winner)
    metrics[2].metric("ELO home", f"{result['elo_home_win_probability']:.1%}")
    metrics[3].metric("ELO winner", elo_winner)
    metrics[4].metric(
        "Projected score",
        f"{result['forecasted_home_points']:.1f} - {result['forecasted_away_points']:.1f}",
    )


def render_model_tab() -> None:
    st.subheader("Model")
    metadata = _load_metadata()
    if metadata is None:
        st.info("Model metadata is not available yet.")
        return
    classifier_metrics = metadata.get("metrics", {})
    elo_metrics = metadata.get("elo_baseline_metrics", {})
    comparison_rows = []
    for label, metrics in [("Classifier", classifier_metrics), ("ELO baseline", elo_metrics)]:
        if metrics:
            comparison_rows.append(
                {
                    "Signal": label,
                    "Accuracy": metrics.get("accuracy"),
                    "ROC AUC": metrics.get("roc_auc"),
                    "Log loss": metrics.get("log_loss"),
                    "Brier score": metrics.get("brier_score"),
                }
            )
    if comparison_rows:
        st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True, hide_index=True)
    st.caption(
        f"{metadata.get('model_name', 'unknown')} {metadata.get('model_version', '')} trained {metadata.get('trained_at', '')}"
    )
    chart_cols = st.columns(2)
    roc_path = _chart_path("roc_curve.png")
    calibration_path = _chart_path("calibration_curve.png")
    if roc_path.exists():
        chart_cols[0].image(str(roc_path), caption="ROC curve")
    if calibration_path.exists():
        chart_cols[1].image(str(calibration_path), caption="Calibration curve")


def _render_operation_result(result: dict[str, Any]) -> None:
    st.success("Operation completed.")
    st.json(result)


def _run_dashboard_operation(label: str, fn: Any) -> None:
    try:
        with st.spinner(f"{label} running..."):
            result = fn()
    except Exception as exc:
        st.error(f"{label} failed: {exc}")
        return
    _render_operation_result(result)


def render_operations_tab() -> None:
    st.subheader("Operations")
    st.caption("Run routine pipeline jobs from the dashboard. Historical backfills remain CLI-only because they can run for hours.")
    defaults = st.columns(2)
    season = defaults[0].text_input("Season", value="2025-26", key="operations_season")
    predict_date = defaults[1].date_input("Prediction date", value=date.today(), key="operations_predict_date")

    st.markdown("#### Routine Refresh")
    refresh_cols = st.columns(2)
    if refresh_cols[0].button("Refresh data", use_container_width=True):
        _run_dashboard_operation("Refresh data", lambda: run_refresh(season, predict_date))
    if refresh_cols[1].button("Refresh + retrain", use_container_width=True):
        _run_dashboard_operation("Refresh and retrain", lambda: run_refresh_full(season, predict_date))

    st.markdown("#### Pipeline Steps")
    step_cols = st.columns(4)
    if step_cols[0].button("Ingest", use_container_width=True):
        _run_dashboard_operation("Ingest", lambda: run_ingest(season))
    if step_cols[1].button("Build features", use_container_width=True):
        _run_dashboard_operation("Build features", run_features)
    if step_cols[2].button("Forecast", use_container_width=True):
        _run_dashboard_operation("Forecast", run_forecast)
    if step_cols[3].button("Predict", use_container_width=True):
        _run_dashboard_operation("Predict", lambda: run_predict(predict_date))

    model_cols = st.columns(3)
    if model_cols[0].button("Train", use_container_width=True):
        _run_dashboard_operation("Train", run_train)
    if model_cols[1].button("Evaluate", use_container_width=True):
        _run_dashboard_operation("Evaluate", run_evaluate)
    if model_cols[2].button("Full pipeline", use_container_width=True):
        _run_dashboard_operation("Full pipeline", lambda: run_full_pipeline(season, predict_date))
    if st.button("Refresh upcoming injury context", use_container_width=True):
        _run_dashboard_operation("Refresh upcoming injury context", lambda: run_upcoming_context(predict_date))


def _chat_history(history_key: str = "chat_history") -> list[dict[str, Any]]:
    return st.session_state.setdefault(history_key, [])


def _render_chat_answer(answer: dict[str, Any]) -> None:
    st.markdown(str(answer["summary"]))
    rows = list(answer.get("rows", []))
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True, hide_index=True)
    sql = answer.get("sql")
    if sql:
        with st.expander("Generated SQL"):
            st.code(str(sql), language="sql")


def _submit_chat_question(question: str, history_key: str = "chat_history") -> None:
    history = _chat_history(history_key)
    history.append({"role": "user", "content": question})
    try:
        answer = answer_question(question)
        history.append({"role": "assistant", "answer": answer.model_dump(mode="json")})
    except Exception as exc:
        history.append({"role": "assistant", "error": _chat_error_message(exc)})


def _chat_error_message(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 404:
        return (
            f"Ollama is reachable, but the local model `{settings.ollama_model}` is not available. "
            f"Install it with `ollama pull {settings.ollama_model}` or change `OLLAMA_MODEL`."
        )
    if isinstance(exc, httpx.HTTPError):
        return f"Could not reach Ollama at `{settings.ollama_base_url}`."
    return str(exc)


def render_chat_interface(
    quick_prompts: list[str],
    input_placeholder: str,
    history_key: str = "chat_history",
) -> None:
    st.markdown(
        f'<p class="chat-note">Local model: <strong>{settings.ollama_model}</strong>. Ask for historical analysis or a future matchup prediction.</p>',
        unsafe_allow_html=True,
    )
    prompt_cols = st.columns(len(quick_prompts))
    for column, prompt in zip(prompt_cols, quick_prompts, strict=True):
        if column.button(prompt, use_container_width=True):
            with st.spinner("Processing question with the local model and database..."):
                _submit_chat_question(prompt, history_key)
            st.rerun()

    with st.form(f"{history_key}_form", clear_on_submit=True):
        form_cols = st.columns([5, 1])
        question = form_cols[0].text_input(
            "Question",
            placeholder=input_placeholder,
            label_visibility="collapsed",
        )
        submitted = form_cols[1].form_submit_button("Ask", use_container_width=True)
    if submitted and question.strip():
        with st.spinner("Processing question with the local model and database..."):
            _submit_chat_question(question.strip(), history_key)
        st.rerun()

    for message in _chat_history(history_key):
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(str(message["content"]))
            elif "answer" in message:
                _render_chat_answer(dict(message["answer"]))
            else:
                st.error(str(message["error"]))

def render_chat_tab() -> None:
    st.subheader("Ask Data")
    render_chat_interface(
        quick_prompts=[
            "Which players scored the most points per game for BOS in 2025-26?",
            "Show the highest ELO teams right now.",
            "Who is favored if BOS hosts NYK on 2026-01-15?",
        ],
        input_placeholder="Ask about team trends, predictions, or referee assignments",
    )


def main() -> None:
    st.set_page_config(page_title="NBA Predictor", layout="wide")
    _inject_styles()
    st.markdown('<h1 class="dashboard-title">NBA Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p class="dashboard-subtitle">Team form, matchup predictions, and local analytics.</p>', unsafe_allow_html=True)
    render_overview()
    teams_tab, players_tab, upcoming_tab, matchup_tab, chat_tab, operations_tab, model_tab = st.tabs(
        ["Teams", "Players", "Upcoming", "Matchup", "Ask Data", "Operations", "Model"]
    )
    with teams_tab:
        render_teams_tab()
    with players_tab:
        render_players_tab()
    with upcoming_tab:
        render_upcoming_tab()
    with matchup_tab:
        render_matchup_tab()
    with chat_tab:
        render_chat_tab()
    with operations_tab:
        render_operations_tab()
    with model_tab:
        render_model_tab()


if __name__ == "__main__":
    main()
