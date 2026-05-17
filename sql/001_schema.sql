CREATE TABLE IF NOT EXISTS teams (
    team_id BIGINT PRIMARY KEY,
    abbreviation TEXT NOT NULL,
    full_name TEXT,
    city TEXT,
    nickname TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS team_season_identities (
    id BIGSERIAL PRIMARY KEY,
    team_id BIGINT NOT NULL,
    season TEXT NOT NULL,
    abbreviation TEXT NOT NULL,
    full_name TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(team_id, season)
);

CREATE TABLE IF NOT EXISTS players (
    player_id BIGINT PRIMARY KEY,
    full_name TEXT NOT NULL,
    first_name TEXT,
    last_name TEXT,
    is_active BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    season TEXT NOT NULL,
    game_date DATE NOT NULL,
    home_team_id BIGINT NOT NULL,
    away_team_id BIGINT NOT NULL,
    home_score INTEGER,
    away_score INTEGER,
    home_team_win BOOLEAN,
    season_type TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS player_game_stats (
    id BIGSERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    player_id BIGINT NOT NULL,
    team_id BIGINT NOT NULL,
    game_date DATE NOT NULL,
    season TEXT NOT NULL,
    season_type TEXT,
    matchup TEXT,
    is_home BOOLEAN,
    won BOOLEAN,
    minutes INTEGER,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    field_goal_pct DOUBLE PRECISION,
    three_point_pct DOUBLE PRECISION,
    free_throw_pct DOUBLE PRECISION,
    plus_minus DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(game_id, player_id)
);

CREATE TABLE IF NOT EXISTS play_by_play_events (
    id BIGSERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    season TEXT NOT NULL,
    game_date DATE NOT NULL,
    action_number INTEGER NOT NULL,
    action_id BIGINT,
    period INTEGER,
    clock TEXT,
    team_id BIGINT,
    team_tricode TEXT,
    person_id BIGINT,
    player_name TEXT,
    player_name_i TEXT,
    x_legacy DOUBLE PRECISION,
    y_legacy DOUBLE PRECISION,
    shot_distance DOUBLE PRECISION,
    shot_result TEXT,
    is_field_goal BOOLEAN,
    score_home INTEGER,
    score_away INTEGER,
    points_total INTEGER,
    location TEXT,
    description TEXT,
    action_type TEXT,
    sub_type TEXT,
    video_available BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(game_id, action_number)
);

CREATE TABLE IF NOT EXISTS play_by_play_sync_state (
    game_id TEXT PRIMARY KEY,
    fetched_at TIMESTAMP NOT NULL DEFAULT NOW(),
    event_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS team_rosters (
    id BIGSERIAL PRIMARY KEY,
    team_id BIGINT NOT NULL,
    season TEXT NOT NULL,
    player_id BIGINT NOT NULL,
    jersey_number TEXT,
    position TEXT,
    height TEXT,
    weight TEXT,
    birth_date DATE,
    age DOUBLE PRECISION,
    experience TEXT,
    school TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(team_id, season, player_id)
);

CREATE TABLE IF NOT EXISTS coaches (
    coach_id BIGINT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    coach_name TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS team_coaches (
    id BIGSERIAL PRIMARY KEY,
    team_id BIGINT NOT NULL,
    season TEXT NOT NULL,
    coach_id BIGINT NOT NULL,
    is_assistant BOOLEAN,
    coach_type TEXT,
    sort_sequence INTEGER,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(team_id, season, coach_id)
);

CREATE TABLE IF NOT EXISTS team_game_stats (
    id BIGSERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    team_id BIGINT NOT NULL,
    opponent_team_id BIGINT NOT NULL,
    game_date DATE NOT NULL,
    is_home BOOLEAN NOT NULL,
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    field_goal_pct DOUBLE PRECISION,
    three_point_pct DOUBLE PRECISION,
    free_throw_pct DOUBLE PRECISION,
    offensive_rating DOUBLE PRECISION,
    defensive_rating DOUBLE PRECISION,
    pace DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(game_id, team_id)
);

CREATE TABLE IF NOT EXISTS referees (
    referee_id BIGINT PRIMARY KEY,
    first_name TEXT,
    last_name TEXT,
    jersey_num TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS game_officials (
    game_id TEXT NOT NULL,
    referee_id BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    PRIMARY KEY (game_id, referee_id)
);

CREATE TABLE IF NOT EXISTS team_daily_features (
    id BIGSERIAL PRIMARY KEY,
    team_id BIGINT NOT NULL,
    feature_date DATE NOT NULL,
    games_played INTEGER,
    win_pct DOUBLE PRECISION,
    last_5_win_pct DOUBLE PRECISION,
    last_10_win_pct DOUBLE PRECISION,
    avg_points_last_5 DOUBLE PRECISION,
    avg_points_last_10 DOUBLE PRECISION,
    avg_rebounds_last_10 DOUBLE PRECISION,
    avg_assists_last_10 DOUBLE PRECISION,
    avg_turnovers_last_10 DOUBLE PRECISION,
    avg_off_rating_last_10 DOUBLE PRECISION,
    avg_def_rating_last_10 DOUBLE PRECISION,
    avg_pace_last_10 DOUBLE PRECISION,
    elo_rating DOUBLE PRECISION,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(team_id, feature_date)
);

CREATE TABLE IF NOT EXISTS team_elo_history (
    id BIGSERIAL PRIMARY KEY,
    game_id TEXT NOT NULL,
    team_id BIGINT NOT NULL,
    opponent_team_id BIGINT NOT NULL,
    game_date DATE NOT NULL,
    is_home BOOLEAN NOT NULL,
    won BOOLEAN NOT NULL,
    pregame_elo DOUBLE PRECISION NOT NULL,
    postgame_elo DOUBLE PRECISION NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(game_id, team_id)
);

CREATE TABLE IF NOT EXISTS game_features (
    game_id TEXT PRIMARY KEY,
    game_date DATE NOT NULL,
    season TEXT NOT NULL,
    home_team_id BIGINT NOT NULL,
    away_team_id BIGINT NOT NULL,
    home_win_pct DOUBLE PRECISION,
    away_win_pct DOUBLE PRECISION,
    win_pct_diff DOUBLE PRECISION,
    home_last_10_win_pct DOUBLE PRECISION,
    away_last_10_win_pct DOUBLE PRECISION,
    last_10_win_pct_diff DOUBLE PRECISION,
    home_avg_points_last_10 DOUBLE PRECISION,
    away_avg_points_last_10 DOUBLE PRECISION,
    avg_points_diff DOUBLE PRECISION,
    home_avg_off_rating_last_10 DOUBLE PRECISION,
    away_avg_off_rating_last_10 DOUBLE PRECISION,
    off_rating_diff DOUBLE PRECISION,
    home_avg_def_rating_last_10 DOUBLE PRECISION,
    away_avg_def_rating_last_10 DOUBLE PRECISION,
    def_rating_diff DOUBLE PRECISION,
    home_elo DOUBLE PRECISION,
    away_elo DOUBLE PRECISION,
    elo_diff DOUBLE PRECISION,
    rest_days_home INTEGER,
    rest_days_away INTEGER,
    rest_days_diff INTEGER,
    home_back_to_back BOOLEAN,
    away_back_to_back BOOLEAN,
    home_team_win BOOLEAN,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS team_metric_forecasts (
    id BIGSERIAL PRIMARY KEY,
    team_id BIGINT NOT NULL,
    forecast_date DATE NOT NULL,
    metric_name TEXT NOT NULL,
    forecast_value DOUBLE PRECISION,
    forecast_p10 DOUBLE PRECISION,
    forecast_p50 DOUBLE PRECISION,
    forecast_p90 DOUBLE PRECISION,
    model_name TEXT DEFAULT 'timesfm',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(team_id, forecast_date, metric_name, model_name)
);

CREATE TABLE IF NOT EXISTS game_predictions (
    id BIGSERIAL PRIMARY KEY,
    game_id TEXT NOT NULL UNIQUE,
    prediction_date TIMESTAMP DEFAULT NOW(),
    home_team_id BIGINT NOT NULL,
    away_team_id BIGINT NOT NULL,
    home_win_probability DOUBLE PRECISION NOT NULL,
    away_win_probability DOUBLE PRECISION NOT NULL,
    predicted_winner_team_id BIGINT NOT NULL,
    forecasted_home_points DOUBLE PRECISION,
    forecasted_away_points DOUBLE PRECISION,
    model_name TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS scheduler_sync_state (
    job_name TEXT PRIMARY KEY,
    last_successful_sync TIMESTAMP NOT NULL,
    updated_at TIMESTAMP DEFAULT NOW()
);
