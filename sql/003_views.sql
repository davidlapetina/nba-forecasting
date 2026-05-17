CREATE OR REPLACE VIEW latest_team_features AS
SELECT DISTINCT ON (team_id)
    team_id,
    feature_date,
    games_played,
    win_pct,
    last_5_win_pct,
    last_10_win_pct,
    avg_points_last_5,
    avg_points_last_10,
    avg_rebounds_last_10,
    avg_assists_last_10,
    avg_turnovers_last_10,
    avg_off_rating_last_10,
    avg_def_rating_last_10,
    avg_pace_last_10,
    elo_rating
FROM team_daily_features
ORDER BY team_id, feature_date DESC;

CREATE OR REPLACE VIEW latest_team_elo AS
SELECT DISTINCT ON (team_id)
    team_id,
    game_date,
    postgame_elo AS elo_rating
FROM team_elo_history
ORDER BY team_id, game_date DESC, game_id DESC;

CREATE OR REPLACE VIEW latest_player_team AS
SELECT DISTINCT ON (player_id)
    player_id,
    team_id,
    game_date,
    season
FROM player_game_stats
ORDER BY player_id, game_date DESC, game_id DESC;

CREATE OR REPLACE VIEW player_season_summary AS
SELECT
    s.player_id,
    p.full_name,
    s.team_id,
    s.season,
    count(*) AS games,
    round(avg(s.minutes)::numeric, 1) AS avg_minutes,
    round(avg(s.points)::numeric, 1) AS avg_points,
    round(avg(s.rebounds)::numeric, 1) AS avg_rebounds,
    round(avg(s.assists)::numeric, 1) AS avg_assists,
    round(avg(s.plus_minus)::numeric, 1) AS avg_plus_minus
FROM player_game_stats s
JOIN players p ON p.player_id = s.player_id
GROUP BY s.player_id, p.full_name, s.team_id, s.season;
