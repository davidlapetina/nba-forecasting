# NBA OSS Game Predictor

Self-hosted NBA analytics and game prediction stack built from public data and open-source components only.

The system ingests NBA team, player, roster, referee, and schedule data, stores it locally in PostgreSQL, builds leakage-safe rolling features, forecasts numeric team trends with TimesFM or a rolling fallback, trains a win-probability classifier, and serves predictions through FastAPI plus a Streamlit dashboard.

```text
nba_api
  |
  v
PostgreSQL
  |
  v
Feature Engineering
  |
  v
TimesFM Forecasts
  |
  v
LightGBM / XGBoost Classifier
  |
  v
FastAPI + Streamlit
  |
  v
Predictions, ELO history, player cross-checks, chatbot analytics
```

## What Is Included

- PostgreSQL schema for teams, games, team stats, player stats, player availability comments, play-by-play events, team context summaries, referees, rosters, coaches, forecasts, features, ELO history, predictions, and scheduler sync state
- Historical NBA ingestion from `nba_api`, plus near-term daily-scoreboard refreshes for active upcoming games
- Franchise ELO history from the first recorded game onward
- Leakage-safe rolling team features and pregame game features
- TimesFM metric forecasting with a rolling-average fallback if TimesFM is unavailable
- LightGBM classifier by default, with optional XGBoost support
- FastAPI endpoints for health, model metadata, predictions, direct matchup prediction, and chatbot queries
- Streamlit dashboard with:
  - one team tile per franchise
  - player search and season detail dashboard
  - upcoming games with saved team context summaries
  - all-time and seasonal ELO charts
  - recent games
  - saved predictions
  - team/player reconciliation tables
  - matchup prediction UI
  - local Ollama-backed natural-language analytics
- Daily refresh and weekly retraining jobs via Docker Compose profiles

## Core Design

TimesFM is not the game-result classifier.

TimesFM forecasts numeric univariate team signals such as:

- points
- offensive rating
- defensive rating
- pace
- rebounds
- assists
- turnovers
- field-goal percentage
- three-point percentage

The classifier then combines forecasts with leakage-safe rolling historical features, all-time head-to-head context, current-season matchup history, and playoff-series state before predicting the probability that the home team wins. Direct matchup predictions also expose the independent ELO and head-to-head baseline probabilities so the learned model can be compared against simpler signals instead of hiding them inside the feature set.

The system now stores richer player availability data:

- historical zero-minute player-game rows from the season logs
- explicit postgame DNP-style comments from per-game player box scores when that enrichment job is run
- advisory upcoming-team summaries generated from the latest official NBA injury-report PDF plus recent headline feeds

The advisory summaries are visible in the dashboard but are **not** direct classifier inputs. A recent zero-minute availability feature was tested and rejected from the active model because it did not improve validation.

Franchise ELO starts at `2500` for each franchise's first recorded game. Ratings update after every played game, and the full pregame/postgame series is stored in `team_elo_history`. The pregame value is copied into feature tables for leakage-safe model training.

Historical franchise labels are stored separately in `team_season_identities`, so analytics can show season-correct abbreviations such as `SEA`, `NJN`, and `VAN`, plus same-code historical names such as `Washington Bullets` and `Charlotte Bobcats`, while the core `teams` table still keeps each current franchise identity for present-day operations.
`make features` refreshes this table from already stored local player logs, so rebuilding historical labels does not require another network backfill.

## Current Status

- Historical play-by-play ingestion is implemented for the supported era beginning in `1996-97`, with resumable per-game sync state.
- Player dashboards, optional Superset support, upcoming-game display, official injury summaries, and headline summaries are implemented.
- Daily refresh now updates the season schedule, near-term daily scoreboard feed, player availability comments, and play-by-play before rebuilding features and predictions.
- Explicit DNP comments are stored when available from per-game box scores; the enrichment is resumable and only revisits newly completed games after the initial backfill.
- Injury/news summaries are advisory dashboard context for now. They are intentionally not fed directly into the classifier until we have enough outcome history to validate a calibrated effect.

## Prerequisites

Required:

- Docker Desktop or Docker Engine with Compose support
- Python `3.11+` for local commands outside containers
- `make`

Optional but recommended:

- Ollama running locally if you want the chatbot UI
- Local TimesFM installation if you want model-based forecasts instead of rolling-average fallback

## Quick Start

### Start Everything From a Local Snapshot

This is the fastest path for your own deployments if you already have a locally created dump file and want a working system without re-downloading the historical NBA data.

1. Create your environment file:

   ```bash
   cp .env.example .env
   ```

2. Start PostgreSQL only:

   ```bash
   docker compose up -d postgres
   ```

3. Restore the database snapshot:

   ```bash
   docker compose cp data/snapshots/nba_predictor_2026-05-16.dump postgres:/tmp/nba_predictor.dump
   docker compose exec -T postgres pg_restore -U nba -d nba_predictor --clean --if-exists /tmp/nba_predictor.dump
   ```

4. Start the API and dashboard:

   ```bash
   docker compose --profile dashboard up -d api dashboard
   ```

5. Open:

   ```text
   API:       http://localhost:8000
   Dashboard: http://localhost:8501
   ```

6. Verify:

   ```bash
   curl http://localhost:8000/health
   curl http://localhost:8000/model/info
   ```

### Start the Full Stack Including Scheduled Jobs

To start PostgreSQL, API, dashboard, the daily data refresh job, and the weekly retraining job together:

```bash
docker compose --profile dashboard --profile scheduler up -d
```

That is the single command to start "all" services in this project once `.env` exists and the database has either been restored or populated.

### Start Superset For Local BI

Superset is optional and runs under the `analytics` profile with its own metadata database:

```bash
docker compose --profile analytics up -d superset-db superset
```

Open `http://localhost:8088` and sign in with `admin` / `admin`. Add the NBA PostgreSQL database with:

```text
postgresql+psycopg2://nba:nba@postgres:5432/nba_predictor
```

The `player_season_summary` database view is included for quickly building player-focused Superset charts and dashboards.

### Start Only the Application Services

If you want the API and dashboard but not the background scheduler containers:

```bash
docker compose --profile dashboard up -d postgres api dashboard
```

## First-Time Setup Without the Snapshot

Use this path if you want to build the database yourself from NBA.com-backed public endpoints.

1. Create `.env`:

   ```bash
   cp .env.example .env
   ```

2. Install local Python dependencies:

   ```bash
   make install
   ```

3. Start PostgreSQL:

   ```bash
   make db-up
   ```

4. Ingest one season:

   ```bash
   make ingest SEASON=2025-26
   make ingest-player-availability SEASON=2025-26
   ```

5. Build model-ready tables and predictions:

   ```bash
   make features
   make forecast
   make train
   make evaluate
   make predict PREDICT_DATE=2026-01-15
   ```

6. Start the application:

   ```bash
   docker compose --profile dashboard up -d api dashboard
   ```

## Full Historical Backfill

To rebuild the historical team/game layer from league start:

```bash
make backfill START_SEASON=1946-47 END_SEASON=2025-26
```

To import playoff games, team logs, player logs, and advanced metrics for an existing historical range:

```bash
make backfill-playoffs START_SEASON=1946-47 END_SEASON=2025-26
```

To backfill historical player game logs:

```bash
make backfill-players START_SEASON=1946-47 END_SEASON=2025-26
make backfill-players-playoffs START_SEASON=1946-47 END_SEASON=2025-26
```

To enrich one season with explicit box-score availability comments such as DNP annotations:

```bash
make ingest-player-availability SEASON=2025-26
```

To ingest play-by-play for one season or backfill it across the available play-by-play era:

```bash
make ingest-play-by-play SEASON=2025-26
make backfill-play-by-play START_SEASON=1996-97 END_SEASON=2025-26
```

To ingest referee assignments:

```bash
make backfill-officials START_SEASON=1946-47 END_SEASON=2025-26
```

Roster and coach ingestion is intentionally separate because it uses slower team-by-team requests:

```bash
make ingest-rosters SEASON=2025-26
```

Notes:

- Historical backfill is idempotent.
- The standard daily refresh ingests both regular-season and playoff logs for the active season.
- Player availability comment enrichment requires one player-box-score request per completed game, but it keeps per-game sync state so later daily refreshes only fetch newly completed games.
- NBA play-by-play coverage begins with the `1996-97` season; earlier seasons cannot be populated from this source.
- Full historical play-by-play backfill is much slower than game/player backfill because it makes one request per completed game.
- NBA.com endpoints can be slow, rate-limited, or change over time.
- Player and team history are much deeper than coach history because the public coach data returned by the roster endpoint is source-dependent.
- Some advanced metrics are unavailable in early seasons even when the games themselves exist.
- The latest official injury report and team-news summaries are advisory context for upcoming games; they are not currently used to change win probabilities.

## Environment Variables

Default values from `.env.example`:

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=nba_predictor
POSTGRES_USER=nba
POSTGRES_PASSWORD=nba

MODEL_DIR=./models
DATA_DIR=./data

CLASSIFIER_MODEL=lightgbm
TIMESFM_MODEL_VERSION=timesfm-2.5-200m

API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
REFRESH_INTERVAL_HOURS=24

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

Inside Docker, the Compose services override:

```bash
POSTGRES_HOST=postgres
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

## TimesFM Installation

TimesFM is optional at runtime because the forecasting pipeline falls back to rolling averages when the model is unavailable.

To install TimesFM locally:

```bash
git clone https://github.com/google-research/timesfm.git
cd timesfm
uv venv
source .venv/bin/activate
uv pip install -e .[torch]
```

The loader expects the TimesFM 2.5 torch path and attempts:

```python
TimesFM_2p5_200M_torch.from_pretrained(...)
```

## Ollama Chatbot Setup

The chatbot is local-only. It uses Ollama with `llama3.2` by default.

1. Install and start Ollama locally.
2. Pull the model:

   ```bash
   ollama pull llama3.2
   ```

3. Confirm it is available:

   ```bash
   curl http://localhost:11434/api/tags
   ```

4. Start the dashboard:

   ```bash
   docker compose --profile dashboard up -d dashboard
   ```

The chatbot translates supported questions into constrained read-only SQL or routes future-matchup questions through the prediction pipeline. Deterministic common queries, such as current ELO rankings, season-best records, and simple player scoring questions, bypass free-form SQL generation for reliability.

## Day-to-Day Operations

### Refresh Today's Data Once

```bash
make refresh SEASON=2025-26 PREDICT_DATE=2026-01-15
```

This runs:

- schedule ingestion
- completed games ingestion
- team logs
- box-score advanced metrics
- player logs
- current-season roster and coach sync
- ELO rebuild
- rolling features
- forecasts
- saved predictions

### Refresh and Retrain Once

```bash
make refresh-full SEASON=2025-26 PREDICT_DATE=2026-01-15
```

### Run Scheduled Services

```bash
docker compose --profile scheduler up -d daily-refresh full-refresh
```

Scheduler behavior:

- `daily-refresh` runs every 24 hours by default.
- `full-refresh` runs every 168 hours by default and retrains the classifier.
- Successful sync timestamps are stored in PostgreSQL.
- After downtime, refresh jobs catch up every NBA season touched since the last successful sync before rebuilding derived tables.

## API

### Health

```bash
curl http://localhost:8000/health
```

```json
{"status":"ok"}
```

### Model Metadata

```bash
curl http://localhost:8000/model/info
```

### Saved Predictions

```bash
curl "http://localhost:8000/predictions?date=2026-01-15&team=BOS&limit=10"
```

### Predict a Matchup

```bash
curl -X POST http://localhost:8000/predict/matchup \
  -H "Content-Type: application/json" \
  -d '{"home_team":"BOS","away_team":"NYK","game_date":"2026-01-15"}'
```

Example:

```json
{
  "home_team": "BOS",
  "away_team": "NYK",
  "home_win_probability": 0.64,
  "away_win_probability": 0.36,
  "predicted_winner": "BOS",
  "forecasted_home_points": 116.2,
  "forecasted_away_points": 110.8
}
```

### Ask a Natural-Language Question

```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question":"Which players scored the most points per game for BOS in 2025-26?"}'
```

## Dashboard

Open:

```text
http://localhost:8501
```

Main views:

- `Teams`: one tile per franchise with current ELO, last five results, and last game date
- `Upcoming`: future games plus saved team injury/news context summaries
- team detail: all-time ELO, season ELO, recent games, saved predictions, player/team reconciliation, coach assignments, team-scoped chat
- `Matchup`: direct matchup prediction form
  - shows the model pick alongside ELO, head-to-head, same-season meetings, and active playoff-series score when applicable
- `Ask Data`: general natural-language analytics with a visible question input, quick prompts, result tables, and generated SQL inspection
- `Operations`: buttons for routine ingest, feature, forecast, train, evaluate, predict, refresh, refresh-and-retrain, and full-pipeline jobs
- `Model`: latest saved evaluation metrics and plots

## Useful Commands

```bash
make install
make db-up
make ingest SEASON=2025-26
make ingest-officials SEASON=2025-26
make ingest-players SEASON=2025-26
make ingest-player-availability SEASON=2025-26
make ingest-play-by-play SEASON=2025-26
make ingest-rosters SEASON=2025-26
make backfill START_SEASON=1946-47 END_SEASON=2025-26
make backfill-players START_SEASON=1946-47 END_SEASON=2025-26
make backfill-play-by-play START_SEASON=1996-97 END_SEASON=2025-26
make backfill-officials START_SEASON=1946-47 END_SEASON=2025-26
make features
make forecast
make train
make evaluate
make predict PREDICT_DATE=2026-01-15
make refresh SEASON=2025-26 PREDICT_DATE=2026-01-15
make refresh-full SEASON=2025-26 PREDICT_DATE=2026-01-15
make dashboard
make test
make full-pipeline
```

## Local Development

Run the API outside Docker:

```bash
make api
```

Run the dashboard outside Docker:

```bash
make dashboard
```

Run tests:

```bash
make test
```

Evaluation outputs are written to:

```bash
data/processed/evaluation/
```

Model artifacts are written to:

```bash
models/classifier/
```

## Repository Layout

```text
.
|- docker-compose.yml
|- Dockerfile
|- Dockerfile.dashboard
|- Makefile
|- sql/
|- src/nba_predictor/
|  |- ingest/
|  |- features/
|  |- forecast/
|  |- train/
|  |- predict/
|  |- api/
|  |- dashboard/
|  |- analytics/
|  `- jobs/
|- models/
|- data/
|  |- raw/
|  |- processed/
|  |- predictions/
|  `- snapshots/
`- tests/
```

## Troubleshooting

### PostgreSQL Is Not Reachable

Check:

```bash
docker compose ps
docker compose logs postgres
```

If you are running local Python commands, `POSTGRES_HOST=localhost` should be correct. Inside Docker Compose services, the host is `postgres`.

### Chatbot Returns an Ollama Error

Check:

```bash
curl http://localhost:11434/api/tags
ollama pull llama3.2
```

The dashboard and API containers expect Ollama on the host at `host.docker.internal:11434`.

### TimesFM Is Not Installed

The forecasting job should still complete by using rolling-average fallback forecasts. Install TimesFM only if you need foundation-model forecasts.

### Snapshot Restore Leaves Old Objects Around

Use:

```bash
docker compose exec -T postgres pg_restore -U nba -d nba_predictor --clean --if-exists /tmp/nba_predictor.dump
```

### Fresh Source Changes Trigger Slow Docker Rebuilds

The current Dockerfiles copy source before dependency installation, so image rebuilds reinstall dependencies after source edits. This is functional but slower than necessary and is a good future optimization before frequent development.

## Data and Modeling Limitations

- NBA.com data access may be rate-limited or change over time.
- This repository distributes software, not NBA.com-derived data. Review upstream terms before sharing any locally collected datasets.
- Predictions are not betting advice.
- Injuries, trades, lineup changes, and rest decisions can strongly affect accuracy.
- TimesFM forecasts numerical team trends, not match outcomes directly.
- Model quality depends heavily on clean historical data and leakage-free features.
- Historical player and roster coverage is broader than historical coach coverage in the public source.

## Roadmap

- Improve container layer caching for faster rebuilds
- Add richer player trend views and player-level forecasting
- Add lineup and injury-aware features when reliable OSS sources are available
- Add release automation for code artifacts
- Expand dashboard filters and historical comparison tools

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
