.PHONY: install db-up api dashboard ingest ingest-officials ingest-players ingest-player-availability ingest-play-by-play ingest-rosters backfill backfill-players backfill-play-by-play backfill-officials features forecast train evaluate predict refresh refresh-full test full-pipeline

SEASON ?= 2025-26
START_SEASON ?= 1946-47
END_SEASON ?= $(SEASON)
PREDICT_DATE ?= $(shell date +%F)
PYTHON ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
PIP ?= $(PYTHON) -m pip

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

db-up:
	docker compose up -d postgres

api:
	$(PYTHON) -m uvicorn nba_predictor.api.main:app --host 0.0.0.0 --port 8000 --reload

ingest:
	$(PYTHON) -m nba_predictor.ingest.ingest_schedule --season $(SEASON)
	$(PYTHON) -m nba_predictor.ingest.ingest_games --season $(SEASON)
	$(PYTHON) -m nba_predictor.ingest.ingest_team_logs --season $(SEASON)
	$(PYTHON) -m nba_predictor.ingest.ingest_box_scores --season $(SEASON)
	$(PYTHON) -m nba_predictor.ingest.ingest_players --season $(SEASON)
	$(PYTHON) -m nba_predictor.ingest.ingest_rosters --season $(SEASON)

ingest-officials:
	$(PYTHON) -m nba_predictor.ingest.ingest_box_scores --season $(SEASON) --include-officials

ingest-players:
	$(PYTHON) -m nba_predictor.ingest.ingest_players --season $(SEASON)

ingest-player-availability:
	$(PYTHON) -c "from nba_predictor.ingest.ingest_players import ingest_player_availability_comments; print(ingest_player_availability_comments('$(SEASON)'))"

ingest-play-by-play:
	$(PYTHON) -m nba_predictor.ingest.ingest_play_by_play --season $(SEASON)

ingest-rosters:
	$(PYTHON) -m nba_predictor.ingest.ingest_rosters --season $(SEASON)

backfill:
	$(PYTHON) -m nba_predictor.ingest.ingest_history --start-season $(START_SEASON) --end-season $(END_SEASON)

backfill-players:
	$(PYTHON) -m nba_predictor.ingest.ingest_player_history --start-season $(START_SEASON) --end-season $(END_SEASON)

backfill-play-by-play:
	$(PYTHON) -m nba_predictor.ingest.ingest_play_by_play --start-season $(START_SEASON) --end-season $(END_SEASON)

backfill-officials:
	$(PYTHON) -m nba_predictor.ingest.ingest_history --start-season $(START_SEASON) --end-season $(END_SEASON) --skip-box-scores --include-officials

features:
	$(PYTHON) -m nba_predictor.features.build_team_season_identities
	$(PYTHON) -m nba_predictor.features.build_team_elo_history
	$(PYTHON) -m nba_predictor.features.build_team_daily_features
	$(PYTHON) -m nba_predictor.features.build_game_features

forecast:
	$(PYTHON) -m nba_predictor.forecast.forecast_team_metrics

train:
	$(PYTHON) -m nba_predictor.train.train_classifier

evaluate:
	$(PYTHON) -m nba_predictor.train.evaluate_model

predict:
	$(PYTHON) -m nba_predictor.predict.predict_games --date $(PREDICT_DATE)

dashboard:
	$(PYTHON) -m streamlit run src/nba_predictor/dashboard/app.py

refresh:
	$(PYTHON) -m nba_predictor.jobs.refresh_pipeline --season $(SEASON) --predict-date $(PREDICT_DATE) --run-once

refresh-full:
	$(PYTHON) -m nba_predictor.jobs.refresh_pipeline --season $(SEASON) --predict-date $(PREDICT_DATE) --retrain --run-once

test:
	$(PYTHON) -m pytest -q

full-pipeline:
	make ingest
	make features
	make forecast
	make train
	make evaluate
	make predict
