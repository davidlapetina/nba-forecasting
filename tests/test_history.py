from nba_predictor.ingest.ingest_history import season_range


def test_season_range_includes_start_and_end() -> None:
    assert season_range("1946-47", "1948-49") == ["1946-47", "1947-48", "1948-49"]
