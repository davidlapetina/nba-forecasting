from nba_predictor.ingest.ingest_player_history import ingest_player_history


def test_ingest_player_history_calls_each_season(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "nba_predictor.ingest.ingest_player_history.ingest_players",
        lambda season, season_type="Regular Season": calls.append((season_type, season)) or 10,
    )
    summary = ingest_player_history("1946-47", "1947-48")

    assert calls == [
        ("Regular Season", "1946-47"),
        ("Regular Season", "1947-48"),
    ]
    assert summary["1947-48"] == {"player_logs": 10}


def test_ingest_player_history_can_include_playoffs(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "nba_predictor.ingest.ingest_player_history.ingest_players",
        lambda season, season_type="Regular Season": calls.append((season_type, season)) or 10,
    )

    summary = ingest_player_history("2025-26", "2025-26", include_playoffs=True)

    assert calls == [("Regular Season", "2025-26"), ("Playoffs", "2025-26")]
    assert summary["2025-26"] == {"player_logs": 10, "playoff_player_logs": 10}


def test_ingest_player_history_can_import_only_playoffs(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "nba_predictor.ingest.ingest_player_history.ingest_players",
        lambda season, season_type="Regular Season": calls.append((season_type, season)) or 10,
    )

    summary = ingest_player_history("2025-26", "2025-26", playoffs_only=True)

    assert calls == [("Playoffs", "2025-26")]
    assert summary["2025-26"] == {"playoff_player_logs": 10}
