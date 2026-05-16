from nba_predictor.ingest.ingest_player_history import ingest_player_history


def test_ingest_player_history_calls_each_season(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    monkeypatch.setattr(
        "nba_predictor.ingest.ingest_player_history.ingest_players",
        lambda season: calls.append(("players", season)) or 10,
    )
    summary = ingest_player_history("1946-47", "1947-48")

    assert calls == [
        ("players", "1946-47"),
        ("players", "1947-48"),
    ]
    assert summary["1947-48"] == {"player_logs": 10}
