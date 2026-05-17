from __future__ import annotations

import re
from datetime import date
from io import BytesIO
from urllib.parse import quote_plus
from xml.etree import ElementTree

import httpx
from pypdf import PdfReader
from sqlalchemy import and_, select

from nba_predictor.analytics.chatbot import OllamaClient
from nba_predictor.db import games, get_engine, team_context_summaries, teams, upsert_rows

INJURY_REPORT_PAGE = "https://official.nba.com/nba-injury-report-2025-26-season/"
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}


def _get(url: str, timeout_seconds: float) -> httpx.Response:
    last_error: Exception | None = None
    for _ in range(3):
        try:
            return httpx.get(
                url,
                headers=REQUEST_HEADERS,
                timeout=httpx.Timeout(timeout_seconds, connect=10.0),
                follow_redirects=True,
            )
        except httpx.HTTPError as exc:
            last_error = exc
    raise RuntimeError(f"failed to fetch {url}") from last_error


def latest_injury_report_url(html: str) -> str:
    urls = re.findall(r"href=[\"']([^\"']+\.pdf)[\"']", html, flags=re.IGNORECASE)
    if not urls:
        raise ValueError("no injury report PDF links found")
    return urls[-1]


def extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(BytesIO(content))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _upcoming_teams(target_date: date) -> dict[int, tuple[str, str]]:
    upcoming_ids = (
        select(games.c.home_team_id.label("team_id"))
        .where(
            and_(
                games.c.game_date >= target_date,
                games.c.home_score.is_(None),
                games.c.away_score.is_(None),
            )
        )
        .union(
            select(games.c.away_team_id.label("team_id")).where(
                and_(
                    games.c.game_date >= target_date,
                    games.c.home_score.is_(None),
                    games.c.away_score.is_(None),
                )
            )
        )
        .subquery()
    )
    stmt = select(teams.c.team_id, teams.c.abbreviation, teams.c.full_name).join(
        upcoming_ids,
        upcoming_ids.c.team_id == teams.c.team_id,
    )
    with get_engine().connect() as conn:
        return {
            int(row.team_id): (str(row.abbreviation), str(row.full_name or row.abbreviation))
            for row in conn.execute(stmt)
        }


def _summary_prompt(team_code: str, report_text: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Summarize only the official injury-report information relevant to the requested NBA team. "
                "Return at most three concise sentences. Mention listed players and statuses when present. "
                "If the report has no clear entry for the team, say that no clear official injury entry was found."
            ),
        },
        {
            "role": "user",
            "content": f"Team: {team_code}\n\nOfficial injury report text:\n{report_text}",
        },
    ]


def refresh_upcoming_injury_context(target_date: date | None = None, client: OllamaClient | None = None) -> int:
    target_date = target_date or date.today()
    ollama = client or OllamaClient()
    html = _get(INJURY_REPORT_PAGE, timeout_seconds=20.0).text
    report_url = latest_injury_report_url(html)
    report_bytes = _get(report_url, timeout_seconds=30.0).content
    report_text = extract_pdf_text(report_bytes)
    rows = []
    for team_id, (team_code, _) in _upcoming_teams(target_date).items():
        summary = ollama.chat(_summary_prompt(team_code, report_text))
        rows.append(
            {
                "team_id": team_id,
                "summary_date": target_date,
                "source_kind": "official_injury_report",
                "source_payload": report_url,
                "summary": summary,
            }
        )
    return upsert_rows(
        get_engine(),
        team_context_summaries,
        rows,
        ["team_id", "summary_date", "source_kind"],
        ["source_payload", "summary"],
    )


def latest_team_news_titles(team_name: str) -> list[str]:
    query = quote_plus(f'"{team_name}" NBA')
    response = _get(
        f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en",
        timeout_seconds=20.0,
    )
    root = ElementTree.fromstring(response.text)
    return [
        str(item.findtext("title"))
        for item in root.findall("./channel/item")[:8]
        if item.findtext("title")
    ]


def _news_summary_prompt(team_name: str, titles: list[str]) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Summarize the supplied basketball news headlines in at most three concise sentences. "
                "Focus on injuries, rotation changes, morale, or recent team context when mentioned. "
                "Do not invent facts beyond the headlines."
            ),
        },
        {
            "role": "user",
            "content": f"Team: {team_name}\n\nHeadlines:\n" + "\n".join(f"- {title}" for title in titles),
        },
    ]


def refresh_upcoming_news_context(target_date: date | None = None, client: OllamaClient | None = None) -> int:
    target_date = target_date or date.today()
    ollama = client or OllamaClient()
    rows = []
    for team_id, (_, team_name) in _upcoming_teams(target_date).items():
        titles = latest_team_news_titles(team_name)
        if not titles:
            continue
        rows.append(
            {
                "team_id": team_id,
                "summary_date": target_date,
                "source_kind": "news_headlines",
                "source_payload": "\n".join(titles),
                "summary": ollama.chat(_news_summary_prompt(team_name, titles)),
            }
        )
    return upsert_rows(
        get_engine(),
        team_context_summaries,
        rows,
        ["team_id", "summary_date", "source_kind"],
        ["source_payload", "summary"],
    )
