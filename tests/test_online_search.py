import pytest
from app.search.online import _clean_text, _infer_court_and_year, search_online
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_clean_text():
    raw = "<b>Supreme Court</b> of India &amp; Others <div class='extra'>text</div>"
    cleaned = _clean_text(raw)
    assert "Supreme Court of India & Others text" == cleaned


def test_infer_court_and_year():
    court, year = _infer_court_and_year(
        "Arnesh Kumar vs State Of Bihar on 2 July, 2014",
        "Supreme Court of India judgment on Section 498A IPC"
    )
    assert court == "Supreme Court of India"
    assert year == 2014


def test_search_online():
    import asyncio
    results = asyncio.run(search_online("anticipatory bail section 438 CrPC Arnesh Kumar", top_k=3))
    assert isinstance(results, list)
    if results:
        assert results[0]["type"] == "online"
        assert "url" in results[0]
        assert "case_name" in results[0]



def test_api_search_online():
    resp = client.get("/api/v1/search?q=anticipatory%20bail&k=5")
    assert resp.status_code == 200
    data = resp.json()
    assert "results" in data
    assert data["source"] == "online"
    assert isinstance(data["results"], list)


