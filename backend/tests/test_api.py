from __future__ import annotations

from tests.conftest import StubEngine


def test_health_reports_all_engines(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert {"toxicity", "ai_detection", "fact_check"} <= set(body["engines"])


def test_health_reports_live_research_channel(client):
    """Live research is a declared channel even when it is switched off, so a reader can
    always tell whether anything left the machine."""
    live = client.get("/health").json()["engines"]["live_research"]
    assert live["status"] == "off"
    # and an off channel must not drag the overall status down
    assert client.get("/health").json()["status"] == "ok"


def test_analyze_returns_all_three_vectors(client):
    body = client.post("/analyze", json={"text": "The moon landing was faked."}).json()
    for key in ("trust_score", "band", "confidence", "ai_detection", "toxicity",
                "fact_check", "explanation", "timing_ms", "text_hash", "analyzed_at"):
        assert key in body, key
    assert body["ai_detection"]["status"] == "ok"
    assert body["toxicity"]["status"] == "ok"
    assert body["fact_check"]["status"] == "ok"
    assert 0 <= body["trust_score"] <= 100


def test_engines_run_concurrently_not_serially(client):
    """R4: with three ~50ms engines, total must be far below their sum."""
    timing = client.post("/analyze", json={"text": "x" * 200}).json()["timing_ms"]
    serial = timing["ai_detection"] + timing["toxicity"] + timing["fact_check"]
    assert timing["total"] < serial, (timing, serial)
    assert timing["speedup_vs_serial"] > 1.5


def test_empty_text_is_rejected_with_the_limit_named(client):
    resp = client.post("/analyze", json={"text": "   "})
    assert resp.status_code == 422
    assert "5000" in resp.text


def test_oversized_text_is_rejected(client):
    resp = client.post("/analyze", json={"text": "a" * 5001})
    assert resp.status_code == 422
    assert "5000" in resp.text


def test_one_dead_engine_degrades_only_that_vector(client_factory):
    with client_factory(toxicity=StubEngine("toxicity", fail="model file missing")) as c:
        body = c.post("/analyze", json={"text": "Some claim."}).json()
    assert body["toxicity"]["status"] == "unavailable"
    assert "model file missing" in body["toxicity"]["error"]
    assert body["ai_detection"]["status"] == "ok"      # others unaffected
    assert body["fact_check"]["status"] == "ok"
    assert 0 <= body["trust_score"] <= 100             # still a usable report


def test_explanation_covers_every_vector(client):
    body = client.post("/analyze", json={"text": "A claim to check."}).json()
    factors = {e["factor"] for e in body["explanation"]}
    assert {"ai_detection", "toxicity", "fact_check"} <= factors
