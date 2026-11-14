"""R2 and R12: a broken engine degrades one vector; it never takes the server down.

R12 specifically covers the startup path - a model that cannot be loaded must be reported
by name at /health, and the failure must surface as an `unavailable` block on the very
first request, rather than as a crash somewhere mid-analysis.
"""
from __future__ import annotations

from tests.conftest import StubEngine


# ---------- R12: failure at load time ----------

def test_server_starts_even_when_a_model_fails_to_load(client_factory):
    broken = StubEngine("ai_detection", fail_load="No such file: pytorch_model.bin")
    with client_factory(ai_detection=broken) as c:
        health = c.get("/health").json()
    assert health["status"] == "degraded"          # not "down", not a crash
    assert health["engines"]["ai_detection"]["status"] == "failed"
    assert "pytorch_model.bin" in health["engines"]["ai_detection"]["error"]
    assert health["engines"]["toxicity"]["status"] == "loaded"


def test_load_failure_surfaces_as_unavailable_on_the_first_request(client_factory):
    broken = StubEngine("fact_check", fail_load="corpus/embedding mismatch")
    with client_factory(fact_check=broken) as c:
        body = c.post("/analyze", json={"text": "Some claim to check."}).json()
    assert body["fact_check"]["status"] == "unavailable"
    assert "corpus/embedding mismatch" in body["fact_check"]["error"]
    assert body["toxicity"]["status"] == "ok"
    assert 0 <= body["trust_score"] <= 100


def test_every_engine_failing_to_load_returns_503_not_a_fake_score(client_factory):
    with client_factory(
        toxicity=StubEngine("toxicity", fail_load="gone"),
        ai_detection=StubEngine("ai_detection", fail_load="gone"),
        fact_check=StubEngine("fact_check", fail_load="gone"),
    ) as c:
        assert c.get("/health").json()["status"] == "down"
        resp = c.post("/analyze", json={"text": "Some claim."})
    assert resp.status_code == 503                 # fail loudly, never invent a score
    assert "engine_errors" in resp.json()["detail"]


# ---------- R2: failure at inference time ----------

def test_inference_failure_degrades_only_that_vector(client_factory):
    with client_factory(toxicity=StubEngine("toxicity", fail="CUDA out of memory")) as c:
        body = c.post("/analyze", json={"text": "A claim."}).json()
    assert body["toxicity"]["status"] == "unavailable"
    assert "CUDA out of memory" in body["toxicity"]["error"]
    assert body["ai_detection"]["status"] == "ok"
    assert body["fact_check"]["status"] == "ok"


def test_a_failed_vector_is_explained_never_silently_dropped(client_factory):
    with client_factory(ai_detection=StubEngine("ai_detection", fail="tokenizer died")) as c:
        body = c.post("/analyze", json={"text": "A claim."}).json()
    entry = next(e for e in body["explanation"] if e["factor"] == "ai_detection")
    assert entry["direction"] == "neutral"
    assert entry["weight"] == 0.0
    assert "tokenizer died" in entry["detail"]


def test_two_failed_engines_still_produce_a_report_from_the_third(client_factory):
    with client_factory(
        toxicity=StubEngine("toxicity", fail="boom"),
        ai_detection=StubEngine("ai_detection", fail_load="boom"),
    ) as c:
        body = c.post("/analyze", json={"text": "A claim."}).json()
    assert body["fact_check"]["status"] == "ok"
    assert 0 <= body["trust_score"] <= 100
    assert body["confidence"] > 0                  # the surviving engine still counts
