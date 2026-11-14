"""Stub engines: the orchestration, fusion and API layers are tested without any model.

Model-dependent behaviour is covered separately by scripts/smoke_models.py (Phase 1).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.engines.base import Engine  # noqa: E402


class StubEngine(Engine):
    def __init__(self, name: str, payload: dict[str, Any] | None = None,
                 fail: str | None = None, fail_load: str | None = None,
                 delay: float = 0.05) -> None:
        self.name = name
        self.payload = payload or {}
        self.fail = fail
        self.fail_load = fail_load
        self.delay = delay
        self.corpus_size = 3

    def load(self) -> None:
        if self.fail_load:
            raise OSError(self.fail_load)

    def analyze(self, text: str) -> dict[str, Any]:
        time.sleep(self.delay)          # so parallel-vs-serial timing is observable
        if self.fail:
            raise RuntimeError(self.fail)
        return dict(self.payload)

    def describe(self) -> dict[str, Any]:
        return {"model": f"stub-{self.name}", "quantized": False, "labels": {}}


def make_engines(**overrides: Engine) -> dict[str, Engine]:
    base = {
        "toxicity": StubEngine("toxicity", {
            "score": 0.1, "flagged": False, "label": "non-offensive",
            "confidence": 0.8, "model": "stub"}),
        "ai_detection": StubEngine("ai_detection", {
            "probability": 0.2, "label": "likely_human", "confidence": 0.6,
            "model": "stub", "note": "stub"}),
        "fact_check": StubEngine("fact_check", {
            "verdict": "false", "matched_claim": "A false claim.", "similarity": 0.9,
            "publisher": "StubCheck", "source_url": "https://example.invalid/1",
            "alternatives": [], "source": "local_corpus", "note": "stub"}),
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _no_live_research(monkeypatch):
    """The suite must not download or run the entailment model. Live research has its own
    tests with a stubbed researcher and NLI."""
    monkeypatch.setenv("TRUTHLENS_LIVE_RESEARCH", "0")


@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app.store import Store

    # Settings is a frozen dataclass, so redirect the store constructor rather than the path.
    monkeypatch.setattr(main_mod, "Store", lambda _path: Store(tmp_path / "t.db"))
    monkeypatch.setattr(main_mod, "_build_engines", make_engines)
    main_mod.STATE["engines"] = {}
    main_mod.STATE["load_errors"] = {}
    with TestClient(main_mod.app) as c:
        yield c


@pytest.fixture
def client_factory(tmp_path, monkeypatch):
    """Build a client whose engine set the test chooses."""
    from fastapi.testclient import TestClient

    import app.main as main_mod
    from app.store import Store

    def _factory(**overrides):
        monkeypatch.setattr(main_mod, "Store", lambda _path: Store(tmp_path / "t2.db"))
        monkeypatch.setattr(main_mod, "_build_engines", lambda: make_engines(**overrides))
        main_mod.STATE["engines"] = {}
        main_mod.STATE["load_errors"] = {}
        return TestClient(main_mod.app)

    return _factory
