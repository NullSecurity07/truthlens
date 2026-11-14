"""TruthLens gateway.

Mirrors the proposal's architecture (section 4): the client sends text to a FastAPI
gateway, the gateway dispatches PARALLEL asynchronous requests to three analysis engines,
and a result aggregator compiles a unified confidence metric. The only change is that the
engines are local models on this machine instead of third-party APIs.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .aggregator import aggregate
from .config import settings
from .engines.ai_detect import AIDetectionEngine
from .engines.base import EngineResult
from .engines.factcheck import FactCheckEngine
from .engines.nli import NLIEngine
from .research import WikipediaResearcher
from .engines.toxicity import ToxicityEngine
from .schemas import AnalyzeRequest, AnalyzeResponse, HealthResponse, HistoryResponse
from .store import Store

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s")
log = logging.getLogger("truthlens")

STATE: dict[str, Any] = {"engines": {}, "load_errors": {}, "store": None, "nli": None}


def live_research_enabled() -> bool:
    """Read at call time, not import time, so it can be switched without a code change
    and so tests can run without pulling a 140 MB entailment model."""
    import os
    return os.getenv("TRUTHLENS_LIVE_RESEARCH", "1") not in ("0", "false", "False")


def _build_engines() -> dict[str, Any]:
    """Build the three analysis engines. The NLI model is a fourth model, loaded only when
    live research is switched on, and attached to the fact-check engine rather than being
    its own vector — it answers the same question by a weaker method."""
    fact = FactCheckEngine(
        settings.embedding_model, device=settings.device,
        threshold=settings.verdict_threshold, top_k=settings.top_k,
        google_api_key=settings.google_api_key)
    return {
        "toxicity": ToxicityEngine(
            settings.toxicity_model, device=settings.device,
            quantize=settings.quantize and settings.quantize_toxicity,
            max_tokens=settings.max_tokens, flag_threshold=settings.toxicity_flag_threshold),
        "ai_detection": AIDetectionEngine(
            settings.ai_detect_model, device=settings.device,
            quantize=settings.quantize and settings.quantize_ai_detection,
            max_tokens=settings.max_tokens, flag_threshold=settings.ai_flag_threshold),
        "fact_check": fact,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading engines (device=%s quantize=%s)", settings.device, settings.quantize)
    for name, engine in _build_engines().items():
        try:
            engine.load()
            STATE["engines"][name] = engine
        except Exception as exc:  # noqa: BLE001 - a failed engine must not stop the server
            log.exception("engine %s failed to load", name)
            STATE["load_errors"][name] = f"{type(exc).__name__}: {exc}"

    # Live research is strictly additive: if the NLI model will not load, the product is
    # exactly the local tool it was before, and says so at /health.
    fact = STATE["engines"].get("fact_check")
    if live_research_enabled() and fact is not None:
        try:
            nli = NLIEngine(settings.nli_model, device=settings.device,
                            quantize=settings.quantize,
                            decision_threshold=settings.nli_threshold)
            nli.load()
            fact.nli = nli
            fact.researcher = WikipediaResearcher(
                fact.model, timeout=settings.live_timeout,
                max_pages=settings.live_max_pages,
                max_passages=settings.live_max_passages)
            STATE["nli"] = nli
            log.info("live research enabled (wikipedia + %s)", settings.nli_model)
        except Exception as exc:  # noqa: BLE001
            log.exception("live research unavailable")
            STATE["load_errors"]["live_research"] = f"{type(exc).__name__}: {exc}"
    else:
        log.info("live research disabled by configuration")
    STATE["store"] = Store(settings.db_path)
    log.info("ready: %d/%d engines loaded", len(STATE["engines"]), 3)
    yield
    if STATE["store"]:
        STATE["store"].close()


app = FastAPI(title="TruthLens", version="1.0.0", lifespan=lifespan,
              description="Local, key-free AI moderation and claim checking.")

# Local-only service: the browser extension calls from a chrome-extension:// origin and
# the dashboard from localhost:5173. Both are permitted; nothing here is internet-facing.
app.add_middleware(CORSMiddleware, allow_origin_regex=".*", allow_credentials=False,
                   allow_methods=["*"], allow_headers=["*"])


def _unavailable(name: str) -> EngineResult:
    return EngineResult("unavailable", {},
                        STATE["load_errors"].get(name, "engine not loaded"), 0)


async def _run(name: str, text: str) -> EngineResult:
    engine = STATE["engines"].get(name)
    if engine is None:
        return _unavailable(name)
    # to_thread gives real parallelism here: torch releases the GIL during the forward pass
    return await asyncio.to_thread(engine.run, text)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    engines: dict[str, Any] = {}
    for name in ("toxicity", "ai_detection", "fact_check"):
        engine = STATE["engines"].get(name)
        if engine is None:
            engines[name] = {"status": "failed", "error": STATE["load_errors"].get(name)}
        elif hasattr(engine, "describe"):
            engines[name] = {"status": "loaded", **engine.describe()}
        else:
            engines[name] = {"status": "loaded", "model": settings.embedding_model,
                             "corpus_claims": getattr(engine, "corpus_size", 0)}
    nli = STATE.get("nli")
    engines["live_research"] = (
        {"status": "loaded", **nli.describe(), "source": "en.wikipedia.org"} if nli
        else {"status": "off" if not live_research_enabled() else "failed",
              "error": STATE["load_errors"].get("live_research")})
    loaded = sum(1 for name, e in engines.items()
                 if e["status"] == "loaded" and name != "live_research")
    status = "ok" if loaded == 3 else ("degraded" if loaded else "down")
    fact = STATE["engines"].get("fact_check")
    return HealthResponse(status=status, device=settings.device,
                          quantized=settings.quantize_toxicity or settings.quantize_ai_detection,
                          engines=engines,
                          corpus_claims=getattr(fact, "corpus_size", 0) if fact else 0)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest) -> AnalyzeResponse:
    if not STATE["engines"]:
        raise HTTPException(503, detail={
            "message": "No analysis engine is loaded.",
            "engine_errors": STATE["load_errors"],
        })

    text = req.text
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    ai, tox, fact = await asyncio.gather(
        _run("ai_detection", text), _run("toxicity", text), _run("fact_check", text))
    total_ms = round((loop.time() - t0) * 1000)

    fused = aggregate(ai, tox, fact)
    serial = ai.elapsed_ms + tox.elapsed_ms + fact.elapsed_ms
    report: dict[str, Any] = {
        **fused,
        "ai_detection": {"status": ai.status, "error": ai.error,
                         "elapsed_ms": ai.elapsed_ms, **ai.data},
        "toxicity": {"status": tox.status, "error": tox.error,
                     "elapsed_ms": tox.elapsed_ms, **tox.data},
        "fact_check": {"status": fact.status, "error": fact.error,
                       "elapsed_ms": fact.elapsed_ms, **fact.data},
        "timing_ms": {"ai_detection": ai.elapsed_ms, "toxicity": tox.elapsed_ms,
                      "fact_check": fact.elapsed_ms, "total": total_ms,
                      "speedup_vs_serial": round(serial / total_ms, 2) if total_ms else 1.0},
        "text_hash": "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:32],
        "excerpt": text[:280],
        "analyzed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # drop engine-internal keys the response model does not declare
    for block in ("ai_detection", "toxicity", "fact_check"):
        report[block] = {k: v for k, v in report[block].items()
                         if k not in ("distribution", "corpus_size")}
    try:
        STATE["store"].save(report, req.url)
    except Exception:  # noqa: BLE001 - history is a convenience, never fail the analysis
        log.exception("failed to persist analysis")
    return AnalyzeResponse(**report)


@app.get("/history", response_model=HistoryResponse)
async def history(limit: int = Query(50, ge=1, le=500)) -> HistoryResponse:
    items, total = STATE["store"].history(limit)
    return HistoryResponse(items=items, total=total)


@app.get("/report/{analysis_id}")
async def report(analysis_id: int) -> dict[str, Any]:
    found = STATE["store"].report(analysis_id)
    if found is None:
        raise HTTPException(404, detail=f"No analysis with id {analysis_id}")
    return found
