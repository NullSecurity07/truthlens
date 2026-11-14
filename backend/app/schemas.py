"""Wire contract. Consumed by the browser extension AND the dashboard - a frozen surface."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from .config import settings


class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="The user-selected text to analyse.")
    url: str | None = Field(None, description="Page the selection came from, for history.")

    @field_validator("text")
    @classmethod
    def _non_empty_and_bounded(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Provide between 1 and %d characters of text." % settings.max_chars)
        if len(stripped) > settings.max_chars:
            raise ValueError(
                "Selection is %d characters; the limit is %d. Select a shorter passage."
                % (len(stripped), settings.max_chars)
            )
        return stripped


class EngineBlock(BaseModel):
    status: Literal["ok", "unavailable"]
    error: str | None = None
    elapsed_ms: int = 0


class AIDetectionBlock(EngineBlock):
    probability: float | None = None          # P(machine-generated)
    label: str | None = None                  # likely_ai | likely_human | uncertain
    confidence: float | None = None
    model: str | None = None
    note: str | None = None


class ToxicityBlock(EngineBlock):
    score: float | None = None                # P(offensive/toxic)
    flagged: bool | None = None
    label: str | None = None
    confidence: float | None = None
    model: str | None = None


class FactMatch(BaseModel):
    claim: str
    verdict: str
    publisher: str | None = None
    source_url: str | None = None
    similarity: float


class LiveEvidence(BaseModel):
    """Wikipedia looked up at request time and read by an entailment model.

    Deliberately separate from `matched_claim`: that is a professional's published verdict,
    this is a model's reading of an encyclopedia."""
    status: str                               # supported|contradicted|inconclusive|
                                              # no_evidence|disabled
    confidence: float | None = None
    relevance: float | None = None            # how related the passage was to the claim
    evidence: float | None = None             # confidence x relevance
    source_title: str | None = None
    source_url: str | None = None
    passage: str | None = None
    passages: list[dict[str, Any]] = []
    note: str | None = None
    error: str | None = None


class ClaimResult(BaseModel):
    """One extracted claim and what the corpus says about it."""
    text: str
    index: int
    checkworthiness: float
    signals: list[str] = []
    verdict: str                              # true|false|mixed|related|no_match
    rating: str | None = None
    matched_claim: str | None = None
    publisher: str | None = None
    source_url: str | None = None
    similarity: float | None = None
    source: str | None = None
    warnings: list[str] = []                  # contradiction-guard findings
    explanation: str | None = None
    alternatives: list[FactMatch] = []
    live: LiveEvidence | None = None


class FactCheckBlock(EngineBlock):
    verdict: str | None = None                # true|false|mixed|related|no_match
    rating: str | None = None                 # the original 6-way PolitiFact rating
    claims: list[ClaimResult] = []            # per-claim breakdown
    claims_found: int = 0
    claims_matched: int = 0
    live_used: bool = False
    live_findings: int = 0
    live: LiveEvidence | None = None
    matched_claim: str | None = None
    publisher: str | None = None
    source_url: str | None = None
    similarity: float | None = None
    warnings: list[str] = []
    alternatives: list[FactMatch] = []
    source: str | None = None                 # local_corpus | google_factcheck | none
    note: str | None = None


class Explanation(BaseModel):
    factor: str
    direction: Literal["lowers", "raises", "neutral"]
    weight: float
    detail: str


class Timing(BaseModel):
    ai_detection: int = 0
    toxicity: int = 0
    fact_check: int = 0
    total: int = 0
    speedup_vs_serial: float = 1.0


class AnalyzeResponse(BaseModel):
    trust_score: int = Field(..., ge=0, le=100)
    band: Literal["low", "caution", "trustworthy", "insufficient_evidence"]
    confidence: float = Field(..., ge=0.0, le=1.0, description="How much evidence backed the score.")
    ai_detection: AIDetectionBlock
    toxicity: ToxicityBlock
    fact_check: FactCheckBlock
    explanation: list[Explanation]
    timing_ms: Timing
    text_hash: str
    excerpt: str
    analyzed_at: str


class HistoryItem(BaseModel):
    id: int
    created_at: str
    text_excerpt: str
    trust_score: int
    band: str
    ai_probability: float | None
    toxicity_score: float | None
    fact_verdict: str | None
    source_url: str | None
    url: str | None


class HistoryResponse(BaseModel):
    items: list[HistoryItem]
    total: int


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "down"]
    device: str
    quantized: bool
    engines: dict[str, Any]
    corpus_claims: int
