"""Central configuration. Every tunable lives here, nothing is hardcoded downstream."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

_RESOLVED = DATA_DIR / "resolved_models.json"


def _resolved(slot: str, default: str) -> str:
    """Phase 1's smoke test records which checkpoint actually loaded. Prefer it."""
    try:
        return json.loads(_RESOLVED.read_text())[slot]["model_id"]
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    # --- models (resolved by scripts/smoke_models.py, overridable by env) ---
    toxicity_model: str = field(
        default_factory=lambda: os.getenv(
            "TRUTHLENS_TOXICITY_MODEL",
            _resolved("toxicity", "cardiffnlp/twitter-roberta-base-offensive"),
        )
    )
    ai_detect_model: str = field(
        default_factory=lambda: os.getenv(
            "TRUTHLENS_AI_MODEL",
            _resolved("ai_detection", "openai-community/roberta-base-openai-detector"),
        )
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv(
            "TRUTHLENS_EMBED_MODEL",
            _resolved("embedding", "sentence-transformers/all-MiniLM-L6-v2"),
        )
    )

    # --- runtime ---
    device: str = field(default_factory=lambda: os.getenv("TRUTHLENS_DEVICE", "cpu"))

    # Quantization is decided PER ENGINE from measurement, not applied blanket-fashion.
    # scripts/bench.py on this machine:
    #   toxicity      31.9ms -> 10.9ms  (2.94x)  label agreement 100%   -> enable
    #   ai_detection  35.0ms -> 10.7ms  (3.27x)  label agreement  80%   -> disable
    # The AI detector's int8 outputs drift by a mean |dp| of 0.074 (max 0.118), enough to
    # push borderline cases across the 0.70 threshold. It is already the least reliable
    # vector and already fast at fp32, so accuracy wins over the extra 24ms.
    quantize: bool = field(
        default_factory=lambda: os.getenv("TRUTHLENS_QUANTIZE", "1") not in ("0", "false", "False")
    )
    quantize_toxicity: bool = field(
        default_factory=lambda: os.getenv("TRUTHLENS_QUANTIZE_TOXICITY", "1")
        not in ("0", "false", "False")
    )
    quantize_ai_detection: bool = field(
        default_factory=lambda: os.getenv("TRUTHLENS_QUANTIZE_AI", "0")
        not in ("0", "false", "False")
    )
    max_chars: int = 5000
    max_tokens: int = 512

    # --- fact check ---
    # Both thresholds come from scripts/calibrate_threshold.py, measured on 400 perturbed
    # corpus claims plus hard negatives. Summary of that run:
    #   genuine paraphrases      median similarity 0.904,  p10 0.763
    #   unrelated everyday text  max    similarity 0.502
    #   best NON-matching claim  mean   similarity 0.621,  p90 0.756,  p99 0.876
    # Re-measured after the corpus grew 22,928 -> 42,277. Growth cuts both ways: recall
    # improved (top-1 98.2%) but the chance that some OTHER claim is a close match rose,
    # so false matches at a 0.80 gate went 2.25% -> 6%. The verdict bar therefore rose
    # with the corpus: 0.85 leaves ~2% and keeps the honest fallback of showing the
    # related claim without asserting its verdict.
    # A wrong verdict is far worse than a missing one, so verdicts are only asserted well
    # above the "best wrong claim" distribution. Matches in between are surfaced as
    # RELATED CONTEXT, explicitly not a verdict - which keeps the recall useful without
    # ever putting a stranger's rating on the user's sentence.
    verdict_threshold: float = field(
        default_factory=lambda: float(os.getenv("TRUTHLENS_VERDICT_THRESHOLD", "0.85"))
    )
    related_threshold: float = field(
        default_factory=lambda: float(os.getenv("TRUTHLENS_RELATED_THRESHOLD", "0.62"))
    )
    # kept as an alias so older callers and tests keep working
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("TRUTHLENS_VERDICT_THRESHOLD", "0.85"))
    )
    top_k: int = 3
    max_claims: int = 6

    # --- live research (the one feature that leaves this machine) ---
    # Search terms derived from the selected text are sent to Wikipedia. Set
    # TRUTHLENS_LIVE_RESEARCH=0 to keep everything local; the rest of the product is
    # unaffected. Every result that used it says so in the UI.
    live_research: bool = field(
        default_factory=lambda: os.getenv("TRUTHLENS_LIVE_RESEARCH", "1")
        not in ("0", "false", "False")
    )
    live_timeout: float = field(
        default_factory=lambda: float(os.getenv("TRUTHLENS_LIVE_TIMEOUT", "4.0"))
    )
    live_max_claims: int = 2      # research only the most check-worthy claims: latency
    # Kept deliberately NARROW, and the wider setting was tried and reverted.
    # Widening to 4 pages / 8 passages was measured and made things WORSE: checking
    # "Barack Obama was born in Kenya" it admitted a sentence from the *Auma Obama*
    # article ("patron of the International Storymoja Festivals in Kenya") which the
    # entailment model read as a contradiction at 0.98, replacing the correct citation of
    # Obama's actual birth record. A wider net catches more marginal passages, and a
    # confident reading of a marginal passage is exactly the failure mode of this design.
    live_max_pages: int = 3
    live_max_passages: int = 4
    nli_model: str = field(
        default_factory=lambda: os.getenv(
            "TRUTHLENS_NLI_MODEL",
            _resolved("nli", "cross-encoder/nli-deberta-v3-small"))
    )
    nli_threshold: float = field(
        default_factory=lambda: float(os.getenv("TRUTHLENS_NLI_THRESHOLD", "0.60"))
    )
    # A model reading an encyclopedia is weaker evidence than a professional's published
    # verdict, so live findings never vote at full strength.
    live_confidence_ceiling: float = 0.55
    # A passage must actually be about the claim before its entailment reading counts.
    # Measured failure this guards against: a bibliography line at 0.33 similarity was
    # read as a contradiction at 0.74 confidence and knocked a TRUE claim down to 23/100,
    # while the genuinely relevant passage at 0.52 correctly said "neutral".
    live_min_similarity: float = field(
        default_factory=lambda: float(os.getenv("TRUTHLENS_LIVE_MIN_SIM", "0.45")))
    # combined score = entailment confidence x passage relevance.
    # Raised 0.40 -> 0.60 from measured cases, because every live-research failure so far
    # has been the same shape: a CONFIDENT reading of an insufficiently relevant passage.
    #   0.722  Eiffel Tower lead sentence vs "the Eiffel Tower is in Berlin"   -> keep
    #   0.640  Obama's birth record vs "Obama was born in Kenya"               -> keep
    #   0.550  an Auma Obama festival sentence, read as a contradiction at 0.98 -> BLOCK
    #   0.240  a Swedish bibliography line, read as a contradiction at 0.74     -> BLOCK
    # The two blocked cases produced the right verdict for the wrong reason, which for a
    # tool whose whole output is "here is the evidence" is its own kind of failure.
    live_min_evidence: float = field(
        default_factory=lambda: float(os.getenv("TRUTHLENS_LIVE_MIN_EVIDENCE", "0.60")))
    google_api_key: str | None = field(
        default_factory=lambda: os.getenv("GOOGLE_FACTCHECK_API_KEY") or None
    )

    # --- aggregation weights (documented in aggregator.py) ---
    weight_fact: float = 0.50
    weight_toxicity: float = 0.30
    weight_ai: float = 0.20

    # --- flags ---
    toxicity_flag_threshold: float = 0.50
    ai_flag_threshold: float = 0.70

    # --- paths ---
    db_path: Path = DATA_DIR / "truthlens.db"
    corpus_path: Path = DATA_DIR / "claims.jsonl"
    embeddings_path: Path = DATA_DIR / "claims_emb.npy"


settings = Settings()
