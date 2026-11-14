from __future__ import annotations

from app.aggregator import aggregate
from app.engines.base import EngineResult


def ok(data):
    return EngineResult("ok", data, None, 10)


def dead(msg="boom"):
    return EngineResult("unavailable", {}, msg, 1)


AI_CLEAN = ok({"probability": 0.05, "confidence": 0.9})
TOX_CLEAN = ok({"score": 0.02, "confidence": 0.95})
FACT_TRUE = ok({"verdict": "true", "similarity": 0.9, "publisher": "P"})
FACT_FALSE = ok({"verdict": "false", "similarity": 0.9, "publisher": "P"})
FACT_NOMATCH = ok({"verdict": "no_match", "similarity": 0.3})


def test_clean_content_scores_high():
    out = aggregate(AI_CLEAN, TOX_CLEAN, FACT_TRUE)
    assert out["trust_score"] >= 90
    assert out["band"] == "trustworthy"


def test_debunked_claim_scores_low():
    out = aggregate(AI_CLEAN, TOX_CLEAN, FACT_FALSE)
    assert out["trust_score"] < 70
    assert any(e["factor"] == "fact_check" and e["direction"] == "lowers"
               for e in out["explanation"])


def test_no_match_abstains_rather_than_voting_neutral():
    """A no_match must not push the score toward the middle - it must not vote at all."""
    with_match = aggregate(AI_CLEAN, TOX_CLEAN, FACT_TRUE)["trust_score"]
    without = aggregate(AI_CLEAN, TOX_CLEAN, FACT_NOMATCH)["trust_score"]
    # both are clean on the other two vectors, so both stay high...
    assert without >= 85
    # ...but confidence must drop, because less evidence backed the number
    assert (aggregate(AI_CLEAN, TOX_CLEAN, FACT_NOMATCH)["confidence"]
            < aggregate(AI_CLEAN, TOX_CLEAN, FACT_TRUE)["confidence"])
    assert with_match >= without


def test_low_confidence_engine_has_less_influence():
    confident = aggregate(AI_CLEAN, ok({"score": 0.9, "confidence": 0.9}), FACT_NOMATCH)
    unsure = aggregate(AI_CLEAN, ok({"score": 0.9, "confidence": 0.1}), FACT_NOMATCH)
    assert confident["trust_score"] < unsure["trust_score"]


def test_all_engines_dead_returns_neutral_placeholder_not_a_measurement():
    out = aggregate(dead(), dead(), dead())
    assert out["trust_score"] == 50
    assert out["confidence"] == 0.0
    assert any("placeholder" in e["detail"] for e in out["explanation"])


def test_dead_engine_is_explained_not_hidden():
    out = aggregate(dead("model missing"), TOX_CLEAN, FACT_TRUE)
    detail = next(e["detail"] for e in out["explanation"] if e["factor"] == "ai_detection")
    assert "model missing" in detail


def test_score_is_bounded():
    for fact in (FACT_TRUE, FACT_FALSE, FACT_NOMATCH):
        out = aggregate(ok({"probability": 1.0, "confidence": 1.0}),
                        ok({"score": 1.0, "confidence": 1.0}), fact)
        assert 0 <= out["trust_score"] <= 100


def test_quantization_defaults_encode_the_measured_tradeoff():
    """Pins the per-engine decision from scripts/bench.py so nobody 'optimises' it back.

    toxicity int8: 2.94x faster, 100% label agreement -> on
    ai_detect int8: 3.27x faster, only 80% agreement  -> off
    """
    from app.config import settings

    assert settings.quantize_toxicity is True
    assert settings.quantize_ai_detection is False


FACT_RELATED = ok({"verdict": "related", "similarity": 0.823, "publisher": "P",
                   "warnings": []})
FACT_LIVE_SUPPORTED = ok({"verdict": "no_match", "similarity": 0.45,
                          "live": {"status": "supported", "confidence": 0.95,
                                   "source_title": "Eiffel Tower"}})


def test_top_band_requires_evidence_about_truth():
    """Regression: "Barack Obama was born in Kenya, not the United States." reported
    78/100 "Likely trustworthy" on 0.20 confidence. The fact-check vector had abstained -
    the match sat at 0.823, below the 0.85 verdict bar - so only toxicity and AI-detection
    voted. Toxicity measures TONE and the detector measures AUTHORSHIP. Neither measures
    TRUTH, so neither can license a claim about trustworthiness, however clean the text."""
    out = aggregate(AI_CLEAN, TOX_CLEAN, FACT_NOMATCH)
    assert out["trust_score"] >= 85, "the score itself is unchanged"
    assert out["band"] == "insufficient_evidence"


def test_related_only_cannot_license_the_top_band():
    """`related` is explicitly not a verdict, so it must not buy the top band either."""
    assert aggregate(AI_CLEAN, TOX_CLEAN, FACT_RELATED)["band"] == "insufficient_evidence"


def test_a_published_verdict_licenses_the_top_band():
    assert aggregate(AI_CLEAN, TOX_CLEAN, FACT_TRUE)["band"] == "trustworthy"


def test_a_live_finding_licenses_the_top_band():
    """Live research is weaker evidence, but it IS evidence about truth."""
    assert aggregate(AI_CLEAN, TOX_CLEAN, FACT_LIVE_SUPPORTED)["band"] == "trustworthy"


def test_a_low_score_is_still_low_without_fact_evidence():
    """The gate blocks OVERCLAIMING trust. A strong toxicity signal is real evidence of a
    problem, so the bottom bands are untouched."""
    out = aggregate(AI_CLEAN, ok({"score": 1.0, "confidence": 1.0}), FACT_NOMATCH)
    assert out["trust_score"] < 40
    assert out["band"] == "low"


def test_the_withheld_band_is_explained_never_silent():
    out = aggregate(AI_CLEAN, TOX_CLEAN, FACT_NOMATCH)
    detail = " ".join(e["detail"] for e in out["explanation"] if e["factor"] == "overall")
    assert "trustworthy" in detail.lower()
