"""Fact-check retrieval, tiering and the contradiction guard.

A fake encoder lets each test dial an EXACT cosine similarity, so the tier boundaries are
tested precisely and no model download is needed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.factcheck import FactCheckEngine, contradiction_signals

FLAT = "The earth is flat and NASA lied about it."
BOIL = "Water boils at 100 degrees at sea level."

CLAIMS = [
    {"claim": FLAT, "rating": "pants-fire", "verdict": "false",
     "publisher": "StubCheck", "source_url": "https://example.invalid/1"},
    {"claim": BOIL, "rating": "true", "verdict": "true",
     "publisher": "StubCheck", "source_url": "https://example.invalid/2"},
]


def at_similarity(sim: float, axis: int) -> np.ndarray:
    """A unit vector whose cosine with basis vector `axis` is exactly `sim`."""
    v = np.zeros(3, dtype=np.float32)
    v[axis] = sim
    v[2] = math.sqrt(max(0.0, 1.0 - sim * sim))
    return v


class DialEncoder:
    """Maps a substring to a chosen similarity against one of the two corpus claims."""

    def __init__(self, rules):
        self.rules = rules            # list of (substring, sim, axis)

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kw):
        out = []
        for t in texts:
            low = t.lower()
            for needle, sim, axis in self.rules:
                if needle in low:
                    out.append(at_similarity(sim, axis))
                    break
            else:
                out.append(at_similarity(0.05, 0))
        return np.stack(out)


def build(rules, **kwargs) -> FactCheckEngine:
    engine = FactCheckEngine("fake", top_k=2, **kwargs)
    engine.model = DialEncoder(rules)
    engine.claims = CLAIMS
    engine.matrix = np.stack([at_similarity(1.0, 0), at_similarity(1.0, 1)])
    return engine


# ---------------- tiering ----------------

def test_high_similarity_asserts_the_verdict():
    engine = build([("flat", 0.95, 0)])
    out = engine.analyze("The earth is flat and NASA lied about it.")
    assert out["verdict"] == "false"
    assert out["source_url"] == "https://example.invalid/1"
    assert out["claims_matched"] == 1


def test_middle_similarity_is_related_context_never_a_verdict():
    """0.70 is above the relatedness floor but below the verdict threshold."""
    engine = build([("flat", 0.70, 0)])
    out = engine.analyze("The earth is flat and NASA lied about it.")
    assert out["verdict"] == "related"
    assert out["matched_claim"] == FLAT          # still shown, as context
    assert "NOT a verdict" in out["note"]
    assert out["claims_matched"] == 0            # and it does not count as a match


def test_low_similarity_is_no_match():
    engine = build([("flat", 0.20, 0)])
    out = engine.analyze("The earth is flat and NASA lied about it.")
    assert out["verdict"] == "no_match"
    assert out["matched_claim"] is None
    assert "not evidence the claim is true" in out["note"]


@pytest.mark.parametrize("sim,expected", [
    (0.95, "false"), (0.85, "false"),        # at and above the verdict threshold
    (0.84, "related"), (0.62, "related"),    # the related band
    (0.61, "no_match"),                      # below the relatedness floor
])
def test_tier_boundaries(sim, expected):
    engine = build([("flat", sim, 0)])
    assert engine.analyze("The earth is flat and NASA lied about it.")["verdict"] == expected


# ---------------- contradiction guard ----------------

def test_negation_mismatch_blocks_a_verdict():
    """Measured: negated claims retrieve the original at HIGHER similarity than genuine
    paraphrases. The neural score cannot catch this, so the symbolic guard must."""
    engine = build([("flat", 0.97, 0)])
    out = engine.analyze("The earth is not flat and NASA did not lie about it.")
    assert out["verdict"] == "related"           # downgraded despite 0.97 similarity
    assert any("negation mismatch" in w for w in out["warnings"])


def test_numeric_mismatch_blocks_a_verdict():
    engine = build([("boils", 0.97, 1)])
    out = engine.analyze("Water boils at 250 degrees at sea level.")
    assert out["verdict"] == "related"
    assert any("numeric mismatch" in w for w in out["warnings"])


def test_matching_numbers_do_not_trigger_the_guard():
    engine = build([("boils", 0.97, 1)])
    out = engine.analyze("Water boils at 100 degrees at sea level.")
    assert out["verdict"] == "true"
    assert out["warnings"] == []


@pytest.mark.parametrize("query,matched,expect", [
    ("Obama grew up in Kenya.", "Obama grew up in Kenya.", 0),
    ("Obama did not grow up in Kenya.", "Obama grew up in Kenya.", 1),
    ("Unemployment fell 3 percent.", "Unemployment fell 8 percent.", 1),
    ("Unemployment fell 3 percent.", "Unemployment fell 3 percent last year.", 0),
])
def test_contradiction_signals_unit(query, matched, expect):
    assert len(contradiction_signals(query, matched)) == expect


# ---------------- multi-claim ----------------

def test_each_sentence_is_checked_separately():
    engine = build([("flat", 0.95, 0), ("boils", 0.95, 1)])
    out = engine.analyze(FLAT + " " + BOIL)
    assert out["claims_found"] == 2
    assert {c["verdict"] for c in out["claims"]} == {"false", "true"}


def test_the_worst_verdict_drives_the_overall_result():
    """One debunked sentence in an otherwise clean paragraph must still pull it down."""
    engine = build([("flat", 0.95, 0), ("boils", 0.95, 1)])
    out = engine.analyze(BOIL + " " + FLAT)
    assert out["verdict"] == "false"             # not 'true', despite a true claim present
    assert out["matched_claim"] == FLAT


def test_claims_carry_their_checkworthiness_and_signals():
    engine = build([("flat", 0.95, 0)])
    claim = engine.analyze(FLAT)["claims"][0]
    assert 0.0 <= claim["checkworthiness"] <= 1.0
    assert claim["signals"]
    assert claim["text"] == FLAT


def test_note_reports_how_many_claims_were_checked():
    engine = build([("flat", 0.95, 0), ("boils", 0.95, 1)])
    assert "2 claims checked" in engine.analyze(FLAT + " " + BOIL)["note"]


# ---------------- degradation ----------------

def test_empty_corpus_degrades_honestly():
    engine = build([("flat", 0.95, 0)])
    engine.claims, engine.matrix = [], None
    out = engine.analyze(FLAT)
    assert out["verdict"] == "no_match"
    assert out["source"] == "none"


def test_google_path_is_skipped_without_a_key():
    engine = build([("flat", 0.95, 0)])
    assert engine.google_api_key is None
    assert engine._google("anything") == []


# ---------------- negation scoping ----------------

@pytest.mark.parametrize("query,matched,expect_warning", [
    # contrastive "not" ASSERTS the claim - must NOT trip the guard
    ("Obama was born in Kenya, not the United States.", "Says Obama grew up in Kenya.", False),
    ("It was Kenya, not Hawaii.", "It was Kenya.", False),
    # clause-scoped negation REVERSES the claim - must trip the guard
    ("Obama did not grow up in Kenya.", "Obama grew up in Kenya.", True),
    ("Obama was not born in Kenya.", "Obama was born in Kenya.", True),
    ("Obama didn't grow up in Kenya.", "Obama grew up in Kenya.", True),
    ("Obama never lived in Kenya.", "Obama lived in Kenya.", True),
])
def test_only_clause_scoped_negation_trips_the_guard(query, matched, expect_warning):
    warnings = contradiction_signals(query, matched)
    assert bool(any("negation" in w for w in warnings)) is expect_warning


# ---------------- the case that justifies the guard existing ----------------

def test_antivax_claim_never_inherits_the_debunking_claims_true_rating():
    """Real data, real numbers. This is the failure the guard exists to prevent.

    Corpus entry (LIAR2/PolitiFact), rated TRUE:
        'Science is "very certain" that the measles vaccine doesn't cause autism.'
    User text:
        'The measles vaccine causes autism.'
    Measured similarity with all-MiniLM-L6-v2: 0.8601 - ABOVE the 0.85 verdict threshold.

    Without the guard, TruthLens would have reported the anti-vaccine claim as TRUE,
    because the retrieved claim is rated true and the embedding cannot see the negation.
    """
    user_text = "The measles vaccine causes autism."
    corpus_claim = ('Science is "very certain" that the measles vaccine doesn\'t '
                    'cause autism.')

    warnings = contradiction_signals(user_text, corpus_claim)
    assert warnings, "the guard must fire on this pair"
    assert any("negation mismatch" in w for w in warnings)

    # and end-to-end, at the measured similarity, the verdict must NOT be asserted
    engine = FactCheckEngine("fake", top_k=1)
    engine.model = DialEncoder([("vaccine", 0.8601, 0)])
    engine.claims = [{"claim": corpus_claim, "rating": "true", "verdict": "true",
                      "publisher": "PolitiFact", "source_url": "https://example.invalid/v"}]
    engine.matrix = np.stack([at_similarity(1.0, 0)])

    out = engine.analyze(user_text)
    assert out["verdict"] == "related"      # NOT "true"
    assert out["claims_matched"] == 0
    assert any("negation" in w for w in out["warnings"])
