"""Claim extraction: sentence segmentation and check-worthiness scoring."""
from __future__ import annotations

import pytest

from app.claims import extract_claims, score_claim, split_sentences


# ---------------- segmentation ----------------

@pytest.mark.parametrize("text,expected", [
    ("One claim here. Another claim there.", 2),
    ("Dr. Smith said the rate fell.", 1),                 # title abbreviation
    ("The U.S. economy grew last year.", 1),              # dotted acronym
    ("It rose 3.2 percent in March.", 1),                 # decimal number
    ("Is it true? Yes it is. No it isn't!", 3),
    ("Mr. Jones and Mrs. Lee met Prof. Chen.", 1),
])
def test_sentence_segmentation(text, expected):
    assert len(split_sentences(text)) == expected


def test_blank_lines_separate_claims():
    assert len(split_sentences("First block of text\n\nSecond block of text")) == 2


def test_empty_input_yields_nothing():
    assert split_sentences("   ") == []
    assert extract_claims("   ") == []


# ---------------- check-worthiness ----------------

def test_factual_assertions_outscore_opinions():
    fact, _ = score_claim("The unemployment rate fell 3.2 percent in March 2024.")
    opinion, _ = score_claim("I think that is a really beautiful result.")
    assert fact > opinion


def test_questions_are_penalised():
    statement, _ = score_claim("The economy added 250,000 jobs last year.")
    question, _ = score_claim("Did the economy add 250,000 jobs last year?")
    assert question < statement


@pytest.mark.parametrize("sentence,signal", [
    ("Inflation hit 9 percent.", "contains a number"),
    ("Growth reached 5 million dollars.", "contains a quantity or unit"),
    ("The claim about Kenya was examined.", "names a person, place or organisation"),
    ("The policy was signed last week.", "uses an assertive verb"),
    ("Is that correct?", "is a question"),
    ("I think it is fine.", "reads as opinion, not assertion"),
])
def test_signals_are_reported_for_transparency(sentence, signal):
    _, signals = score_claim(sentence)
    assert signal in signals


def test_score_is_bounded():
    for s in ["?", "I think maybe perhaps I hope?", "A" * 500,
              "The Bureau reported 250,000 jobs and 9.1 percent inflation in March 2024."]:
        score, _ = score_claim(s)
        assert 0.0 <= score <= 1.0


# ---------------- extraction ----------------

def test_opinions_and_questions_are_filtered_out():
    text = ("Dr. Smith said the unemployment rate fell 3.2 percent last year. "
            "I think that is great news. "
            "Is that really true? "
            "The U.S. economy added 250,000 jobs in March.")
    kept = [c.text for c in extract_claims(text)]
    assert len(kept) == 2
    assert all("I think" not in k and not k.endswith("?") for k in kept)


def test_claims_come_back_best_first():
    text = "It happened. The Bureau reported 250,000 new jobs in March 2024."
    claims = extract_claims(text)
    assert claims[0].score >= claims[-1].score
    assert "250,000" in claims[0].text


def test_never_returns_empty_for_real_input():
    """Even unpromising text yields the best candidate, with an explanation of why."""
    claims = extract_claims("hmm ok")
    assert len(claims) == 1
    assert any("below the check-worthiness threshold" in s for s in claims[0].signals)


def test_respects_the_claim_cap():
    text = " ".join(f"The agency reported {n}00 new cases in March." for n in range(1, 12))
    assert len(extract_claims(text, max_claims=4)) == 4


# ---------------- regression ----------------

@pytest.mark.parametrize("sentence", [
    "Vaccines cause autism in children.",          # base-form verb, no number, no proper noun
    "The election was stolen.",
    "Masks do nothing to stop transmission.",
    "Climate change is a hoax invented overseas.",
])
def test_plain_declarative_claims_are_never_skipped(sentence):
    """Regression: an earlier whitelist-of-verbs scorer silently dropped these.

    English verbs cannot be enumerated, so the scorer must assume a declarative sentence
    is checkable and subtract only for disqualifying features.
    """
    claims = extract_claims(sentence)
    assert len(claims) == 1
    assert claims[0].text == sentence
    assert claims[0].score >= 0.40
    assert "below the check-worthiness threshold" not in " ".join(claims[0].signals)


@pytest.mark.parametrize("sentence,signal", [
    ("The vaccine might cause mild side effects.", "is hedged rather than asserted"),
    ("Please click the link below.", "is an instruction, not an assertion"),
])
def test_hedged_and_imperative_text_is_demoted(sentence, signal):
    score, signals = score_claim(sentence)
    assert signal in signals
    assert score < 0.40
