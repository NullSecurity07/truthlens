"""Claim extraction: turn a free-text selection into individual checkable claims.

WHY THIS EXISTS
---------------
The fact-check corpus is indexed at *sentence* granularity - every PolitiFact statement is
one sentence. If we embed a whole paragraph into a single vector, that vector is the
average of every sentence in it, so a paragraph containing one checkable falsehood plus
four neutral sentences resembles neither the falsehood nor anything else. The signal is
diluted below the match threshold. Matching sentence-by-sentence restores it.

TWO STAGES
----------
1. `split_sentences`  - segment the text, without dragging in an NLP dependency.
2. `score_claim`      - rank each sentence by "check-worthiness": is this the kind of
                        statement a fact-checker would even look at? Questions, opinions
                        and pleasantries are not. This is a transparent, inspectable
                        heuristic - every contribution is reported in the response, so a
                        reader can disagree with a specific rule rather than a black box.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Abbreviations that end in '.' but do not end a sentence.
_ABBREV = (
    r"Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Lt|Gen|Sen|Rep|Gov|Col|Capt|Rev"
    r"|vs|etc|approx|est|Inc|Ltd|Co|Corp|Univ|Dept|Fig|No|al"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
)
# A placeholder that cannot occur in ordinary prose, used to hide abbreviation dots
# from the sentence splitter and then restored afterwards.
_SENTINEL = "<!DOT!>"
_PROTECT = re.compile(rf"\b({_ABBREV})\.", re.IGNORECASE)
_PROTECT_DOTTED = re.compile(r"\b([A-Za-z](?:\.[A-Za-z])+)\.")   # U.S.  e.g.  a.m.
_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"'“(\[]?[A-Z0-9])")

# Signals that a sentence asserts a checkable fact.
_NUMBER = re.compile(r"\d")
_QUANTITY = re.compile(
    r"(\d+(\.\d+)?\s*(%|percent|per cent|million|billion|trillion|thousand|kg|km|miles|"
    r"dollars|usd|eur|inr|years?|months?|days?))|([$£€₹]\s*\d)",
    re.IGNORECASE)
_PROPER = re.compile(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{2,}")
_FACT_VERB = re.compile(
    r"\b(is|are|was|were|has|have|had|will|would|did|does|said|says|claimed|claims|"
    r"reported|reports|found|shows|showed|caused|causes|increased|decreased|rose|fell|"
    r"voted|passed|signed|killed|created|banned|announced|admitted|denied)\b",
    re.IGNORECASE)
_OPINION = re.compile(
    r"\b(i think|i believe|i feel|in my opinion|imho|we should|you should|must be|"
    r"probably|maybe|perhaps|i guess|i hope|i wish|seems like|feels like|"
    r"beautiful|ugly|lovely|awesome|amazing|terrible)\b", re.IGNORECASE)
_FIRST_PERSON = re.compile(r"\b(i|my|me|mine)\b", re.IGNORECASE)
_HEDGE = re.compile(r"\b(might|could|may|possibly|allegedly|reportedly|rumou?red|"
                    r"some say|apparently)\b", re.IGNORECASE)
_IMPERATIVE = re.compile(r"^\s*(please|click|subscribe|share|read|check out|buy|"
                         r"follow|sign up|download|watch)\b", re.IGNORECASE)

MIN_WORDS = 4
MAX_WORDS = 80
CHECKWORTHY_THRESHOLD = 0.40
MAX_CLAIMS = 6


@dataclass
class Claim:
    text: str
    index: int
    score: float
    signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"text": self.text, "index": self.index,
                "checkworthiness": round(self.score, 3), "signals": self.signals}


def split_sentences(text: str) -> list[str]:
    """Segment into sentences. Abbreviations are masked so 'Dr. Smith' stays one sentence."""
    protected = _PROTECT.sub(lambda m: m.group(1) + _SENTINEL, text.strip())
    protected = _PROTECT_DOTTED.sub(lambda m: m.group(1).replace(".", _SENTINEL) + _SENTINEL,
                                    protected)
    out: list[str] = []
    for part in _SPLIT.split(protected):
        for chunk in re.split(r"\n{2,}", part):        # blank lines also separate claims
            chunk = chunk.replace(_SENTINEL, ".").strip()
            if chunk:
                out.append(chunk)
    return out


def score_claim(sentence: str) -> tuple[float, list[str]]:
    """Return (checkworthiness in 0..1, the human-readable reasons for it)."""
    words = sentence.split()
    # BLACKLIST, NOT WHITELIST. An earlier version required a verb from a fixed list and
    # therefore silently skipped "Vaccines cause autism in children." - the exact kind of
    # claim this tool exists for - because the list had "causes" but not "cause". English
    # verbs cannot be enumerated, so a declarative sentence is assumed checkable and points
    # are SUBTRACTED for the things that genuinely disqualify it. An unseen verb now costs
    # nothing; the worst case is checking a harmless sentence and reporting no_match.
    score, signals = 0.45, []

    if _NUMBER.search(sentence):
        score += 0.18
        signals.append("contains a number")
    if _QUANTITY.search(sentence):
        score += 0.12
        signals.append("contains a quantity or unit")
    if _PROPER.search(sentence):
        score += 0.15
        signals.append("names a person, place or organisation")
    if _FACT_VERB.search(sentence):
        score += 0.10
        signals.append("uses an assertive verb")
    if _HEDGE.search(sentence):
        score -= 0.15
        signals.append("is hedged rather than asserted")

    if sentence.rstrip().endswith("?"):
        score -= 0.35
        signals.append("is a question")
    if _OPINION.search(sentence):
        score -= 0.28
        signals.append("reads as opinion, not assertion")
    if _FIRST_PERSON.search(sentence):
        score -= 0.18
        signals.append("is first-person")
    if _IMPERATIVE.search(sentence):
        score -= 0.20
        signals.append("is an instruction, not an assertion")
    if len(words) < MIN_WORDS:
        score -= 0.30
        signals.append(f"shorter than {MIN_WORDS} words")
    if len(words) > MAX_WORDS:
        score -= 0.20
        signals.append(f"longer than {MAX_WORDS} words")

    return max(0.0, min(1.0, score)), signals


def extract_claims(text: str, max_claims: int = MAX_CLAIMS,
                   threshold: float = CHECKWORTHY_THRESHOLD) -> list[Claim]:
    """Extract check-worthy claims, best first.

    Never returns empty for non-empty input: if nothing clears the threshold we still
    return the single best sentence, so the user always sees that something was checked
    and why it scored low, rather than a silent nothing.
    """
    sentences = split_sentences(text)
    if not sentences:
        return []

    scored = []
    for i, sentence in enumerate(sentences):
        score, signals = score_claim(sentence)
        scored.append(Claim(text=sentence, index=i, score=score, signals=signals))

    checkworthy = [c for c in scored if c.score >= threshold]
    if not checkworthy:
        best = max(scored, key=lambda c: c.score)
        best.signals.append("below the check-worthiness threshold; checked anyway as the "
                            "best available candidate")
        return [best]

    checkworthy.sort(key=lambda c: (-c.score, c.index))
    return checkworthy[:max_claims]
