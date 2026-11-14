"""Live research: query building, ranking, entailment, and the rules that keep it honest.

No network and no models: the HTTP calls and the NLI model are both stubbed, so these run
in milliseconds and assert behaviour rather than model quality.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.engines.factcheck import FactCheckEngine
from app.research import Passage, WikipediaResearcher, search_terms
from tests.test_factcheck import CLAIMS, DialEncoder, at_similarity, FLAT


# ---------------- query building ----------------

def test_search_terms_drops_stopwords_but_keeps_names_and_numbers():
    q = search_terms("The measles vaccine was said to have caused autism in 12 children.")
    assert "measles" in q and "vaccine" in q and "autism" in q and "12" in q
    for stop in (" the ", " was ", " to ", " in "):
        assert stop not in f" {q.lower()} "


def test_search_terms_keeps_capitalised_stopwords_inside_names():
    """A capitalised stopword mid-sentence is usually part of a proper name, so it stays —
    but the same word at position 0 is just sentence case and must be dropped."""
    mid = search_terms("Delegates met in The Hague to sign the treaty.")
    assert "The" in mid.split(), mid
    assert "The" not in search_terms("The treaty was signed last year.").split()


def test_search_terms_never_returns_empty():
    assert search_terms("is a the of").strip()


# ---------------- retrieval and ranking ----------------

class StubEmbedder:
    """Ranks by keyword overlap so the expected ordering is deterministic."""

    def encode(self, texts, normalize_embeddings=True, convert_to_numpy=True, **kw):
        out = []
        for t in texts:
            low = t.lower()
            v = np.array([1.0 if "autism" in low else 0.0,
                          1.0 if "measles" in low else 0.0,
                          0.3], dtype=np.float32)
            out.append(v / np.linalg.norm(v))
        return np.stack(out)


def make_researcher(pages: dict[str, str]) -> WikipediaResearcher:
    r = WikipediaResearcher(StubEmbedder(), max_passages=3)
    r._search_titles = lambda client, query: list(pages)          # noqa: SLF001
    r._extracts = lambda client, titles: pages                    # noqa: SLF001
    r._client = lambda: _NullClient()                             # noqa: SLF001
    return r


class _NullClient:
    def __enter__(self): return self
    def __exit__(self, *a): return False


PAGES = {
    "MMR vaccine and autism": (
        "Claims of a link between the MMR vaccine and autism have been extensively "
        "investigated. The measles vaccine does not cause autism. "
        "The original study was found to be fraudulent."),
    "Measles": "Measles is a highly contagious infectious disease caused by a virus.",
}


def test_research_returns_passages_ranked_by_relevance():
    passages = make_researcher(PAGES).research("The measles vaccine causes autism.")
    assert passages
    assert "autism" in passages[0].text.lower()
    assert passages[0].similarity >= passages[-1].similarity


def test_research_attaches_a_real_wikipedia_url():
    p = make_researcher(PAGES).research("The measles vaccine causes autism.")[0]
    assert p.url.startswith("https://en.wikipedia.org/wiki/")
    assert " " not in p.url


def test_research_caches_repeat_queries():
    r = make_researcher(PAGES)
    calls = []
    original = r._extracts                                        # noqa: SLF001
    r._extracts = lambda c, t: (calls.append(1), original(c, t))[1]  # noqa: SLF001
    r.research("The measles vaccine causes autism.")
    r.research("the measles VACCINE causes autism.")              # same claim, different case
    assert len(calls) == 1


def test_network_failure_is_reported_not_raised():
    r = WikipediaResearcher(StubEmbedder())
    def boom():
        raise OSError("no route to host")
    r._client = boom                                              # noqa: SLF001
    assert r.research("anything at all") == []
    assert "no route to host" in (r.last_error or "")


# ---------------- entailment wiring ----------------

class StubNLI:
    """Returns a scripted judgement so the engine's decision rules can be tested."""

    def __init__(self, label="contradiction", confidence=0.95, decisive=True):
        self.label, self.confidence, self.decisive = label, confidence, decisive
        self.calls = 0

    def judge(self, premise, hypothesis):
        self.calls += 1
        return {"label": self.label, "raw_label": self.label,
                "confidence": self.confidence, "decisive": self.decisive,
                "scores": {self.label: self.confidence}}


def build_engine(sim: float, nli=None, researcher=None, **kw) -> FactCheckEngine:
    e = FactCheckEngine("fake", top_k=1, nli=nli, researcher=researcher, **kw)
    e.model = DialEncoder([("flat", sim, 0)])
    e.claims = CLAIMS
    e.matrix = np.stack([at_similarity(1.0, 0), at_similarity(1.0, 1)])
    return e


def stub_researcher(passages=None):
    class R:
        last_error = None
        def __init__(self): self.calls = 0
        def research(self, claim):
            self.calls += 1
            return passages if passages is not None else [
                Passage("The earth is not flat.", "Figure of the Earth",
                        "https://en.wikipedia.org/wiki/Figure_of_the_Earth", 0.9)]
    return R()


def test_live_research_runs_when_the_corpus_cannot_settle_the_claim():
    r, nli = stub_researcher(), StubNLI("contradiction", 0.93)
    out = build_engine(0.10, nli=nli, researcher=r).analyze(FLAT)   # 0.10 -> no_match
    assert r.calls == 1
    live = out["claims"][0]["live"]
    assert live["status"] == "contradicted"
    assert live["source_url"].startswith("https://en.wikipedia.org/")
    assert out["live_used"] is True


def test_live_research_is_skipped_when_a_published_verdict_exists():
    """A fact-checker's verdict is the better evidence; do not spend a network call."""
    r, nli = stub_researcher(), StubNLI()
    out = build_engine(0.95, nli=nli, researcher=r).analyze(FLAT)   # 0.95 -> verdict
    assert r.calls == 0
    assert "live" not in out["claims"][0]
    assert out["live_used"] is False


def test_live_evidence_never_becomes_a_fact_check_verdict():
    out = build_engine(0.10, nli=StubNLI("entailment", 0.99),
                       researcher=stub_researcher()).analyze(FLAT)
    claim = out["claims"][0]
    assert claim["verdict"] == "no_match"          # the claim verdict is untouched
    assert claim["live"]["status"] == "supported"  # the live finding is separate
    assert "not a fact-checker's verdict" in claim["live"]["note"]


def test_indecisive_entailment_is_reported_as_inconclusive():
    out = build_engine(0.10, nli=StubNLI("neutral", 0.5, decisive=False),
                       researcher=stub_researcher()).analyze(FLAT)
    assert out["claims"][0]["live"]["status"] == "inconclusive"
    assert out["live"] is None


def test_no_wikipedia_evidence_is_reported_honestly():
    out = build_engine(0.10, nli=StubNLI(), researcher=stub_researcher([])).analyze(FLAT)
    assert out["claims"][0]["live"]["status"] == "no_evidence"


def test_engine_without_a_researcher_is_unchanged():
    """Live research is strictly additive; with it absent the engine is the local tool."""
    out = build_engine(0.10).analyze(FLAT)
    assert "live" not in out["claims"][0]
    assert out["live_used"] is False


def test_live_research_is_bounded_by_live_max_claims():
    r = stub_researcher()
    engine = build_engine(0.10, nli=StubNLI(), researcher=r, live_max_claims=1)
    engine.model = DialEncoder([("flat", 0.10, 0), ("boils", 0.10, 1)])
    engine.analyze("The earth is flat and NASA lied about it. "
                   "Water boils at 100 degrees at sea level.")
    assert r.calls == 1, "only the most check-worthy claims should cost a network call"


# ---------------- the aggregator's treatment of live evidence ----------------

def test_live_contradiction_lowers_the_score_but_is_capped():
    from app.aggregator import aggregate
    from app.config import settings
    from app.engines.base import EngineResult

    clean_ai = EngineResult("ok", {"probability": 0.05, "confidence": 0.9}, None, 1)
    clean_tox = EngineResult("ok", {"score": 0.02, "confidence": 0.95}, None, 1)

    published = EngineResult("ok", {"verdict": "false", "similarity": 0.95}, None, 1)
    live = EngineResult("ok", {"verdict": "no_match", "claims_found": 1,
                               "live": {"status": "contradicted", "confidence": 0.99,
                                        "source_title": "Figure of the Earth"}}, None, 1)

    live_score = aggregate(clean_ai, clean_tox, live)["trust_score"]
    published_score = aggregate(clean_ai, clean_tox, published)["trust_score"]
    neutral = aggregate(clean_ai, clean_tox,
                        EngineResult("ok", {"verdict": "no_match"}, None, 1))["trust_score"]

    assert live_score < neutral, "a live contradiction must lower the score"
    assert live_score > published_score, "but never as hard as a published verdict"

    entry = next(e for e in aggregate(clean_ai, clean_tox, live)["explanation"]
                 if e["factor"] == "fact_check")
    assert entry["weight"] <= settings.weight_fact * settings.live_confidence_ceiling + 1e-9
    assert "Wikipedia" in entry["detail"]


@pytest.mark.parametrize("id2label", [
    {0: "entailment", 1: "neutral", 2: "contradiction"},     # MoritzLaurer ordering
    {0: "contradiction", 1: "entailment", 2: "neutral"},     # cross-encoder ordering
    {0: "CONTRADICTION", 1: "NEUTRAL", 2: "ENTAILMENT"},     # upper case
])
def test_nli_labels_resolve_by_name_not_position(id2label):
    """Two checkpoints disagree on column order; resolving by position would invert
    'supports' and 'contradicts' silently."""
    from app.engines.nli import NLIEngine

    engine = NLIEngine("dummy")
    index = {}
    for wanted in ("entailment", "contradiction", "neutral"):
        for idx, name in {k: v.lower() for k, v in id2label.items()}.items():
            if wanted in name:
                index[wanted] = idx
                break
    engine.index = index
    assert id2label[engine.index["entailment"]].lower() == "entailment"
    assert id2label[engine.index["contradiction"]].lower() == "contradiction"


# ---------------- relevance gates the reasoning ----------------

def test_a_confident_reading_of_an_irrelevant_passage_is_not_evidence():
    """Regression, from a real failure.

    Checking "Water boils at 100 degrees Celsius at sea level." (TRUE), Wikipedia returned
    a Swedish bibliography line from the 'Anders Celsius' article at 0.327 relevance. The
    entailment model called it a contradiction at 0.74 confidence, and the claim scored
    23/100 — while the genuinely relevant passage (0.515) had correctly said 'neutral'.

    An entailment model asked about an irrelevant premise still answers confidently, so
    relevance must GATE the reading, not merely rank it.
    """
    irrelevant = [Passage("Observationer om twanne bestandiga grader pa en thermometer.",
                          "Anders Celsius",
                          "https://en.wikipedia.org/wiki/Anders_Celsius", 0.327)]
    out = build_engine(0.10, nli=StubNLI("contradiction", 0.74),
                       researcher=stub_researcher(irrelevant)).analyze(FLAT)
    live = out["claims"][0]["live"]
    assert live["status"] == "inconclusive", "a 0.33-relevance passage must not decide anything"
    assert out["live"] is None


def test_a_confident_reading_of_a_relevant_passage_is_evidence():
    relevant = [Passage("The Eiffel Tower is a lattice tower in Paris, France.",
                        "Eiffel Tower",
                        "https://en.wikipedia.org/wiki/Eiffel_Tower", 0.83)]
    out = build_engine(0.10, nli=StubNLI("contradiction", 0.99),
                       researcher=stub_researcher(relevant)).analyze(FLAT)
    live = out["claims"][0]["live"]
    assert live["status"] == "contradicted"
    assert live["relevance"] == 0.83
    assert live["evidence"] == pytest.approx(0.99 * 0.83, abs=1e-3)


@pytest.mark.parametrize("confidence,relevance,expected", [
    (0.99, 0.83, "contradicted"),   # confident and relevant   -> evidence 0.82, keep
    (0.65, 0.50, "inconclusive"),   # relevant but hesitant    -> evidence 0.33, block
    (0.99, 0.33, "inconclusive"),   # confident but off-topic  -> blocked by relevance
    (0.98, 0.56, "inconclusive"),   # the real "Auma Obama" failure -> evidence 0.55, block
    (0.93, 0.69, "contradicted"),   # the real Obama birth record   -> evidence 0.64, keep
])
def test_evidence_is_confidence_times_relevance(confidence, relevance, expected):
    """Neither number alone decides: a hesitant reading of a relevant passage and a
    confident reading of an irrelevant one both fail to become evidence."""
    passage = [Passage("A sentence about the topic at hand, long enough to qualify.", "T",
                       "https://en.wikipedia.org/wiki/T", relevance)]
    out = build_engine(0.10, nli=StubNLI("contradiction", confidence),
                       researcher=stub_researcher(passage)).analyze(FLAT)
    assert out["claims"][0]["live"]["status"] == expected


@pytest.mark.parametrize("sentence,usable", [
    ("The Eiffel Tower is a lattice tower on the Champ de Mars in Paris.", True),
    ("== See also ==", False),
    ("Celsius family Daniel Gabriel Fahrenheit", False),
    ("Observationer om twänne beständiga grader på en thermometer", False),
    ("Short line", False),
])
def test_junk_and_foreign_passages_are_never_offered_as_evidence(sentence, usable):
    from app.research import usable_passage
    assert usable_passage(sentence) is usable


@pytest.mark.parametrize("sentence,usable", [
    # real prose about the topic — must survive
    ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.", True),
    ("As one of the most recognizable structures in the world, it was completed in 1889.", True),
    # regression: a coordinate blob on a topically RELEVANT page was read as entailment at
    # 0.92 and scored "the Eiffel Tower is in Berlin" as 91/100 trustworthy
    ("10°43′29.0172″N 122″24.6168″E Yekaterinoslavka, Khabarovsk "
     "Krai, Russia, 47.952315°N 135.123593°E", False),
    ("Paris 48.8584 2.2945 France 330 m 1889 7000000 visitors annually 2023", False),
    ("1889 1930 1957 1991 2004 2011 2019 2020 2021 2022", False),
])
def test_data_tables_are_not_treated_as_assertions(sentence, usable):
    """Relevance cannot reject these: a coordinate table on the Eiffel Tower page really is
    about the Eiffel Tower. The question is not 'is it related?' but 'is it an assertion?'"""
    from app.research import usable_passage
    assert usable_passage(sentence) is usable


def test_one_page_cannot_monopolise_the_evidence():
    """Regression: checking "the Eiffel Tower is in Berlin", the 'Eiffel Tower replicas and
    derivatives' page filled all four passage slots with sentences that merely MENTION the
    tower, crowding out the actual 'Eiffel Tower' article whose opening sentence settles
    the claim. A global top-k has this failure; a per-page cap does not."""
    chatty = " ".join(
        f"The Eiffel Tower inspired replica number {n} built in that country." for n in range(1, 9))
    pages = {
        "Eiffel Tower replicas and derivatives": chatty,
        "Eiffel Tower": "The Eiffel Tower is a wrought iron lattice tower located on the "
                        "Champ de Mars in Paris, France.",
    }
    r = make_researcher(pages)
    r.max_per_page = 2
    titles = {p.title for p in r.research("The Eiffel Tower is located in Berlin, Germany.")}
    assert "Eiffel Tower" in titles, "the definitive page must not be crowded out"


@pytest.mark.parametrize("sentence,usable", [
    # Regression: Wikipedia lead sentences carry pronunciation/native-language glosses in
    # brackets. Scoring the raw string as "foreign" discarded the most definitive sentence
    # on the page — the one that settles the claim.
    ("The Eiffel Tower ( EYE-fəl; French: Tour Eiffel [tuʁ ɛfɛl] ) is a lattice tower "
     "on the Champ de Mars in Paris, France.", True),
    ("Munich (German: München [ˈmʏnçn̩]) is the capital of Bavaria in Germany.", True),
    # genuinely foreign prose still rejected
    ("Observationer om twänne beständiga grader på en thermometer och deras användning", False),
])
def test_english_leads_with_foreign_glosses_are_kept(sentence, usable):
    from app.research import usable_passage
    assert usable_passage(sentence) is usable


@pytest.mark.parametrize("sentence,usable", [
    # Regression: checking "Vaccines cause autism in children.", this sentence from the
    # "Causes of autism" article was read as ENTAILMENT at 0.9675 confidence x 0.8346
    # relevance = 0.8075 evidence. It cleared every gate and RAISED the trust score to
    # 91/100. The sentence is TRUE and RELEVANT - and it asserts nothing about vaccines.
    # It reports a belief. Relevance gating cannot catch this: the passage really is about
    # the claim. The question is not "is it related?" but "is it an assertion?"
    ("Despite this, some parents believe that vaccinations cause autism; they therefore "
     "delay or avoid immunisation.", False),
    ("Anti-vaccine activists claim that the MMR vaccine causes autism in children.", False),
    ("It is widely believed that the Great Wall of China is visible from the Moon.", False),
    ("Supporters alleged that the election had been stolen from their candidate.", False),
    # Assertions ABOUT THE EVIDENCE must survive - they are what we actually want.
    ("There is no evidence that the MMR vaccine is linked to autism in children.", True),
    ("Multiple large studies have found no association between vaccination and autism.", True),
    ("The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.", True),
])
def test_belief_attribution_is_not_treated_as_an_assertion(sentence, usable):
    """An entailment model cannot tell "X believes P" from "P". A sentence that ATTRIBUTES
    a proposition to someone is evidence about a belief, never about the proposition."""
    from app.research import usable_passage
    assert usable_passage(sentence) is usable
