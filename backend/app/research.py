"""Live research: look a claim up in Wikipedia, right now, and find the sentences that
bear on it.

WHY THIS EXISTS
---------------
A fact-check corpus only covers claims that professional fact-checkers already examined,
so most sentences come back `no_match`. That is honest but unhelpful. Live research adds a
second, clearly-weaker channel: search Wikipedia for the claim, pull the passages that are
actually about it, and let an entailment model say whether they support or contradict it.

WHAT THIS IS NOT
----------------
This is NOT a fact-check. A fact-check is a human professional's published verdict. This
is a model's reading of an encyclopedia. The two are kept in separate fields with separate
vocabulary all the way to the UI, and live evidence is weighted lower in the score. It can
never overwrite a published verdict.

PRIVACY
-------
This is the one part of TruthLens that leaves the machine. Search terms derived from the
selected text are sent to Wikipedia. It is off by a single setting, the UI says so on every
result that used it, and the rest of the product still works with it disabled.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import numpy as np

from .claims import split_sentences

log = logging.getLogger("truthlens.research")

WIKI_API = "https://en.wikipedia.org/w/api.php"
WIKI_PAGE = "https://en.wikipedia.org/wiki/"
USER_AGENT = ("TruthLens/1.0 (local fact-checking tool; "
              "https://github.com/ - student project)")

# Words that make a Wikipedia search worse rather than better.
_STOP = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had", "will", "would", "shall", "should",
    "can", "could", "may", "might", "must", "of", "to", "in", "on", "at", "by",
    "for", "with", "about", "into", "from", "up", "down", "out", "over", "under",
    "then", "than", "so", "such", "that", "this", "these", "those", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "we", "you", "i", "my",
    "and", "or", "but", "not", "no", "as", "if", "because", "while", "there",
    "here", "very", "really", "just", "also", "only", "even", "said", "says",
}
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]+|\d[\d,.]*")

# Wikipedia extracts contain section headers, "See also" lists and foreign-language
# bibliography entries. None of those are evidence about anything, and an entailment model
# asked about them will still answer confidently, so they are removed before ranking.
_JUNK = re.compile(r"^\s*=|=\s*$|^\s*\*|^[A-Z][a-z]+ family$")
_ENGLISH_MARKERS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "that", "for",
    "on", "with", "has", "have", "had", "not", "this", "it", "as", "by", "from",
    "and", "but", "at", "which", "its", "or", "be", "been", "than", "when",
}


# Coordinates, measurement tables and infobox dumps. These sit on topically relevant pages,
# so a relevance score cannot reject them - but they are not assertions, and an entailment
# model handed a table of numbers still returns a confident label. Measured failure: a
# coordinate blob on "Eiffel Tower replicas and derivatives" was read as ENTAILMENT at 0.92
# for the claim "The Eiffel Tower is located in Berlin", scoring it 91/100.
# A sentence that ATTRIBUTES a proposition to someone is evidence about a BELIEF, not
# about the proposition. An entailment model cannot make that distinction: measured, the
# "Causes of autism" sentence "some parents believe that vaccinations cause autism" was
# read as ENTAILMENT at 0.9675 for the claim "Vaccines cause autism in children", cleared
# every relevance gate (0.8346 relevance, 0.8075 evidence) and RAISED the trust score to
# 91/100. Relevance cannot reject it - the passage genuinely is about the claim - so the
# attribution has to be recognised symbolically, the same way negation is in factcheck.py.
_ATTRIBUTION = re.compile(
    r"\b(?:believe[sd]?|believing|claim(?:s|ed|ing)?|allege[sd]?|alleging|"
    r"assert(?:s|ed|ing)?|contend(?:s|ed|ing)?|insist(?:s|ed|ing)?|"
    r"speculate[sd]?|purport(?:s|ed|ing)?|rumou?r(?:s|ed)?)\b",
    re.IGNORECASE)
_PARENTHETICAL = re.compile(r"\([^)]*\)|\[[^\]]*\]")
_COORDINATE = re.compile(r"[0-9]\s*[°′″]|\d+\.\d{4,}|\bN\b\s*\d|\bkm2\b")


def usable_passage(sentence: str) -> bool:
    """A sentence is evidence only if it is PROSE, in English, and a sane length.

    Three separate questions, and all three matter:
      is it long enough to say something?   -> word count
      is it a statement, not a data table?  -> prose ratio + coordinate check
      is it English?                        -> ascii ratio + function words
      does it ASSERT, or merely REPORT      -> attribution check
        that someone believes it?
    """
    words = sentence.split()
    if not (6 <= len(words) <= 70):
        return False
    if _JUNK.search(sentence) or _COORDINATE.search(sentence):
        return False
    if _ATTRIBUTION.search(sentence):
        return False

    # Measure "is this English" with parenthetical glosses removed. Wikipedia lead
    # sentences routinely carry pronunciation and native-language spellings in brackets —
    # "The Eiffel Tower ( EYE-fəl; French: Tour Eiffel [tuʁ ɛfɛl] ) is a lattice tower..."
    # Scoring the raw string pushed that below the threshold and threw away the single most
    # definitive sentence on the page, which is exactly the one worth keeping.
    stripped = _PARENTHETICAL.sub(" ", sentence)
    ascii_ratio = sum(c.isascii() for c in stripped) / max(1, len(stripped))
    if ascii_ratio < 0.92:                      # genuinely foreign text
        return False

    # Prose has mostly word-shaped tokens. Tables, coordinate lists and infobox dumps do not.
    wordish = sum(1 for w in words if len(re.sub(r"[^A-Za-z]", "", w)) >= 2)
    if wordish / len(words) < 0.65:
        return False
    if sum(c.isdigit() for c in sentence) / max(1, len(sentence)) > 0.20:
        return False

    lowered = {w.strip(".,;:()\"'").lower() for w in words}
    return len(lowered & _ENGLISH_MARKERS) >= 2


@dataclass
class Passage:
    text: str
    title: str
    url: str
    similarity: float

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "title": self.title, "url": self.url,
                "similarity": round(self.similarity, 4)}


def search_terms(claim: str, max_terms: int = 10) -> str:
    """Turn a sentence into a search query: drop stopwords, keep names and numbers.

    A capitalised stopword is kept because it is usually part of a proper name
    ("The Hague") — EXCEPT at the start of the sentence, where the capital says nothing
    about the word. Without that exception every query begins with a useless "The".
    """
    kept: list[str] = []
    for position, token in enumerate(_WORD.findall(claim)):
        low = token.lower()
        looks_like_a_name = token[0].isupper() and position > 0
        if low in _STOP and not looks_like_a_name:
            continue
        if token not in kept:
            kept.append(token)
        if len(kept) >= max_terms:
            break
    return " ".join(kept) if kept else claim[:120]


class WikipediaResearcher:
    """Searches Wikipedia and ranks its sentences against the claim."""

    def __init__(self, embedder, timeout: float = 4.0, max_pages: int = 3,
                 max_passages: int = 4, cache_size: int = 256,
                 max_per_page: int = 2) -> None:
        self.embedder = embedder          # the SentenceTransformer already loaded
        self.timeout = timeout
        self.max_pages = max_pages
        self.max_passages = max_passages
        self.max_per_page = max_per_page
        self._cache: dict[str, list[Passage]] = {}
        self._cache_size = cache_size
        self.last_error: str | None = None

    # ---------- network ----------

    def _client(self):
        import httpx
        return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=self.timeout)

    def _search_titles(self, client, query: str) -> list[str]:
        resp = client.get(WIKI_API, params={
            "action": "query", "list": "search", "srsearch": query,
            "srlimit": self.max_pages, "format": "json", "srprop": ""})
        resp.raise_for_status()
        return [hit["title"] for hit in resp.json().get("query", {}).get("search", [])]

    def _extracts(self, client, titles: list[str]) -> dict[str, str]:
        if not titles:
            return {}
        resp = client.get(WIKI_API, params={
            "action": "query", "prop": "extracts", "titles": "|".join(titles),
            "explaintext": 1, "format": "json", "exlimit": len(titles)})
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        return {p["title"]: (p.get("extract") or "")
                for p in pages.values() if p.get("title")}

    # ---------- public ----------

    def research(self, claim: str) -> list[Passage]:
        """Return the Wikipedia sentences most related to this claim, best first."""
        key = claim.strip().lower()
        if key in self._cache:
            return self._cache[key]

        self.last_error = None
        try:
            with self._client() as client:
                titles = self._search_titles(client, search_terms(claim))
                extracts = self._extracts(client, titles)
        except Exception as exc:  # noqa: BLE001 - offline is a normal state, not a crash
            self.last_error = f"{type(exc).__name__}: {str(exc)[:120]}"
            log.warning("wikipedia lookup failed: %s", self.last_error)
            return []

        candidates: list[tuple[str, str]] = []      # (sentence, page title)
        for title, text in extracts.items():
            for sentence in split_sentences(text):
                if usable_passage(sentence):
                    candidates.append((sentence, title))
        if not candidates:
            return []

        # Rank every candidate sentence against the claim with the embedding model that
        # is already loaded — no extra model, no extra download.
        vectors = self.embedder.encode(
            [claim] + [c[0] for c in candidates], normalize_embeddings=True,
            convert_to_numpy=True, batch_size=128).astype(np.float32)
        sims = vectors[1:] @ vectors[0]

        # DIVERSIFY BY PAGE. Taking a global top-k lets one article monopolise every slot:
        # checking "the Eiffel Tower is in Berlin", the "Eiffel Tower replicas and
        # derivatives" page filled all four slots with sentences that merely MENTION the
        # tower, crowding out the actual "Eiffel Tower" article whose opening sentence
        # settles the claim. Capping how many passages any one page may contribute keeps
        # the lower-ranked but more definitive page represented.
        per_page: dict[str, int] = {}
        chosen: list[int] = []
        for i in np.argsort(-sims):
            title = candidates[i][1]
            if per_page.get(title, 0) >= self.max_per_page:
                continue
            per_page[title] = per_page.get(title, 0) + 1
            chosen.append(int(i))
            if len(chosen) >= self.max_passages:
                break

        passages = [
            Passage(text=candidates[i][0], title=candidates[i][1],
                    url=WIKI_PAGE + candidates[i][1].replace(" ", "_"),
                    similarity=float(sims[i]))
            for i in chosen
        ]
        if len(self._cache) >= self._cache_size:
            self._cache.clear()
        self._cache[key] = passages
        return passages
