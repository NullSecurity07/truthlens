"""Fact-check vector: per-claim dense retrieval over a local corpus, with a guard.

WHY IT IS BUILT THIS WAY (all three points are measured, see scripts/calibrate_threshold.py)

1. PER-CLAIM, NOT PER-SELECTION.
   The corpus is indexed at sentence granularity. Embedding a whole paragraph averages its
   sentences together and dilutes the signal below any threshold. So the selection is first
   split into individual check-worthy claims (app/claims.py) and each is matched separately.

2. TIERED OUTCOMES, NOT A SINGLE THRESHOLD.
   Measured on 400 perturbed corpus claims:
       genuine paraphrase      median similarity 0.900
       unrelated everyday text max    similarity 0.497
       best NON-matching claim mean   similarity 0.610, p90 0.745
   A single cut-off must therefore either assert verdicts on wrong claims or reject real
   paraphrases. Instead there are three outcomes:
       >= verdict_threshold (0.80)  -> assert the fact-checker's verdict
       >= related_threshold (0.62)  -> show as RELATED CONTEXT, explicitly not a verdict
       below                        -> no_match
   The middle tier is the honest answer to "close, but I will not put someone else's
   rating on your sentence".

3. A CONTRADICTION GUARD ON TOP OF THE NEURAL SCORE.
   Sentence embeddings encode topic, not truth conditions. Measured: 100% of claims that
   were NEGATED or had a number changed still retrieved the original, at mean similarity
   0.959 - HIGHER than genuine paraphrases. A threshold cannot fix this, because the bad
   case scores better than the good case. So a symbolic check compares negation count and
   numeric content between the query and the matched claim; a mismatch downgrades a
   verdict to "related", because the verdict provably may not transfer.

If GOOGLE_FACTCHECK_API_KEY is present the live API is queried too and preferred. It is
never required and never prompted for.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

import numpy as np

from ..claims import extract_claims
from ..config import settings
from .base import Engine

log = logging.getLogger("truthlens.engines.factcheck")

RATING_TO_VERDICT = {
    "false": "false", "pants on fire": "false", "pants-fire": "false", "incorrect": "false",
    "true": "true", "correct": "true", "accurate": "true",
    "half true": "mixed", "half-true": "mixed", "mostly true": "mixed",
    "mostly-true": "mixed", "mostly false": "mixed", "mostly-false": "mixed",
    "misleading": "mixed", "mixture": "mixed",
}
# Ordered worst-first: the overall verdict for a selection is its worst asserted claim.
VERDICT_SEVERITY = {"false": 3, "mixed": 2, "true": 1, "related": 0, "no_match": 0}

# Only CLAUSE-SCOPED negation counts. A bare "not" is often contrastive rather than
# negating: "born in Kenya, not the United States" ASSERTS the claim, while "did not grow
# up in Kenya" REVERSES it. Counting bare tokens conflated the two and produced a false
# guard trip on a genuine match, so negation must be bound to an auxiliary verb.
_NEGATION = re.compile(
    r"\b(?:is|are|am|was|were|be|been|being|has|have|had|do|does|did|will|would|can|"
    r"could|shall|should|may|might|must)\s+not\b"
    r"|\bn't\b|\w+n't\b"
    r"|\bcannot\b|\bnever\b|\bno longer\b|\bnot at all\b"
    r"|\bdenies\b|\bdenied\b|\bdebunked\b|\buntrue\b",
    re.IGNORECASE)
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def _numbers(text: str) -> set[float]:
    out: set[float] = set()
    for raw in _NUMBER.findall(text):
        try:
            out.add(float(raw.replace(",", "")))
        except ValueError:
            continue
    return out


def contradiction_signals(query: str, matched: str) -> list[str]:
    """Symbolic differences that mean the matched claim's verdict may not transfer.

    This exists because the neural score cannot see either difference - both mutations
    scored HIGHER than genuine paraphrases in calibration.
    """
    signals: list[str] = []

    q_neg, m_neg = len(_NEGATION.findall(query)), len(_NEGATION.findall(matched))
    if q_neg != m_neg:
        signals.append(
            f"negation mismatch: your text has {q_neg} negation word(s), the "
            f"fact-checked claim has {m_neg}. These may be opposite assertions.")

    q_num, m_num = _numbers(query), _numbers(matched)
    if q_num and m_num and not (q_num & m_num):
        signals.append(
            f"numeric mismatch: your text cites {sorted(q_num)}, the fact-checked claim "
            f"cites {sorted(m_num)}. The rating applies to the other figure.")

    return signals


class FactCheckEngine(Engine):
    name = "fact_check"

    def __init__(self, embedding_model: str, device: str = "cpu",
                 threshold: float | None = None, top_k: int = 3,
                 google_api_key: str | None = None,
                 related_threshold: float | None = None,
                 max_claims: int | None = None,
                 researcher: Any = None, nli: Any = None,
                 live_max_claims: int | None = None) -> None:
        self.embedding_model = embedding_model
        self.device = device
        self.verdict_threshold = (threshold if threshold is not None
                                  else settings.verdict_threshold)
        self.related_threshold = (related_threshold if related_threshold is not None
                                  else settings.related_threshold)
        # keep the old attribute name working for existing callers/tests
        self.threshold = self.verdict_threshold
        self.top_k = top_k
        self.max_claims = max_claims if max_claims is not None else settings.max_claims
        self.google_api_key = google_api_key
        # Optional live-research pair. Absent -> the engine is purely local, unchanged.
        self.researcher = researcher
        self.nli = nli
        self.live_max_claims = (live_max_claims if live_max_claims is not None
                                else settings.live_max_claims)
        self.model = None
        self.claims: list[dict[str, Any]] = []
        self.matrix: np.ndarray | None = None

    # ---------- lifecycle ----------

    def load(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(self.embedding_model, device=self.device)
        if settings.corpus_path.exists() and settings.embeddings_path.exists():
            with settings.corpus_path.open(encoding="utf-8") as fh:
                self.claims = [json.loads(line) for line in fh if line.strip()]
            self.matrix = np.load(settings.embeddings_path)
            if len(self.claims) != self.matrix.shape[0]:
                raise RuntimeError(
                    f"corpus/embedding mismatch: {len(self.claims)} claims vs "
                    f"{self.matrix.shape[0]} vectors. Re-run scripts/build_corpus.py")
            log.info("fact-check corpus loaded: %d claims, dim=%d",
                     len(self.claims), self.matrix.shape[1])
        else:
            log.warning("no local corpus - fact-check runs in no-corpus mode. "
                        "Run scripts/build_corpus.py to enable it.")

    @property
    def corpus_size(self) -> int:
        return len(self.claims)

    # ---------- retrieval ----------

    def _search_batch(self, texts: list[str]) -> list[list[dict[str, Any]]]:
        """Embed every claim in ONE forward pass, then rank all of them at once."""
        if self.matrix is None or not self.claims or not texts:
            return [[] for _ in texts]
        qv = self.model.encode(texts, normalize_embeddings=True,
                               convert_to_numpy=True).astype(np.float32)
        sims = qv @ self.matrix.T                      # (n_queries, n_corpus), cosine
        results = []
        k = min(self.top_k, sims.shape[1])
        for row in sims:
            idx = np.argpartition(-row, k - 1)[:k]
            idx = idx[np.argsort(-row[idx])]
            results.append([{**self.claims[i], "similarity": round(float(row[i]), 4)}
                            for i in idx])
        return results

    def _google(self, text: str) -> list[dict[str, Any]]:
        """Optional. Only runs when a free API key is present in the environment."""
        if not self.google_api_key:
            return []
        import httpx

        try:
            resp = httpx.get(
                "https://factchecktools.googleapis.com/v1alpha1/claims:search",
                params={"query": text[:200], "key": self.google_api_key, "pageSize": 5},
                timeout=3.0)
            resp.raise_for_status()
            out = []
            for claim in resp.json().get("claims", []):
                for review in claim.get("claimReview", []):
                    rating = (review.get("textualRating") or "").lower()
                    out.append({
                        "claim": claim.get("text", ""), "rating": rating,
                        "verdict": RATING_TO_VERDICT.get(rating, "mixed"),
                        "publisher": (review.get("publisher") or {}).get("name"),
                        "source_url": review.get("url"), "similarity": None})
            return out
        except Exception as exc:  # noqa: BLE001 - optional path must never break analysis
            log.warning("google fact-check lookup failed: %s", exc)
            return []

    # ---------- per-claim decision ----------

    def _judge(self, claim_text: str, hits: list[dict[str, Any]],
               remote: list[dict[str, Any]]) -> dict[str, Any]:
        if remote:
            best = remote[0]
            return {
                "verdict": best["verdict"], "rating": best.get("rating"),
                "matched_claim": best["claim"], "publisher": best.get("publisher"),
                "source_url": best.get("source_url"), "similarity": None,
                "source": "google_factcheck", "warnings": [],
                "explanation": "Matched against the live Google Fact Check Tools API.",
            }

        if not hits:
            return {"verdict": "no_match", "similarity": None, "source": "none",
                    "warnings": [], "matched_claim": None,
                    "explanation": "No fact-check corpus is loaded."}

        best = hits[0]
        sim = best["similarity"]

        if sim < self.related_threshold:
            return {
                "verdict": "no_match", "similarity": sim, "source": "local_corpus",
                "matched_claim": None, "warnings": [],
                "explanation": (
                    f"Nothing in the corpus resembles this claim (closest was {sim:.2f}, "
                    f"below the {self.related_threshold:.2f} relatedness threshold). "
                    f"That is not evidence the claim is true - only that no fact-checker "
                    f"in this dataset has examined it."),
            }

        warnings = contradiction_signals(claim_text, best["claim"])
        asserts_verdict = sim >= self.verdict_threshold and not warnings

        if asserts_verdict:
            return {
                "verdict": best["verdict"], "rating": best.get("rating"),
                "matched_claim": best["claim"], "publisher": best.get("publisher"),
                "source_url": best.get("source_url"), "similarity": sim,
                "source": "local_corpus", "warnings": [],
                "explanation": (
                    f"Matches a claim rated '{best.get('rating')}' by "
                    f"{best.get('publisher')} at {sim:.2f} similarity."),
            }

        if warnings:
            reason = ("The wording differs in a way the verdict may not survive: "
                      + " ".join(warnings))
        else:
            reason = (f"Similarity {sim:.2f} is below the {self.verdict_threshold:.2f} "
                      f"threshold required to assert a verdict, so this is shown as "
                      f"context only.")
        return {
            "verdict": "related", "rating": best.get("rating"),
            "matched_claim": best["claim"], "publisher": best.get("publisher"),
            "source_url": best.get("source_url"), "similarity": sim,
            "source": "local_corpus", "warnings": warnings,
            "explanation": ("A related fact-checked claim was found, but this is NOT a "
                            "verdict on your text. " + reason),
        }

    # ---------- live research ----------

    def _research(self, claim_text: str) -> dict[str, Any]:
        """Look the claim up in Wikipedia now and ask an entailment model about it.

        Deliberately NOT called a fact-check: it is a model reading an encyclopedia, so it
        gets its own vocabulary (supported / contradicted / inconclusive) and can never
        overwrite a published verdict.
        """
        if not (self.researcher and self.nli):
            return {"status": "disabled"}

        passages = self.researcher.research(claim_text)
        if not passages:
            err = getattr(self.researcher, "last_error", None)
            return {
                "status": "no_evidence", "error": err, "passages": [],
                "note": ("Live research was unavailable, so only local evidence was used."
                         if err else
                         "No Wikipedia passage about this claim was found."),
            }

        judged = [{**p.to_dict(), "nli": self.nli.judge(p.text, claim_text)}
                  for p in passages]

        # An entailment model asked about an irrelevant passage still answers confidently,
        # so RELEVANCE GATES THE REASONING: a passage must be about the claim before its
        # reading counts, and the evidence score multiplies the two together.
        for j in judged:
            j["evidence"] = round(j["nli"]["confidence"] * j["similarity"], 4)

        qualified = [
            j for j in judged
            if j["similarity"] >= settings.live_min_similarity
            and j["nli"]["decisive"]
            and j["nli"]["label"] in ("entailment", "contradiction")
            and j["evidence"] >= settings.live_min_evidence
        ]
        if not qualified:
            near = max((j["similarity"] for j in judged), default=0.0)
            return {"status": "inconclusive", "passages": judged,
                    "note": (f"Wikipedia passages were found (closest {near:.2f} relevance), "
                             f"but none was both relevant enough and clear enough to count "
                             f"as evidence.")}

        # Among passages that qualified, take the strongest evidence; contradiction breaks
        # ties, because a source refuting a claim is the more actionable finding.
        best = max(qualified, key=lambda j: (j["evidence"],
                                             j["nli"]["label"] == "contradiction"))
        supported = best["nli"]["label"] == "entailment"
        return {
            "status": "supported" if supported else "contradicted",
            "confidence": best["nli"]["confidence"],
            "relevance": best["similarity"],
            "evidence": best["evidence"],
            "source_title": best["title"], "source_url": best["url"],
            "passage": best["text"], "passages": judged,
            "note": (f"Wikipedia's \u201c{best['title']}\u201d article "
                     f"{'supports' if supported else 'contradicts'} this claim "
                     f"({best['nli']['confidence']:.0%} model confidence). This is a model "
                     f"reading an encyclopedia, not a fact-checker's verdict."),
        }

    # ---------- public ----------

    def analyze(self, text: str) -> dict[str, Any]:
        extracted = extract_claims(text, max_claims=self.max_claims)
        if not extracted:
            return {"verdict": "no_match", "claims": [], "claims_found": 0,
                    "source": "local_corpus" if self.claims else "none",
                    "note": "No checkable claim was found in this text.",
                    "corpus_size": self.corpus_size}

        texts = [c.text for c in extracted]
        hits = self._search_batch(texts)
        results = []
        for claim, claim_hits in zip(extracted, hits):
            remote = self._google(claim.text)
            judged = self._judge(claim.text, claim_hits, remote)
            results.append({
                **claim.to_dict(), **judged,
                "alternatives": [self._alt(a) for a in claim_hits[1:]],
            })

        # Claims the corpus could not settle get looked up live — most check-worthy first,
        # and only a couple, because each one costs a network round trip.
        if self.researcher and self.nli:
            unsettled = [r for r in results
                         if r["verdict"] in ("no_match", "related")][: self.live_max_claims]
            for r in unsettled:
                r["live"] = self._research(r["text"])

        # The selection's verdict is its WORST asserted claim: one debunked sentence in an
        # otherwise clean paragraph should still pull the score down.
        asserted = [r for r in results if r["verdict"] in ("false", "mixed", "true")]
        if asserted:
            worst = max(asserted, key=lambda r: (VERDICT_SEVERITY[r["verdict"]],
                                                 r.get("similarity") or 0))
            overall, driver = worst["verdict"], worst
        elif any(r["verdict"] == "related" for r in results):
            driver = next(r for r in results if r["verdict"] == "related")
            overall = "related"
        else:
            driver, overall = results[0], "no_match"

        live_findings = [r["live"] for r in results
                         if r.get("live", {}).get("status") in ("supported", "contradicted")]
        note = driver.get("explanation", "")
        if len(results) > 1:
            note = (f"{len(results)} claims checked; "
                    f"{len(asserted)} matched a fact-check. " + note)

        return {
            "verdict": overall,
            "claims": results,
            "claims_found": len(results),
            "claims_matched": len(asserted),
            "live_used": any("live" in r for r in results),
            "live_findings": len(live_findings),
            "live": (max(live_findings, key=lambda f: f.get("confidence", 0))
                     if live_findings else None),
            "matched_claim": driver.get("matched_claim"),
            "rating": driver.get("rating"),
            "publisher": driver.get("publisher"),
            "source_url": driver.get("source_url"),
            "similarity": driver.get("similarity"),
            "warnings": driver.get("warnings", []),
            "alternatives": driver.get("alternatives", []),
            "source": driver.get("source", "local_corpus"),
            "note": note,
            "corpus_size": self.corpus_size,
        }

    @staticmethod
    def _alt(a: dict[str, Any]) -> dict[str, Any]:
        return {"claim": a.get("claim", ""), "verdict": a.get("verdict", "mixed"),
                "publisher": a.get("publisher"), "source_url": a.get("source_url"),
                "similarity": a.get("similarity") or 0.0}
