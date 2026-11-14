"""Calibrate the fact-check similarity threshold from data instead of guessing it.

THE PROBLEM
-----------
`settings.similarity_threshold` decides when a retrieved claim counts as "the same claim"
and therefore whether we report a verdict at all. Set it too low and unrelated sentences
inherit a stranger's verdict - the worst possible failure for a fact-checking tool. Set it
too high and nothing ever matches. The initial 0.55 was a guess; this script replaces it
with a measured value.

THE EXPERIMENT
--------------
POSITIVES  Take real corpus claims and perturb them the way a user would naturally
           paraphrase while quoting: drop ~20% of words, lowercase, strip punctuation.
           A correct system should still retrieve the original claim. We record the
           similarity of the top hit AND whether that hit is the right one.

NEGATIVES  Everyday sentences about cooking, weather, pets and code, which certainly have
           never been fact-checked by PolitiFact. A correct system must return no_match.
           We record the top-1 similarity - this is the score the threshold must exclude.

Then sweep candidate thresholds and report precision / recall / F1, plus the false-match
rate, which matters more than F1 here: a wrong verdict is worse than a missing one.

Run:  ../.venv/bin/python scripts/calibrate_threshold.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

SEED = 20260818
N_POSITIVE = 400
NEG_TEMPLATES = [
    "I reheated the leftover pasta and it tasted better the second day.",
    "The rain finally stopped just before we left for the market.",
    "My cat knocked the plant off the windowsill again this morning.",
    "He refactored the parser so the tests would stop timing out.",
    "We repainted the garden fence a pale blue colour last Tuesday.",
    "The bus was late so I walked the rest of the way home.",
    "She borrowed my charger and forgot to give it back.",
    "The bread needs another twenty minutes in the oven.",
    "I finally finished the novel I started over the summer.",
    "The neighbours were playing music until well past midnight.",
    "My laptop battery drains much faster since the last update.",
    "We planted tomatoes along the back wall where the sun hits.",
    "The dog refuses to walk past the house with the loud gate.",
    "I rewrote the query using a join and it runs much quicker now.",
    "There was a queue outside the bakery when I cycled past.",
    "He always forgets to close the window before it rains.",
    "The coffee machine started making a strange grinding noise.",
    "I moved the bookshelf to the other side of the bedroom.",
    "We watched the sunset from the hill behind the college.",
    "The printer jammed twice while I was making copies.",
]


def perturb(text: str, rng: random.Random) -> str:
    """Paraphrase-ish corruption: drop ~20% of words, lowercase, strip punctuation."""
    words = [w.strip(".,!?;:\"'") for w in text.split()]
    keep = [w for w in words if rng.random() > 0.20]
    if len(keep) < 3:
        keep = words[:5]
    return " ".join(keep).lower()


def main() -> int:
    if not (settings.corpus_path.exists() and settings.embeddings_path.exists()):
        print("Corpus missing. Run scripts/build_corpus.py first.")
        return 1

    rng = random.Random(SEED)
    with settings.corpus_path.open(encoding="utf-8") as fh:
        claims = [json.loads(line) for line in fh if line.strip()]
    matrix = np.load(settings.embeddings_path)
    print(f"corpus: {len(claims)} claims, dim {matrix.shape[1]}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(settings.embedding_model, device=settings.device)

    # ---------- positives ----------
    idxs = rng.sample(range(len(claims)), N_POSITIVE)
    queries = [perturb(claims[i]["claim"], rng) for i in idxs]
    qv = model.encode(queries, normalize_embeddings=True, convert_to_numpy=True,
                      batch_size=128).astype(np.float32)
    sims = qv @ matrix.T                       # (N_POSITIVE, corpus)
    top = sims.argmax(axis=1)
    pos_scores = sims.max(axis=1)
    correct = np.array([top[k] == idxs[k] for k in range(N_POSITIVE)])
    print(f"\npositives  n={N_POSITIVE}")
    print(f"  top-1 is the original claim: {correct.mean() * 100:.1f}%")
    print(f"  similarity  mean {pos_scores.mean():.3f}  median {np.median(pos_scores):.3f}"
          f"  p10 {np.percentile(pos_scores, 10):.3f}")

    # ---------- negatives ----------
    nv = model.encode(NEG_TEMPLATES, normalize_embeddings=True, convert_to_numpy=True
                      ).astype(np.float32)
    neg_scores = (nv @ matrix.T).max(axis=1)
    print(f"\nnegatives  n={len(NEG_TEMPLATES)}  (everyday sentences, never fact-checked)")
    print(f"  similarity  mean {neg_scores.mean():.3f}  max {neg_scores.max():.3f}"
          f"  p90 {np.percentile(neg_scores, 90):.3f}")

    # ---------- HARD negatives ----------
    # The dangerous failure is not an everyday sentence matching; it is a RELATED but
    # DIFFERENT claim inheriting a stranger's verdict. For each positive query, take the
    # best hit that is NOT the correct claim: these are the hardest possible negatives,
    # drawn from the same topics and vocabulary.
    hard = np.array([np.partition(sims[k], -2)[-2] if top[k] == idxs[k] else sims[k].max()
                     for k in range(N_POSITIVE)])
    print(f"\nHARD negatives  n={N_POSITIVE}  (best NON-matching corpus claim per query)")
    print(f"  similarity  mean {hard.mean():.3f}  p90 {np.percentile(hard, 90):.3f}"
          f"  p99 {np.percentile(hard, 99):.3f}  max {hard.max():.3f}")

    # ---------- MUTATED claims: the failure that actually matters ----------
    # Sentence embeddings are famously weak at negation and at numeric difference.
    # "Obama grew up in Kenya" and "Obama did NOT grow up in Kenya" are near-identical
    # vectors but opposite claims. If a user quotes the corrected version, a naive system
    # hands them the debunked version's verdict. Measure how often that happens.
    def mutate(text, rng):
        words = text.split()
        nums = [i for i, w in enumerate(words) if any(ch.isdigit() for ch in w)]
        if nums and rng.random() < 0.5:                       # change a number
            i = rng.choice(nums)
            digits = "".join(ch for ch in words[i] if ch.isdigit())
            new = str(int(digits) * 3 + 7) if digits else "42"
            words[i] = words[i].replace(digits, new)
            return " ".join(words), "number changed"
        for i, w in enumerate(words):                          # insert a negation
            if w.lower() in ("is", "was", "are", "were", "has", "have", "did", "does"):
                words.insert(i + 1, "not")
                return " ".join(words), "negated"
        return "It is false that " + text[0].lower() + text[1:], "prefixed with a denial"

    mut_texts, mut_kinds = [], []
    for k in idxs[:200]:
        t_, kind = mutate(claims[k]["claim"], rng)
        mut_texts.append(t_)
        mut_kinds.append(kind)
    mv = model.encode(mut_texts, normalize_embeddings=True, convert_to_numpy=True,
                      batch_size=128).astype(np.float32)
    mut_sims = mv @ matrix.T
    mut_top = mut_sims.argmax(axis=1)
    mut_scores = mut_sims.max(axis=1)
    still_original = np.array([mut_top[k] == idxs[k] for k in range(len(mut_texts))])
    print(f"\nMUTATED claims  n={len(mut_texts)}  (materially DIFFERENT assertions)")
    print(f"  still retrieve the ORIGINAL as top-1: {still_original.mean() * 100:.1f}%")
    print(f"  similarity  mean {mut_scores.mean():.3f}  median {np.median(mut_scores):.3f}")
    print("  -> embeddings cannot see negation or numeric change; a threshold alone")
    print("     CANNOT fix this. Hence the contradiction guard in engines/factcheck.py.")

    # ---------- sweep ----------
    print(f"\n{'thresh':>7}{'recall':>9}{'precision':>11}{'F1':>8}{'easy FP':>9}{'hard FP':>9}")
    best = None
    for t in [round(x * 0.05, 2) for x in range(6, 19)]:
        tp = int(((pos_scores >= t) & correct).sum())
        fn = N_POSITIVE - tp
        fp = int((neg_scores >= t).sum())
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 1.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        hard_fp = int((hard >= t).sum())
        mark = ""
        # A wrong verdict is far worse than a missing one, so the chosen threshold must
        # exclude the HARD negatives too, not just the easy ones.
        if fp == 0 and hard_fp / N_POSITIVE <= 0.01 and best is None:
            best = t
            mark = "  <- first threshold with <=1% hard false matches"
        print(f"{t:>7.2f}{recall:>9.3f}{precision:>11.3f}{f1:>8.3f}{fp:>9d}"
              f"{hard_fp:>9d}{mark}")

    print(f"\nRECOMMENDED similarity_threshold = {best}")
    print(f"current setting = {settings.similarity_threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
