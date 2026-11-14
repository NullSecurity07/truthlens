"""Build the local fact-check corpus + embedding index. No API key, one-time, offline after.

SOURCES (all public, free, no token)
------------------------------------
1. SinclairSchneider/datacommons_org_factcheck  (~90k rows, 1,243 publishers)
   Google Data Commons' aggregation of ClaimReview markup. Every row carries the REAL
   article URL of the fact-check, which is why it is processed first and wins ties.

2. chengxuphd/liar2  (~23k rows, PolitiFact)
   Statement-level PolitiFact data. Its label column is undocumented; the mapping below
   was recovered empirically (see verify_labels.py) rather than assumed.

3. tdiggelm/climate_fever  (1,535 rows)
   Climate claims labelled against Wikipedia evidence — adds topical range the other two
   lack.

TWO FILTERS, BOTH DELIBERATE
----------------------------
* ENGLISH ONLY. The embedding model, the toxicity model and the AI detector are all
  English-trained, so a non-English claim in the index would match badly and score
  unreliably. Roughly 60% of the Data Commons rows are dropped by this. Removing the
  restriction means switching to a multilingual embedding model and re-running
  calibrate_threshold.py; it is recorded as future work, not silently half-done.

* INTERPRETABLE RATINGS ONLY. Publishers use 11,782 distinct rating strings. A verdict we
  cannot read is a verdict we cannot report, so a row whose rating does not normalise is
  dropped rather than guessed at. The script prints how many that was.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import unicodedata
import urllib.parse
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.config import settings  # noqa: E402

# --- LIAR2 labels: empirically recovered, see scripts/verify_labels.py -------------
LIAR2_LABELS = {0: "pants-fire", 1: "false", 2: "mostly-false",
                3: "half-true", 4: "mostly-true", 5: "true"}
LIAR2_VERDICT = {"pants-fire": "false", "false": "false", "mostly-false": "mixed",
                 "half-true": "mixed", "mostly-true": "mixed", "true": "true"}

# --- Climate-FEVER claim labels (documented by the dataset) ------------------------
CFEVER_VERDICT = {0: "true", 1: "false", 2: "mixed", 3: None}   # 3 = not enough info
CFEVER_RATING = {0: "supported", 1: "refuted", 2: "disputed", 3: "not enough info"}

# --- rating normaliser: ordered, first match wins ----------------------------------
# Ordering matters: "mostly false" must be tested before the bare "false" rule, or every
# hedged rating would collapse to a flat falsehood.
RATING_RULES: list[tuple[str, str]] = [
    (r"pants on fire|pants-on-fire", "false"),
    (r"\b(four|4)[- ]pinocchio", "false"),
    (r"\b(one|1|two|2|three|3)[- ]pinocchio", "mixed"),
    (r"mostly[- ]false|partly[- ]false|partially[- ]false|part(ly)?[- ]true|"
     r"mostly[- ]true|largely[- ]true|half[- ]true|barely[- ]true", "mixed"),
    (r"mixture|mixed|misleading|missing context|needs context|lacks context|"
     r"out of context|miscaptioned|exaggerat|unsupported|unproven|unverified|"
     r"inconclusive|disputed|altered|edited|satire|flip[- ]?flop|outdated", "mixed"),
    (r"\b(false|fake|hoax|fabricat|debunk|incorrect|untrue|not true|baseless|"
     r"no evidence|misinformation|scam|impostor|spoof)\b", "false"),
    (r"\b(true|correct|accurate|legitimate|verified|confirmed|genuine)\b", "true"),
]
_STRIP_PREFIX = re.compile(
    r"^\s*(fact crescendo rating:?|rating:?|verdict:?|our rating:?|claim:?)\s*", re.I)

# --- a cheap but effective English check --------------------------------------------
_ENGLISH_MARKERS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "in", "that", "for",
    "on", "with", "has", "have", "had", "not", "this", "it", "as", "by", "from",
    "said", "says", "he", "she", "they", "we", "you", "and", "but", "at", "his", "her",
}
_WORD = re.compile(r"[a-z']+")


def normalise_rating(raw: object) -> tuple[str, str] | None:
    """(verdict, cleaned original rating) or None when the rating is not interpretable."""
    if raw is None:
        return None
    text = unicodedata.normalize("NFKC", str(raw)).strip()
    if not text:
        return None
    lowered = _STRIP_PREFIX.sub("", text).lower()
    for pattern, verdict in RATING_RULES:
        if re.search(pattern, lowered):
            return verdict, text[:60]
    return None


def is_english(text: str, min_markers: int = 3) -> bool:
    if not text or len(text) < 25:
        return False
    words = set(_WORD.findall(text.lower()))
    return len(words & _ENGLISH_MARKERS) >= min_markers


def publisher_from(url: str | None) -> str | None:
    if not url:
        return None
    try:
        host = urllib.parse.urlparse(url).netloc.lower()
    except ValueError:
        return None
    return host[4:] if host.startswith("www.") else host or None


# ------------------------------------------------------------------ sources

def from_datacommons(stats: Counter) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("SinclairSchneider/datacommons_org_factcheck",
                           "data/train-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(path).to_pydict()
    rows: list[dict] = []
    for claim, rating, url, author, date in zip(
            table["claimText"], table["reviewRating"], table["reviewURL"],
            table["claimAuthorName"], table["reviewDatePublished"]):
        claim = (claim or "").strip()
        stats["datacommons_seen"] += 1
        if not is_english(claim):
            stats["dropped_not_english"] += 1
            continue
        norm = normalise_rating(rating)
        if norm is None:
            stats["dropped_unreadable_rating"] += 1
            continue
        verdict, original = norm
        rows.append({
            "claim": claim, "rating": original, "verdict": verdict,
            "speaker": (author or "").strip() or None,
            "date": str(date)[:10] if date else None,
            "publisher": publisher_from(url) or "unknown publisher",
            "source_url": url or None,
            "source_dataset": "datacommons",
        })
    return rows


def from_liar2(stats: Counter) -> list[dict]:
    from huggingface_hub import hf_hub_download

    rows: list[dict] = []
    for split in ("train.csv", "valid.csv", "test.csv"):
        try:
            path = hf_hub_download("chengxuphd/liar2", split, repo_type="dataset")
        except Exception as exc:  # noqa: BLE001
            print(f"    ! {split}: {type(exc).__name__}")
            continue
        with open(path, encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                claim = (row.get("statement") or "").strip()
                stats["liar2_seen"] += 1
                if not claim:
                    continue
                try:
                    rating = LIAR2_LABELS[int(row["label"])]
                except (KeyError, TypeError, ValueError):
                    continue
                rows.append({
                    "claim": claim, "rating": rating, "verdict": LIAR2_VERDICT[rating],
                    "speaker": (row.get("speaker") or "").strip() or None,
                    "date": (row.get("date") or "").strip() or None,
                    "publisher": "politifact.com",
                    # LIAR2 ships no article URL, so this is a SEARCH link and is
                    # labelled as such in the record rather than passed off as the source.
                    "source_url": "https://www.politifact.com/search/?q="
                                  + urllib.parse.quote_plus(claim[:120]),
                    "source_is_search": True,
                    "source_dataset": "liar2",
                })
    return rows


def from_climate_fever(stats: Counter) -> list[dict]:
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    path = hf_hub_download("tdiggelm/climate_fever",
                           "data/test-00000-of-00001.parquet", repo_type="dataset")
    table = pq.read_table(path).to_pydict()
    rows: list[dict] = []
    for claim, label in zip(table["claim"], table["claim_label"]):
        stats["climate_fever_seen"] += 1
        verdict = CFEVER_VERDICT.get(int(label))
        if verdict is None:            # "not enough info" is not a verdict
            stats["dropped_no_verdict"] += 1
            continue
        rows.append({
            "claim": (claim or "").strip(), "rating": CFEVER_RATING[int(label)],
            "verdict": verdict, "speaker": None, "date": None,
            "publisher": "Climate-FEVER (Wikipedia-evidenced)",
            "source_url": "https://huggingface.co/datasets/tdiggelm/climate_fever",
            "source_dataset": "climate_fever",
        })
    return rows


# ------------------------------------------------------------------ main

def main() -> int:
    stats: Counter = Counter()
    print("Building the local fact-check corpus\n")

    collected: list[dict] = []
    for name, loader in (("Data Commons ClaimReview", from_datacommons),
                         ("LIAR2 (PolitiFact)", from_liar2),
                         ("Climate-FEVER", from_climate_fever)):
        try:
            rows = loader(stats)
        except Exception as exc:  # noqa: BLE001 - one bad source must not stop the build
            print(f"  ! {name}: {type(exc).__name__}: {str(exc)[:90]}")
            continue
        print(f"  {name:28s} +{len(rows):6d} usable")
        collected.extend(rows)

    if not collected:
        print("\nFAILED: no rows fetched. The fact-check engine runs in no-corpus mode.")
        return 1

    # De-duplicate on the claim text. Sources are processed best-provenance-first, so the
    # record kept is the one with a real article URL rather than a search link.
    seen: set[str] = set()
    unique: list[dict] = []
    for row in collected:
        key = re.sub(r"\W+", " ", row["claim"].lower()).strip()
        if not key or key in seen:
            stats["dropped_duplicate"] += 1
            continue
        seen.add(key)
        unique.append(row)

    print(f"\n  {len(collected)} rows -> {len(unique)} unique claims")
    print("  dropped: "
          f"{stats['dropped_not_english']} non-English, "
          f"{stats['dropped_unreadable_rating']} unreadable rating, "
          f"{stats['dropped_no_verdict']} no verdict, "
          f"{stats['dropped_duplicate']} duplicates")

    with settings.corpus_path.open("w", encoding="utf-8") as fh:
        for row in unique:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {settings.corpus_path}")

    print(f"\n  embedding {len(unique)} claims with {settings.embedding_model} ...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model, device=settings.device)
    vecs = model.encode([r["claim"] for r in unique], batch_size=128,
                        show_progress_bar=True, normalize_embeddings=True,
                        convert_to_numpy=True).astype(np.float32)
    np.save(settings.embeddings_path, vecs)
    print(f"  wrote {settings.embeddings_path}  shape={vecs.shape}"
          f"  ({vecs.nbytes / 1e6:.0f} MB)")

    verdicts = Counter(r["verdict"] for r in unique)
    sources = Counter(r["source_dataset"] for r in unique)
    publishers = len({r["publisher"] for r in unique})
    print(f"\nCORPUS: {len(unique)} claims from {publishers} publishers")
    print(f"  verdicts: {dict(verdicts)}")
    print(f"  sources:  {dict(sources)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
