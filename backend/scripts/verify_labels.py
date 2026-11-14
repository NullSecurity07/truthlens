"""Reproduce the empirical LIAR2 label-mapping derivation used by build_corpus.py.

Identity: when a speaker's entire credit history contains exactly one verdict, the single
non-zero credit column must be that row's own label. Prints agreement per label.
"""
from __future__ import annotations

import collections
import csv

from huggingface_hub import hf_hub_download

COLS = ["true_counts", "mostly_true_counts", "half_true_counts",
        "mostly_false_counts", "false_counts", "pants_on_fire_counts"]
EXPECTED = {"0": "pants_on_fire_counts", "1": "false_counts", "2": "mostly_false_counts",
            "3": "half_true_counts", "4": "mostly_true_counts", "5": "true_counts"}


def main() -> int:
    path = hf_hub_download("chengxuphd/liar2", "train.csv", repo_type="dataset")
    matches: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    with open(path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                vals = [int(float(row[c] or 0)) for c in COLS]
            except (KeyError, TypeError, ValueError):
                continue
            if sum(vals) == 1:
                matches[row["label"]][COLS[vals.index(1)]] += 1

    ok = True
    for label in sorted(matches):
        total = sum(matches[label].values())
        best, count = matches[label].most_common(1)[0]
        agree = count / total * 100
        flag = "OK " if best == EXPECTED[label] and agree > 95 else "MISMATCH"
        ok &= flag == "OK "
        print(f"{flag} label {label} -> {best:22s} {count}/{total} = {agree:.0f}%")
    print("\nmapping verified" if ok else "\nMAPPING CHANGED - update build_corpus.LABEL_NAMES")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
