"""Phase 1 smoke test: prove all three checkpoints load and infer locally, no token.

Run:  ../.venv/bin/python scripts/smoke_models.py
Exits non-zero if any candidate (primary + fallbacks) fails for a slot.
"""
import os, sys, time, json

os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

CANDIDATES = {
    "toxicity": [
        "cardiffnlp/twitter-roberta-base-offensive",
        "unitary/toxic-bert",
        "martin-ha/toxic-comment-model",
    ],
    "ai_detection": [
        "openai-community/roberta-base-openai-detector",
        "Hello-SimpleAI/chatgpt-detector-roberta",
        "roberta-base-openai-detector",
    ],
    "embedding": [
        "sentence-transformers/all-MiniLM-L6-v2",
        "sentence-transformers/paraphrase-MiniLM-L3-v2",
    ],
    "nli": [
        "cross-encoder/nli-deberta-v3-small",
        "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli",
        "typeform/distilbert-base-uncased-mnli",
    ],
}

SAMPLE = "The government secretly replaced all the water supply with mind control chemicals."
NLI_PAIR = ("The measles vaccine does not cause autism.", "The measles vaccine causes autism.")


def try_classifier(model_id):
    from transformers import pipeline
    t0 = time.perf_counter()
    pipe = pipeline("text-classification", model=model_id, device=-1, top_k=None)
    load_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    out = pipe(SAMPLE)
    infer_ms = (time.perf_counter() - t1) * 1000
    return {"load_ms": round(load_ms), "infer_ms": round(infer_ms), "output": out}


def try_nli(model_id):
    import sys as _sys, os as _os
    _sys.path.insert(0, _os.path.join(_os.path.dirname(__file__), ".."))
    from app.engines.nli import NLIEngine
    t0 = time.perf_counter()
    engine = NLIEngine(model_id)
    engine.load()
    load_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    out = engine.judge(*NLI_PAIR)
    infer_ms = (time.perf_counter() - t1) * 1000
    return {"load_ms": round(load_ms), "infer_ms": round(infer_ms), "output": out}


def try_embedder(model_id):
    from sentence_transformers import SentenceTransformer
    t0 = time.perf_counter()
    m = SentenceTransformer(model_id, device="cpu")
    load_ms = (time.perf_counter() - t0) * 1000
    t1 = time.perf_counter()
    v = m.encode([SAMPLE, "Water supply contains chemicals."])
    infer_ms = (time.perf_counter() - t1) * 1000
    return {"load_ms": round(load_ms), "infer_ms": round(infer_ms),
            "output": f"shape={tuple(v.shape)}"}


def main():
    resolved, failed = {}, []
    for slot, ids in CANDIDATES.items():
        runner = {"embedding": try_embedder, "nli": try_nli}.get(slot, try_classifier)
        for model_id in ids:
            print(f"\n[{slot}] trying {model_id} ...", flush=True)
            try:
                res = runner(model_id)
            except Exception as exc:  # noqa: BLE001 - we want the reason printed
                print(f"  FAILED: {type(exc).__name__}: {str(exc)[:200]}")
                continue
            print(f"  OK  load={res['load_ms']}ms  infer={res['infer_ms']}ms")
            print(f"  out: {json.dumps(res['output'])[:300]}")
            resolved[slot] = {"model_id": model_id, **{k: v for k, v in res.items() if k != "output"}}
            break
        else:
            failed.append(slot)

    print("\n" + "=" * 60)
    print("RESOLVED:", json.dumps(resolved, indent=2))
    if failed:
        print("UNRESOLVED SLOTS:", failed)
        sys.exit(1)
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "resolved_models.json"), "w") as fh:
        json.dump(resolved, fh, indent=2)
    print("wrote data/resolved_models.json")


if __name__ == "__main__":
    main()
