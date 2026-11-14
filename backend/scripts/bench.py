"""Benchmark: INT8 dynamic quantization vs fp32, and end-to-end API latency.

Produces the numbers quoted in README.md. Two independent measurements:

  1. Per-model inference latency, fp32 vs int8, on identical inputs, plus how often the
     two agree on the predicted label - a speedup that changes predictions is not a win.
  2. End-to-end /analyze latency against a running server, if one is up.

Usage:
    ../.venv/bin/python scripts/bench.py            # models only
    ../.venv/bin/python scripts/bench.py --api      # also hit http://localhost:8000
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import settings              # noqa: E402
from app.engines.ai_detect import AIDetectionEngine  # noqa: E402
from app.engines.toxicity import ToxicityEngine      # noqa: E402

SAMPLES = [
    "The unemployment rate has fallen every single month for the last three years, which is "
    "the longest continuous decline recorded in modern economic history.",
    "You are completely worthless and everyone who agrees with you is an idiot.",
    "Researchers at the university published a peer-reviewed study on soil carbon capture.",
    "This new policy will destroy the country and anyone supporting it should be ashamed.",
    "Water freezes at zero degrees Celsius under standard atmospheric pressure.",
]
REPEATS = 6


def time_engine(engine, samples, repeats):
    for s in samples[:2]:                     # warm-up: first call pays lazy-init costs
        engine.analyze(s)
    times, labels = [], []
    for _ in range(repeats):
        for s in samples:
            t0 = time.perf_counter()
            out = engine.analyze(s)
            times.append((time.perf_counter() - t0) * 1000)
            labels.append(out.get("label"))
    return times, labels


def bench_model(cls, model_id, title):
    print(f"\n{title}\n  model: {model_id}")
    rows = {}
    for quantize in (False, True):
        engine = cls(model_id, device=settings.device, quantize=quantize)
        engine.load()
        times, labels = time_engine(engine, SAMPLES, REPEATS)
        rows["int8" if quantize else "fp32"] = {
            "median": statistics.median(times),
            "mean": statistics.fmean(times),
            "p95": sorted(times)[int(len(times) * 0.95) - 1],
            "labels": labels,
            "applied": engine.quantized,
            "note": engine.quantize_note,
        }
    fp32, int8 = rows["fp32"], rows["int8"]
    agree = sum(a == b for a, b in zip(fp32["labels"], int8["labels"])) / len(fp32["labels"])
    print(f"  fp32   median {fp32['median']:7.1f} ms   mean {fp32['mean']:7.1f}   p95 {fp32['p95']:7.1f}")
    print(f"  int8   median {int8['median']:7.1f} ms   mean {int8['mean']:7.1f}   p95 {int8['p95']:7.1f}"
          f"   ({int8['note']})")
    print(f"  speedup {fp32['median'] / int8['median']:.2f}x   label agreement {agree * 100:.0f}%")
    return {"title": title, "fp32": fp32["median"], "int8": int8["median"], "agreement": agree}


def bench_api(url="http://localhost:8000"):
    import httpx

    text = " ".join(SAMPLES[0].split()[:50])
    print(f"\nEnd-to-end /analyze ({len(text.split())} words)")
    try:
        httpx.get(f"{url}/health", timeout=5).raise_for_status()
    except Exception as exc:  # noqa: BLE001
        print(f"  server not reachable at {url}: {exc}")
        return None
    times, payload = [], {"text": text}
    for i in range(10):
        t0 = time.perf_counter()
        r = httpx.post(f"{url}/analyze", json=payload, timeout=60)
        r.raise_for_status()
        times.append((time.perf_counter() - t0) * 1000)
        if i == 0:
            body = r.json()
            print(f"  server-reported timing: {body['timing_ms']}")
    med = statistics.median(times)
    print(f"  client-observed median {med:.0f} ms over {len(times)} runs "
          f"(p95 {sorted(times)[int(len(times)*0.95)-1]:.0f} ms)")
    print(f"  R11 target <= 2000 ms: {'PASS' if med <= 2000 else 'FAIL'}")
    return med


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", action="store_true", help="also benchmark a running server")
    args = ap.parse_args()

    print("=" * 68)
    print(f"TruthLens benchmark   device={settings.device}   samples={len(SAMPLES)} x {REPEATS}")
    print("=" * 68)
    results = [
        bench_model(ToxicityEngine, settings.toxicity_model, "Toxicity engine"),
        bench_model(AIDetectionEngine, settings.ai_detect_model, "AI-detection engine"),
    ]
    print("\n" + "-" * 68)
    print(f"{'engine':<24}{'fp32 (ms)':>12}{'int8 (ms)':>12}{'speedup':>10}{'agreement':>12}")
    for r in results:
        print(f"{r['title']:<24}{r['fp32']:>12.1f}{r['int8']:>12.1f}"
              f"{r['fp32'] / r['int8']:>9.2f}x{r['agreement'] * 100:>11.0f}%")
    if args.api:
        bench_api()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
