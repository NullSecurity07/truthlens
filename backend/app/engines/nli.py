"""Natural Language Inference: does this passage support or contradict this claim?

This is the piece that turns retrieval into reasoning. Dense retrieval can only say "these
two sentences are about the same thing" — which is exactly why a negated claim matched its
own debunking at 0.96 similarity. An entailment model reads the pair directionally:

    premise    = a sentence from Wikipedia
    hypothesis = the user's claim
    output     = entailment | contradiction | neutral

So "the measles vaccine does not cause autism" (premise) against "the measles vaccine
causes autism" (hypothesis) is a CONTRADICTION, which similarity alone can never see.

Label order differs between checkpoints — MoritzLaurer's models are
{entailment, neutral, contradiction} while the cross-encoder ones are
{contradiction, entailment, neutral} — so labels are resolved by NAME, never by position,
the same way the toxicity and detector engines do it.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from .base import Engine, quantize_dynamic

log = logging.getLogger("truthlens.engines.nli")

# The claim must clear this before we report support or contradiction; anything softer is
# reported as inconclusive rather than dressed up as a finding.
DEFAULT_DECISION_THRESHOLD = 0.60


class NLIEngine(Engine):
    name = "nli"

    def __init__(self, model_id: str, device: str = "cpu", quantize: bool = True,
                 max_tokens: int = 256,
                 decision_threshold: float = DEFAULT_DECISION_THRESHOLD) -> None:
        self.model_id = model_id
        self.device = device
        self.quantize = quantize
        self.max_tokens = max_tokens
        self.decision_threshold = decision_threshold
        self.tokenizer = None
        self.model = None
        self.index: dict[str, int] = {}      # entailment/contradiction/neutral -> column
        self.quantized = False
        self.quantize_note = ""
        self.load_ms = 0

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        model.eval()
        if self.quantize and self.device == "cpu":
            # Some checkpoints ship fp16 weights; dynamic quantization needs fp32 input or
            # it fails at the first matmul with "expected Float but found Half".
            model = model.float()
            model, self.quantized, self.quantize_note = quantize_dynamic(model)
        else:
            self.quantize_note = "quantization disabled"
        if self.device != "cpu":
            model = model.to(self.device)
        self.model = model

        raw = getattr(model.config, "id2label", None) or {}
        id2label = {int(k): str(v).lower() for k, v in raw.items()}
        for wanted in ("entailment", "contradiction", "neutral"):
            for idx, name in id2label.items():
                if wanted in name:
                    self.index[wanted] = idx
                    break
        missing = {"entailment", "contradiction", "neutral"} - set(self.index)
        if missing:
            raise RuntimeError(
                f"{self.model_id} does not expose {sorted(missing)} in id2label ({id2label}); "
                f"refusing to guess which column means what")
        self.load_ms = round((time.perf_counter() - t0) * 1000)
        log.info("nli loaded %s in %dms (%s) labels=%s",
                 self.model_id, self.load_ms, self.quantize_note, self.index)

    def judge(self, premise: str, hypothesis: str) -> dict[str, Any]:
        """One (premise, hypothesis) pair -> label plus the full distribution."""
        import torch

        enc = self.tokenizer(premise, hypothesis, return_tensors="pt",
                             truncation=True, max_length=self.max_tokens)
        if self.device != "cpu":
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits.float(), dim=-1).tolist()

        scores = {name: float(probs[i]) for name, i in self.index.items()}
        best = max(scores, key=scores.get)
        confident = scores[best] >= self.decision_threshold
        return {
            "label": best if confident else "inconclusive",
            "raw_label": best,
            "confidence": round(scores[best], 4),
            "scores": {k: round(v, 4) for k, v in scores.items()},
            "decisive": confident,
        }

    def analyze(self, text: str) -> dict[str, Any]:      # pragma: no cover
        raise NotImplementedError("NLIEngine is used through judge(), not run()")

    def describe(self) -> dict[str, Any]:
        return {"model": self.model_id, "quantized": self.quantized,
                "quantize_note": self.quantize_note, "labels": self.index,
                "decision_threshold": self.decision_threshold, "load_ms": self.load_ms}
