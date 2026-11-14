"""Engine interface + a shared local-transformer classifier.

Design notes (these are the "we own the inference" parts, not API glue):
  * We load tokenizer + model directly and run the forward pass ourselves, so the
    softmax, the truncation policy and the label mapping are all visible and testable.
  * INT8 dynamic quantization is applied to every nn.Linear on CPU. This is a real,
    measurable optimisation - scripts/bench.py reports the speedup.
  * Label mapping is resolved from the checkpoint's own id2label by NAME, never by
    position, because two checkpoints in the same slot disagree on ordering. If the
    checkpoint only exposes generic LABEL_0/LABEL_1 names we fall back to position and
    say so in `mapping_source`, rather than silently guessing.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("truthlens.engines")


@dataclass
class EngineResult:
    status: str                       # "ok" | "unavailable"
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    elapsed_ms: int = 0


class Engine:
    """Every engine loads once at startup and is called per request."""

    name: str = "engine"

    def load(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    def analyze(self, text: str) -> dict[str, Any]:  # pragma: no cover - overridden
        raise NotImplementedError

    def run(self, text: str) -> EngineResult:
        """Never raises. A broken engine degrades one vector, not the whole request."""
        t0 = time.perf_counter()
        try:
            data = self.analyze(text)
            return EngineResult("ok", data, None, round((time.perf_counter() - t0) * 1000))
        except Exception as exc:  # noqa: BLE001 - deliberate: surface, don't crash
            log.exception("%s failed", self.name)
            return EngineResult(
                "unavailable",
                {},
                f"{type(exc).__name__}: {exc}",
                round((time.perf_counter() - t0) * 1000),
            )


def quantize_dynamic(model):
    """INT8-quantize every Linear layer. Returns (model, applied: bool, note: str)."""
    import torch

    for path in ("ao.quantization", "quantization"):
        try:
            mod = torch
            for part in path.split("."):
                mod = getattr(mod, part)
            qmodel = mod.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
            return qmodel, True, f"int8 dynamic via torch.{path}"
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
    return model, False, f"quantization unavailable ({last}); running fp32"


class HFClassifier(Engine):
    """Local sequence-classification model with explicit softmax and label resolution."""

    #: subclasses list the label substrings that mean "the risky class", in priority order
    positive_aliases: tuple[str, ...] = ()
    #: fallback index for the risky class when the checkpoint has no meaningful names
    positive_index_default: int = 1

    def __init__(self, model_id: str, device: str = "cpu", quantize: bool = True,
                 max_tokens: int = 512) -> None:
        self.model_id = model_id
        self.device = device
        self.quantize = quantize
        self.max_tokens = max_tokens
        self.tokenizer = None
        self.model = None
        self.id2label: dict[int, str] = {}
        self.positive_index: int = self.positive_index_default
        self.mapping_source: str = "unset"
        self.quantized: bool = False
        self.quantize_note: str = ""
        self.load_ms: int = 0

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
        model.eval()

        if self.quantize and self.device == "cpu":
            # Some checkpoints ship fp16 weights; dynamic quantization needs fp32 or it
            # fails at the first matmul with "expected Float but found Half".
            model = model.float()
            model, self.quantized, self.quantize_note = quantize_dynamic(model)
        else:
            self.quantize_note = "quantization disabled"

        if self.device != "cpu":
            model = model.to(self.device)

        self.model = model
        raw = getattr(model.config, "id2label", None) or {}
        self.id2label = {int(k): str(v) for k, v in raw.items()}
        self._resolve_positive_index()
        self.load_ms = round((time.perf_counter() - t0) * 1000)
        log.info(
            "%s loaded %s in %dms (%s) labels=%s positive=%s via %s",
            self.name, self.model_id, self.load_ms, self.quantize_note,
            self.id2label, self.positive_index, self.mapping_source,
        )

    #: label names starting with any of these are the NEGATED class and must never match
    #: an alias - 'offensive' is a substring of 'non-offensive', which would silently
    #: invert the whole vector. This bit us once; the test suite now pins it.
    NEGATIONS = ("non", "not", "no-", "no_", "neutral", "clean", "real", "human")

    @staticmethod
    def _is_negated(name: str) -> bool:
        return name.startswith(HFClassifier.NEGATIONS)

    def _resolve_positive_index(self) -> None:
        lowered = {idx: name.lower().strip() for idx, name in self.id2label.items()}
        for alias in self.positive_aliases:
            for idx, name in lowered.items():
                if self._is_negated(name):
                    continue
                if alias in name:
                    self.positive_index, self.mapping_source = idx, f"name match '{alias}'"
                    return
        # Nothing matched positively. If exactly one label is negated, the other is the
        # positive class - a safer inference than falling back to a fixed position.
        negated = [i for i, n in lowered.items() if self._is_negated(n)]
        if len(lowered) == 2 and len(negated) == 1:
            self.positive_index = next(i for i in lowered if i not in negated)
            self.mapping_source = f"complement of negated label '{lowered[negated[0]]}'"
            return
        generic = all(n.startswith("label_") for n in lowered.values()) if lowered else True
        self.positive_index = self.positive_index_default
        self.mapping_source = (
            "positional default (checkpoint exposes only generic labels)"
            if generic else "positional default (no alias matched)"
        )

    def probabilities(self, text: str) -> dict[str, float]:
        import torch

        enc = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=self.max_tokens
        )
        if self.device != "cpu":
            enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = self.model(**enc).logits[0]
        probs = torch.softmax(logits.float(), dim=-1).tolist()
        return {self.id2label.get(i, f"LABEL_{i}"): float(p) for i, p in enumerate(probs)}

    def positive_probability(self, text: str) -> tuple[float, dict[str, float]]:
        probs = self.probabilities(text)
        name = self.id2label.get(self.positive_index, f"LABEL_{self.positive_index}")
        return probs.get(name, 0.0), probs

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "quantized": self.quantized,
            "quantize_note": self.quantize_note,
            "labels": self.id2label,
            "positive_label": self.id2label.get(self.positive_index),
            "label_mapping_source": self.mapping_source,
            "load_ms": self.load_ms,
        }
