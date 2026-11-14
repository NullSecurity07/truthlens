"""Hate-speech / toxicity vector - proposal section 4's 'Hate Speech Model'."""
from __future__ import annotations

from typing import Any

from .base import HFClassifier


class ToxicityEngine(HFClassifier):
    name = "toxicity"
    # checkpoints in this slot variously call the risky class offensive/toxic/hate/abusive
    positive_aliases = ("offensive", "toxic", "hate", "abusive", "insult")
    positive_index_default = 1

    def __init__(self, *args: Any, flag_threshold: float = 0.5, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.flag_threshold = flag_threshold

    def analyze(self, text: str) -> dict[str, Any]:
        score, probs = self.positive_probability(text)
        return {
            "score": round(score, 4),
            "flagged": score >= self.flag_threshold,
            "label": "offensive" if score >= self.flag_threshold else "non-offensive",
            # confidence = how far from the 50/50 coin flip, rescaled to 0..1
            "confidence": round(abs(score - 0.5) * 2, 4),
            "distribution": {k: round(v, 4) for k, v in probs.items()},
            "model": self.model_id,
        }
