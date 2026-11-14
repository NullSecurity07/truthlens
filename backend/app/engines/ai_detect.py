"""AI-generated-text vector - proposal section 4's 'AI Detection Engine'.

Honesty note carried into the response: detectors of this family are unreliable on
short or human-edited text. We return the raw probability and an explicit `note`
instead of asserting authorship, and the aggregator weights this vector lowest.
"""
from __future__ import annotations

from typing import Any

from .base import HFClassifier

SHORT_TEXT_WORDS = 25
NOTE = (
    "AI-text detection is probabilistic and degrades on short or edited passages. "
    "Treat this as a signal, not proof of authorship."
)


class AIDetectionEngine(HFClassifier):
    name = "ai_detection"
    # 'fake' is what the OpenAI GPT-2 output detector calls machine-generated text
    # deliberately no bare "ai": it is a substring of ordinary words like "plain".
    # "real"/"human" are handled by HFClassifier.NEGATIONS, not listed here.
    positive_aliases = ("fake", "machine", "generated", "synthetic", "chatgpt", "gpt")
    positive_index_default = 0

    def __init__(self, *args: Any, flag_threshold: float = 0.7, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.flag_threshold = flag_threshold

    def analyze(self, text: str) -> dict[str, Any]:
        prob, probs = self.positive_probability(text)
        words = len(text.split())
        if prob >= self.flag_threshold:
            label = "likely_ai"
        elif prob <= 1 - self.flag_threshold:
            label = "likely_human"
        else:
            label = "uncertain"
        confidence = abs(prob - 0.5) * 2
        note = NOTE
        if words < SHORT_TEXT_WORDS:
            confidence *= 0.5  # short text: halve how much this vector is trusted
            note = f"Only {words} words - too short for a reliable estimate. " + NOTE
        return {
            "probability": round(prob, 4),
            "label": label,
            "confidence": round(confidence, 4),
            "distribution": {k: round(v, 4) for k, v in probs.items()},
            "model": self.model_id,
            "note": note,
        }
