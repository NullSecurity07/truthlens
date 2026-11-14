"""Pins the label-inversion bug: 'offensive' is a substring of 'non-offensive'.

Getting this wrong inverts the entire toxicity vector while every test still 'passes',
so it is pinned explicitly rather than left to integration testing.
"""
from __future__ import annotations

import pytest

from app.engines.ai_detect import AIDetectionEngine
from app.engines.toxicity import ToxicityEngine


def _resolve(cls, id2label):
    engine = cls("dummy-model")
    engine.id2label = id2label
    engine._resolve_positive_index()
    return engine


@pytest.mark.parametrize("id2label,expected_name", [
    ({0: "non-offensive", 1: "offensive"}, "offensive"),
    ({0: "offensive", 1: "non-offensive"}, "offensive"),      # reversed ordering
    ({0: "NON_TOXIC", 1: "TOXIC"}, "TOXIC"),
    ({0: "clean", 1: "hate"}, "hate"),
])
def test_toxicity_never_selects_the_negated_label(id2label, expected_name):
    engine = _resolve(ToxicityEngine, id2label)
    assert engine.id2label[engine.positive_index] == expected_name


@pytest.mark.parametrize("id2label,expected_name", [
    ({0: "Real", 1: "Fake"}, "Fake"),
    ({0: "Fake", 1: "Real"}, "Fake"),                          # reversed ordering
    ({0: "human", 1: "machine"}, "machine"),
    ({0: "human-written", 1: "chatgpt"}, "chatgpt"),
])
def test_ai_detection_selects_the_machine_label(id2label, expected_name):
    engine = _resolve(AIDetectionEngine, id2label)
    assert engine.id2label[engine.positive_index] == expected_name


def test_generic_labels_fall_back_to_position_and_say_so():
    engine = _resolve(ToxicityEngine, {0: "LABEL_0", 1: "LABEL_1"})
    assert engine.positive_index == ToxicityEngine.positive_index_default
    assert "positional default" in engine.mapping_source


def test_complement_inference_when_only_the_negative_label_is_named():
    engine = _resolve(ToxicityEngine, {0: "non-offensive", 1: "LABEL_1"})
    assert engine.positive_index == 1
    assert "complement" in engine.mapping_source
