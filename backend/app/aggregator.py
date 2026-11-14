"""Explainable evidence fusion.

The Trust Score is NOT a bag of arbitrary weights. Each engine emits a *risk* in [0,1]
and a *confidence* in [0,1]; confidence scales that engine's influence, so an uncertain
engine quietly stops voting instead of dragging the score toward the middle. Engines that
abstain (fact-check with no match) or crash contribute nothing at all rather than a
neutral 0.5, which would silently bias every score toward "caution".

    effective_weight_i = base_weight_i * confidence_i
    risk               = sum(effective_i * risk_i) / sum(effective_i)
    trust_score        = round((1 - risk) * 100)
    overall_confidence = sum(effective_i) / sum(base_weight_i over engines that reported)

The BAND is not a pure function of the score. Only the fact-check vector measures truth,
so when it abstains the top band is withheld and the report reads `insufficient_evidence` -
the score is still shown, because it is an honest summary of the evidence that did exist.

Base weights (config.py) encode a deliberate editorial stance, stated so a reader can
disagree with it explicitly:
    fact-check 0.50 - a documented verdict from a professional fact-checker is the
                      strongest evidence available about whether a claim is true.
    toxicity   0.30 - toxicity is measured reliably but says nothing about truth.
    ai-text    0.20 - weakest: detectors are unreliable, and machine-written text is
                      not automatically false.
"""
from __future__ import annotations

from typing import Any

from .config import settings
from .engines.base import EngineResult

FACT_RISK = {"false": 1.0, "mixed": 0.5, "true": 0.0}
BAND_EDGES = ((70, "trustworthy"), (40, "caution"), (0, "low"))


def _band(score: int) -> str:
    for edge, name in BAND_EDGES:
        if score >= edge:
            return name
    return "low"


def aggregate(ai: EngineResult, tox: EngineResult, fact: EngineResult) -> dict[str, Any]:
    contributions: list[dict[str, Any]] = []
    explanation: list[dict[str, Any]] = []
    available_base = 0.0

    # ---- fact-check ----
    if fact.status == "ok":
        available_base += settings.weight_fact
        verdict = fact.data.get("verdict")
        if verdict in FACT_RISK:
            sim = fact.data.get("similarity")
            conf = float(sim) if sim is not None else 0.9   # live API match: no score exposed
            contributions.append({"name": "fact_check", "risk": FACT_RISK[verdict],
                                  "base": settings.weight_fact, "conf": conf})
            explanation.append({
                "factor": "fact_check",
                "direction": {"false": "lowers", "mixed": "lowers", "true": "raises"}[verdict],
                "weight": round(settings.weight_fact * conf, 3),
                "detail": (
                    f"{fact.data.get('claims_matched', 1)} of "
                    f"{fact.data.get('claims_found', 1)} extracted claim(s) matched a "
                    f"fact-check rated '{fact.data.get('rating', verdict)}' by "
                    f"{fact.data.get('publisher')}"
                    + (f" (similarity {sim:.2f})." if sim is not None else ".")),
            })
        elif fact.data.get("live") and fact.data["live"].get("status") in (
                "supported", "contradicted"):
            # No published verdict, but live research reached a decisive reading. It votes,
            # capped well below a human fact-checker's confidence, and the explanation says
            # exactly what produced it.
            live = fact.data["live"]
            contradicted = live["status"] == "contradicted"
            conf = min(float(live.get("confidence", 0.0)),
                       settings.live_confidence_ceiling)
            contributions.append({"name": "fact_check", "risk": 1.0 if contradicted else 0.0,
                                  "base": settings.weight_fact, "conf": conf})
            explanation.append({
                "factor": "fact_check",
                "direction": "lowers" if contradicted else "raises",
                "weight": round(settings.weight_fact * conf, 3),
                "detail": (f"No published fact-check matched, so the claim was looked up "
                           f"live: Wikipedia's \u201c{live.get('source_title')}\u201d article "
                           f"{'contradicts' if contradicted else 'supports'} it "
                           f"({float(live.get('confidence', 0)):.0%} model confidence). "
                           f"Weighted below a human verdict because it is a model reading "
                           f"an encyclopedia."),
            })
        elif verdict == "related":
            # A related-but-not-equivalent claim, or one the contradiction guard flagged.
            # It informs the reader but must NOT vote: putting a stranger's rating on the
            # user's sentence is the worst failure this tool can have.
            warn = fact.data.get("warnings") or []
            explanation.append({
                "factor": "fact_check", "direction": "neutral", "weight": 0.0,
                "detail": ("A related fact-checked claim was found but is not equivalent, "
                           "so it did not affect the score. "
                           + (" ".join(warn) if warn else
                              "Similarity was below the verdict threshold.")),
            })
        else:
            found = fact.data.get("claims_found", 0)
            explanation.append({
                "factor": "fact_check", "direction": "neutral", "weight": 0.0,
                "detail": (f"{found} claim(s) were extracted and checked; none matched a "
                           f"fact-checked claim, so this vector abstained. Absence of a "
                           f"match is not evidence of truth."),
            })
    else:
        explanation.append({"factor": "fact_check", "direction": "neutral", "weight": 0.0,
                            "detail": f"Engine unavailable: {fact.error}"})

    # ---- toxicity ----
    if tox.status == "ok":
        available_base += settings.weight_toxicity
        score = float(tox.data.get("score", 0.0))
        conf = float(tox.data.get("confidence", 0.0))
        contributions.append({"name": "toxicity", "risk": score,
                              "base": settings.weight_toxicity, "conf": conf})
        explanation.append({
            "factor": "toxicity",
            "direction": "lowers" if score >= settings.toxicity_flag_threshold else "raises",
            "weight": round(settings.weight_toxicity * conf, 3),
            "detail": f"Offensive-language probability {score:.2f} "
                      f"({'above' if score >= settings.toxicity_flag_threshold else 'below'} "
                      f"the {settings.toxicity_flag_threshold:.2f} flag threshold).",
        })
    else:
        explanation.append({"factor": "toxicity", "direction": "neutral", "weight": 0.0,
                            "detail": f"Engine unavailable: {tox.error}"})

    # ---- ai detection ----
    if ai.status == "ok":
        available_base += settings.weight_ai
        prob = float(ai.data.get("probability", 0.0))
        conf = float(ai.data.get("confidence", 0.0))
        contributions.append({"name": "ai_detection", "risk": prob,
                              "base": settings.weight_ai, "conf": conf})
        explanation.append({
            "factor": "ai_detection",
            "direction": "lowers" if prob >= settings.ai_flag_threshold else "raises",
            "weight": round(settings.weight_ai * conf, 3),
            "detail": f"Machine-generated probability {prob:.2f}. Machine-written text is "
                      f"not automatically false; this is weighted lowest of the three.",
        })
    else:
        explanation.append({"factor": "ai_detection", "direction": "neutral", "weight": 0.0,
                            "detail": f"Engine unavailable: {ai.error}"})

    # ---- fusion ----
    total_w = sum(c["base"] * c["conf"] for c in contributions)
    if total_w <= 0:
        return {
            "trust_score": 50, "band": "caution", "confidence": 0.0,
            "explanation": explanation + [{
                "factor": "overall", "direction": "neutral", "weight": 0.0,
                "detail": "No engine produced usable evidence, so the score is a neutral "
                          "placeholder rather than a measurement.",
            }],
        }

    risk = sum(c["base"] * c["conf"] * c["risk"] for c in contributions) / total_w
    score = max(0, min(100, round((1.0 - risk) * 100)))
    confidence = round(total_w / available_base, 4) if available_base else 0.0
    band = _band(score)

    # A BAND IS A CLAIM ABOUT TRUSTWORTHINESS, AND ONLY ONE VECTOR MEASURES TRUTH.
    # Toxicity measures tone; the detector measures authorship. A clean, human-sounding
    # sentence scores high on both while nothing at all is known about whether it is true,
    # so the arithmetic can reach the top band on evidence that cannot support it.
    # Measured: "Barack Obama was born in Kenya, not the United States." reported 78/100
    # "Likely trustworthy" on 0.20 confidence, because its match sat at 0.823 - below the
    # 0.85 verdict bar - and the fact-check vector correctly abstained.
    # The score is left exactly as computed; only the LABEL is withheld. This is the fix
    # for limitation 6 in the README.
    if band == "trustworthy" and not any(c["name"] == "fact_check" for c in contributions):
        band = "insufficient_evidence"
        explanation.append({
            "factor": "overall", "direction": "neutral", "weight": 0.0,
            "detail": ("Not labelled trustworthy: no fact-check or live source spoke to "
                       "this text, and the toxicity and AI-detection vectors measure tone "
                       "and authorship rather than truth. The score reflects only that the "
                       "text is clean and human-sounding."),
        })

    return {"trust_score": score, "band": band, "confidence": confidence,
            "explanation": explanation}
