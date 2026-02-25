from __future__ import annotations

from typing import Dict, Any, Tuple

import numpy as np


def _get_scalar(features: Dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """Safely fetch a scalar from a dict, trying multiple keys."""
    for k in keys:
        if k in features:
            v = features[k]
            try:
                return float(np.asarray(v).item())
            except Exception:
                continue
    return float(default)


# Valid 6-class CREMA-D emotions (must match NUANCED_STATE_LABELS in fusion_head)
VALID_STATES = frozenset({"Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"})


def _is_neutral(neural_prediction: str) -> bool:
    """True if neural prediction is Neutral."""
    if neural_prediction is None:
        return False
    return neural_prediction.strip() == "Neutral"


def classify_nuanced_state(
    neural_prediction: str,
    neural_confidence: float,
    fau_data: Dict[str, float],
    body_data: Dict[str, Any],
    audio_data: Dict[str, Any],
    synchrony_incongruent: bool = False,
) -> Tuple[str, float, str]:
    """
    Hybrid Gate: applies physical overrides to neural prediction.
    Base emotions from neural: Angry, Disgust, Fear, Happy, Neutral, Sad.
    Rule overrides can refine to nuanced states: Frustration, Panic, Contempt.

    Args:
        neural_prediction: State string from the neural classifier (6-class).
        neural_confidence: Confidence [0, 1] from the neural model.
        fau_data: FAU intensities, keys FAU12, FAU6, FAU4 (float [0, 1]).
        body_data: Rigidity, tilt, touching, tapping, posture. Expected keys:
            - "Shoulder_Rigidity" (float [0, 1])
            - "Head_Tilt" (float or degrees)
            - "Self_Touching_Hands" (bool)
            - "Finger_Tapping" (bool)
            - "posture_asymmetry", "posture_slump"/"is_slumped", "shoulders_raised" (bool)
            - "lean" (str: 'forward', 'back', 'neutral')
        audio_data: At least "jitter" (float [0, 1]) for stress rule.
        synchrony_incongruent: True when Visual_Peak follows Audio_Peak by >300ms;
            biases result toward Happy (sarcastic) or Neutral (fake smile).

    Returns:
        final_state: str (base emotion or nuanced: Frustration, Panic, Contempt)
        confidence: float in [0, 1]
        logic_source: "Neural", "Override", "Rule-Override", or "Posture-Override"
    """
    fau12 = float(fau_data.get("FAU12", 0.0))
    fau6 = float(fau_data.get("FAU6", 0.0))
    fau4 = float(fau_data.get("FAU4", 0.0))
    fau_intensity = max(fau12, fau6, fau4)

    rigidity = _get_scalar(body_data, "Shoulder_Rigidity", "rigidity", default=0.0)
    head_tilt = _get_scalar(body_data, "Head_Tilt", "head_tilt", default=0.0)
    posture_asymmetry = body_data.get("posture_asymmetry", False)
    is_slumped = body_data.get("is_slumped", body_data.get("posture_slump", False))
    shoulders_raised = body_data.get("shoulders_raised", False)
    lean = body_data.get("lean", "neutral")
    if isinstance(lean, str):
        lean = lean.strip().lower()
    touching = body_data.get("Self_Touching_Hands", False)
    if isinstance(touching, (int, float)):
        touching = bool(touching)
    tapping = body_data.get("Finger_Tapping", False)
    if isinstance(tapping, (int, float)):
        tapping = bool(tapping)

    audio_jitter = _get_scalar(audio_data, "jitter", "stress", "audio_stress", default=0.0)
    audio_jitter = float(np.clip(audio_jitter, 0.0, 1.0))

    # Ensure neural prediction is valid; fallback to Neutral if unknown
    if neural_prediction not in VALID_STATES:
        neural_prediction = "Neutral"

    # Physical overrides (order matters: first match wins)
    # All overrides map to 6 base emotions.

    # Temporal synchrony: Visual peak followed audio peak by >300ms → Incongruent
    if synchrony_incongruent:
        if fau_intensity > 0.35:
            return "Happy", 0.82, "Override"  # Sarcastic smile
        return "Neutral", 0.82, "Override"  # Fake smile

    # Boredom-like: Self_Touching_Hands AND low face intensity → Neutral
    if touching and fau_intensity < 0.3:
        return "Neutral", 0.85, "Override"

    # Hiding Stress: Neural says Neutral BUT rigid shoulders + high jitter → Fear
    if _is_neutral(neural_prediction) and rigidity > 0.85 and audio_jitter > 0.6:
        return "Fear", 0.88, "Override"

    # Awkwardness-like: Self_Touching_Hands AND smile → Neutral (uncomfortable)
    if touching and fau12 > 0.4:
        return "Neutral", 0.85, "Override"

    # Controlled Annoyance: Finger_Tapping AND neural says Neutral → Angry
    if tapping and _is_neutral(neural_prediction):
        return "Angry", 0.85, "Override"

    # Contempt: Neutral/Happy + posture asymmetry or head tilt (Posture-Override)
    if neural_prediction in ("Neutral", "Happy") and (
        posture_asymmetry or head_tilt > 10
    ):
        return "Contempt", 0.85, "Posture-Override"

    # Panic: Fear + shoulders raised + self-touching (Posture-Override)
    if neural_prediction == "Fear" and shoulders_raised and touching:
        return "Panic", 0.90, "Posture-Override"

    # Frustration: Angry/Sad + lean forward + fidgeting (Posture-Override)
    if neural_prediction in ("Angry", "Sad") and lean == "forward" and (
        tapping or touching
    ):
        return "Frustration", 0.85, "Posture-Override"

    # Confidence Boosters for base emotions (no state change)
    final_confidence = neural_confidence
    if neural_prediction == "Sad" and is_slumped:
        final_confidence = 0.95
    elif neural_prediction == "Angry" and lean == "forward":
        final_confidence = 0.95
    elif neural_prediction == "Disgust" and lean in ("back", "backward"):
        final_confidence = 0.95

    return neural_prediction, final_confidence, "Neural"


__all__ = ["classify_nuanced_state"]

