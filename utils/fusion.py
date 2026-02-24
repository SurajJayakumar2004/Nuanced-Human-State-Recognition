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
    All overrides return one of the 6 base emotions: Angry, Disgust, Fear, Happy, Neutral, Sad.

    Args:
        neural_prediction: State string from the neural classifier (6-class).
        neural_confidence: Confidence [0, 1] from the neural model.
        fau_data: FAU intensities, keys FAU12, FAU6, FAU4 (float [0, 1]).
        body_data: Rigidity, tilt, touching, tapping. Expected keys:
            - "Shoulder_Rigidity" (float [0, 1])
            - "Head_Tilt" (float [0, 1])
            - "Self_Touching_Hands" (bool)
            - "Finger_Tapping" (bool)
        audio_data: At least "jitter" (float [0, 1]) for stress rule.
        synchrony_incongruent: True when Visual_Peak follows Audio_Peak by >300ms;
            biases result toward Happy (sarcastic) or Neutral (fake smile).

    Returns:
        final_state: str (one of 6 base emotions)
        confidence: float in [0, 1]
        logic_source: "Neural" or "Override"
    """
    fau12 = float(fau_data.get("FAU12", 0.0))
    fau6 = float(fau_data.get("FAU6", 0.0))
    fau4 = float(fau_data.get("FAU4", 0.0))
    fau_intensity = max(fau12, fau6, fau4)

    rigidity = _get_scalar(body_data, "Shoulder_Rigidity", "rigidity", default=0.0)
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

    return neural_prediction, neural_confidence, "Neural"


__all__ = ["classify_nuanced_state"]

