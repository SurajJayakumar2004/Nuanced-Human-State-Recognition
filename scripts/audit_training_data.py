#!/usr/bin/env python3
"""
Audit training data before training the temporal GMF+LSTM model.

Scans Data/features/*.pkl for:
- Modal integrity: temporal shapes visual_latent (15, 384), audio_latent (15, 768),
  visual_fau (15, 3), audio_geometric (15, 2)
- Class distribution via CREMA-D stem → nuanced state mapping
- Corruption / missing keys
- Summary: valid vs total, class counts, average geometric features (mean over 15-step sequences)
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Project root on path (for consistency; we duplicate mapping to avoid loading torch)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Same labels as fusion_head / train_temporal_model (no torch import)
NUANCED_STATE_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
]


def map_filename_to_label(stem: str) -> str:
    """Map CREMA-D stem to 6 base emotions. Mirrors train_temporal_model."""
    parts = stem.split("_")
    if len(parts) < 3:
        return "Neutral"
    emo = parts[2].upper()
    mapping = {
        "ANG": "Angry",
        "DIS": "Disgust",
        "FEA": "Fear",
        "HAP": "Happy",
        "NEU": "Neutral",
        "SAD": "Sad",
    }
    return mapping.get(emo, "Neutral")


FEATURES_DIR = Path("Data/features")

# Required target shapes for temporal sequences (validated in check_modal_integrity)
VISUAL_LATENT_SHAPE = (15, 384)   # visual_latent
AUDIO_LATENT_SHAPE = (15, 768)   # audio_latent
VISUAL_FAU_SHAPE = (15, 3)       # visual_fau [FAU12, FAU6, FAU4] per step
AUDIO_GEOMETRIC_SHAPE = (15, 2)  # audio_geometric [jitter, intensity] per step


def _shape_of(obj: Any) -> Tuple[int, ...] | None:
    """Return shape as tuple if array-like, else None."""
    s = getattr(obj, "shape", None)
    if s is not None:
        return tuple(s)
    try:
        if hasattr(obj, "__len__") and len(obj) > 0 and hasattr(obj[0], "__len__"):
            return (len(obj), len(obj[0]))
    except Exception:
        pass
    return None


def check_modal_integrity(feat: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    Verify required keys and temporal shapes. Returns (is_valid, list of error messages).
    """
    errors: List[str] = []

    # visual_latent: shape (15, 384)
    if "visual_latent" not in feat:
        errors.append("missing 'visual_latent'")
    else:
        v = feat["visual_latent"]
        try:
            shape = _shape_of(v)
            if shape is not None and shape != VISUAL_LATENT_SHAPE:
                errors.append(f"visual_latent shape {shape} != {VISUAL_LATENT_SHAPE}")
        except Exception as e:
            errors.append(f"visual_latent check failed: {e}")

    # audio_latent: shape (15, 768), top-level
    al = feat.get("audio_latent")
    if al is None:
        errors.append("missing 'audio_latent'")
    else:
        try:
            shape = _shape_of(al)
            if shape is not None and shape != AUDIO_LATENT_SHAPE:
                errors.append(f"audio_latent shape {shape} != {AUDIO_LATENT_SHAPE}")
        except Exception as e:
            errors.append(f"audio_latent check failed: {e}")

    # visual_fau: shape (15, 3)
    vfau = feat.get("visual_fau")
    if vfau is None:
        errors.append("missing 'visual_fau'")
    else:
        try:
            shape = _shape_of(vfau)
            if shape is not None and shape != VISUAL_FAU_SHAPE:
                errors.append(f"visual_fau shape {shape} != {VISUAL_FAU_SHAPE}")
        except Exception as e:
            errors.append(f"visual_fau check failed: {e}")

    # audio_geometric: shape (15, 2) [jitter, intensity per step]
    ageo = feat.get("audio_geometric")
    if ageo is None:
        errors.append("missing 'audio_geometric'")
    else:
        try:
            shape = _shape_of(ageo)
            if shape is not None and shape != AUDIO_GEOMETRIC_SHAPE:
                errors.append(f"audio_geometric shape {shape} != {AUDIO_GEOMETRIC_SHAPE}")
        except Exception as e:
            errors.append(f"audio_geometric check failed: {e}")

    return len(errors) == 0, errors


def _mean_over_axis0(seq: Any, cols: int) -> List[float]:
    """Mean over first axis (sequence). seq is (T, cols) as list or array. Avoids 'if not seq' for NumPy arrays."""
    if seq is None or cols <= 0:
        return [0.0] * cols
    try:
        n = len(seq)
    except TypeError:
        return [0.0] * cols
    if n == 0:
        return [0.0] * cols
    if hasattr(seq, "mean"):
        m = seq.mean(axis=0)
        return [float(m[i]) for i in range(cols)]
    # list of rows
    rows = list(seq)
    n = len(rows)
    out = [0.0] * cols
    for r in rows:
        for c in range(min(cols, len(r))):
            out[c] += float(r[c])
    return [out[c] / n for c in range(cols)] if n else out


def get_geometric(feat: Dict[str, Any]) -> Tuple[float, float, float, float, float]:
    """
    Extract mean geometric features from temporal sequences for audit reporting.
    - visual_fau (15, 3): mean over the array -> overall FAU12, FAU6, FAU4 intensities.
    - audio_geometric (15, 2): mean over the array -> average jitter and intensity.
    """
    vfau = feat.get("visual_fau")
    ageo = feat.get("audio_geometric")
    if vfau is not None:
        fau_mean = _mean_over_axis0(vfau, 3)
        fau12, fau6, fau4 = fau_mean[0], fau_mean[1], fau_mean[2]
    else:
        fau12 = fau6 = fau4 = 0.0
    if ageo is not None:
        geo_mean = _mean_over_axis0(ageo, 2)
        jitter, intensity = geo_mean[0], geo_mean[1]
    else:
        jitter = intensity = 0.0
    return fau12, fau6, fau4, jitter, intensity


def run_audit() -> None:
    if not FEATURES_DIR.exists():
        print(f"[ERROR] Features directory not found: {FEATURES_DIR}")
        sys.exit(1)

    pkl_files = sorted(FEATURES_DIR.glob("*.pkl"))
    total_files = len(pkl_files)
    print(f"[INFO] Scanning {total_files} files in {FEATURES_DIR} ...\n")

    valid_files: List[Path] = []
    corrupted: List[Tuple[Path, str]] = []  # (path, reason)
    class_counts: Dict[str, int] = {label: 0 for label in NUANCED_STATE_LABELS}
    geometric_sums: List[float] = [0.0] * 5  # fau12, fau6, fau4, jitter, intensity
    geometric_count = 0

    for idx, path in enumerate(pkl_files):
        if (idx + 1) % 1000 == 0:
            print(f"[INFO] Processed {idx + 1}/{total_files} ...")
        stem = path.stem
        try:
            with open(path, "rb") as f:
                feat = pickle.load(f)
        except Exception as e:
            corrupted.append((path, f"unreadable: {e}"))
            continue

        if not isinstance(feat, dict):
            corrupted.append((path, "root is not a dict"))
            continue

        ok, errs = check_modal_integrity(feat)
        if not ok:
            corrupted.append((path, "; ".join(errs)))
            continue

        valid_files.append(path)
        label = map_filename_to_label(stem)
        if label not in class_counts:
            class_counts[label] = 0
        class_counts[label] += 1

        g = get_geometric(feat)
        for i in range(5):
            geometric_sums[i] += g[i]
        geometric_count += 1

    # ---- Report corruption ----
    if corrupted:
        print("=" * 60)
        print("CORRUPTION / MISSING KEYS")
        print("=" * 60)
        for path, reason in corrupted[:50]:  # cap at 50
            print(f"  {path.name}: {reason}")
        if len(corrupted) > 50:
            print(f"  ... and {len(corrupted) - 50} more.")
        print(f"Total problematic: {len(corrupted)}\n")
    else:
        print("No corrupted or incomplete files.\n")

    # ---- Summary ----
    valid_count = len(valid_files)
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total files:     {total_files}")
    print(f"  Valid files:     {valid_count}")
    print(f"  Invalid/Corrupt: {total_files - valid_count}")
    if total_files > 0:
        pct = 100.0 * valid_count / total_files
        print(f"  Valid %:         {pct:.1f}%")
    print()

    # Class distribution
    print("CLASS DISTRIBUTION (Sequence potential by nuanced state)")
    print("-" * 60)
    print(f"  {'Label':<32} {'Count':>8}  {'%':>8}")
    print("-" * 60)
    for label in NUANCED_STATE_LABELS:
        count = class_counts.get(label, 0)
        pct = (100.0 * count / valid_count) if valid_count else 0.0
        print(f"  {label:<32} {count:>8}  {pct:>7.1f}%")
    print("-" * 60)
    print(f"  {'TOTAL':<32} {valid_count:>8}  {100.0:>7.1f}%")
    print()

    # Average geometric features (mean over (15, N) sequences, then over samples)
    print("AVERAGE GEOMETRIC FEATURES (mean over 15-step seq then over samples)")
    print("-" * 60)
    names = ["FAU12", "FAU6", "FAU4", "jitter", "intensity"]
    if geometric_count > 0:
        print(f"  {'Feature':<12} {'Mean':>10}  ({geometric_count} valid samples, (15,N) seqs)")
        print("-" * 60)
        for i, name in enumerate(names):
            mean = geometric_sums[i] / geometric_count
            print(f"  {name:<12} {mean:>10.4f}")
    else:
        print("  No valid samples to compute averages.")
    print("-" * 60)


if __name__ == "__main__":
    run_audit()
