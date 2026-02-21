import pickle
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ------------------------------------------------------------------
# Inline copy of thresholds + map_to_label so we don't need torch
# ------------------------------------------------------------------
_INTEN_HI    = 0.060
_INTEN_LO    = 0.025
_JITTER_HI   = 0.110
_FAU12_SMILE = 0.490

NUANCED_STATE_LABELS = [
    "Fake / Polite Face", "Hiding Stress", "Deep Focus", "Sarcasm",
    "Confusion", "Boredom", "Awkwardness", "Controlled Annoyance",
    "Relief", "Mixed Feelings",
]


def _resolve(intensity):
    if intensity >= _INTEN_HI:  return "HI"
    if intensity >= _INTEN_LO:  return "MD"
    return "LO"


def map_to_label(stem, audio, visual):
    parts = stem.split("_")
    emo   = parts[2].upper() if len(parts) >= 3 else "NEU"
    inten = parts[3].upper() if len(parts) >= 4 else "XX"
    jitter    = float(audio.get("jitter",    0.0))
    intensity = float(audio.get("intensity", 0.0))
    fau12     = float(visual.get("FAU12",    0.0))
    if inten == "XX":
        inten = _resolve(intensity)
    if emo == "NEU":
        return {"LO": "Boredom", "MD": "Deep Focus"}.get(inten, "Confusion")
    if emo == "HAP":
        if inten == "LO":                           return "Fake / Polite Face"
        if jitter >= _JITTER_HI or inten == "HI":  return "Sarcasm"
        return "Fake / Polite Face"
    if emo == "ANG":    return "Controlled Annoyance"
    if emo == "SAD":    return "Hiding Stress" if inten == "HI" else "Relief"
    if emo == "FEA":
        if inten == "HI":   return "Hiding Stress"
        if inten == "LO":   return "Boredom"
        return "Hiding Stress"
    if emo == "DIS":    return "Awkwardness"
    return "Mixed Feelings"


# ------------------------------------------------------------------
SAMPLE = 500   # None = full dataset; set int for quick check

paths = sorted(Path("Data/features").glob("*.pkl"))
if SAMPLE:
    paths = paths[:SAMPLE]

print(f"Checking {len(paths)} files ...")
labels = []
for p in paths:
    d = pickle.load(open(p, "rb"))
    labels.append(map_to_label(p.stem, d.get("audio", {}), d.get("visual_fau", {})))

counts = Counter(labels)
total  = len(labels)
print(f"\nClass distribution (N={total}):")
for lbl in NUANCED_STATE_LABELS:
    c   = counts.get(lbl, 0)
    bar = "#" * int(c / total * 40)
    print(f"  {lbl:30s}: {c:5d} ({c / total * 100:5.1f}%)  {bar}")
print(f"\n  catch-all 'Mixed Feelings': {counts.get('Mixed Feelings', 0)}")
