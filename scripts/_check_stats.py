import pickle
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ------------------------------------------------------------------
# 6-class CREMA-D labels (matches fusion_head / train_temporal_model)
# ------------------------------------------------------------------
NUANCED_STATE_LABELS = [
    "Angry",
    "Disgust",
    "Fear",
    "Happy",
    "Neutral",
    "Sad",
]


def map_to_label(stem, audio, visual):
    """Map CREMA-D stem to 6 base emotions. Uses stem only (same as train_temporal_model)."""
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
print(f"\n  catch-all 'Neutral': {counts.get('Neutral', 0)}")
