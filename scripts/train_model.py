from __future__ import annotations

import glob
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Ensure the project root is on sys.path so `models` and `utils` are importable
# regardless of how this script is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models.fusion_head import (
    NuancedStateClassifier,
    NUANCED_STATE_LABELS,
)


FEATURES_DIR = Path("Data/features")
WEIGHTS_DIR = Path("weights")
WEIGHTS_PATH = WEIGHTS_DIR / "fusion_v1.pth"

# TEMP: limit number of feature files for quick smoke-test training.
# Set to None to use the full dataset for production training.
MAX_FILES: int | None = None

# Balanced training: take this many samples per class (equal distribution).
# Set to None to use full (imbalanced) dataset.
BALANCED_SAMPLES_PER_CLASS: int | None = 90
RANDOM_SEED: int = 42


def map_filename_to_label(stem: str) -> str:
    """
    Map CREMA-D style filename stem to a nuanced state label.

    Example stem: 1001_DFA_HAP_LO
    CREMA-D uses NEU+XX (not NEU+MD/LO/HI); we split NEU+XX across Deep Focus / Confusion / Boredom.
    """
    parts = stem.split("_")
    if len(parts) < 4:
        return "Mixed Feelings"

    _, _, emo, inten = parts[:4]

    emo = emo.upper()
    inten = inten.upper()

    # 1. Fake / Polite Face: Happy face but low voice energy (polite smile)
    if emo == "HAP" and inten == "LO":
        return "Fake / Polite Face"

    # 2. Hiding Stress: Sad/fear face but high vocal stress (masking internal pressure)
    if emo in {"SAD", "FEA"} and inten == "HI":
        return "Hiding Stress"

    # 3–5. CREMA-D has NEU+XX (no MD/LO/HI). Split by hash so all three get examples.
    if emo == "NEU" and inten == "XX":
        h = hash(stem) % 3
        if h == 0:
            return "Deep Focus"
        if h == 1:
            return "Confusion"
        return "Boredom"

    # 3. Deep Focus (also from ANG+MD: controlled/focused)
    if emo == "NEU" and inten == "MD":
        return "Deep Focus"
    if emo == "ANG" and inten == "MD":
        return "Deep Focus"

    # 4. Sarcasm: Happy face but high intensity voice (contradictory signal)
    if emo == "HAP" and inten == "HI":
        return "Sarcasm"

    # 5. Confusion: Neutral face but high vocal stress
    if emo == "NEU" and inten == "HI":
        return "Confusion"
    if emo == "ANG" and inten == "HI":
        return "Confusion"

    # 6. Boredom: Neutral face, low energy voice
    if emo == "NEU" and inten == "LO":
        return "Boredom"
    if emo == "DIS" and inten == "LO":
        return "Boredom"

    # 7. Awkwardness: Disgust/unease face, medium voice (uncomfortable)
    if emo == "DIS" and inten == "MD":
        return "Awkwardness"

    # 8. Controlled Annoyance: Angry face but low voice (suppressing irritation)
    if emo == "ANG" and inten == "LO":
        return "Controlled Annoyance"

    # 9. Relief: Sad face but low voice (stress releasing)
    if emo == "SAD" and inten == "LO":
        return "Relief"

    # 10. Mixed Feelings: HAP+MD, DIS+HI, FEA+LO, FEA+MD, SAD+MD, etc.
    return "Mixed Feelings"


def label_to_index(label: str) -> int:
    try:
        return NUANCED_STATE_LABELS.index(label)
    except ValueError:
        # Unknown label -> map to Mixed Feelings
        return NUANCED_STATE_LABELS.index("Mixed Feelings")


def get_balanced_paths(features_dir: Path, n_per_class: int, seed: int) -> List[Path]:
    """
    Group paths by label, then randomly sample n_per_class from each class
    (or all if a class has fewer). Returns a single shuffled list of paths.
    """
    from collections import defaultdict
    import random

    all_paths = sorted(features_dir.glob("*.pkl"))
    by_label: Dict[str, List[Path]] = defaultdict(list)
    for p in all_paths:
        label = map_filename_to_label(p.stem)
        by_label[label].append(p)

    rng = random.Random(seed)
    balanced: List[Path] = []
    for label in NUANCED_STATE_LABELS:
        paths = by_label.get(label, [])
        n_take = min(n_per_class, len(paths)) if paths else 0
        if n_take > 0:
            balanced.extend(rng.sample(paths, n_take))

    rng.shuffle(balanced)
    return balanced


class CremaDataset(Dataset):
    """
    Dataset for CREMA-D feature .pkl files produced by scripts/preprocess_crema.py.

    Each item contains:
        x: concatenated feature vector [FAU12, FAU6, FAU4, jitter, intensity]
        y: class index in [0, 9]
    """

    def __init__(self, features_dir: Path) -> None:
        self.features_dir = features_dir
        if BALANCED_SAMPLES_PER_CLASS is not None:
            self.paths = get_balanced_paths(
                features_dir, BALANCED_SAMPLES_PER_CLASS, RANDOM_SEED
            )
        else:
            all_paths = sorted(features_dir.glob("*.pkl"))
            if MAX_FILES is not None:
                all_paths = all_paths[:MAX_FILES]
            self.paths = all_paths
        if not self.paths:
            raise RuntimeError(f"No .pkl feature files found in {features_dir}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        stem = path.stem  # e.g., 1001_DFA_HAP_LO

        with open(path, "rb") as f:
            feat: Dict[str, Dict] = pickle.load(f)

        visual = feat.get("visual_fau", {})
        audio = feat.get("audio", {})

        fau12 = float(visual.get("FAU12", 0.0))
        fau6 = float(visual.get("FAU6", 0.0))
        fau4 = float(visual.get("FAU4", 0.0))
        jitter = float(audio.get("jitter", 0.0))
        intensity = float(audio.get("intensity", 0.0))

        x_np = np.asarray([fau12, fau6, fau4, jitter, intensity], dtype=np.float32)  # [5]
        x = torch.from_numpy(x_np)  # [5]

        label_str = map_filename_to_label(stem)
        y_idx = label_to_index(label_str)
        y = torch.tensor(y_idx, dtype=torch.long)

        return x, y


def get_device() -> torch.device:
    """
    Prefer MPS on M1 Mac, fallback to CUDA or CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-3,
) -> None:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    dataset = CremaDataset(FEATURES_DIR)

    if BALANCED_SAMPLES_PER_CLASS is not None:
        print(f"[INFO] Balanced sampling: {BALANCED_SAMPLES_PER_CLASS} samples per class (seed={RANDOM_SEED})")

    # Log class distribution
    from collections import Counter
    label_counts = Counter()
    for path in dataset.paths:
        stem = path.stem
        label = map_filename_to_label(stem)
        label_counts[label] += 1
    
    print(f"[INFO] Dataset size: {len(dataset)}")
    print("[INFO] Class distribution:")
    for label in NUANCED_STATE_LABELS:
        count = label_counts.get(label, 0)
        pct = (count / len(dataset) * 100) if len(dataset) > 0 else 0.0
        print(f"  {label:30s}: {count:4d} ({pct:5.1f}%)")
    
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )

    device = get_device()
    print(f"[INFO] Using device: {device}")

    input_dim = 5  # [FAU12, FAU6, FAU4, jitter, intensity]
    model = NuancedStateClassifier(input_dim=input_dim, hidden_dim=64).to(device)

    # Optional: Use class weights to handle imbalanced dataset
    # Compute inverse frequency weights
    class_weights = []
    for label in NUANCED_STATE_LABELS:
        count = label_counts.get(label, 1)
        weight = len(dataset) / max(count, 1)  # Inverse frequency
        class_weights.append(weight)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        correct = 0
        total = 0

        for x, y in dataloader:
            x = x.to(device)  # [batch, 5]
            y = y.to(device)  # [batch]

            optimizer.zero_grad()
            logits = model(x)  # [batch, 10]
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * x.size(0)

            preds = torch.argmax(logits, dim=-1)
            correct += int((preds == y).sum().item())
            total += int(y.size(0))

        avg_loss = epoch_loss / max(total, 1)
        acc = correct / max(total, 1)
        print(
            f"[EPOCH {epoch:02d}/{epochs}] "
            f"loss={avg_loss:.4f} acc={acc:.4f} (N={total})"
        )

    torch.save(model.state_dict(), WEIGHTS_PATH)
    print(f"[INFO] Saved model weights to {WEIGHTS_PATH}")


def main() -> None:
    train()


if __name__ == "__main__":
    main()

