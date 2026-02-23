#!/usr/bin/env python3
"""
RHNS v2.0 — Core training script for the GMF + Conflict Attention + LSTM model.

1. Data: Load (15, 384) visual, (15, 768) audio, (15, 5) geometric sequences from .pkl.
2. Weighted sampling: Class weights so Mixed Feelings (73.1%) and Sarcasm (1.2%) are seen
   equally by the model each epoch (WeightedRandomSampler).
3. Architecture: NuancedStateClassifier; Xavier uniform init for all linear and LSTM layers.
4. M1 (MPS): device='mps', AdamW + OneCycleLR, batch_size 64 or 128; float32 only (no autocast) for Metal stability.
5. Visualization: training_progress_v2.png each epoch; confusion_matrix_v2.png after training.
6. Checkpointing: Save only best weights to weights/fusion_v2_best.pth.
"""

from __future__ import annotations

import pickle
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from models.fusion_head import (
    NuancedStateClassifier,
    NUANCED_STATE_LABELS,
)


FEATURES_DIR = Path("Data/features")
WEIGHTS_DIR = Path("weights")
REPORTS_DIR = Path("reports")
BEST_WEIGHTS_PATH = WEIGHTS_DIR / "fusion_v2_best.pth"
PROGRESS_PLOT_PATH = REPORTS_DIR / "training_progress_v2.png"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix_v2.png"
RANDOM_SEED = 42
DEFAULT_BATCH_SIZE = 64  # 64 or 128 for M1 Unified Memory
SEQ_LEN = 15
VISUAL_DIM = 384
AUDIO_DIM = 768
GEOMETRIC_DIM = 5  # FAU12, FAU6, FAU4, jitter, intensity


def map_filename_to_label(stem: str) -> str:
    """CREMA-D stem -> nuanced state (same logic as train_model / audit)."""
    parts = stem.split("_")
    if len(parts) < 4:
        return "Mixed Feelings"
    _, _, emo, inten = parts[:4]
    emo, inten = emo.upper(), inten.upper()
    if emo == "HAP" and inten == "LO":
        return "Fake / Polite Face"
    if emo in {"SAD", "FEA"} and inten == "HI":
        return "Hiding Stress"
    if emo == "NEU" and inten == "XX":
        h = hash(stem) % 3
        if h == 0:
            return "Deep Focus"
        if h == 1:
            return "Confusion"
        return "Boredom"
    if emo == "NEU" and inten == "MD":
        return "Deep Focus"
    if emo == "ANG" and inten == "MD":
        return "Deep Focus"
    if emo == "HAP" and inten == "HI":
        return "Sarcasm"
    if emo == "NEU" and inten == "HI":
        return "Confusion"
    if emo == "ANG" and inten == "HI":
        return "Confusion"
    if emo == "NEU" and inten == "LO":
        return "Boredom"
    if emo == "DIS" and inten == "LO":
        return "Boredom"
    if emo == "DIS" and inten == "MD":
        return "Awkwardness"
    if emo == "ANG" and inten == "LO":
        return "Controlled Annoyance"
    if emo == "SAD" and inten == "LO":
        return "Relief"
    return "Mixed Feelings"


def label_to_index(label: str) -> int:
    try:
        return NUANCED_STATE_LABELS.index(label)
    except ValueError:
        return NUANCED_STATE_LABELS.index("Mixed Feelings")


def get_device(force_mps: bool = True) -> torch.device:
    """M1 optimization: force MPS when available; else CUDA, then CPU."""
    if force_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _xavier_lstm_and_gated(module: nn.Module) -> None:
    """Xavier uniform for LSTM and gating/projection layers."""
    for name, param in module.named_parameters():
        if "weight" in name and param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        elif "bias" in name:
            nn.init.zeros_(param)


class Visualizer:
    """
    Plots train/val loss and accuracy; saves to reports/training_progress_v2.png.
    Draws a 'Best Model' vertical line at the epoch with lowest val_loss to highlight
    overfitting (divergence when val_loss rises while train_loss falls).
    """

    def __init__(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_accs: List[float],
        val_accs: List[float],
        save_path: Path,
    ) -> None:
        self.train_losses = train_losses
        self.val_losses = val_losses
        self.train_accs = train_accs
        self.val_accs = val_accs
        self.save_path = save_path

    def plot(self, best_epoch: int | None = None) -> None:
        """Create figure with Loss and Accuracy subplots; draw Best Model line; save."""
        if not self.train_losses:
            return
        epochs_x = list(range(1, len(self.train_losses) + 1))
        fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        # Loss: train vs val — distinct colors so divergence is clear
        ax_loss.plot(epochs_x, self.train_losses, "b-", label="Train Loss", linewidth=2)
        ax_loss.plot(epochs_x, self.val_losses, "r-", label="Val Loss", linewidth=2)
        if best_epoch is not None and 1 <= best_epoch <= len(epochs_x):
            ax_loss.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5, label="Best Model (lowest val loss)")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title("Train vs Val Loss (divergence = overfitting risk)")
        ax_loss.legend(loc="upper right")
        ax_loss.grid(True, alpha=0.3)

        # Accuracy
        ax_acc.plot(epochs_x, self.train_accs, "b-", label="Train Acc", linewidth=2)
        ax_acc.plot(epochs_x, self.val_accs, "r-", label="Val Acc", linewidth=2)
        if best_epoch is not None and 1 <= best_epoch <= len(epochs_x):
            ax_acc.axvline(x=best_epoch, color="green", linestyle="--", linewidth=1.5)
        ax_acc.set_ylabel("Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_title("Train vs Val Accuracy")
        ax_acc.legend(loc="lower right")
        ax_acc.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_confusion_matrix(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    save_path: Path,
) -> None:
    """Run best model on val set (float32), plot confusion matrix heatmap with seaborn."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    with torch.no_grad():
        for v, a, g, y in val_loader:
            v, a, g = v.to(device), a.to(device), g.to(device)
            logits, _ = model(v, a, g)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(y.cpu().numpy().tolist())
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    n_classes = len(NUANCED_STATE_LABELS)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=NUANCED_STATE_LABELS,
        yticklabels=NUANCED_STATE_LABELS,
        ax=ax,
        cbar_kws={"label": "Count"},
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("RHNS v2.0 — Confusion Matrix (best model on validation set)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class CremaTemporalDataset(Dataset):
    """
    Load temporal sequences from .pkl: visual_latent (15, 384), audio_latent (15, 768),
    geometric (15, 5) = concat(visual_fau (15,3), audio_geometric (15,2)).
    """

    def __init__(self, paths: List[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        stem = path.stem
        with open(path, "rb") as f:
            feat = pickle.load(f)

        # visual_latent: (15, 384)
        v = feat.get("visual_latent")
        if v is None:
            v = np.zeros((SEQ_LEN, VISUAL_DIM), dtype=np.float32)
        else:
            v = np.asarray(v, dtype=np.float32)
        if v.shape != (SEQ_LEN, VISUAL_DIM):
            v = np.zeros((SEQ_LEN, VISUAL_DIM), dtype=np.float32)

        # audio_latent: (15, 768)
        a = feat.get("audio_latent")
        if a is None:
            a = np.zeros((SEQ_LEN, AUDIO_DIM), dtype=np.float32)
        else:
            a = np.asarray(a, dtype=np.float32)
        if a.shape != (SEQ_LEN, AUDIO_DIM):
            a = np.zeros((SEQ_LEN, AUDIO_DIM), dtype=np.float32)

        # geometric: (15, 5) = [FAU12, FAU6, FAU4, jitter, intensity] per step
        vfau = feat.get("visual_fau")
        ageo = feat.get("audio_geometric")
        try:
            vfau = np.asarray(vfau, dtype=np.float32)
            if vfau.shape != (SEQ_LEN, 3):
                vfau = np.zeros((SEQ_LEN, 3), dtype=np.float32)
        except (TypeError, ValueError):
            vfau = np.zeros((SEQ_LEN, 3), dtype=np.float32)
        try:
            ageo = np.asarray(ageo, dtype=np.float32)
            if ageo.shape != (SEQ_LEN, 2):
                ageo = np.zeros((SEQ_LEN, 2), dtype=np.float32)
        except (TypeError, ValueError):
            ageo = np.zeros((SEQ_LEN, 2), dtype=np.float32)
        g = np.concatenate([vfau, ageo], axis=1).astype(np.float32)  # (15, 5)

        v_t = torch.from_numpy(v)   # (15, 384)
        a_t = torch.from_numpy(a)   # (15, 768)
        g_t = torch.from_numpy(g)   # (15, 5)
        y_idx = label_to_index(map_filename_to_label(stem))
        y = torch.tensor(y_idx, dtype=torch.long)
        return v_t, a_t, g_t, y


def build_paths_and_weights(
    features_dir: Path, seed: int
) -> Tuple[List[Path], List[Path], torch.Tensor, List[float]]:
    """
    Scan .pkl, compute 80/20 train/val split. Class weights and per-sample weights
    (1 / class_count) so WeightedRandomSampler gives balanced exposure: e.g. Mixed
    Feelings (73.1%) and Sarcasm (1.2%) are seen equally by the model each epoch.
    Returns (train_paths, val_paths, class_weights_tensor, sample_weights_for_sampler).
    """
    all_paths = sorted(features_dir.glob("*.pkl"))
    if not all_paths:
        raise RuntimeError(f"No .pkl files in {features_dir}")

    rng = random.Random(seed)
    rng.shuffle(all_paths)
    n = len(all_paths)
    n_train = int(0.8 * n)
    train_paths = all_paths[:n_train]
    val_paths = all_paths[n_train:]

    # Class counts over training set only (for weights and sampler)
    class_counts: Dict[str, int] = defaultdict(int)
    for p in train_paths:
        class_counts[map_filename_to_label(p.stem)] += 1

    num_classes = len(NUANCED_STATE_LABELS)
    counts_per_class = [
        max(class_counts.get(label, 1), 1) for label in NUANCED_STATE_LABELS
    ]
    # Inverse frequency for CrossEntropyLoss
    total = sum(counts_per_class)
    class_weights = torch.tensor(
        [total / c for c in counts_per_class], dtype=torch.float32
    )

    # Per-sample weight for WeightedRandomSampler (1 / count of that class)
    label_to_weight: Dict[str, float] = {
        label: 1.0 / max(class_counts.get(label, 1), 1)
        for label in NUANCED_STATE_LABELS
    }
    sample_weights = [
        label_to_weight[map_filename_to_label(p.stem)] for p in train_paths
    ]

    return train_paths, val_paths, class_weights, sample_weights


GRAD_CLIP_NORM = 1.0


def train(
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 3e-4,
) -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device(force_mps=True)
    print(f"[INFO] Device: {device} (MPS forced for M1)")

    train_paths, val_paths, class_weights, sample_weights = build_paths_and_weights(
        FEATURES_DIR, RANDOM_SEED
    )
    class_weights = class_weights.to(device)
    print(f"[INFO] Train samples: {len(train_paths)}, Val samples: {len(val_paths)}")

    train_ds = CremaTemporalDataset(train_paths)
    val_ds = CremaTemporalDataset(val_paths)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = NuancedStateClassifier(
        visual_dim=VISUAL_DIM,
        audio_dim=AUDIO_DIM,
        hidden_dim=256,
    ).to(device)
    _xavier_lstm_and_gated(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

    # Float32 only (no autocast) for MPS stability
    best_val_acc = 0.0
    best_val_loss = float("inf")
    best_epoch_by_val_loss: int | None = None  # for visualizer vertical line
    train_losses: List[float] = []
    val_losses: List[float] = []
    train_accs: List[float] = []
    val_accs: List[float] = []
    visualizer = Visualizer(
        train_losses, val_losses, train_accs, val_accs, PROGRESS_PLOT_PATH
    )

    print("-" * 88)
    print(f"{'EPOCH':<10} {'Train Loss':<14} {'Train Acc':<12} {'Val Loss':<12} {'Val Acc':<10}")
    print("-" * 88)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for v, a, g, y in train_loader:
            v, a, g, y = v.to(device), a.to(device), g.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(v, a, g)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            scheduler.step()
            train_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for v, a, g, y in val_loader:
                v, a, g, y = v.to(device), a.to(device), g.to(device), y.to(device)
                logits, _ = model(v, a, g)
                loss = criterion(logits, y)
                val_loss += loss.item() * y.size(0)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)
        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch_by_val_loss = epoch
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), BEST_WEIGHTS_PATH)
            best_mark = " (best)"
        else:
            best_mark = ""

        visualizer.plot(best_epoch=best_epoch_by_val_loss)

        # Epoch table
        print(
            f"{epoch:02d}/{epochs}    "
            f"{train_loss:<14.4f} {train_acc:<12.4f} "
            f"{val_loss:<12.4f} {val_acc:<10.4f}{best_mark}"
        )

    print("-" * 88)
    print(f"[INFO] Best validation accuracy: {best_val_acc:.4f}")
    print(f"[INFO] Saved best model to {BEST_WEIGHTS_PATH}")
    print(f"[INFO] Training progress plot saved to {PROGRESS_PLOT_PATH}")

    # Confusion matrix: load best weights and run on validation set (float32)
    model.load_state_dict(torch.load(BEST_WEIGHTS_PATH, map_location=device))
    plot_confusion_matrix(model, val_loader, device, CONFUSION_MATRIX_PATH)
    print(f"[INFO] Confusion matrix saved to {CONFUSION_MATRIX_PATH}")


def main() -> None:
    train(epochs=30, batch_size=DEFAULT_BATCH_SIZE, lr=3e-4)


if __name__ == "__main__":
    main()
