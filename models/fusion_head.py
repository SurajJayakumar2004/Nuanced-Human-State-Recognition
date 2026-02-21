from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


NUANCED_STATE_LABELS: List[str] = [
    "Fake / Polite Face",
    "Hiding Stress",
    "Deep Focus",
    "Sarcasm",
    "Confusion",
    "Boredom",
    "Awkwardness",
    "Controlled Annoyance",
    "Relief",
    "Mixed Feelings",
]


@dataclass
class NuancedStatePrediction:
    label: str
    confidence: float
    probabilities: np.ndarray  # [10]


class NuancedStateClassifier(nn.Module):
    """
    Fusion head for RNHS v1.0.

    Takes a concatenated feature vector of:
      - Visual FAUs (e.g., [FAU12, FAU6, FAU4])
      - Audio features (e.g., [jitter, intensity])
    and predicts one of 10 nuanced states defined in NUANCED_STATE_LABELS.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        """
        Args:
            input_dim: Dimension of the concatenated feature vector.
                       For FAU12/6/4 + jitter + intensity, this is 5.
            hidden_dim: Hidden dimension for the MLP.
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Simple MLP fusion head
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, len(NUANCED_STATE_LABELS))  # [batch, 10]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor of shape [batch, input_dim]

        Returns:
            logits: torch.Tensor of shape [batch, 10]
        """
        x = self.fc1(x)  # [batch, hidden_dim]
        x = F.relu(x)
        x = self.fc2(x)  # [batch, hidden_dim]
        x = F.relu(x)
        logits = self.out(x)  # [batch, 10]
        return logits

    @torch.no_grad()
    def predict(self, features: torch.Tensor) -> NuancedStatePrediction:
        """
        Predict the nuanced state for a single feature vector.

        Args:
            features: torch.Tensor of shape [input_dim] or [1, input_dim].

        Returns:
            NuancedStatePrediction with:
              - label: str
              - confidence: float (max softmax probability)
              - probabilities: np.ndarray of shape [10]
        """
        if features.ndim == 1:
            features = features.unsqueeze(0)  # [1, input_dim]

        logits = self.forward(features)  # [1, 10]
        probs = F.softmax(logits, dim=-1)[0]  # [10]

        conf, idx = torch.max(probs, dim=-1)
        label = NUANCED_STATE_LABELS[int(idx.item())]

        return NuancedStatePrediction(
            label=label,
            confidence=float(conf.item()),
            probabilities=probs.cpu().numpy(),
        )


__all__ = ["NuancedStateClassifier", "NUANCED_STATE_LABELS", "NuancedStatePrediction"]

