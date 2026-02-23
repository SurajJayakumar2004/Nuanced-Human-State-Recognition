from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

# Temporal context: sequence length and LSTM hidden size
SEQ_LEN = 15
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 1

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
    conflict_score: float  # [0, 1], higher when visual and audio latents disagree


class NuancedStateClassifier(nn.Module):
    """
    Fusion head with Gated Multimodal Fusion (GMF) and Conflict Attention.

    Projects visual (384-D) and audio (768-D) latents into a shared hidden space,
    gates them with a learned scalar g, fuses as h_fused = g*h_visual + (1-g)*h_audio,
    and computes a conflict score from cosine similarity between the two modalities.
    Geometric features (FAU + jitter + intensity) are concatenated with h_fused for
    the final classification.
    """

    def __init__(
        self,
        visual_dim: int = 384,
        audio_dim: int = 768,
        hidden_dim: int = 256,
        geometric_dim: int = 5,
        mlp_hidden_dim: int = 64,
        lstm_hidden_size: int = LSTM_HIDDEN_SIZE,
        lstm_num_layers: int = LSTM_NUM_LAYERS,
    ) -> None:
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        self.hidden_dim = hidden_dim
        self.geometric_dim = geometric_dim

        # Projections into shared space
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Gating network: concat(h_visual, h_audio) -> scalar g in [0, 1]
        self.gate_fc = nn.Linear(2 * hidden_dim, 1)

        # Temporal context: LSTM over sequence of h_fused (256-D)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
        )

        # Classifier: LSTM last hidden (lstm_hidden_size) + last-step geometric -> logits
        self.fc1 = nn.Linear(lstm_hidden_size + geometric_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.out = nn.Linear(mlp_hidden_dim, len(NUANCED_STATE_LABELS))

    def _fuse_and_conflict(
        self,
        visual_latent: torch.Tensor,
        audio_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project both modalities, compute gate g, fused vector, and conflict score.

        Returns:
            h_fused: [batch, hidden_dim]
            conflict_score: [batch] in [0, 1], higher when modalities are far apart
        """
        h_visual = self.visual_proj(visual_latent)  # [B, hidden_dim]
        h_audio = self.audio_proj(audio_latent)     # [B, hidden_dim]

        gate_in = torch.cat([h_visual, h_audio], dim=-1)  # [B, 2*hidden_dim]
        g = torch.sigmoid(self.gate_fc(gate_in))          # [B, 1]

        h_fused = g * h_visual + (1.0 - g) * h_audio     # [B, hidden_dim]

        # Conflict attention: cosine similarity between h_visual and h_audio
        # similarity in [-1, 1]; far apart -> low similarity -> high conflict
        cos_sim = F.cosine_similarity(h_visual, h_audio, dim=1)  # [B]
        conflict_score = (1.0 - cos_sim).clamp(0.0, 1.0)  # [B]

        return h_fused, conflict_score

    def forward(
        self,
        visual_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        geometric: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_latent: [batch, seq_len, visual_dim] or [batch, visual_dim]
            audio_latent: [batch, seq_len, audio_dim] or [batch, audio_dim]
            geometric: [batch, seq_len, geometric_dim] or [batch, geometric_dim]

        Returns:
            logits: [batch, 10]
            conflict_score: [batch]
        """
        # Support both sequence (3D) and single-step (2D) input
        if visual_latent.dim() == 2:
            visual_latent = visual_latent.unsqueeze(1)
            audio_latent = audio_latent.unsqueeze(1)
            geometric = geometric.unsqueeze(1)

        B, T, _ = visual_latent.shape
        # Flatten batch and time for per-step fusion
        v_flat = visual_latent.reshape(B * T, -1)
        a_flat = audio_latent.reshape(B * T, -1)
        h_fused_flat, conflict_flat = self._fuse_and_conflict(v_flat, a_flat)
        h_fused = h_fused_flat.reshape(B, T, self.hidden_dim)   # [B, T, 256]
        conflict_score = conflict_flat.reshape(B, T)             # [B, T]

        # LSTM over time; use last timestep's output
        lstm_out, _ = self.lstm(h_fused)   # [B, T, lstm_hidden_size]
        last_hidden = lstm_out[:, -1, :]   # [B, lstm_hidden_size]
        last_conflict = conflict_score[:, -1]   # [B]
        last_geometric = geometric[:, -1, :]   # [B, geometric_dim]

        x = torch.cat([last_hidden, last_geometric], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.out(x)
        return logits, last_conflict

    @torch.no_grad()
    def predict(
        self,
        visual_latent: Union[torch.Tensor, np.ndarray],
        audio_latent: Union[torch.Tensor, np.ndarray],
        geometric_features: Union[torch.Tensor, np.ndarray],
    ) -> NuancedStatePrediction:
        """
        Predict the nuanced state from a single sample or a sequence.

        Args:
            visual_latent: (visual_dim,), (1, visual_dim), or (1, seq_len, visual_dim)
            audio_latent: (audio_dim,), (1, audio_dim), or (1, seq_len, audio_dim)
            geometric_features: (geometric_dim,), (1, geometric_dim), or (1, seq_len, geometric_dim)

        Returns:
            NuancedStatePrediction with label, confidence, probabilities, conflict_score.
        """
        dev = next(self.parameters()).device
        if isinstance(visual_latent, np.ndarray):
            visual_latent = torch.from_numpy(visual_latent.astype(np.float32))
        if isinstance(audio_latent, np.ndarray):
            audio_latent = torch.from_numpy(audio_latent.astype(np.float32))
        if isinstance(geometric_features, np.ndarray):
            geometric_features = torch.from_numpy(geometric_features.astype(np.float32))

        visual_latent = visual_latent.to(dev)
        audio_latent = audio_latent.to(dev)
        geometric_features = geometric_features.to(dev)

        if visual_latent.ndim == 1:
            visual_latent = visual_latent.unsqueeze(0).unsqueeze(0)  # [1, 1, V]
        elif visual_latent.ndim == 2:
            visual_latent = visual_latent.unsqueeze(0)  # [1, T, V]
        if audio_latent.ndim == 1:
            audio_latent = audio_latent.unsqueeze(0).unsqueeze(0)
        elif audio_latent.ndim == 2:
            audio_latent = audio_latent.unsqueeze(0)
        if geometric_features.ndim == 1:
            geometric_features = geometric_features.unsqueeze(0).unsqueeze(0)
        elif geometric_features.ndim == 2:
            geometric_features = geometric_features.unsqueeze(0)

        logits, conflict_scores = self.forward(
            visual_latent, audio_latent, geometric_features
        )
        probs = F.softmax(logits, dim=-1)[0]
        conflict_score = float(conflict_scores[0].item())

        conf, idx = torch.max(probs, dim=-1)
        label = NUANCED_STATE_LABELS[int(idx.item())]

        return NuancedStatePrediction(
            label=label,
            confidence=float(conf.item()),
            probabilities=probs.cpu().numpy(),
            conflict_score=conflict_score,
        )


__all__ = ["NuancedStateClassifier", "NUANCED_STATE_LABELS", "NuancedStatePrediction"]
