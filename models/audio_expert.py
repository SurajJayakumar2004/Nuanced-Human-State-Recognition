from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import librosa
from transformers import Wav2Vec2Model


class AudioExpert:
    """
    AudioExpert for 1-second mono 16kHz chunks.

    For each chunk it computes:
      - RMS energy (normalized to [0, 1]) as intensity
      - Vocal jitter (Pitch Perturbation Quotient) from an F0 track
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        # Device: prefer MPS, then CUDA, then CPU
        self.device = torch.device(
            "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._wav2vec2 = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")
        self._wav2vec2.to(self.device)
        self._wav2vec2.eval()

    def _to_float_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """
        Convert audio to float32 in [-1, 1].
        """
        if audio_data.ndim != 1:
            raise ValueError(
                f"AudioExpert expects 1D mono audio, got shape {audio_data.shape}"
            )

        if np.issubdtype(audio_data.dtype, np.integer):
            max_val = np.iinfo(audio_data.dtype).max
            audio = audio_data.astype(np.float32) / float(max_val)
        else:
            audio = audio_data.astype(np.float32)
        return audio

    def _compute_intensity(self, audio: np.ndarray) -> float:
        """
        RMS energy over the chunk, normalized to [0, 1].
        """
        if audio.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(audio ** 2)))
        # audio is already in [-1, 1], so rms is naturally in [0, 1]
        intensity = float(np.clip(rms, 0.0, 1.0))
        return intensity

    def _compute_f0_track(self, audio: np.ndarray) -> np.ndarray:
        """
        Frame-wise F0 track using librosa.pyin if available, otherwise piptrack.

        Returns:
            f0_track: np.ndarray of shape [frames] with F0 in Hz (0 for unvoiced).
        """
        if audio.size == 0:
            return np.zeros((0,), dtype=np.float32)

        try:
            # Use pyin for more stable F0 when available
            f0, _, _ = librosa.pyin(
                audio,
                fmin=librosa.note_to_hz("C2"),
                fmax=librosa.note_to_hz("C7"),
                sr=self.sample_rate,
            )  # [frames]
            f0 = np.nan_to_num(f0, nan=0.0).astype(np.float32)
            return f0
        except Exception:
            # Fallback to piptrack
            stft = np.abs(
                librosa.stft(y=audio, n_fft=1024, hop_length=256)
            )  # [freq_bins, frames]
            pitches, magnitudes = librosa.piptrack(
                S=stft, sr=self.sample_rate
            )  # [freq_bins, frames]

            f0_track = []
            for mag_col, pitch_col in zip(magnitudes.T, pitches.T):
                idx = mag_col.argmax()
                f0 = pitch_col[idx]
                f0_track.append(float(f0))

            return np.asarray(f0_track, dtype=np.float32)  # [frames]

    def _compute_jitter_ppq(self, f0_track: np.ndarray) -> float:
        """
        Compute Pitch Perturbation Quotient (PPQ) style jitter from an F0 track.

        Jitter_PPQ ≈ mean(|Ti - Tmean|) / Tmean, where Ti are pitch periods.
        """
        if f0_track.size == 0:
            return 0.0

        # Keep only voiced frames
        voiced_f0 = f0_track[f0_track > 0.0]
        if voiced_f0.size < 2:
            return 0.0

        periods = 1.0 / voiced_f0  # [frames], seconds
        T_mean = float(periods.mean())
        if T_mean <= 0.0:
            return 0.0

        # Mean absolute deviation of periods, normalized by mean period.
        jitter_abs = np.abs(periods - T_mean).mean()
        jitter_ppq = float(jitter_abs / T_mean)

        # Clamp to a reasonable [0, 1] range for downstream fusion logic.
        return float(np.clip(jitter_ppq, 0.0, 1.0))

    def get_latent_embeddings(self, audio_np: np.ndarray) -> np.ndarray:
        """
        Process a 1-second 16kHz chunk through Wav2Vec 2.0 and return
        mean-pooled hidden states as a feature vector.

        Args:
            audio_np: np.ndarray of shape [T], mono 16kHz (e.g. T=16000).

        Returns:
            np.ndarray of shape (768,) dtype float32.
        """
        audio = self._to_float_audio(audio_np)
        if audio.size == 0:
            return np.zeros(768, dtype=np.float32)
        # [T] -> [1, T]
        tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._wav2vec2(tensor)
        # last_hidden_state: [1, seq_len, 768]
        hidden = out.last_hidden_state
        pooled = hidden.mean(dim=1).squeeze(0).cpu().numpy().astype(np.float32)
        return pooled

    def get_latent_sequence(self, audio_np: np.ndarray, num_steps: int = 15) -> np.ndarray:
        """
        Return hidden states as a sequence of num_steps timesteps to align with visual.
        Samples (or interpolates) from last_hidden_state [1, seq_len, 768] to get [num_steps, 768].
        """
        audio = self._to_float_audio(audio_np)
        if audio.size == 0:
            return np.zeros((num_steps, 768), dtype=np.float32)
        tensor = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self._wav2vec2(tensor)
        hidden = out.last_hidden_state  # [1, seq_len, 768]
        seq_len = hidden.size(1)
        if seq_len == 0:
            return np.zeros((num_steps, 768), dtype=np.float32)
        hidden = hidden.squeeze(0).cpu().numpy()  # [seq_len, 768]
        if seq_len >= num_steps:
            # Sample num_steps indices uniformly
            indices = np.linspace(0, seq_len - 1, num_steps, dtype=np.int64)
            return hidden[indices].astype(np.float32)
        # Fewer than num_steps: interpolate (linear) to num_steps
        x_old = np.linspace(0, 1, seq_len)
        x_new = np.linspace(0, 1, num_steps)
        out = np.zeros((num_steps, 768), dtype=np.float32)
        for d in range(768):
            out[:, d] = np.interp(x_new, x_old, hidden[:, d])
        return out

    def extract_features(self, audio_np: np.ndarray) -> Dict[str, Any]:
        """
        Extract normalized intensity, jitter, and Wav2Vec2 latent from a 1-second 16kHz buffer.

        Args:
            audio_np: np.ndarray of shape [T], mono 16kHz, typically T ~= 16000.

        Returns:
            Dict with "jitter", "intensity" (float), and "audio_latent" (np.ndarray shape (768,)).
        """
        audio = self._to_float_audio(audio_np)

        intensity = self._compute_intensity(audio)
        f0_track = self._compute_f0_track(audio)
        jitter = self._compute_jitter_ppq(f0_track)
        audio_latent = self.get_latent_embeddings(audio_np)

        return {
            "jitter": jitter,
            "intensity": intensity,
            "audio_latent": audio_latent,
        }


__all__ = ["AudioExpert"]



