from __future__ import annotations

from typing import Dict

import numpy as np
import librosa


class AudioExpert:
    """
    AudioExpert for 1-second mono 16kHz chunks.

    For each chunk it computes:
      - RMS energy (normalized to [0, 1]) as intensity
      - Vocal jitter (Pitch Perturbation Quotient) from an F0 track
    """

    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate

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

    def extract_features(self, audio_np: np.ndarray) -> Dict[str, float]:
        """
        Extract normalized intensity and jitter from a 1-second 16kHz buffer.

        Args:
            audio_np: np.ndarray of shape [T], mono 16kHz, typically T ~= 16000.

        Returns:
            Dict[str, float]:
                {
                    "jitter": float in [0, 1],
                    "intensity": float in [0, 1],
                }
        """
        audio = self._to_float_audio(audio_np)

        intensity = self._compute_intensity(audio)
        f0_track = self._compute_f0_track(audio)
        jitter = self._compute_jitter_ppq(f0_track)

        return {
            "jitter": jitter,
            "intensity": intensity,
        }


__all__ = ["AudioExpert"]



