from __future__ import annotations

import os
import pickle
import subprocess
from pathlib import Path
from typing import Dict, List

import cv2
import librosa
import numpy as np

from models.visual_expert import VisualExpert
from models.audio_expert import AudioExpert


RAW_VIDEO_DIR = Path("Data/raw/video")
RAW_AUDIO_DIR = Path("Data/raw/audio")  # Assumed CREMA-D audio location
PROC_VIDEO_DIR = Path("Data/processed/video")
FEATURES_DIR = Path("Data/features")


def ensure_dirs() -> None:
    PROC_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)


def convert_flv_to_mp4() -> None:
    """
    Convert all .flv files in Data/raw/video/ to .mp4 in Data/processed/video/
    using ffmpeg via subprocess.
    """
    if not RAW_VIDEO_DIR.exists():
        print(f"[WARN] RAW_VIDEO_DIR not found: {RAW_VIDEO_DIR}")
        return

    flv_files = sorted(RAW_VIDEO_DIR.glob("*.flv"))
    if not flv_files:
        print(f"[INFO] No .flv files found in {RAW_VIDEO_DIR}")
        return

    print(f"[INFO] Converting {len(flv_files)} .flv files to .mp4 ...")

    for flv_path in flv_files:
        stem = flv_path.stem  # e.g., 1001_DFA_ANG_XX
        out_path = PROC_VIDEO_DIR / f"{stem}.mp4"

        cmd = [
            "ffmpeg",
            "-y",  # overwrite
            "-i",
            str(flv_path),
            str(out_path),
        ]
        print(f"[ffmpeg] {flv_path.name} -> {out_path.name}")
        try:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ffmpeg failed for {flv_path}: {e}")


def aggregate_fau_features(
    frames_fau: List[Dict[str, float]]
) -> Dict[str, float]:
    """
    Aggregate per-frame FAU dictionaries into a single mean FAU dictionary.
    """
    if not frames_fau:
        return {"FAU12": 0.0, "FAU6": 0.0, "FAU4": 0.0}

    keys = ["FAU12", "FAU6", "FAU4"]
    agg: Dict[str, float] = {}
    for k in keys:
        values = [float(f.get(k, 0.0)) for f in frames_fau]
        agg[k] = float(np.mean(values)) if values else 0.0
    return agg


def extract_visual_features(
    video_path: Path, visual_expert: VisualExpert, frame_stride: int = 5
) -> Dict[str, float]:
    """
    Run VisualExpert over a subset of frames and aggregate FAU intensities.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Unable to open video: {video_path}")
        return {"FAU12": 0.0, "FAU6": 0.0, "FAU4": 0.0}

    frame_idx = 0
    fau_frames: List[Dict[str, float]] = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_stride == 0:
            fau = visual_expert.get_fau_intensities(frame)
            fau_frames.append(fau)

        frame_idx += 1

    cap.release()

    return aggregate_fau_features(fau_frames)


def extract_audio_features(
    audio_path: Path, audio_expert: AudioExpert
) -> Dict[str, float]:
    """
    Run AudioExpert over the full audio clip resampled to 16kHz mono.
    """
    if not audio_path.exists():
        print(f"[WARN] Missing audio file: {audio_path}")
        return {"jitter": 0.0, "intensity": 0.0}

    # Load entire clip at 16kHz mono
    audio_np, sr = librosa.load(str(audio_path), sr=audio_expert.sample_rate, mono=True)
    if audio_np.size == 0:
        return {"jitter": 0.0, "intensity": 0.0}

    return audio_expert.extract_features(audio_np)


def process_pairs() -> None:
    """
    For each synchronized ID (e.g., 1001_DFA_ANG_XX), run VisualExpert and AudioExpert
    and save features as a .pkl file in Data/features/ with the ID as the filename.
    """
    visual_expert = VisualExpert()
    audio_expert = AudioExpert(sample_rate=16000)

    mp4_files = sorted(PROC_VIDEO_DIR.glob("*.mp4"))
    if not mp4_files:
        print(f"[INFO] No .mp4 files found in {PROC_VIDEO_DIR}")
        return

    print(f"[INFO] Extracting features for {len(mp4_files)} video/audio pairs ...")

    for video_path in mp4_files:
        stem = video_path.stem  # e.g., 1001_DFA_ANG_XX
        audio_path = RAW_AUDIO_DIR / f"{stem}.wav"
        out_path = FEATURES_DIR / f"{stem}.pkl"

        if out_path.exists():
            print(f"[SKIP] Features already exist for {stem}")
            continue

        print(f"[FEAT] Processing ID: {stem}")

        visual_feats = extract_visual_features(video_path, visual_expert)
        audio_feats = extract_audio_features(audio_path, audio_expert)

        features = {
            "id": stem,
            "visual_fau": visual_feats,
            "audio": audio_feats,
        }

        with open(out_path, "wb") as f:
            pickle.dump(features, f)


def main() -> None:
    ensure_dirs()
    convert_flv_to_mp4()
    process_pairs()


if __name__ == "__main__":
    main()

