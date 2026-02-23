import os
import pickle

# Enable MPS fallback for macOS (avoid ops that don't support MPS yet)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# Import your experts
from models.visual_expert import VisualExpert
from models.audio_expert import AudioExpert

# --- Configuration ---
PROC_VIDEO_DIR = Path("Data/processed/video")
RAW_AUDIO_DIR = Path("Data/raw/audio")
FEATURES_DIR = Path("Data/features")
SAMPLE_RATE = 16000
FRAME_STRIDE = 5  # Process every 5th frame
SEQ_LEN = 15  # Temporal sequence length (visual and audio)

FAU_KEYS = ["FAU12", "FAU6", "FAU4"]
VISUAL_LATENT_DIM = 384


def _to_sequence_15(
    latent_frames: List[np.ndarray],
    fau_frames: List[Dict[str, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert lists of per-frame latents (384,) and FAU dicts to fixed (15, 384) and (15, 3).
    If more than 15: take middle 15. If fewer: zero-pad to 15.
    """
    n = len(latent_frames)
    out_latent = np.zeros((SEQ_LEN, VISUAL_LATENT_DIM), dtype=np.float32)
    out_fau = np.zeros((SEQ_LEN, 3), dtype=np.float32)

    if n == 0:
        return out_latent, out_fau

    if n >= SEQ_LEN:
        start = (n - SEQ_LEN) // 2
        for i in range(SEQ_LEN):
            out_latent[i] = latent_frames[start + i]
            d = fau_frames[start + i]
            out_fau[i] = [float(d.get(k, 0.0)) for k in FAU_KEYS]
    else:
        for i in range(n):
            out_latent[i] = latent_frames[i]
            d = fau_frames[i]
            out_fau[i] = [float(d.get(k, 0.0)) for k in FAU_KEYS]
        # rest stays zero (zero-padding)

    return out_latent, out_fau


def extract_video_data(video_path: Path, visual_expert: VisualExpert) -> Dict[str, Any]:
    """
    Extracts temporal sequences: visual_latent (15, 384), visual_fau (15, 3).
    Uses middle-15 sliding window if >15 stride frames; zero-pads if <15.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {
            "visual_fau": np.zeros((SEQ_LEN, 3), dtype=np.float32),
            "visual_latent": np.zeros((SEQ_LEN, VISUAL_LATENT_DIM), dtype=np.float32),
        }

    fau_frames: List[Dict[str, float]] = []
    latent_frames: List[np.ndarray] = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % FRAME_STRIDE == 0:
            summary = visual_expert.get_visual_summary(frame)
            fau_frames.append({k: summary.get(k, 0.0) for k in FAU_KEYS})
            latent_frames.append(visual_expert.get_latent_embeddings(frame))
        frame_idx += 1
    cap.release()

    visual_latent_seq, visual_fau_seq = _to_sequence_15(latent_frames, fau_frames)
    return {
        "visual_fau": visual_fau_seq,
        "visual_latent": visual_latent_seq,
    }

def run_extraction(limit: int = None):
    """Main loop to process video/audio pairs."""
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    
    visual_expert = VisualExpert()
    audio_expert = AudioExpert(sample_rate=SAMPLE_RATE)
    
    mp4_files = sorted(PROC_VIDEO_DIR.glob("*.mp4"))
    if limit:
        print(f"[INFO] Running in TEST MODE. Limiting to {limit} files.")
        mp4_files = mp4_files[:limit]

    print(f"[INFO] Found {len(mp4_files)} files. Starting extraction...")

    for video_path in mp4_files:
        stem = video_path.stem
        audio_path = RAW_AUDIO_DIR / f"{stem}.wav"
        out_path = FEATURES_DIR / f"{stem}.pkl"

        # Check if audio exists
        if not audio_path.exists():
            print(f"[WARN] Missing audio for {stem}, skipping.")
            continue

        print(f"[PROCESS] ID: {stem}")

        # 1. Visual extraction: sequences (15, 384) and (15, 3)
        v_data = extract_video_data(video_path, visual_expert)
        visual_fau_seq = v_data["visual_fau"]   # (15, 3)
        visual_latent_seq = v_data["visual_latent"]  # (15, 384)

        # 2. Audio: latent sequence (15, 768) and geometric (15, 2) [jitter, intensity]
        audio_np = np.fromfile(audio_path, dtype=np.int16)
        audio_float = audio_expert._to_float_audio(audio_np)
        a_data = audio_expert.extract_features(audio_float)
        audio_latent_seq = audio_expert.get_latent_sequence(audio_float, num_steps=SEQ_LEN)  # (15, 768)
        jitter = float(a_data.get("jitter", 0.0))
        intensity = float(a_data.get("intensity", 0.0))
        audio_geo_seq = np.tile([jitter, intensity], (SEQ_LEN, 1)).astype(np.float32)  # (15, 2)

        # 3. Final feature dictionary (temporal sequences)
        combined_features = {
            "id": stem,
            "visual_fau": visual_fau_seq,       # (15, 3)
            "visual_latent": visual_latent_seq,  # (15, 384)
            "audio_latent": audio_latent_seq,    # (15, 768)
            "audio_geometric": audio_geo_seq,    # (15, 2) [jitter, intensity]
        }

        # 4. Save
        with open(out_path, "wb") as f:
            pickle.dump(combined_features, f)

if __name__ == "__main__":
    # Change to None to run all 7442 files after testing
    run_extraction(limit=None)