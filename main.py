import threading
import signal
import time
from collections import deque, Counter
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import pyaudio
import mediapipe as mp
import torch

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose

from models.visual_expert import VisualExpert
from models.audio_expert import AudioExpert
from models.fusion_head import NuancedStateClassifier, SEQ_LEN
from utils.fusion import classify_nuanced_state

import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

class CameraThread(threading.Thread):
    """
    Captures frames from the default camera at 640x480 and updates a shared container.
    """

    def __init__(
        self,
        frame_lock: threading.Lock,
        frame_container: Dict[str, Any],
        stop_event: threading.Event,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        super().__init__(daemon=True)
        self.frame_lock = frame_lock
        self.frame_container = frame_container
        self.stop_event = stop_event
        self.camera_index = camera_index
        self.width = width
        self.height = height

    def run(self) -> None:
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not cap.isOpened():
            print("[CameraThread] ERROR: Unable to open camera.")
            self.stop_event.set()
            return

        print("[CameraThread] Started.")

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    print("[CameraThread] WARNING: Failed to read frame.")
                    time.sleep(0.01)
                    continue

                with self.frame_lock:
                    self.frame_container["frame"] = frame.copy()

                # Small sleep helps avoid maxing out CPU
                time.sleep(0.005)
        finally:
            cap.release()
            print("[CameraThread] Stopped and camera released.")


class AudioThread(threading.Thread):
    """
    Captures 1-second audio chunks at 16kHz using PyAudio and updates a shared container.
    """

    def __init__(
        self,
        audio_lock: threading.Lock,
        audio_container: Dict[str, Any],
        stop_event: threading.Event,
        rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
    ) -> None:
        super().__init__(daemon=True)
        self.audio_lock = audio_lock
        self.audio_container = audio_container
        self.stop_event = stop_event
        self.rate = rate
        self.channels = channels
        self.chunk_size = chunk_size

    def run(self) -> None:
        pa = pyaudio.PyAudio()
        stream = None

        try:
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=self.channels,
                rate=self.rate,
                input=True,
                frames_per_buffer=self.chunk_size,
            )
            print("[AudioThread] Microphone stream opened at 16kHz.")

            bytes_per_sample = 2 * self.channels  # int16 = 2 bytes

            while not self.stop_event.is_set():
                frames = []
                collected_samples = 0

                # Collect approximately 1 second of audio (rate samples)
                while (
                    collected_samples < self.rate
                    and not self.stop_event.is_set()
                ):
                    data = stream.read(
                        self.chunk_size, exception_on_overflow=False
                    )
                    frames.append(data)
                    collected_samples += len(data) // bytes_per_sample

                if not frames:
                    continue

                audio_bytes = b"".join(frames)

                with self.audio_lock:
                    self.audio_container["chunk"] = audio_bytes
                    self.audio_container["sample_rate"] = self.rate

        except Exception as e:
            print(f"[AudioThread] ERROR: {e}")
            self.stop_event.set()
        finally:
            if stream is not None:
                stream.stop_stream()
                stream.close()
            pa.terminate()
            print("[AudioThread] Stopped and microphone released.")


class InferenceThread(threading.Thread):
    """
    Dummy inference thread that consumes the latest AV data and produces:
      - Nuanced State label (e.g., 'Sarcasm')
      - Conflict Score based on cosine distance between dummy embeddings.
    """

    def __init__(
        self,
        frame_lock: threading.Lock,
        frame_container: Dict[str, Any],
        audio_lock: threading.Lock,
        audio_container: Dict[str, Any],
        visual_expert: VisualExpert,
        audio_expert: AudioExpert,
        state_lock: threading.Lock,
        state_container: Dict[str, Any],
        stop_event: threading.Event,
        control_flags: Optional[Dict[str, Any]] = None,
        interval_s: float = 0.2,
    ) -> None:
        super().__init__(daemon=True)
        self.frame_lock = frame_lock
        self.frame_container = frame_container
        self.audio_lock = audio_lock
        self.audio_container = audio_container
        self.visual_expert = visual_expert
        self.audio_expert = audio_expert
        self.state_lock = state_lock
        self.state_container = state_container
        self.stop_event = stop_event
        self.interval_s = interval_s
        self.control_flags = control_flags if control_flags is not None else {}

        # 1-second rolling buffers for temporal synchrony (FAU vs Audio peak timing)
        self._sync_window_s = 1.0
        self._fau_intensity_buffer: List[Tuple[float, float]] = []  # (timestamp, value)
        self._audio_intensity_buffer: List[Tuple[float, float]] = []  # (timestamp, value)

        # Rolling sequence buffer for temporal context (last SEQ_LEN frames)
        self._seq_visual: deque = deque(maxlen=SEQ_LEN)
        self._seq_audio: deque = deque(maxlen=SEQ_LEN)
        self._seq_geometric: deque = deque(maxlen=SEQ_LEN)

        # Initialize neural fusion head (NuancedStateClassifier) on device.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.classifier = NuancedStateClassifier(
            visual_dim=384,
            audio_dim=768,
            hidden_dim=256,
        ).to(self.device)
        weights_path = "weights/fusion_v2_best.pth"
        try:
            state_dict = torch.load(weights_path, map_location=self.device)
            self.classifier.load_state_dict(state_dict)
            self.classifier.eval()
            print(
                f"[InferenceThread] Loaded fusion weights from {weights_path} on {self.device}."
            )
        except Exception as e:
            print(
                f"[InferenceThread] WARNING: Failed to load weights from {weights_path}: {e}"
            )
            print(
                "[InferenceThread] Using randomly initialized classifier instead."
            )

    @staticmethod
    def _audio_bytes_to_np(audio_bytes: bytes) -> np.ndarray:
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)  # [T]
        return audio_np

    def _update_sync_buffers(self, fau_intensity: float, audio_intensity: float) -> None:
        """Append current FAU and audio intensity to 1s rolling buffers; drop older entries."""
        now = time.monotonic()
        cutoff = now - self._sync_window_s
        self._fau_intensity_buffer.append((now, fau_intensity))
        self._audio_intensity_buffer.append((now, audio_intensity))
        self._fau_intensity_buffer = [(t, v) for t, v in self._fau_intensity_buffer if t >= cutoff]
        self._audio_intensity_buffer = [(t, v) for t, v in self._audio_intensity_buffer if t >= cutoff]

    def check_synchrony(self) -> Tuple[bool, float]:
        """
        Compare peak timing of face vs. voice in the last 1s window.
        Returns (is_incongruent, delay_sec).
        Incongruent when Visual_Peak follows Audio_Peak by more than 300ms.
        """
        INCONGRUENT_DELAY_THRESHOLD_S = 0.3
        if len(self._fau_intensity_buffer) < 2 or len(self._audio_intensity_buffer) < 2:
            return False, 0.0
        # Time of peak (argmax by value) in each buffer
        _, fau_vals = zip(*self._fau_intensity_buffer)
        _, audio_vals = zip(*self._audio_intensity_buffer)
        visual_peak_idx = int(np.argmax(fau_vals))
        audio_peak_idx = int(np.argmax(audio_vals))
        visual_peak_t = self._fau_intensity_buffer[visual_peak_idx][0]
        audio_peak_t = self._audio_intensity_buffer[audio_peak_idx][0]
        delay = visual_peak_t - audio_peak_t  # positive => visual followed audio
        is_incongruent = delay > INCONGRUENT_DELAY_THRESHOLD_S
        return is_incongruent, delay

    @staticmethod
    def _detect_micro_expression(geometric_buffer: np.ndarray) -> bool:
        """
        Detect if a specific FAU (e.g. FAU4 brow lowerer) spiked and vanished
        within the window. geometric_buffer: [T, 5] with cols [fau12, fau6, fau4, jitter, intensity].
        """
        if geometric_buffer is None or geometric_buffer.shape[0] < 3:
            return False
        FAU4_IDX = 2
        SPIKE_THRESHOLD = 0.5
        VANISH_THRESHOLD = 0.35
        fau4 = np.asarray(geometric_buffer[:, FAU4_IDX], dtype=np.float32)
        peak_val = float(np.max(fau4))
        if peak_val < SPIKE_THRESHOLD:
            return False
        peak_idx = int(np.argmax(fau4))
        # Must have "vanished" by the end: last value low, and peak was not at the very last step
        last_val = float(fau4[-1])
        if last_val > VANISH_THRESHOLD:
            return False
        if peak_idx >= fau4.size - 1:
            return False
        return True

    def run(self) -> None:
        print("[InferenceThread] Started AV feature + neural fusion loop.")

        while not self.stop_event.is_set():
            frame: Optional[np.ndarray] = None
            audio_bytes: Optional[bytes] = None

            with self.frame_lock:
                frame = self.frame_container.get("frame", None)

            with self.audio_lock:
                audio_bytes = self.audio_container.get("chunk", None)

                if frame is not None and audio_bytes is not None:
                    try:
                        # Full visual summary (FAU + Pose + Hands for hybrid overrides)
                        visual_summary = self.visual_expert.get_visual_summary(frame)
                        fau = {k: v for k, v in visual_summary.items() if k in ("FAU12", "FAU6", "FAU4")}

                        # Audio features (jitter, intensity) over the 1s chunk
                        audio_np_int16 = self._audio_bytes_to_np(audio_bytes)  # [T]
                        audio_features = self.audio_expert.extract_features(audio_np_int16)

                        fau12 = float(visual_summary.get("FAU12", 0.0))
                        fau6 = float(visual_summary.get("FAU6", 0.0))
                        fau4 = float(visual_summary.get("FAU4", 0.0))
                        jitter = float(audio_features.get("jitter", 0.0))
                        intensity = float(audio_features.get("intensity", 0.0))
                        fau_intensity = max(fau12, fau6, fau4)
                        self._update_sync_buffers(fau_intensity, intensity)
                        synchrony_incongruent, sync_delay_sec = self.check_synchrony()

                        visual_latent = visual_summary.get("visual_latent")
                        audio_latent = audio_features.get("audio_latent")
                        if visual_latent is None:
                            visual_latent = np.zeros(384, dtype=np.float32)
                        if audio_latent is None:
                            audio_latent = np.zeros(768, dtype=np.float32)
                        geometric = np.array(
                            [fau12, fau6, fau4, jitter, intensity],
                            dtype=np.float32,
                        )

                        # Append to sequence buffer (rolling last SEQ_LEN frames)
                        self._seq_visual.append(visual_latent.copy())
                        self._seq_audio.append(audio_latent.copy())
                        self._seq_geometric.append(geometric.copy())

                        # Build [1, seq_len, dim] tensors; pad with zeros if buffer not yet full
                        n = len(self._seq_visual)
                        seq_vis = np.zeros((SEQ_LEN, 384), dtype=np.float32)
                        seq_aud = np.zeros((SEQ_LEN, 768), dtype=np.float32)
                        seq_geo = np.zeros((SEQ_LEN, 5), dtype=np.float32)
                        seq_vis[SEQ_LEN - n:] = np.stack(self._seq_visual)
                        seq_aud[SEQ_LEN - n:] = np.stack(self._seq_audio)
                        seq_geo[SEQ_LEN - n:] = np.stack(self._seq_geometric)

                        # Blindfolds: zero visual or audio when toggled
                        if self.control_flags.get("blind_v", False):
                            seq_vis = np.zeros_like(seq_vis)
                        if self.control_flags.get("blind_a", False):
                            seq_aud = np.zeros_like(seq_aud)

                        # Neural prediction (GMF + LSTM + conflict attention) on sequence
                        pred = self.classifier.predict(seq_vis, seq_aud, seq_geo)

                        # Micro-expression: FAU spike-and-vanish in the geometric buffer
                        micro_expression = self._detect_micro_expression(seq_geo)
                        neural_state = pred.label
                        neural_confidence = pred.confidence
                        conflict_score = pred.conflict_score
                        gate_weight = pred.gate_weight

                        # Hybrid Gate (utils.fusion): physical overrides or rule bypass
                        body_data = {
                            "Shoulder_Rigidity": visual_summary.get("Shoulder_Rigidity", 0.0),
                            "Head_Tilt": visual_summary.get("Head_Tilt_deg", visual_summary.get("Head_Tilt", 0.0)),
                            "Self_Touching_Hands": visual_summary.get("Self_Touching_Hands", False),
                            "Finger_Tapping": visual_summary.get("Finger_Tapping", False),
                            "posture_asymmetry": visual_summary.get("posture_asymmetry", False),
                            "posture_slump": visual_summary.get("posture_slump", False),
                            "is_slumped": visual_summary.get("posture_slump", False),
                            "shoulders_raised": visual_summary.get("shoulders_raised", False),
                            "lean": visual_summary.get("lean", "neutral"),
                        }
                        if self.control_flags.get("use_rules", True):
                            audio_data = {"jitter": jitter, "intensity": intensity}
                            state, confidence, logic_source = classify_nuanced_state(
                                neural_state,
                                neural_confidence,
                                fau,
                                body_data,
                                audio_data,
                                synchrony_incongruent=synchrony_incongruent,
                            )
                        else:
                            state = pred.label
                            confidence = pred.confidence
                            logic_source = "Neural (Rules OFF)"

                        with self.state_lock:
                            self.state_container["state"] = state
                            self.state_container["logic_source"] = logic_source
                            self.state_container["neural_state"] = neural_state
                            self.state_container["finger_tapping"] = body_data.get("Finger_Tapping", False)
                            self.state_container["self_touching_hands"] = body_data.get("Self_Touching_Hands", False)
                            self.state_container["hand_on_chin"] = visual_summary.get("hand_on_chin", False)
                            self.state_container["hand_on_left_temple"] = visual_summary.get("hand_on_left_temple", False)
                            self.state_container["hand_on_right_temple"] = visual_summary.get("hand_on_right_temple", False)
                            self.state_container["hand_covering_mouth"] = visual_summary.get("hand_covering_mouth", False)
                            self.state_container["conflict_score"] = conflict_score
                            self.state_container["confidence"] = confidence
                            self.state_container["fau"] = fau
                            self.state_container["audio_summary"] = {
                                "jitter": jitter,
                                "intensity": intensity,
                            }
                            self.state_container["sync_delay"] = sync_delay_sec
                            self.state_container["micro_expression"] = micro_expression
                            self.state_container["gate_weight"] = gate_weight
                            self.state_container["lean"] = visual_summary.get("lean", "center")
                            self.state_container["is_slumped"] = visual_summary.get("posture_slump", False)
                            self.state_container["shoulders_raised"] = visual_summary.get("shoulders_raised", False)
                            self.state_container["posture_asymmetry"] = visual_summary.get("posture_asymmetry", False)
                            self.state_container["face_landmarks"] = visual_summary.get("face_landmarks")
                            self.state_container["pose_landmarks"] = visual_summary.get("pose_landmarks")
                    except Exception as e:
                        print(f"[InferenceThread] ERROR during neural inference: {e}")

            time.sleep(self.interval_s)

        print("[InferenceThread] Stopped.")


# XAI Dashboard layout (1280x720 canvas, 640x480 frame centered)
CANVAS_W, CANVAS_H = 1280, 720
FRAME_W, FRAME_H = 640, 480
FRAME_X = 320
FRAME_Y = 90

# BGR Color Palette (Dark-mode XAI Dashboard)
BG_COLOR = (15, 10, 10)
PANEL_BG = (36, 26, 26)
BORDER_COLOR = (85, 68, 51)
TEXT_MAIN = (255, 255, 255)
TEXT_DIM = (204, 204, 204)
COLOR_NEURAL = (100, 255, 100)
COLOR_OVERRIDE = (255, 255, 0)
COLOR_WARNING = (50, 50, 255)
COLOR_ATTENTION = (0, 200, 255)


def _draw_sci_fi_brackets(
    canvas: np.ndarray, x: int, y: int, w: int, h: int, thickness: int = 20
) -> None:
    """Draw L-shaped brackets at the 4 corners of a rectangle."""
    t = thickness
    # Top-left L
    cv2.rectangle(canvas, (x, y), (x + t, y + 2 * t), BORDER_COLOR, -1)
    cv2.rectangle(canvas, (x, y), (x + 2 * t, y + t), BORDER_COLOR, -1)
    # Top-right L
    cv2.rectangle(canvas, (x + w - t, y), (x + w, y + 2 * t), BORDER_COLOR, -1)
    cv2.rectangle(canvas, (x + w - 2 * t, y), (x + w, y + t), BORDER_COLOR, -1)
    # Bottom-left L
    cv2.rectangle(canvas, (x, y + h - 2 * t), (x + t, y + h), BORDER_COLOR, -1)
    cv2.rectangle(canvas, (x, y + h - t), (x + 2 * t, y + h), BORDER_COLOR, -1)
    # Bottom-right L
    cv2.rectangle(canvas, (x + w - t, y + h - 2 * t), (x + w, y + h), BORDER_COLOR, -1)
    cv2.rectangle(canvas, (x + w - 2 * t, y + h - t), (x + w, y + h), BORDER_COLOR, -1)


def _draw_top_header(
    canvas: np.ndarray,
    state: Optional[str],
    logic_source: Optional[str],
    confidence: Optional[float],
) -> None:
    """Zone 1: Top header (0-90px)."""
    cv2.rectangle(canvas, (0, 0), (1280, 80), PANEL_BG, -1)
    cv2.line(canvas, (0, 80), (1280, 80), BORDER_COLOR, 1)
    # Left: RHNS title
    cv2.putText(
        canvas,
        "REAL-TIME NUANCED HUMAN STATE RECOGNITION",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_DIM,
        1,
        cv2.LINE_AA,
    )
    # Center: State (large)
    state_text = state or "State: --"
    state_color = COLOR_OVERRIDE if logic_source in ("Rule-Override", "Override", "Posture-Override") else COLOR_NEURAL
    (tw, th), _ = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    tx = (1280 - tw) // 2
    cv2.putText(canvas, state_text, (tx, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, state_color, 2, cv2.LINE_AA)
    # Right: Confidence & Source
    conf_val = (confidence or 0.0) * 100
    src = logic_source or "Neural"
    cv2.putText(canvas, f"CONFIDENCE: {conf_val:.1f}%", (1050, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"SOURCE: {src}", (1050, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)


def _draw_left_panel(
    canvas: np.ndarray,
    fau: Dict[str, float],
    lean: str,
    is_slumped: bool,
    shoulders_raised: bool,
    posture_asymmetry: bool,
    finger_tapping: bool,
    self_touching_hands: bool,
    control_flags: Optional[Dict[str, Any]] = None,
) -> None:
    """Zone 2: Left panel - Visual Cortex (x: 10-300, y: 90-570)."""
    x1, y1, x2, y2 = 10, 90, 300, 570
    cv2.rectangle(canvas, (x1, y1), (x2, y2), PANEL_BG, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BORDER_COLOR, 1)
    left_x = 24
    y_pos = 115
    cv2.putText(canvas, "VISUAL CORTEX & POSTURE", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MAIN, 1, cv2.LINE_AA)
    y_pos += 35
    # FAU bars: "FAU12 [||||      ] 0.45"
    for key, label in [("FAU12", "FAU12"), ("FAU6", "FAU6"), ("FAU4", "FAU4")]:
        v = float(fau.get(key, 0.0))
        v = max(0.0, min(1.0, v))
        bar_w = 140
        fill = int(bar_w * v)
        cv2.putText(canvas, f"{label} [", (left_x, y_pos + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.rectangle(canvas, (left_x + 45, y_pos), (left_x + 45 + bar_w, y_pos + 16), (50, 50, 50), -1)
        if fill > 0:
            cv2.rectangle(canvas, (left_x + 45, y_pos), (left_x + 45 + fill, y_pos + 16), COLOR_NEURAL, -1)
        cv2.rectangle(canvas, (left_x + 45, y_pos), (left_x + 45 + bar_w, y_pos + 16), BORDER_COLOR, 1)
        cv2.putText(canvas, f"] {v:.2f}", (left_x + 45 + bar_w + 4, y_pos + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv2.LINE_AA)
        y_pos += 28
    y_pos += 15
    # Posture Matrix
    lean_upper = (lean or "center").upper()
    cv2.putText(canvas, f"LEAN: {lean_upper}", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_MAIN, 1, cv2.LINE_AA)
    y_pos += 25
    slump_color = COLOR_ATTENTION if is_slumped else TEXT_DIM
    cv2.putText(canvas, f"SHOULDERS SLUMPED: {is_slumped}", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, slump_color, 1, cv2.LINE_AA)
    y_pos += 22
    asym_color = COLOR_OVERRIDE if posture_asymmetry else TEXT_DIM
    cv2.putText(canvas, f"ASYMMETRICAL: {posture_asymmetry}", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, asym_color, 1, cv2.LINE_AA)
    y_pos += 35
    # Gestures
    if finger_tapping or self_touching_hands:
        cv2.putText(canvas, "Gestures:", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        y_pos += 22
        if finger_tapping:
            cv2.putText(canvas, "Finger Tapping", (left_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ATTENTION, 1, cv2.LINE_AA)
            y_pos += 20
        if self_touching_hands:
            cv2.putText(canvas, "Hand to Face", (left_x + 10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLOR_ATTENTION, 1, cv2.LINE_AA)

    # Control flag warnings (bottom-right of left panel)
    cf = control_flags or {}
    warn_y = y2 - 75
    if not cf.get("use_rules", True):
        cv2.putText(canvas, "RULES: BYPASSED", (left_x, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARNING, 1, cv2.LINE_AA)
        warn_y += 22
    if cf.get("blind_v", False):
        cv2.putText(canvas, "VISUAL SENSOR: OFFLINE", (left_x, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARNING, 1, cv2.LINE_AA)
        warn_y += 22
    if cf.get("blind_a", False):
        cv2.putText(canvas, "AUDIO SENSOR: OFFLINE", (left_x, warn_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARNING, 1, cv2.LINE_AA)


def _draw_right_panel(
    canvas: np.ndarray,
    audio_summary: Dict[str, float],
    sync_delay_sec: Optional[float],
) -> None:
    """Zone 3: Right panel - Audio Cortex (x: 980-1270, y: 90-570)."""
    x1, y1, x2, y2 = 980, 90, 1270, 570
    cv2.rectangle(canvas, (x1, y1), (x2, y2), PANEL_BG, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BORDER_COLOR, 1)
    rx = 1000
    y_pos = 115
    cv2.putText(canvas, "AUDIO CORTEX", (rx, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.55, TEXT_MAIN, 1, cv2.LINE_AA)
    y_pos += 40
    # VU meter (vertical) for intensity
    intensity = float(audio_summary.get("intensity", 0.0))
    intensity = max(0.0, min(1.0, intensity))
    vu_w, vu_h = 30, 120
    vu_x, vu_y = rx, y_pos
    cv2.rectangle(canvas, (vu_x, vu_y), (vu_x + vu_w, vu_y + vu_h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (vu_x, vu_y), (vu_x + vu_w, vu_y + vu_h), BORDER_COLOR, 1)
    fill_h = int(vu_h * intensity)
    if fill_h > 0:
        # Gradient: green (bottom) -> yellow (middle) -> red (top)
        third = vu_h // 3
        # Draw segments from bottom up
        seg1_h = min(fill_h, third)
        if seg1_h > 0:
            cv2.rectangle(canvas, (vu_x + 2, vu_y + vu_h - seg1_h), (vu_x + vu_w - 2, vu_y + vu_h), (0, 255, 0), -1)
        if fill_h > third:
            seg2_h = min(fill_h - third, third)
            cv2.rectangle(canvas, (vu_x + 2, vu_y + vu_h - third - seg2_h), (vu_x + vu_w - 2, vu_y + vu_h - third), (0, 255, 255), -1)
        if fill_h > 2 * third:
            seg3_h = fill_h - 2 * third
            cv2.rectangle(canvas, (vu_x + 2, vu_y + vu_h - fill_h), (vu_x + vu_w - 2, vu_y + vu_h - 2 * third), (0, 0, 255), -1)
    cv2.putText(canvas, "Intensity", (vu_x, vu_y + vu_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
    y_pos += vu_h + 45
    # Jitter bar (horizontal)
    jitter = float(audio_summary.get("jitter", 0.0))
    jitter = max(0.0, min(1.0, jitter))
    jw, jh = 220, 14
    cv2.putText(canvas, "Jitter", (rx, y_pos - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (rx, y_pos), (rx + jw, y_pos + jh), (30, 30, 30), -1)
    cv2.rectangle(canvas, (rx, y_pos), (rx + jw, y_pos + jh), COLOR_WARNING if jitter > 0.6 else BORDER_COLOR, 2 if jitter > 0.6 else 1)
    fill_w = int(jw * jitter)
    if fill_w > 0:
        c = COLOR_WARNING if jitter > 0.6 else (0, 200, 255)
        cv2.rectangle(canvas, (rx, y_pos), (rx + fill_w, y_pos + jh), c, -1)
    y_pos += 45
    # Sync Delay slider
    delay = float(sync_delay_sec) if sync_delay_sec is not None else 0.0
    delay = max(-0.5, min(0.5, delay))
    sync_w, sync_h = 220, 36
    center_x = rx + sync_w // 2
    cv2.putText(canvas, "AV Sync", (rx, y_pos - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (rx, y_pos), (rx + sync_w, y_pos + sync_h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (rx, y_pos), (rx + sync_w, y_pos + sync_h), BORDER_COLOR, 1)
    cv2.line(canvas, (center_x, y_pos), (center_x, y_pos + sync_h), TEXT_DIM, 1)
    marker_x = int(center_x + delay * (sync_w * 0.9))
    marker_x = max(rx + 2, min(rx + sync_w - 2, marker_x))
    mcolor = COLOR_NEURAL if abs(delay) <= 0.2 else COLOR_WARNING if abs(delay) > 0.3 else COLOR_ATTENTION
    cv2.line(canvas, (marker_x, y_pos), (marker_x, y_pos + sync_h), mcolor, 2)
    ms = int(round(delay * 1000))
    delay_str = f"+{ms}ms" if ms >= 0 else f"{ms}ms"
    cv2.putText(canvas, delay_str, (rx + sync_w // 2 - 20, y_pos + sync_h + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)


def _draw_bottom_panel(
    canvas: np.ndarray,
    gate_weight: float,
    conflict_val: float,
    fps: Optional[float],
) -> None:
    """Zone 4: Bottom panel - Fusion Engine (x: 10-1270, y: 580-710)."""
    x1, y1, x2, y2 = 10, 580, 1270, 710
    cv2.rectangle(canvas, (x1, y1), (x2, y2), PANEL_BG, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), BORDER_COLOR, 1)
    col_w = (x2 - x1) // 3
    # Col 1: Gate Weight
    gw_x, gw_y = 30, 600
    gw_w = 350
    cv2.putText(canvas, "AUDIO DOMINANT", (gw_x, gw_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, "VISUAL DOMINANT", (gw_x + gw_w - 100, gw_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (gw_x, gw_y), (gw_x + gw_w, gw_y + 20), (30, 30, 30), -1)
    cv2.rectangle(canvas, (gw_x, gw_y), (gw_x + gw_w, gw_y + 20), BORDER_COLOR, 1)
    g = max(0.0, min(1.0, gate_weight))
    cx = gw_x + int(gw_w * g)
    cv2.line(canvas, (cx, gw_y), (cx, gw_y + 20), COLOR_OVERRIDE, 2)
    # Col 2: Conflict Engine
    cx_x, cx_y = 420, 595
    cx_w, cx_h = 400, 30
    cv2.putText(canvas, "CONFLICT ENGINE", (cx_x, cx_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_DIM, 1, cv2.LINE_AA)
    border_c = COLOR_WARNING if conflict_val > 0.6 else BORDER_COLOR
    cv2.rectangle(canvas, (cx_x, cx_y), (cx_x + cx_w, cx_y + cx_h), (30, 30, 30), -1)
    cv2.rectangle(canvas, (cx_x, cx_y), (cx_x + cx_w, cx_y + cx_h), border_c, 3 if conflict_val > 0.6 else 1)
    fill = int(cx_w * conflict_val)
    if fill > 0:
        c = COLOR_WARNING if conflict_val > 0.6 else COLOR_ATTENTION if conflict_val > 0.3 else COLOR_NEURAL
        cv2.rectangle(canvas, (cx_x, cx_y), (cx_x + fill, cx_y + cx_h), c, -1)
    if conflict_val > 0.6:
        cv2.putText(canvas, "CONTRADICTION DETECTED", (cx_x, cx_y + cx_h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WARNING, 1, cv2.LINE_AA)
    # Col 3: Telemetry
    tel_x = 860
    fps_str = f"{fps:.1f}" if fps is not None and fps > 0 else "N/A"
    cv2.putText(canvas, f"FPS: {fps_str}", (tel_x, 615), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, "TEMPORAL BUFFER: 15/15 FRAMES", (tel_x, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(canvas, "SMOOTHING: ACTIVE (SMA-5)", (tel_x, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_DIM, 1, cv2.LINE_AA)


def _draw_video_enhancements(
    canvas: np.ndarray,
    detections,
    width: int,
    height: int,
    conflict_val: float,
    micro_expression: bool,
) -> None:
    """Zone 5: Center video - face bbox and micro-expression banner."""
    box_color = COLOR_WARNING if conflict_val > 0.6 else BORDER_COLOR
    if detections:
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            x_min = FRAME_X + int(bbox.xmin * width)
            y_min = FRAME_Y + int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)
            x_min = max(FRAME_X, x_min)
            y_min = max(FRAME_Y, y_min)
            x_max = min(FRAME_X + FRAME_W - 1, x_min + w)
            y_max = min(FRAME_Y + FRAME_H - 1, y_min + h)
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), box_color, 1)
    if micro_expression:
        banner_y = FRAME_Y + FRAME_H - 28
        cv2.rectangle(canvas, (FRAME_X, banner_y), (FRAME_X + FRAME_W, FRAME_Y + FRAME_H), (0, 0, 0), -1)
        cv2.putText(
            canvas,
            "[!] MICRO-EXPRESSION DETECTED",
            (FRAME_X + 80, banner_y + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )


def draw_hud(
    frame: np.ndarray,
    state: Optional[str],
    logic_source: Optional[str],
    neural_state: Optional[str],
    finger_tapping: bool,
    self_touching_hands: bool,
    show_analyzing: bool,
    show_masking: bool,
    show_bored: bool,
    show_micro_expression: bool,
    conflict_score: Optional[float],
    fau: Optional[Dict[str, float]],
    audio_summary: Optional[Dict[str, float]],
    confidence: Optional[float],
    sync_delay_sec: Optional[float],
    detections,
    width: int,
    height: int,
    gate_weight: float = 0.5,
    lean: str = "center",
    is_slumped: bool = False,
    shoulders_raised: bool = False,
    posture_asymmetry: bool = False,
    fps: Optional[float] = None,
    show_landmarks: bool = False,
    face_landmarks: Any = None,
    pose_landmarks: Any = None,
    control_flags: Optional[Dict[str, Any]] = None,
    show_roi: bool = False,
) -> np.ndarray:
    """
    XAI Dashboard v2: 1280x720 dark-mode canvas. Zones: Top header, Left (Visual Cortex),
    Right (Audio Cortex), Bottom (Fusion Engine), Center (video + enhancements).
    """
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = BG_COLOR

    # Place live frame centered at (320, 90)
    if (frame.shape[1], frame.shape[0]) != (FRAME_W, FRAME_H):
        frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))
    else:
        frame_small = frame.copy()

    # Draw landmarks on the 640x480 video frame (before placing on canvas)
    if show_landmarks:
        video_frame = frame_small
        if face_landmarks:
            for face_lm in face_landmarks:
                mp_drawing.draw_landmarks(
                    image=video_frame,
                    landmark_list=face_lm,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style(),
                )
        if pose_landmarks:
            mp_drawing.draw_landmarks(
                image=video_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
            )

    # ROI PIP: face crop as grayscale machine-vision feed in bottom-right of video
    if show_roi and detections:
        fh, fw = frame_small.shape[:2]
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * fw)
            y_min = int(bbox.ymin * fh)
            w = int(bbox.width * fw)
            h = int(bbox.height * fh)
            x_min = max(0, min(frame_small.shape[1] - 1, x_min))
            y_min = max(0, min(frame_small.shape[0] - 1, y_min))
            w = max(1, min(frame_small.shape[1] - x_min, w))
            h = max(1, min(frame_small.shape[0] - y_min, h))
            face_crop = frame_small[y_min : y_min + h, x_min : x_min + w]
            if face_crop.size > 0:
                roi_resized = cv2.resize(face_crop, (112, 112))
                roi_gray = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2GRAY)
                roi_bgr = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)
                # Overlay in bottom-right of 640x480 video
                pip_x = FRAME_W - 112 - 12
                pip_y = FRAME_H - 112 - 12
                frame_small[pip_y : pip_y + 112, pip_x : pip_x + 112] = roi_bgr
            break  # Only first face

    canvas[FRAME_Y : FRAME_Y + FRAME_H, FRAME_X : FRAME_X + FRAME_W] = frame_small

    # Sci-fi brackets around video
    _draw_sci_fi_brackets(canvas, FRAME_X, FRAME_Y, FRAME_W, FRAME_H)

    fau = fau or {}
    audio_summary = audio_summary or {}
    conflict_val = float(conflict_score) if conflict_score is not None else 0.0
    conflict_val = max(0.0, min(1.0, conflict_val))

    # Zone 1: Top Header
    _draw_top_header(canvas, state, logic_source, confidence)

    # Zone 2: Left Panel (Visual Cortex)
    _draw_left_panel(
        canvas, fau, lean or "center", is_slumped, shoulders_raised, posture_asymmetry,
        finger_tapping, self_touching_hands,
        control_flags=control_flags,
    )

    # Zone 3: Right Panel (Audio Cortex)
    _draw_right_panel(canvas, audio_summary, sync_delay_sec)

    # Zone 4: Bottom Panel (Fusion Engine)
    _draw_bottom_panel(canvas, gate_weight, conflict_val, fps)

    # Zone 5: Center Video Enhancements
    _draw_video_enhancements(canvas, detections, width, height, conflict_val, show_micro_expression)

    return canvas


def _draw_sync_visualizer(
    hud_frame: np.ndarray,
    sync_delay_sec: Optional[float],
    width: int,
    height: int,
) -> None:
    """Draw SYNC box (200x50) at bottom left: zero line center, sliding marker, label with delay ms."""
    box_w, box_h = 200, 50
    left = 10
    bottom = height - 22  # above Logic Source line
    top = bottom - box_h
    right = left + box_w
    center_x = left + box_w // 2

    # Box background (dark)
    cv2.rectangle(hud_frame, (left, top), (right, bottom), (40, 40, 40), -1)
    cv2.rectangle(hud_frame, (left, top), (right, bottom), (180, 180, 180), 1)

    # Zero line (perfect synchrony) in center
    cv2.line(
        hud_frame,
        (center_x, top),
        (center_x, bottom),
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    # Map delay_sec to x: positive = face lags = right; negative = voice lags = left
    # Scale: ±0.5s -> full half-width (100 px)
    delay = float(sync_delay_sec) if sync_delay_sec is not None else 0.0
    delay = max(-0.5, min(0.5, delay))
    marker_x = int(center_x + delay * 200)  # 200 px per second
    marker_x = max(left + 2, min(right - 2, marker_x))

    # Color: Green synchronized (-0.2 to 0.2), Red high lag (>0.3)
    abs_delay = abs(delay)
    if abs_delay <= 0.2:
        marker_color = (0, 255, 0)  # Green BGR
    elif abs_delay > 0.3:
        marker_color = (0, 0, 255)  # Red BGR
    else:
        marker_color = (0, 165, 255)  # Orange BGR

    # Sliding marker (vertical line)
    cv2.line(
        hud_frame,
        (marker_x, top),
        (marker_x, bottom),
        marker_color,
        2,
        cv2.LINE_AA,
    )

    # Label: SYNC + delay in ms (e.g. +340ms, -120ms)
    ms = int(round(delay * 1000))
    delay_str = f"+{ms}ms" if ms >= 0 else f"{ms}ms"
    cv2.putText(
        hud_frame,
        f"SYNC {delay_str}",
        (left, top - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return


def main() -> None:
    stop_event = threading.Event()

    frame_lock = threading.Lock()
    audio_lock = threading.Lock()
    state_lock = threading.Lock()

    frame_container: Dict[str, Any] = {}
    audio_container: Dict[str, Any] = {}
    state_container: Dict[str, Any] = {
        "state": None,
        "logic_source": None,
        "neural_state": None,
        "finger_tapping": None,
        "self_touching_hands": None,
        "hand_on_chin": False,
        "hand_on_left_temple": False,
        "hand_on_right_temple": False,
        "hand_covering_mouth": False,
        "conflict_score": None,
        "fau": None,
        "audio_summary": None,
        "confidence": None,
        "sync_delay": 0.0,
        "micro_expression": False,
        "gate_weight": 0.5,
        "lean": "center",
        "is_slumped": False,
        "shoulders_raised": False,
        "posture_asymmetry": False,
    }

    control_flags: Dict[str, Any] = {
        "use_rules": True,
        "blind_v": False,
        "blind_a": False,
    }

    visual_expert = VisualExpert()
    audio_expert = AudioExpert(sample_rate=16000)

    camera_thread = CameraThread(frame_lock, frame_container, stop_event)
    audio_thread = AudioThread(audio_lock, audio_container, stop_event)
    inference_thread = InferenceThread(
        frame_lock,
        frame_container,
        audio_lock,
        audio_container,
        visual_expert,
        audio_expert,
        state_lock,
        state_container,
        stop_event,
        control_flags=control_flags,
    )

    camera_thread.start()
    audio_thread.start()
    inference_thread.start()

    print("[Main] RHNS started. Press 'q' to quit.")

    mp_face_detection = mp.solutions.face_detection

    def handle_sigint(signum, frame):
        print("\n[Main] Caught SIGINT. Shutting down...")
        stop_event.set()

    signal.signal(signal.SIGINT, handle_sigint)

    fps_target = 30.0
    frame_interval = 1.0 / fps_target

    # Persistence counters for gesture labels (only show after >10 consecutive frames)
    GESTURE_PERSIST_FRAMES = 10
    analyzing_count = 0
    masking_count = 0
    bored_count = 0

    # SMA for sync_delay (window=5) to smooth the visual marker
    sync_delay_sma_buffer: deque = deque(maxlen=5)

    # Temporal smoothing buffers for XAI Dashboard (reduce flicker)
    SMOOTHING_WINDOW = 5  # Roughly 1 second of inference data
    state_buffer = deque(maxlen=SMOOTHING_WINDOW)
    neural_state_buffer = deque(maxlen=SMOOTHING_WINDOW)
    gate_buffer = deque(maxlen=SMOOTHING_WINDOW)
    conflict_buffer = deque(maxlen=SMOOTHING_WINDOW)
    confidence_buffer = deque(maxlen=SMOOTHING_WINDOW)

    current_fps: float = 0.0
    show_landmarks: bool = False
    show_roi: bool = False

    ui_face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    )
    ui_pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    with mp_face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.5
    ) as face_detection:
        try:
            last_time = time.perf_counter()
            while not stop_event.is_set():
                with frame_lock:
                    frame = frame_container.get("frame", None)

                if frame is None:
                    time.sleep(0.005)
                    continue

                height, width = frame.shape[:2]

                # Face detection uses RGB images
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detection.process(rgb_frame)
                detections = (
                    results.detections if results.detections else []
                )

                with state_lock:
                    state = state_container.get("state", None)
                    logic_source = state_container.get("logic_source", None)
                    neural_state = state_container.get("neural_state", None)
                    finger_tapping = bool(state_container.get("finger_tapping", False))
                    self_touching_hands = bool(state_container.get("self_touching_hands", False))
                    hand_on_chin = bool(state_container.get("hand_on_chin", False))
                    hand_on_temple = bool(
                        state_container.get("hand_on_left_temple", False)
                        or state_container.get("hand_on_right_temple", False)
                    )
                    hand_covering_mouth = bool(state_container.get("hand_covering_mouth", False))
                    conflict_score = state_container.get(
                        "conflict_score", None
                    )
                    fau = state_container.get("fau", None)
                    audio_summary = state_container.get("audio_summary", None)
                    confidence = state_container.get("confidence", None)
                    sync_delay_sec = state_container.get("sync_delay", 0.0)
                    micro_expression = bool(state_container.get("micro_expression", False))
                    gate_weight = state_container.get("gate_weight", 0.5)
                    lean = state_container.get("lean", "center")
                    is_slumped = bool(state_container.get("is_slumped", False))
                    shoulders_raised = bool(state_container.get("shoulders_raised", False))
                    posture_asymmetry = bool(state_container.get("posture_asymmetry", False))

                # Real-time UI landmarks (decoupled from inference; zero-latency when toggle ON)
                face_landmarks = None
                pose_landmarks = None
                if show_landmarks:
                    rgb_vid = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    face_results = ui_face_mesh.process(rgb_vid)
                    pose_results = ui_pose.process(rgb_vid)
                    if face_results.multi_face_landmarks:
                        face_landmarks = face_results.multi_face_landmarks
                    if pose_results.pose_landmarks:
                        pose_landmarks = pose_results.pose_landmarks

                # Temporal smoothing: append raw values to buffers (only if not None)
                if state is not None:
                    state_buffer.append(state)
                if neural_state is not None:
                    neural_state_buffer.append(neural_state)
                if gate_weight is not None:
                    gate_buffer.append(float(gate_weight))
                if conflict_score is not None:
                    conflict_buffer.append(float(conflict_score))
                if confidence is not None:
                    confidence_buffer.append(float(confidence))

                # Majority vote for categorical values
                smoothed_state = Counter(state_buffer).most_common(1)[0][0] if state_buffer else None
                smoothed_neural_state = Counter(neural_state_buffer).most_common(1)[0][0] if neural_state_buffer else None

                # SMA for numerical values
                smoothed_gate = sum(gate_buffer) / len(gate_buffer) if gate_buffer else 0.5
                smoothed_conflict = sum(conflict_buffer) / len(conflict_buffer) if conflict_buffer else None
                smoothed_confidence = sum(confidence_buffer) / len(confidence_buffer) if confidence_buffer else None

                # Apply SMA to sync_delay for smooth visual marker (window=5 frames)
                sync_delay_sma_buffer.append(
                    float(sync_delay_sec) if sync_delay_sec is not None else 0.0
                )
                sync_delay_smoothed = sum(sync_delay_sma_buffer) / len(sync_delay_sma_buffer)

                # Update persistence: show label only if condition held > GESTURE_PERSIST_FRAMES
                if hand_on_temple:
                    analyzing_count = min(analyzing_count + 1, GESTURE_PERSIST_FRAMES + 1)
                else:
                    analyzing_count = 0
                if hand_covering_mouth:
                    masking_count = min(masking_count + 1, GESTURE_PERSIST_FRAMES + 1)
                else:
                    masking_count = 0
                if hand_on_chin:
                    bored_count = min(bored_count + 1, GESTURE_PERSIST_FRAMES + 1)
                else:
                    bored_count = 0
                show_analyzing = analyzing_count > GESTURE_PERSIST_FRAMES
                show_masking = masking_count > GESTURE_PERSIST_FRAMES
                show_bored = bored_count > GESTURE_PERSIST_FRAMES

                hud_frame = draw_hud(
                    frame,
                    smoothed_state,
                    logic_source,
                    smoothed_neural_state,
                    finger_tapping,
                    self_touching_hands,
                    show_analyzing,
                    show_masking,
                    show_bored,
                    micro_expression,
                    smoothed_conflict,
                    fau,
                    audio_summary,
                    smoothed_confidence,
                    sync_delay_smoothed,
                    detections,
                    width,
                    height,
                    gate_weight=smoothed_gate,
                    lean=lean,
                    is_slumped=is_slumped,
                    shoulders_raised=shoulders_raised,
                    posture_asymmetry=posture_asymmetry,
                    fps=current_fps,
                    show_landmarks=show_landmarks,
                    face_landmarks=face_landmarks,
                    pose_landmarks=pose_landmarks,
                    control_flags=control_flags,
                    show_roi=show_roi,
                )

                cv2.imshow("RHNS - Nuanced State HUD", hud_frame)

                # Maintain ~30 FPS
                elapsed = time.perf_counter() - last_time
                current_fps = 1.0 / elapsed if elapsed > 0 else 0.0
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_time = time.perf_counter()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[Main] 'q' or ESC pressed. Exiting...")
                    stop_event.set()
                elif key == ord("l"):
                    show_landmarks = not show_landmarks
                    print(f"[Main] Landmarks overlay: {'ON' if show_landmarks else 'OFF'}")
                elif key == ord("r"):
                    control_flags["use_rules"] = not control_flags["use_rules"]
                elif key == ord("v"):
                    control_flags["blind_v"] = not control_flags["blind_v"]
                elif key == ord("a"):
                    control_flags["blind_a"] = not control_flags["blind_a"]
                elif key == ord("c"):
                    show_roi = not show_roi
        finally:
            stop_event.set()
            camera_thread.join(timeout=2.0)
            audio_thread.join(timeout=2.0)
            inference_thread.join(timeout=2.0)
            ui_face_mesh.close()
            ui_pose.close()
            cv2.destroyAllWindows()
            print("[Main] Clean shutdown complete.")


if __name__ == "__main__":
    main()
