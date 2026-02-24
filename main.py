import threading
import signal
import time
from collections import deque
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import pyaudio
import mediapipe as mp
import torch

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

                        # Neural prediction (GMF + LSTM + conflict attention) on sequence
                        pred = self.classifier.predict(seq_vis, seq_aud, seq_geo)

                        # Micro-expression: FAU spike-and-vanish in the geometric buffer
                        micro_expression = self._detect_micro_expression(seq_geo)
                        neural_state = pred.label
                        neural_confidence = pred.confidence
                        conflict_score = pred.conflict_score
                        gate_weight = pred.gate_weight

                        # Hybrid Gate (utils.fusion): physical overrides
                        body_data = {
                            "Shoulder_Rigidity": visual_summary.get("Shoulder_Rigidity", 0.0),
                            "Head_Tilt": visual_summary.get("Head_Tilt", 0.0),
                            "Self_Touching_Hands": visual_summary.get("Self_Touching_Hands", False),
                            "Finger_Tapping": visual_summary.get("Finger_Tapping", False),
                        }
                        audio_data = {"jitter": jitter, "intensity": intensity}
                        state, confidence, logic_source = classify_nuanced_state(
                            neural_state,
                            neural_confidence,
                            fau,
                            body_data,
                            audio_data,
                            synchrony_incongruent=synchrony_incongruent,
                        )

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
                    except Exception as e:
                        print(f"[InferenceThread] ERROR during neural inference: {e}")

            time.sleep(self.interval_s)

        print("[InferenceThread] Stopped.")


# XAI Dashboard layout (1280x720 canvas, 640x480 frame centered)
CANVAS_W, CANVAS_H = 1280, 720
FRAME_W, FRAME_H = 640, 480
FRAME_X = (CANVAS_W - FRAME_W) // 2
FRAME_Y = (CANVAS_H - FRAME_H) // 2


def _draw_progress_bar(
    canvas: np.ndarray,
    x: int, y: int, w: int, h: int,
    value: float,
    label: str,
    color_fill: Tuple[int, int, int] = (0, 180, 0),
) -> None:
    """Draw a horizontal progress bar with label. value in [0, 1]."""
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (60, 60, 60), -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (180, 180, 180), 1)
    fill_w = max(0, min(w, int(w * max(0.0, min(1.0, value)))))
    if fill_w > 0:
        cv2.rectangle(canvas, (x, y), (x + fill_w, y + h), color_fill, -1)
    cv2.putText(canvas, label, (x, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


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
) -> np.ndarray:
    """
    XAI Dashboard: 1280x720 canvas, camera frame centered. Left = Visual evidences,
    Right = Audio evidences, Bottom = Fusion brain (gate, conflict, sync), Top = Verdict.
    """
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    # Place live frame in center
    if (frame.shape[1], frame.shape[0]) != (FRAME_W, FRAME_H):
        frame_small = cv2.resize(frame, (FRAME_W, FRAME_H))
    else:
        frame_small = frame
    canvas[FRAME_Y : FRAME_Y + FRAME_H, FRAME_X : FRAME_X + FRAME_W] = frame_small

    fau = fau or {}
    audio_summary = audio_summary or {}
    conflict_val = float(conflict_score) if conflict_score is not None else 0.0
    conflict_val = max(0.0, min(1.0, conflict_val))

    # --- Top center: Verdict (large) ---
    state_text = state or "State: --"
    if logic_source in ("Rule-Override", "Override"):
        state_color = (255, 255, 0)  # Cyan BGR
    else:
        state_color = (0, 255, 0)  # Green BGR
    (tw, th), _ = cv2.getTextSize(state_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)
    tx = (CANVAS_W - tw) // 2
    ty = FRAME_Y - 20
    cv2.putText(canvas, state_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, state_color, 2, cv2.LINE_AA)

    # --- Center: Face bbox (frame coords + offset to canvas) + MICRO-EXPRESSION ---
    if conflict_val < 0.3:
        box_color = (0, 255, 0)
    elif conflict_val < 0.6:
        box_color = (0, 165, 255)
    else:
        box_color = (0, 0, 255)
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
            cv2.rectangle(canvas, (x_min, y_min), (x_max, y_max), box_color, 2)
            if show_micro_expression:
                cv2.putText(
                    canvas, "MICRO-EXPRESSION",
                    (x_min, max(FRAME_Y + 10, y_min - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA,
                )

    # --- Left panel: Visual Evidences ---
    left_x = 24
    bar_w, bar_h = 220, 14
    y_pos = 80
    cv2.putText(canvas, "Visual Evidences", (left_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    _draw_progress_bar(canvas, left_x, y_pos, bar_w, bar_h, float(fau.get("FAU12", 0.0)), "FAU12 (Lip)", (0, 200, 0))
    y_pos += 28
    _draw_progress_bar(canvas, left_x, y_pos, bar_w, bar_h, float(fau.get("FAU6", 0.0)), "FAU6 (Cheek)", (0, 200, 0))
    y_pos += 28
    _draw_progress_bar(canvas, left_x, y_pos, bar_w, bar_h, float(fau.get("FAU4", 0.0)), "FAU4 (Brow)", (0, 200, 0))
    y_pos += 36
    cv2.putText(canvas, "Body language:", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)
    y_pos += 22
    if self_touching_hands:
        cv2.putText(canvas, "Hand to face", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1, cv2.LINE_AA)
        y_pos += 20
    if finger_tapping:
        cv2.putText(canvas, "Tapping", (left_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1, cv2.LINE_AA)

    # --- Right panel: Audio Evidences ---
    right_x = CANVAS_W - 24 - 220
    y_pos = 80
    cv2.putText(canvas, "Audio Evidences", (right_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    _draw_progress_bar(canvas, right_x, y_pos, bar_w, bar_h, float(audio_summary.get("intensity", 0.0)), "Voice Intensity", (200, 100, 0))
    y_pos += 28
    _draw_progress_bar(canvas, right_x, y_pos, bar_w, bar_h, float(audio_summary.get("jitter", 0.0)), "Vocal Jitter", (200, 100, 0))

    # --- Bottom panel: Fusion Brain (gate, conflict, sync) ---
    bottom_y = CANVAS_H - 95
    cv2.putText(canvas, "Fusion Brain", (CANVAS_W // 2 - 60, bottom_y - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    # Gate weight: horizontal gauge (0 = Voice, 1 = Face)
    gw_x, gw_y = 24, bottom_y
    gw_w = 400
    cv2.putText(canvas, "Gate (Face vs Voice)", (gw_x, gw_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    gw_y += 18
    cv2.rectangle(canvas, (gw_x, gw_y), (gw_x + gw_w, gw_y + 16), (50, 50, 50), -1)
    cv2.rectangle(canvas, (gw_x, gw_y), (gw_x + gw_w, gw_y + 16), (150, 150, 150), 1)
    g = max(0.0, min(1.0, gate_weight))
    cx = gw_x + int(gw_w * g)
    cv2.line(canvas, (cx, gw_y), (cx, gw_y + 16), (0, 255, 255), 2)
    cv2.putText(canvas, "Voice", (gw_x, gw_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Face", (gw_x + gw_w - 32, gw_y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    # Conflict score bar
    cx_x, cx_y = gw_x + gw_w + 40, bottom_y + 18
    cx_w = 180
    cv2.putText(canvas, "Conflict Score", (cx_x, cx_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (cx_x, cx_y), (cx_x + cx_w, cx_y + 16), (50, 50, 50), -1)
    cv2.rectangle(canvas, (cx_x, cx_y), (cx_x + cx_w, cx_y + 16), (150, 150, 150), 1)
    fill = int(cx_w * conflict_val)
    if fill > 0:
        ccolor = (0, 0, 255) if conflict_val > 0.6 else (0, 165, 255) if conflict_val > 0.3 else (0, 255, 0)
        cv2.rectangle(canvas, (cx_x, cx_y), (cx_x + fill, cx_y + 16), ccolor, -1)
    # Sync delay (slider in bottom panel)
    sync_x, sync_y = cx_x + cx_w + 40, bottom_y
    sync_w, sync_h = 200, 36
    delay = float(sync_delay_sec) if sync_delay_sec is not None else 0.0
    delay = max(-0.5, min(0.5, delay))
    center_sync = sync_x + sync_w // 2
    cv2.rectangle(canvas, (sync_x, sync_y), (sync_x + sync_w, sync_y + sync_h), (40, 40, 40), -1)
    cv2.rectangle(canvas, (sync_x, sync_y), (sync_x + sync_w, sync_y + sync_h), (180, 180, 180), 1)
    cv2.line(canvas, (center_sync, sync_y), (center_sync, sync_y + sync_h), (200, 200, 200), 1)
    marker_x = int(center_sync + delay * (sync_w * 0.9))
    marker_x = max(sync_x + 2, min(sync_x + sync_w - 2, marker_x))
    mcolor = (0, 255, 0) if abs(delay) <= 0.2 else (0, 0, 255) if abs(delay) > 0.3 else (0, 165, 255)
    cv2.line(canvas, (marker_x, sync_y), (marker_x, sync_y + sync_h), mcolor, 2)
    ms = int(round(delay * 1000))
    delay_str = f"+{ms}ms" if ms >= 0 else f"{ms}ms"
    cv2.putText(canvas, f"Sync {delay_str}", (sync_x, sync_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

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
    )

    camera_thread.start()
    audio_thread.start()
    inference_thread.start()

    print("[Main] RHNS v1.0 started. Press 'q' to quit.")

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

                gate_weight = state_container.get("gate_weight", 0.5)
                hud_frame = draw_hud(
                    frame,
                    state,
                    logic_source,
                    neural_state,
                    finger_tapping,
                    self_touching_hands,
                    show_analyzing,
                    show_masking,
                    show_bored,
                    micro_expression,
                    conflict_score,
                    fau,
                    audio_summary,
                    confidence,
                    sync_delay_smoothed,
                    detections,
                    width,
                    height,
                    gate_weight=gate_weight,
                )

                cv2.imshow("RHNS v1.0 - Nuanced State HUD", hud_frame)

                # Maintain ~30 FPS
                elapsed = time.perf_counter() - last_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
                last_time = time.perf_counter()

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[Main] 'q' or ESC pressed. Exiting...")
                    stop_event.set()
        finally:
            stop_event.set()
            camera_thread.join(timeout=2.0)
            audio_thread.join(timeout=2.0)
            inference_thread.join(timeout=2.0)
            cv2.destroyAllWindows()
            print("[Main] Clean shutdown complete.")


if __name__ == "__main__":
    main()
