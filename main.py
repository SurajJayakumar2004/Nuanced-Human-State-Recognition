import threading
import signal
import time
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import pyaudio
import mediapipe as mp
import torch

from models.visual_expert import VisualExpert
from models.audio_expert import AudioExpert
from models.fusion_head import NuancedStateClassifier
from utils.fusion import classify_nuanced_state


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

        # Initialize neural fusion head (NuancedStateClassifier) on device.
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.classifier = NuancedStateClassifier(input_dim=5, hidden_dim=64).to(
            self.device
        )
        weights_path = "weights/fusion_v1.pth"
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

                        feat_vec = torch.tensor(
                            [fau12, fau6, fau4, jitter, intensity],
                            dtype=torch.float32,
                            device=self.device,
                        )  # [5]

                        # Neural prediction first
                        pred = self.classifier.predict(feat_vec)
                        neural_state = pred.label
                        neural_confidence = pred.confidence

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

                        conflict_score = float((1.0 - confidence) * 2.0)

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
                    except Exception as e:
                        print(f"[InferenceThread] ERROR during neural inference: {e}")

            time.sleep(self.interval_s)

        print("[InferenceThread] Stopped.")


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
    conflict_score: Optional[float],
    fau: Optional[Dict[str, float]],
    audio_summary: Optional[Dict[str, float]],
    confidence: Optional[float],
    sync_delay_sec: Optional[float],
    detections,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Draws the HUD: state (Override = Cyan, Neural = Green), physical labels
    (TAPPING, HAND-TO-FACE), gesture labels (ANALYZING, MASKING, BORED when
    persisted >10 frames), Synchrony Visualizer, Logic Conflict thick box, and Logic Source.
    """
    hud_frame = frame.copy()

    state_text = state or "State: --"
    conflict_text = (
        f"Conflict Score: {conflict_score:.2f}"
        if conflict_score is not None
        else "Conflict Score: --"
    )

    # State text color: Cyan = Override, Green = Neural
    if logic_source in ("Rule-Override", "Override"):
        state_color = (255, 255, 0)  # Cyan (BGR)
    else:
        state_color = (0, 255, 0)  # Green (BGR)

    # Bounding box color by conflict
    if conflict_score is None:
        box_color = (0, 255, 255)  # Yellow for unknown
    elif conflict_score < 0.3:
        box_color = (0, 255, 0)  # Green: Low conflict
    elif conflict_score < 0.6:
        box_color = (0, 165, 255)  # Orange: Moderate conflict
    else:
        box_color = (0, 0, 255)  # Red: High conflict

    # Logic conflict: Neural said something "positive" but Rule says Fake -> thicker box
    logic_conflict = (
        state == "Fake / Polite Face"
        and neural_state is not None
        and neural_state != "Fake / Polite Face"
    )
    box_thickness = 4 if logic_conflict else 2

    # Face bounding box and physical labels next to it
    if detections:
        for detection in detections:
            bbox = detection.location_data.relative_bounding_box
            x_min = int(bbox.xmin * width)
            y_min = int(bbox.ymin * height)
            w = int(bbox.width * width)
            h = int(bbox.height * height)

            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width - 1, x_min + w)
            y_max = min(height - 1, y_min + h)

            cv2.rectangle(
                hud_frame, (x_min, y_min), (x_max, y_max), box_color, box_thickness
            )

            # Physical iconography: labels above/near the face box
            label_y = max(20, y_min - 8)
            if finger_tapping:
                cv2.putText(
                    hud_frame,
                    "TAPPING",
                    (x_min, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 200, 0),  # BGR
                    1,
                    cv2.LINE_AA,
                )
                label_y -= 16
            if self_touching_hands:
                cv2.putText(
                    hud_frame,
                    "HAND-TO-FACE",
                    (x_min, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 200, 0),
                    1,
                    cv2.LINE_AA,
                )
                label_y -= 16
            if show_analyzing:
                cv2.putText(
                    hud_frame,
                    "ANALYZING",
                    (x_min, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 0),  # Cyan BGR
                    1,
                    cv2.LINE_AA,
                )
                label_y -= 16
            if show_masking:
                cv2.putText(
                    hud_frame,
                    "MASKING",
                    (x_min, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 0, 255),  # Red BGR
                    1,
                    cv2.LINE_AA,
                )
                label_y -= 16
            if show_bored:
                cv2.putText(
                    hud_frame,
                    "BORED",
                    (x_min, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 255),  # Yellow BGR
                    1,
                    cv2.LINE_AA,
                )
            if logic_conflict:
                cv2.putText(
                    hud_frame,
                    "Logic Conflict",
                    (x_min, y_max + 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

    if fau is None:
        fau = {}
    if audio_summary is None:
        audio_summary = {}

    jitter = float(audio_summary.get("jitter", 0.0))
    intensity = float(audio_summary.get("intensity", 0.0))
    conf_val = confidence if confidence is not None else 0.0
    source_text = "Rule-Override" if logic_source in ("Rule-Override", "Override") else (logic_source or "Neural")

    # State label (Cyan = Override, Green = Neural)
    cv2.putText(
        hud_frame,
        state_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        state_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        hud_frame,
        conflict_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        hud_frame,
        f"FAU12: {float(fau.get('FAU12', 0.0)):.2f}  "
        f"FAU6: {float(fau.get('FAU6', 0.0)):.2f}  "
        f"FAU4: {float(fau.get('FAU4', 0.0)):.2f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        hud_frame,
        f"Jitter: {jitter:.2f}  Intensity: {intensity:.2f}  Conf: {conf_val:.2f}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
        cv2.LINE_AA,
    )

    # Synchrony Visualizer: 200x50 box at bottom left; zero line center; marker by delay
    _draw_sync_visualizer(hud_frame, sync_delay_sec, width, height)

    # Logic Source at bottom of HUD
    cv2.putText(
        hud_frame,
        f"Logic Source: {source_text}",
        (10, height - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    return hud_frame


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
                    state,
                    logic_source,
                    neural_state,
                    finger_tapping,
                    self_touching_hands,
                    show_analyzing,
                    show_masking,
                    show_bored,
                    conflict_score,
                    fau,
                    audio_summary,
                    confidence,
                    sync_delay_sec,
                    detections,
                    width,
                    height,
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
