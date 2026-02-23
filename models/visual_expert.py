from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch

# Pose landmark indices (MediaPipe Pose)
_POSE_NOSE = 0
_POSE_LEFT_EAR = 7
_POSE_RIGHT_EAR = 8
_POSE_LEFT_SHOULDER = 11
_POSE_RIGHT_SHOULDER = 12

# Hand landmark indices (MediaPipe Hands)
_HAND_WRIST = 0
_HAND_INDEX_TIP = 8
_HAND_MIDDLE_TIP = 12
_HAND_RING_TIP = 16
_HAND_PINKY_TIP = 20

# Face Mesh zone landmark indices (for Hand-to-Face zones)
# Chin: jaw / chin region
_FACE_CHIN_IDS = (152, 148, 176, 377, 400, 378)
# Mouth: lips and mouth opening
_FACE_MOUTH_IDS = (61, 291, 13, 14, 17, 84, 91, 314, 324, 78)
# Left temple / left side of face
_FACE_LEFT_TEMPLE_IDS = (21, 162, 234, 127, 139)
# Right temple / right side of face
_FACE_RIGHT_TEMPLE_IDS = (251, 389, 454, 356, 265)
# Padding (normalized) to expand zone for intersection
_ZONE_PAD = 0.04


@dataclass
class FAUIntensities:
    """Container for normalized FAU intensities in [0, 1]."""

    fau12_lip_corner_puller: float
    fau6_cheek_raiser: float
    fau4_brow_lowerer: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "FAU12": self.fau12_lip_corner_puller,
            "FAU6": self.fau6_cheek_raiser,
            "FAU4": self.fau4_brow_lowerer,
        }


class VisualExpert:
    """
    VisualExpert: Face Mesh (FAUs), Pose (Shoulder_Rigidity, Head_Tilt),
    and Hands (Self_Touching_Hands, Finger_Tapping) for hybrid overrides.
    """

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=static_image_mode,
            model_complexity=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Device: prefer MPS, then CUDA, then CPU
        self.device = torch.device(
            "mps"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )
        self._dinov2 = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True, trust_repo=True
        )
        self._dinov2.to(self.device)
        self._dinov2.eval()
        # ImageNet normalization for DINOv2
        self._imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self._imagenet_std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

        # Temporal buffers for stability / movement (frame-level)
        # Shoulder rigidity: variance of Y-distance between pose landmarks 11 and 12
        self._shoulder_y_distance_buffer: deque = deque(maxlen=15)
        self._fingertip_z_buffer: deque = deque(maxlen=10)
        self._head_tilt_buffer: deque = deque(maxlen=5)

    def _extract_first_face_landmarks(
        self, frame_bgr: np.ndarray
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Runs Face Mesh and returns (xy, z) arrays for the first detected face.

        Returns:
            xy: np.ndarray of shape [num_landmarks, 2] with pixel coordinates.
            z: np.ndarray of shape [num_landmarks] with relative depth.
        """
        h, w = frame_bgr.shape[:2]  # [H, W, C]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)  # [H, W, C]
        results = self._face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        xs = []
        ys = []
        zs = []
        for lm in face_landmarks.landmark:
            xs.append(lm.x * w)
            ys.append(lm.y * h)
            zs.append(lm.z)

        xy = np.stack([xs, ys], axis=-1).astype(np.float32)  # [num_landmarks, 2]
        z = np.asarray(zs, dtype=np.float32)  # [num_landmarks]
        return xy, z

    @staticmethod
    def _safe_norm(v: np.ndarray) -> float:
        """Returns L2 norm of a vector."""
        return float(np.linalg.norm(v))

    @staticmethod
    def _normalize_intensity(value: float, ref: float, scale: float = 1.0) -> float:
        """
        Simple normalization into [0, 1] using a reference distance.

        intensity = clip((value / (ref * scale)), 0, 1).
        """
        if ref <= 1e-6:
            return 0.0
        return float(np.clip(value / (ref * scale), 0.0, 1.0))

    def get_fau_intensities(self, frame_bgr: np.ndarray) -> Dict[str, float]:
        """
        Estimate simple FAU (Action Unit) intensities from a BGR frame.

        Args:
            frame_bgr: np.ndarray of shape [H, W, 3], BGR image from OpenCV.

        Returns:
            Dict[str, float]: Normalized intensities in [0, 1] for:
                - "FAU12": Lip Corner Puller
                - "FAU6": Cheek Raiser
                - "FAU4": Brow Lowerer
        """
        landmarks = self._extract_first_face_landmarks(frame_bgr)
        if landmarks is None:
            return FAUIntensities(0.0, 0.0, 0.0).to_dict()

        xy, _z = landmarks  # xy: [num_landmarks, 2], _z: [num_landmarks]

        # Reference distance: approximate inter-ocular distance (between eye corners).
        # Using MediaPipe indices near the outer eye corners.
        LEFT_EYE_OUTER = 33
        RIGHT_EYE_OUTER = 263

        eye_ref = self._safe_norm(xy[LEFT_EYE_OUTER] - xy[RIGHT_EYE_OUTER])
        if eye_ref <= 1e-6:
            eye_ref = 1.0

        # --- FAU 12: Lip Corner Puller (smile width vs neutral) ---
        # Use mouth corners distance normalized by eye distance.
        LEFT_MOUTH_CORNER = 61
        RIGHT_MOUTH_CORNER = 291
        mouth_width = self._safe_norm(
            xy[RIGHT_MOUTH_CORNER] - xy[LEFT_MOUTH_CORNER]
        )
        fau12 = self._normalize_intensity(mouth_width, eye_ref, scale=1.2)

        # --- FAU 6: Cheek Raiser (eye narrowing) ---
        # Distance between upper and lower eyelid midpoints; smaller distance → stronger activation.
        LEFT_EYE_UPPER = 159
        LEFT_EYE_LOWER = 145
        RIGHT_EYE_UPPER = 386
        RIGHT_EYE_LOWER = 374

        left_eye_opening = self._safe_norm(
            xy[LEFT_EYE_UPPER] - xy[LEFT_EYE_LOWER]
        )
        right_eye_opening = self._safe_norm(
            xy[RIGHT_EYE_UPPER] - xy[RIGHT_EYE_LOWER]
        )
        mean_eye_opening = 0.5 * (left_eye_opening + right_eye_opening)

        # Convert "smaller opening" to "higher intensity" by inverting.
        # Assume neutral opening ≈ 0.25 * eye_ref.
        neutral_opening = 0.25 * eye_ref
        if neutral_opening <= 1e-6:
            fau6 = 0.0
        else:
            raw = np.clip(neutral_opening - mean_eye_opening, 0.0, neutral_opening)
            fau6 = float(raw / neutral_opening)

        # --- FAU 4: Brow Lowerer (brow closer to eye) ---
        # Measure vertical distance between brow and eye; smaller distance → stronger activation.
        LEFT_BROW = 70
        RIGHT_BROW = 300
        LEFT_EYE_CENTER = 159
        RIGHT_EYE_CENTER = 386

        left_brow_eye_dist = self._safe_norm(
            xy[LEFT_BROW] - xy[LEFT_EYE_CENTER]
        )
        right_brow_eye_dist = self._safe_norm(
            xy[RIGHT_BROW] - xy[RIGHT_EYE_CENTER]
        )
        mean_brow_eye = 0.5 * (left_brow_eye_dist + right_brow_eye_dist)

        # Convert "smaller brow-eye distance" to "higher intensity".
        neutral_brow_eye = 0.35 * eye_ref
        if neutral_brow_eye <= 1e-6:
            fau4 = 0.0
        else:
            raw4 = np.clip(neutral_brow_eye - mean_brow_eye, 0.0, neutral_brow_eye)
            fau4 = float(raw4 / neutral_brow_eye)

        intensities = FAUIntensities(
            fau12_lip_corner_puller=fau12,
            fau6_cheek_raiser=fau6,
            fau4_brow_lowerer=fau4,
        )
        return intensities.to_dict()

    def get_latent_embeddings(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Crop the face using Face Mesh landmarks and return a 384-dimensional
        DINOv2 feature vector. Returns zeros if no face is detected.

        Args:
            frame_bgr: BGR image [H, W, 3].

        Returns:
            np.ndarray of shape (384,) dtype float32.
        """
        DINO_SIZE = 224
        EMBED_DIM = 384

        landmarks = self._extract_first_face_landmarks(frame_bgr)
        if landmarks is None:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        xy, _ = landmarks  # xy: [num_landmarks, 2] in pixel coords
        h, w = frame_bgr.shape[:2]

        x_min, y_min = float(xy[:, 0].min()), float(xy[:, 1].min())
        x_max, y_max = float(xy[:, 0].max()), float(xy[:, 1].max())
        margin = 0.2
        w_box = x_max - x_min
        h_box = y_max - y_min
        x_min = max(0, x_min - margin * w_box)
        y_min = max(0, y_min - margin * h_box)
        x_max = min(w, x_max + margin * w_box)
        y_max = min(h, y_max + margin * h_box)
        if x_max <= x_min or y_max <= y_min:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        x_min_i, y_min_i = int(round(x_min)), int(round(y_min))
        x_max_i, y_max_i = int(round(x_max)), int(round(y_max))
        face_crop = frame_bgr[y_min_i:y_max_i, x_min_i:x_max_i]
        if face_crop.size == 0:
            return np.zeros(EMBED_DIM, dtype=np.float32)

        face_resized = cv2.resize(face_crop, (DINO_SIZE, DINO_SIZE), interpolation=cv2.INTER_LINEAR)
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        # [H, W, C] -> [1, C, H, W], float32 [0, 1]
        tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        tensor = tensor.to(self.device)
        tensor = (tensor - self._imagenet_mean) / self._imagenet_std

        with torch.no_grad():
            out = self._dinov2(tensor)
        # out may be (B, 1+N, C) or dict; take CLS token
        if isinstance(out, dict):
            feat = out.get("x_norm_clstoken") or out.get("x") or next(iter(out.values()))
        else:
            feat = out
        if feat.dim() == 3:
            feat = feat[:, 0, :]
        feat = feat.squeeze(0).cpu().numpy().astype(np.float32)
        if feat.size != EMBED_DIM:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        return feat

    def _compute_pose_features(self, frame_rgb: np.ndarray) -> Tuple[float, float]:
        """
        Returns (shoulder_rigidity, head_tilt).
        shoulder_rigidity: [0,1] from variance of Y-distance between POSE 11 and 12 (1 = very rigid).
        head_tilt: [0,1] tilt magnitude (0 = upright, 1 = ~45+ deg).
        """
        h, w = frame_rgb.shape[:2]
        pose_out = self._pose.process(frame_rgb)
        if not pose_out.pose_landmarks:
            if self._shoulder_y_distance_buffer:
                vals = list(self._shoulder_y_distance_buffer)
                var = float(np.var(vals)) if len(vals) > 1 else 0.0
                rigidity = float(np.clip(1.0 - min(var * 50.0, 1.0), 0.0, 1.0))
            else:
                rigidity = 0.0
            return rigidity, 0.0

        lm = pose_out.pose_landmarks.landmark
        # Y-distance between landmarks 11 (left shoulder) and 12 (right shoulder)
        y11 = lm[_POSE_LEFT_SHOULDER].y
        y12 = lm[_POSE_RIGHT_SHOULDER].y
        y_distance = abs(y11 - y12)
        self._shoulder_y_distance_buffer.append(y_distance)

        # Head tilt: angle of ear line vs horizontal (normalized coords)
        le_x, le_y = lm[_POSE_LEFT_EAR].x, lm[_POSE_LEFT_EAR].y
        re_x, re_y = lm[_POSE_RIGHT_EAR].x, lm[_POSE_RIGHT_EAR].y
        dx = re_x - le_x
        dy = re_y - le_y
        angle_rad = np.arctan2(abs(dy), max(abs(dx), 1e-6))
        angle_deg = np.degrees(angle_rad)
        head_tilt_scalar = float(np.clip(angle_deg / 45.0, 0.0, 1.0))
        self._head_tilt_buffer.append(head_tilt_scalar)

        # Shoulder rigidity = 1 - normalized variance of Y-distance (11 vs 12)
        vals = list(self._shoulder_y_distance_buffer)
        var = float(np.var(vals)) if len(vals) > 1 else 0.0
        rigidity = float(np.clip(1.0 - min(var * 50.0, 1.0), 0.0, 1.0))
        return rigidity, head_tilt_scalar

    @staticmethod
    def _get_face_zone_boxes(
        face_xy_norm: np.ndarray,
    ) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Build Hand-to-Face zone boxes (x_min, y_min, x_max, y_max) in normalized [0,1].
        face_xy_norm: [num_landmarks, 2] normalized (x, y) for Face Mesh.
        """
        pad = _ZONE_PAD
        zones: Dict[str, Tuple[float, float, float, float]] = {}

        def box(ids: Tuple[int, ...]) -> Tuple[float, float, float, float]:
            pts = face_xy_norm[list(ids)]
            x_min = float(np.clip(pts[:, 0].min() - pad, 0.0, 1.0))
            y_min = float(np.clip(pts[:, 1].min() - pad, 0.0, 1.0))
            x_max = float(np.clip(pts[:, 0].max() + pad, 0.0, 1.0))
            y_max = float(np.clip(pts[:, 1].max() + pad, 0.0, 1.0))
            return (x_min, y_min, x_max, y_max)

        zones["Chin"] = box(_FACE_CHIN_IDS)
        zones["Mouth"] = box(_FACE_MOUTH_IDS)
        zones["Left_Temple"] = box(_FACE_LEFT_TEMPLE_IDS)
        zones["Right_Temple"] = box(_FACE_RIGHT_TEMPLE_IDS)
        return zones

    @staticmethod
    def _point_in_zone(px: float, py: float, zone: Tuple[float, float, float, float]) -> bool:
        x_min, y_min, x_max, y_max = zone
        return (x_min <= px <= x_max) and (y_min <= py <= y_max)

    def _compute_hand_features(
        self,
        frame_rgb: np.ndarray,
        face_xy: Optional[np.ndarray],
        zone_boxes: Optional[Dict[str, Tuple[float, float, float, float]]] = None,
    ) -> Tuple[bool, bool, bool, bool, bool, bool]:
        """
        Returns (self_touching_hands, finger_tapping, hand_on_chin, hand_on_left_temple,
                 hand_on_right_temple, hand_covering_mouth).
        self_touching: hand-to-face (any zone or bbox) or hand-to-hand proximity.
        Zone flags: True if INDEX_FINGER_TIP or WRIST intersects that Face Mesh zone.
        """
        hands_out = self._hands.process(frame_rgb)
        self_touching = False
        hand_on_chin = False
        hand_on_left_temple = False
        hand_on_right_temple = False
        hand_covering_mouth = False
        fingertip_z_list: List[float] = []

        if not hands_out.multi_hand_landmarks:
            return False, False, False, False, False, False
        # defer finger_tapping computation to after the loop
        h, w = frame_rgb.shape[:2]
        hand_landmarks_list = hands_out.multi_hand_landmarks

        # Face bbox in normalized [0,1] (from face mesh if available)
        face_min = np.array([0.2, 0.2])
        face_max = np.array([0.8, 0.7])
        if face_xy is not None and face_xy.size >= 4:
            face_min = face_xy.min(axis=0)
            face_max = face_xy.max(axis=0)
            margin = 0.08
            face_min = np.clip(face_min - margin, 0.0, 1.0)
            face_max = np.clip(face_max + margin, 1.0, 1.0)

        for hand_landmarks in hand_landmarks_list:
            wrist = hand_landmarks.landmark[_HAND_WRIST]
            idx_tip = hand_landmarks.landmark[_HAND_INDEX_TIP]
            pt_w = (wrist.x, wrist.y)
            pt_i = (idx_tip.x, idx_tip.y)

            # Generic face bbox proximity
            in_face = (
                (face_min[0] <= pt_w[0] <= face_max[0] and face_min[1] <= pt_w[1] <= face_max[1])
                or (face_min[0] <= pt_i[0] <= face_max[0] and face_min[1] <= pt_i[1] <= face_max[1])
            )
            if in_face:
                self_touching = True

            # Hand-to-face zones: INDEX_FINGER_TIP and WRIST vs Chin, Left_Temple, Right_Temple, Mouth
            if zone_boxes:
                for pt in (pt_w, pt_i):
                    if self._point_in_zone(pt[0], pt[1], zone_boxes["Chin"]):
                        hand_on_chin = True
                        self_touching = True
                    if self._point_in_zone(pt[0], pt[1], zone_boxes["Left_Temple"]):
                        hand_on_left_temple = True
                        self_touching = True
                    if self._point_in_zone(pt[0], pt[1], zone_boxes["Right_Temple"]):
                        hand_on_right_temple = True
                        self_touching = True
                    if self._point_in_zone(pt[0], pt[1], zone_boxes["Mouth"]):
                        hand_covering_mouth = True
                        self_touching = True

            for idx in (_HAND_INDEX_TIP, _HAND_MIDDLE_TIP, _HAND_RING_TIP, _HAND_PINKY_TIP):
                fingertip_z_list.append(hand_landmarks.landmark[idx].z)

        # Hand-to-hand: two hands, wrists close
        if len(hand_landmarks_list) >= 2:
            w0 = hand_landmarks_list[0].landmark[_HAND_WRIST]
            w1 = hand_landmarks_list[1].landmark[_HAND_WRIST]
            d = np.sqrt((w0.x - w1.x) ** 2 + (w0.y - w1.y) ** 2 + (w0.z - w1.z) ** 2)
            if d < 0.15:
                self_touching = True

        if fingertip_z_list:
            mean_z = float(np.mean(fingertip_z_list))
            self._fingertip_z_buffer.append(mean_z)

        finger_tapping = False
        if len(self._fingertip_z_buffer) >= 5:
            z_vals = list(self._fingertip_z_buffer)
            if np.var(z_vals) > 0.001:
                finger_tapping = True

        return (
            self_touching,
            finger_tapping,
            hand_on_chin,
            hand_on_left_temple,
            hand_on_right_temple,
            hand_covering_mouth,
        )

    def get_visual_summary(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Full visual summary: FAUs + Pose (Shoulder_Rigidity, Head_Tilt)
        + Hands (Self_Touching_Hands, Finger_Tapping, hand-to-face zone flags).

        Shoulder_Rigidity is the variance of Y-distance between POSE landmarks 11 and 12.

        Returns:
            visual_summary with keys:
                FAU12, FAU6, FAU4 (float),
                Shoulder_Rigidity (float [0,1]),
                Head_Tilt (float [0,1]),
                Self_Touching_Hands (bool),
                Finger_Tapping (bool),
                hand_on_chin, hand_on_left_temple, hand_on_right_temple, hand_covering_mouth (bool).
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        fau = self.get_fau_intensities(frame_bgr)

        # Face landmarks in normalized coords for hand-to-face (optional)
        face_xy_norm: Optional[np.ndarray] = None
        landmarks = self._extract_first_face_landmarks(frame_bgr)
        if landmarks is not None:
            xy, _ = landmarks
            face_xy_norm = np.column_stack([xy[:, 0] / w, xy[:, 1] / h])

        shoulder_rigidity, head_tilt = self._compute_pose_features(frame_rgb)
        zone_boxes = (
            self._get_face_zone_boxes(face_xy_norm)
            if face_xy_norm is not None
            else None
        )
        (
            self_touching,
            finger_tapping,
            hand_on_chin,
            hand_on_left_temple,
            hand_on_right_temple,
            hand_covering_mouth,
        ) = self._compute_hand_features(frame_rgb, face_xy_norm, zone_boxes)

        visual_latent = self.get_latent_embeddings(frame_bgr)

        return {
            **fau,
            "Shoulder_Rigidity": shoulder_rigidity,
            "Head_Tilt": head_tilt,
            "Self_Touching_Hands": self_touching,
            "hand_on_chin": hand_on_chin,
            "hand_on_left_temple": hand_on_left_temple,
            "hand_on_right_temple": hand_on_right_temple,
            "hand_covering_mouth": hand_covering_mouth,
            "Finger_Tapping": finger_tapping,
            "visual_latent": visual_latent,
        }


__all__ = ["VisualExpert", "FAUIntensities"]

