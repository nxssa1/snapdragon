"""Hand landmark backends.

Every backend returns the same thing -- a Hand with a (21, 3) landmark array in
pixel units (x*width, y*height, z*width) -- so the rest of the app does not care
which model produced it:
  onnx       Qualcomm AI Hub MediaPipe-Hand-Detection via ONNX Runtime (onnx_hand.py);
             execution provider QNN on Snapdragon, DirectML or CPU elsewhere.
  mediapipe  the MediaPipe pip package (TFLite on CPU), kept as a fallback.
"""
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Hand:
    landmarks: np.ndarray  # (21, 3) pixel units
    handedness: str        # "Left" / "Right" as seen in the (mirrored) image
    score: float


class MediaPipeTracker:
    name = "mediapipe"

    def __init__(self, max_hands=1, min_detection=0.6, min_tracking=0.5, model_complexity=1, **_):
        import mediapipe as mp
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection,
            min_tracking_confidence=min_tracking,
        )

    def process(self, bgr):
        """Return a list of Hand for one BGR frame."""
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self._hands.process(rgb)
        hands = []
        if res.multi_hand_landmarks:
            for lms, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                pts = np.array([(p.x * w, p.y * h, p.z * w) for p in lms.landmark], dtype=np.float32)
                c = handed.classification[0]
                hands.append(Hand(pts, c.label, float(c.score)))
        return hands

    def close(self):
        self._hands.close()


def _onnx_tracker(**kwargs):
    from mudraguru.onnx_hand import OnnxHandTracker
    return OnnxHandTracker(**kwargs)


BACKENDS = {"onnx": _onnx_tracker, "mediapipe": MediaPipeTracker}


def make_tracker(name="onnx", ep="auto", **kwargs):
    """name: onnx (Qualcomm AI Hub model, execution provider from `ep`) or mediapipe (pip package)."""
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; available: {', '.join(BACKENDS)}")
    if name == "onnx":
        kwargs["ep"] = ep
    return BACKENDS[name](**kwargs)
