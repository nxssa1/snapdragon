"""Qualcomm AI Hub MediaPipe-Hand-Detection (palm detector + hand landmark) via ONNX Runtime.

Models: models/hand_detector.onnx, models/hand_landmark_detector.onnx, models/anchors_palm.npy
(produced by tools/export_hand_onnx.py). Pre/post-processing follows qai_hub_models'
mediapipe_hand app and MediaPipePyTorch (both Apache-2.0 / BSD-3), re-written in NumPy so the
app needs neither torch nor qai-hub-models at run time.

Pipeline per frame:
  no hand being tracked -> palm detector on a 256x256 letterbox -> rotated ROI around the hand
  ROI -> 256x256 crop -> landmark model -> 21 landmarks back in frame pixels
  next frame's ROI comes from these landmarks, so the detector only runs when tracking is lost.
"""
import os
import time

import cv2
import numpy as np

from mudraguru.tracker import Hand

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
RES = 256
# Palm + first finger joints: what MediaPipe uses to place the tracking ROI (fingertips excluded,
# so the box does not grow when fingers stretch out).
ROI_LANDMARKS = [0, 1, 2, 3, 5, 6, 9, 10, 13, 14, 17, 18]

# --ep value -> ONNX Runtime providers, in preference order (CPU always last as the fallback).
EXECUTION_PROVIDERS = {
    "cpu": ["CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
    "qnn": [("QNNExecutionProvider", {"backend_path": "QnnHtp.dll"}), "CPUExecutionProvider"],
}


def _provider_name(p):
    return p[0] if isinstance(p, tuple) else p


def resolve_ep(ep):
    """'auto' -> the best of qnn / dml / cpu that this onnxruntime build has."""
    if ep != "auto":
        return ep
    import onnxruntime as ort
    available = ort.get_available_providers()
    return next(k for k in ("qnn", "dml", "cpu") if _provider_name(EXECUTION_PROVIDERS[k][0]) in available)


def make_session(path, ep):
    import onnxruntime as ort
    ep = resolve_ep(ep)
    if ep not in EXECUTION_PROVIDERS:
        raise ValueError(f"unknown --ep {ep!r}; choose from auto, {', '.join(EXECUTION_PROVIDERS)}")
    available = ort.get_available_providers()
    wanted = EXECUTION_PROVIDERS[ep]
    first = _provider_name(wanted[0])
    if first not in available:
        print(f"[MudraGuru] {first} is not available in this onnxruntime build "
              f"(have: {', '.join(available)}); using CPUExecutionProvider.")
        wanted = ["CPUExecutionProvider"]
    return ort.InferenceSession(path, providers=wanted)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x.astype(np.float64), -100, 100)))


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


class OnnxHandTracker:
    def __init__(self, ep="auto", model_dir=MODEL_DIR, min_detection=0.6, min_tracking=0.5,
                 track=True, **_):
        paths = [os.path.join(model_dir, f) for f in ("hand_detector.onnx", "hand_landmark_detector.onnx",
                                                      "anchors_palm.npy")]
        missing = [p for p in paths if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"Missing {', '.join(missing)}. Run tools/export_hand_onnx.py "
                             "(see README), or use --backend mediapipe.")
        self.detector = make_session(paths[0], ep)
        self.landmarker = make_session(paths[1], ep)
        self.anchors = np.load(paths[2]).astype(np.float32)
        self.min_detection = min_detection
        self.min_landmark = min_tracking
        self.track = track
        self.roi = None  # (cx, cy, size, theta) carried between frames
        self.providers = self.detector.get_providers()
        self.name = "onnx:" + self.providers[0].replace("ExecutionProvider", "")
        self.last_timings = {}

    # ---- detector -------------------------------------------------------------------------
    def _detect(self, rgb):
        h, w = rgb.shape[:2]
        scale = RES / max(h, w)
        nh, nw = round(h * scale), round(w * scale)
        top, left = (RES - nh) // 2, (RES - nw) // 2
        canvas = np.zeros((RES, RES, 3), np.uint8)
        canvas[top:top + nh, left:left + nw] = cv2.resize(rgb, (nw, nh))
        x = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0

        coords, scores = self.detector.run(None, {"image": x})
        scores = _sigmoid(scores[0, :, 0])
        keep = scores >= self.min_detection
        if not keep.any():
            return None
        raw, a, sc = coords[0][keep], self.anchors[keep], scores[keep]

        # Decode: pairs are (cx,cy), (w,h), 7 keypoints; the anchor offset applies to all but (w,h).
        pts = raw.reshape(-1, 9, 2) / RES * a[:, None, 2:4]
        pts[:, [0, 2, 3, 4, 5, 6, 7, 8]] += a[:, None, 0:2]

        # Weighted NMS around the best detection (BlazeFace-style blending), one hand only.
        best = int(np.argmax(sc))
        boxes = np.concatenate([pts[:, 0] - pts[:, 1] / 2, pts[:, 0] + pts[:, 1] / 2], axis=1)
        b = boxes[best]
        iw = np.clip(np.minimum(boxes[:, 2], b[2]) - np.maximum(boxes[:, 0], b[0]), 0, None)
        ih = np.clip(np.minimum(boxes[:, 3], b[3]) - np.maximum(boxes[:, 1], b[1]), 0, None)
        inter = iw * ih
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        iou = inter / (area + area[best] - inter + 1e-9)
        grp = iou > 0.3
        det = (pts[grp] * sc[grp, None, None]).sum(0) / sc[grp].sum()

        # Back to frame pixels.
        det = det * RES
        det[[0, 2, 3, 4, 5, 6, 7, 8]] -= (left, top)
        det /= scale

        center, box_w = det[0], det[1][0]
        wrist, middle = det[2], det[4]  # palm keypoints 0 (wrist centre) and 2 (middle finger base)
        v = middle - wrist
        theta = np.arctan2(wrist[1] - middle[1], wrist[0] - middle[0]) - np.pi / 2
        center = center + 0.5 * box_w * v / (np.linalg.norm(v) + 1e-9)  # the palm box misses the fingers
        return (float(center[0]), float(center[1]), float(box_w * 2.6), float(theta))

    # ---- landmarks ------------------------------------------------------------------------
    def _landmarks(self, rgb, roi):
        cx, cy, size, theta = roi
        corners = (_rot(theta) @ (np.array([[-1, -1], [-1, 1], [1, -1]]).T * size / 2)).T + (cx, cy)
        m = cv2.getAffineTransform(corners.astype(np.float32),
                                   np.float32([[0, 0], [0, RES - 1], [RES - 1, 0]]))
        crop = cv2.warpAffine(rgb, m, (RES, RES))
        x = crop.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        score, lr, lm = self.landmarker.run(None, {"image": x})
        lm = lm[0].astype(np.float64)
        inv = cv2.invertAffineTransform(m)
        xy = lm[:, :2] * RES @ inv[:, :2].T + inv[:, 2]
        z = lm[:, 2] * size  # same pixel scale as x/y
        return np.column_stack([xy, z]).astype(np.float32), float(score[0]), float(lr[0])

    @staticmethod
    def _roi_from_landmarks(lm):
        """Next frame's ROI: box around the hand in its own orientation (as MediaPipe tracking does)."""
        wrist, middle = lm[0, :2], lm[9, :2]
        theta = np.arctan2(wrist[1] - middle[1], wrist[0] - middle[0]) - np.pi / 2
        aligned = lm[ROI_LANDMARKS, :2] @ _rot(-theta).T
        lo, hi = aligned.min(0), aligned.max(0)
        size = float(max(hi - lo) * 2.0)
        center = (lo + hi) / 2 + (0, -0.1 * size)  # nudge toward the fingers
        cx, cy = _rot(theta) @ center
        return (float(cx), float(cy), size, float(theta))

    def process(self, bgr):
        t0 = time.perf_counter()
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        det_ms = 0.0
        roi = self.roi if self.track else None
        if roi is None:
            t = time.perf_counter()
            roi = self._detect(rgb)
            det_ms = 1000 * (time.perf_counter() - t)
        lmk_ms, hands = 0.0, []
        if roi is not None:
            t = time.perf_counter()
            lm, score, lr = self._landmarks(rgb, roi)
            lmk_ms = 1000 * (time.perf_counter() - t)
            if score >= self.min_landmark:
                hands = [Hand(lm, "Right" if lr > 0.5 else "Left", score)]
        self.roi = self._roi_from_landmarks(hands[0].landmarks) if hands else None
        total = 1000 * (time.perf_counter() - t0)
        self.last_timings = {"detector_ms": det_ms, "landmark_ms": lmk_ms,
                             "pre_post_ms": total - det_ms - lmk_ms, "total_ms": total}
        return hands

    def close(self):
        pass
