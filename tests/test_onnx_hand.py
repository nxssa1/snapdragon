"""The Qualcomm AI Hub ONNX backend should agree with the MediaPipe pip package on a real photo.
Run: python -m tests.test_onnx_hand"""
import os

import cv2
import numpy as np

from mudraguru.features import compute_features
from mudraguru.mudras import classify
from mudraguru.tracker import make_tracker

SAMPLE = os.path.join(os.path.dirname(__file__), "sample_hand.jpeg")  # Qualcomm AI Hub sample image


def test_onnx_matches_mediapipe_on_sample():
    img = cv2.imread(SAMPLE)
    ref = make_tracker("mediapipe").process(img)[0].landmarks
    for ep in ("cpu", "auto"):
        tracker = make_tracker("onnx", ep)
        for frame_no in range(5):  # first frame uses the detector, later ones the tracked ROI
            hands = tracker.process(img)
            assert hands, (ep, frame_no)
            err = np.linalg.norm(hands[0].landmarks[:, :2] - ref[:, :2], axis=1).mean()
            assert err < 5.0, (ep, frame_no, err)
        assert classify(compute_features(hands[0].landmarks))[0].name == "Pataka"


def test_no_hand_in_blank_frame():
    tracker = make_tracker("onnx", "cpu")
    assert tracker.process(np.zeros((480, 640, 3), np.uint8)) == []


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
        print("ok ", fn.__name__)
