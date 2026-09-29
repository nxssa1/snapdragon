"""MudraGuru webcam app.

    python app.py                 # detect mode
    python app.py --mode practice
    python app.py --ep cpu        # force an execution provider (auto: qnn > dml > cpu)
    python app.py --backend mediapipe   # fallback: MediaPipe pip package

Keys: M switch detect/practice   N / P next / previous target   D debug features
      C save a feature sample (for tuning)   Q or Esc quit
"""
import argparse
import csv
import os
import sys
import textwrap
import time

import cv2
import numpy as np

from mudraguru.features import FINGER_JOINTS, HAND_EDGES, FeatureSmoother, compute_features
from mudraguru.mudras import MUDRAS, classify
from mudraguru.practice import PracticeSession
from mudraguru.tracker import make_tracker

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
PANEL_W = 400
FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE, GREY, GREEN, AMBER, RED, CYAN = (240, 240, 240), (150, 150, 150), (80, 200, 80), (0, 190, 255), (70, 70, 230), (220, 200, 60)


def open_camera(index, width, height):
    # Media Foundation: 30 FPS on this laptop's webcam, where DirectShow only managed 10.
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF) if sys.platform == "win32" else cv2.VideoCapture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}. Try --camera 1, or close other apps using the webcam.")
    return cap


def landmarks_for_feature(feat):
    """Landmark indices to highlight for a correction on this feature."""
    if feat.startswith("ext_"):
        return FINGER_JOINTS[feat[4:]]
    if feat.startswith("touch_"):
        return (4, FINGER_JOINTS[feat[6:]][-1])
    if feat.startswith("spread_"):
        pair = {"im": ("index", "middle"), "mr": ("middle", "ring"), "rp": ("ring", "pinky")}[feat[7:]]
        return FINGER_JOINTS[pair[0]] + FINGER_JOINTS[pair[1]]
    return FINGER_JOINTS["thumb"]


def draw_hand(img, lm, highlight=()):
    pts = lm[:, :2].astype(int)
    for a, b in HAND_EDGES:
        color = RED if a in highlight and b in highlight else CYAN
        cv2.line(img, tuple(pts[a]), tuple(pts[b]), color, 2, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(img, tuple(p), 6 if i in highlight else 4, RED if i in highlight else WHITE, -1, cv2.LINE_AA)


class Panel:
    def __init__(self, img, x0):
        self.img, self.x, self.y = img, x0 + 16, 34

    def text(self, s, color=WHITE, scale=0.55, thick=1, gap=26):
        cv2.putText(self.img, s, (self.x, self.y), FONT, scale, color, thick, cv2.LINE_AA)
        self.y += gap

    def wrapped(self, s, color=GREY, width=42):
        for line in textwrap.wrap(s, width):
            self.text(line, color, 0.48, gap=21)

    def bar(self, frac, color, w=PANEL_W - 32, h=14):
        x, y = self.x, self.y - 12
        cv2.rectangle(self.img, (x, y), (x + w, y + h), GREY, 1)
        cv2.rectangle(self.img, (x, y), (x + int(w * max(0.0, min(frac, 1.0))), y + h), color, -1)
        self.y += h + 14

    def space(self, px=10):
        self.y += px


def score_color(score):
    return GREEN if score >= 0.7 else AMBER if score >= 0.45 else RED


def render_detect(p, ranked):
    p.text("DETECT", GREY, 0.5)
    if ranked is None:
        p.text("Show one hand to the camera", AMBER, 0.6)
        return ()
    best = ranked[0]
    if best.score < 0.45:
        p.text("No clear mudra", AMBER, 0.75, 2, 32)
        p.text(f"closest: {best.name} {best.score:.0%}", GREY)
    else:
        p.text(best.name, score_color(best.score), 0.95, 2, 34)
        p.text(f"({MUDRAS[best.name]['meaning']})  {best.score:.0%}", GREY)
        p.bar(best.score, score_color(best.score))
    if best.corrections and best.corrections[0].match < 0.9:
        p.text("Fix:", AMBER)
        p.wrapped(best.corrections[0].hint, AMBER)
    p.space()
    p.text("Also close:", GREY, 0.5)
    for s in ranked[1:4]:
        cv2.putText(p.img, f"{s.score:.0%}", (p.x + 160, p.y), FONT, 0.5, GREY, 1, cv2.LINE_AA)
        p.text(f"  {s.name}", GREY, 0.5, gap=22)
    return landmarks_for_feature(best.corrections[0].feature) if best.corrections and best.score >= 0.3 else ()


def render_practice(p, session, ranked, target_score, progress):
    name = session.target
    p.text("PRACTICE  (N/P to change)", GREY, 0.5)
    p.text(name, WHITE, 0.95, 2, 34)
    p.text(f"({MUDRAS[name]['meaning']})", GREY)
    p.wrapped(MUDRAS[name]["description"], GREY)
    p.space()
    highlight = ()
    if session.celebrating:
        p.text("Well done!", GREEN, 1.0, 2, 40)
    elif ranked is None or target_score is None:
        p.text("Show one hand to the camera", AMBER, 0.6)
    else:
        p.text(f"Match {target_score.score:.0%}", score_color(target_score.score), 0.7, 2, 30)
        p.bar(target_score.score, score_color(target_score.score))
        p.text(f"Hold {session.hold_seconds:g}s:", GREY, 0.5)
        p.bar(progress, GREEN)
        if target_score.corrections and target_score.score < session.threshold + 0.2:
            fix = target_score.corrections[0]
            p.text("Fix:", AMBER)
            p.wrapped(fix.hint, AMBER)
            highlight = landmarks_for_feature(fix.feature)
        elif ranked[0].name != name:
            p.text(f"Looks like {ranked[0].name}", AMBER)
    p.space()
    p.text("Your record (passed/tries):", GREY, 0.5)
    for n in MUDRAS:
        color = WHITE if n == name else GREY
        cv2.putText(p.img, session.summary(n), (p.x + 150, p.y), FONT, 0.45, color, 1, cv2.LINE_AA)
        p.text(f"{'>' if n == name else ' '} {n}", color, 0.45, gap=19)
    return highlight


def draw_debug(img, feats):
    y = 20
    for k, v in feats.items():
        cv2.putText(img, f"{k:<12} {v:6.2f}", (10, y + 30), FONT, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        y += 18


def save_sample(feats, label):
    path = os.path.join(LOG_DIR, "feature_samples.csv")
    os.makedirs(LOG_DIR, exist_ok=True)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["timestamp", "label", *feats])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), label, *(f"{v:.4f}" for v in feats.values())])
    return path


def main():
    ap = argparse.ArgumentParser(description="MudraGuru: Bharatanatyam hasta mudra tutor")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--backend", choices=["onnx", "mediapipe"], default="onnx",
                    help="onnx = Qualcomm AI Hub MediaPipe-Hand model; mediapipe = pip package fallback")
    ap.add_argument("--ep", choices=["auto", "qnn", "dml", "cpu"], default="auto",
                    help="ONNX Runtime execution provider for --backend onnx (auto: qnn > dml > cpu)")
    ap.add_argument("--mode", choices=["detect", "practice"], default="detect")
    ap.add_argument("--hold", type=float, default=2.0, help="seconds to hold a mudra in practice mode")
    ap.add_argument("--threshold", type=float, default=0.7, help="match needed to count as holding")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    cap = open_camera(args.camera, args.width, args.height)
    tracker = make_tracker(args.backend, args.ep)
    smoother = FeatureSmoother(alpha=0.5)
    session = PracticeSession(MUDRAS, os.path.join(LOG_DIR, "practice_log.csv"), args.hold, args.threshold)
    mode, debug, fps, last = args.mode, False, 0.0, time.perf_counter()
    flash_until, flash_msg = 0.0, ""
    win = "MudraGuru"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera stopped returning frames.")
            break
        if not args.no_mirror:
            frame = cv2.flip(frame, 1)
        hands = tracker.process(frame)

        feats = ranked = None
        if hands:
            feats = smoother(compute_features(hands[0].landmarks))
            ranked = classify(feats)
        else:
            smoother.reset()

        h, w = frame.shape[:2]
        canvas = np.full((max(h, 720), w + PANEL_W, 3), 32, np.uint8)
        canvas[:h, :w] = frame
        panel = Panel(canvas, w)
        if mode == "practice":
            target_score, progress, _ = session.update(ranked)
            highlight = render_practice(panel, session, ranked, target_score, progress)
        else:
            highlight = render_detect(panel, ranked)
        if hands:
            draw_hand(canvas, hands[0].landmarks, set(highlight))
        if debug and feats:
            draw_debug(canvas, feats)

        now = time.perf_counter()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6)) if fps else 1.0 / max(now - last, 1e-6)
        last = now
        cv2.putText(canvas, f"{fps:4.1f} FPS  [{tracker.name}]", (10, 22), FONT, 0.6, GREEN, 2, cv2.LINE_AA)
        if time.monotonic() < flash_until:
            cv2.putText(canvas, flash_msg, (10, h - 16), FONT, 0.6, AMBER, 2, cv2.LINE_AA)
        cv2.imshow(win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord("m"):
            mode = "practice" if mode == "detect" else "detect"
            if mode == "practice":
                session._start_attempt()
        elif key == ord("d"):
            debug = not debug
        elif key == ord("n") and mode == "practice":
            session.next()
        elif key == ord("p") and mode == "practice":
            session.prev()
        elif key == ord("c") and feats:
            label = session.target if mode == "practice" else ranked[0].name
            save_sample(feats, label)
            flash_msg, flash_until = f"saved sample labelled {label}", time.monotonic() + 1.5

    cap.release()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
