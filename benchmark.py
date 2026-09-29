"""Per-stage latency and FPS of the MudraGuru pipeline for every available execution provider.

    python benchmark.py                      # sample image, all available EPs, 200 frames each
    python benchmark.py --source camera      # live webcam frames instead
    python benchmark.py --eps cpu qnn        # pick EPs (on Snapdragon: qnn = Hexagon NPU)
    python benchmark.py --coach 5            # also time 5 Qwen3 coaching generations

Writes bench/results.md (and prints the table). Stages:
  detector   palm detector (only runs when tracking is lost -- see the "detect every frame" rows)
  landmark   21-point hand landmark model
  pre/post   letterbox, ROI crop, anchor decode, NMS, un-projection (NumPy/OpenCV)
  rules      features + scoring all 10 mudras
"""
import argparse
import datetime
import os
import platform
import subprocess
import sys
import time

import cv2
import numpy as np

from mudraguru.features import compute_features
from mudraguru.mudras import classify
from mudraguru.tracker import make_tracker

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLE = os.path.join(HERE, "tests", "sample_hand.jpeg")

# Published by Qualcomm AI Hub for these exact models (qai-hub-models 0.63.0, mediapipe_hand/perf.yaml,
# float precision, ONNX Runtime + QNN on the Hexagon NPU).
AIHUB_REFERENCE = [
    ("Snapdragon X Elite CRD", 0.861, 1.106),
    ("Snapdragon X2 Elite CRD", 0.475, 0.685),
]


def machine_info():
    cpu = platform.processor()
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                cpu = winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        except OSError:
            pass
    gpu = ""
    if sys.platform == "win32":
        try:
            gpu = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_VideoController).Name -join ', '"],
                                 capture_output=True, text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    import onnxruntime as ort
    return {
        "CPU": cpu, "GPU": gpu or "n/a", "OS": f"{platform.system()} {platform.version()} ({platform.machine()})",
        "Python": platform.python_version(), "onnxruntime": f"{ort.__version__} {ort.get_available_providers()}",
    }


def sample_frames(n, width=1280, height=720):
    """The AI Hub sample hand, scaled into a 1280x720 frame (same every time -> repeatable)."""
    img = cv2.imread(SAMPLE)
    s = height / img.shape[0]
    hand = cv2.resize(img, (int(img.shape[1] * s), height))
    frame = np.full((height, width, 3), 90, np.uint8)
    x0 = (width - hand.shape[1]) // 2
    frame[:, x0:x0 + hand.shape[1]] = hand
    for _ in range(n):
        yield frame


def camera_frames(n, index):
    from app import open_camera
    cap = open_camera(index, 1280, 720)
    try:
        for _ in range(n):
            ok, frame = cap.read()
            if not ok:
                break
            yield cv2.flip(frame, 1)
    finally:
        cap.release()


def pct(values, q):
    return float(np.percentile(values, q)) if values else float("nan")


def run_config(backend, ep, track, frames, warmup=10):
    kwargs = {"track": track} if backend == "onnx" else {}
    tracker = make_tracker(backend, ep or "cpu", **kwargs)
    rows = {"detector": [], "landmark": [], "prepost": [], "rules": [], "total": []}
    seen = n = 0
    t_start = None
    for i, frame in enumerate(frames):
        if i == warmup:
            t_start = time.perf_counter()
        t0 = time.perf_counter()
        hands = tracker.process(frame)
        t1 = time.perf_counter()
        if hands:
            classify(compute_features(hands[0].landmarks))
        t2 = time.perf_counter()
        if i < warmup:
            continue
        n += 1
        seen += bool(hands)
        tm = getattr(tracker, "last_timings", {})
        rows["detector"].append(tm.get("detector_ms", float("nan")))
        rows["landmark"].append(tm.get("landmark_ms", float("nan")))
        rows["prepost"].append(tm.get("pre_post_ms", float("nan")))
        rows["rules"].append(1000 * (t2 - t1))
        rows["total"].append(1000 * (t2 - t0))
    wall = time.perf_counter() - t_start if t_start else float("nan")
    tracker.close()
    name = getattr(tracker, "name", backend)
    return {"backend": backend, "ep": name, "mode": ("tracking" if track else "detect every frame")
            if backend == "onnx" else "built-in tracking", "n": n, "seen": seen, "wall": wall, **rows}


def fmt(values):
    vals = [v for v in values if not np.isnan(v)]
    if not vals:
        return "-"
    return f"{np.median(vals):.2f}"


def time_coach(n):
    import tempfile

    from mudraguru.coach import Coach, all_pairs
    # Throwaway cache: generate() is timed directly, and the app's real cache is left alone.
    coach = Coach(cache_path=os.path.join(tempfile.mkdtemp(), "coach_cache.json"), prewarm=False)
    while coach.status == "loading":
        time.sleep(0.2)
    if coach.status != "ready":
        return None
    times = []
    for mudra, hint in all_pairs()[:n]:
        t = time.perf_counter()
        coach.generate(mudra, hint)
        times.append(time.perf_counter() - t)
    coach.close()
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=200)
    ap.add_argument("--source", choices=["sample", "camera"], default="sample")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--eps", nargs="+", choices=["cpu", "dml", "qnn"], help="default: all available")
    ap.add_argument("--no-mediapipe", action="store_true", help="skip the MediaPipe pip baseline")
    ap.add_argument("--coach", type=int, default=0, metavar="N", help="also time N coaching generations")
    ap.add_argument("--out", default=os.path.join(HERE, "bench", "results.md"))
    args = ap.parse_args()

    import onnxruntime as ort
    available = ort.get_available_providers()
    provider_of = {"cpu": "CPUExecutionProvider", "dml": "DmlExecutionProvider", "qnn": "QNNExecutionProvider"}
    eps = args.eps or [e for e in ("qnn", "dml", "cpu") if provider_of[e] in available]
    skipped = [e for e in eps if provider_of[e] not in available]
    eps = [e for e in eps if e not in skipped]

    def frames():
        total = args.frames + 10
        return camera_frames(total, args.camera) if args.source == "camera" else sample_frames(total)

    results = []
    for ep in eps:
        for track in (True, False):
            print(f"onnx / {ep} / {'tracking' if track else 'detect every frame'} ...", flush=True)
            results.append(run_config("onnx", ep, track, frames()))
    if not args.no_mediapipe:
        print("mediapipe pip baseline ...", flush=True)
        results.append(run_config("mediapipe", None, True, frames()))
    coach_times = time_coach(args.coach) if args.coach else None

    info = machine_info()
    lines = [
        "# MudraGuru benchmark",
        "",
        f"Generated {datetime.datetime.now():%Y-%m-%d %H:%M} with `python benchmark.py {' '.join(sys.argv[1:])}`.",
        "",
        *[f"- **{k}:** {v}" for k, v in info.items()],
        f"- **Input:** {'webcam' if args.source == 'camera' else 'AI Hub sample hand in a 1280x720 frame'}, "
        f"{args.frames} timed frames after 10 warm-up frames",
        "",
        "Median milliseconds per frame. FPS is the processing rate (camera excluded).",
        "",
        "| Backend | Execution provider | Mode | Detector | Landmark | Pre/post | Rules | Total | p90 total | FPS | Hand found |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        fps = r["n"] / r["wall"] if r["wall"] else float("nan")
        lines.append(f"| {r['backend']} | {r['ep']} | {r['mode']} | {fmt(r['detector'])} | {fmt(r['landmark'])} | "
                     f"{fmt(r['prepost'])} | {fmt(r['rules'])} | {fmt(r['total'])} | {pct(r['total'], 90):.2f} | "
                     f"{fps:.0f} | {r['seen']}/{r['n']} |")
    if skipped:
        lines += ["", f"Not available in this onnxruntime build: {', '.join(skipped)}."]
    if coach_times:
        lines += ["", f"**Coaching line** (Qwen3-0.6B, transformers, CPU, 4 threads): median "
                      f"{np.median(coach_times):.1f} s per new sentence over {len(coach_times)} generations "
                      "(runs on a background thread; cached per mudra + correction)."]
    lines += [
        "",
        "## Reference: Snapdragon NPU (published by Qualcomm AI Hub)",
        "",
        "Same two models, float, ONNX Runtime with the QNN execution provider on the Hexagon NPU "
        "(qai-hub-models 0.63.0 `mediapipe_hand/perf.yaml`):",
        "",
        "| Device | Detector ms | Landmark ms | Both |",
        "|---|---:|---:|---:|",
        *[f"| {d} | {a:.3f} | {b:.3f} | {a + b:.3f} |" for d, a, b in AIHUB_REFERENCE],
        "",
        "## Getting Snapdragon NPU numbers yourself",
        "",
        "**On a Snapdragon X Elite / X Plus laptop** (Windows on ARM):",
        "",
        "1. Install ARM64 Python 3.11+ (python.org \"Windows ARM64\" installer) and clone this repo.",
        "2. `python -m venv .venv` then `.venv\\Scripts\\pip install numpy opencv-python onnxruntime-qnn`",
        "   (mediapipe has no Windows ARM64 wheel, so add `--no-mediapipe` below).",
        "3. `.venv\\Scripts\\python benchmark.py --eps qnn cpu --no-mediapipe` -- `qnn` runs on the Hexagon NPU "
        "through `QnnHtp.dll`, which ships inside onnxruntime-qnn.",
        "4. Run the app with `python app.py --ep qnn --no-coach`; the FPS overlay shows `onnx:QNN`.",
        "",
        "**Without a Snapdragon machine, on Qualcomm AI Hub cloud devices** (free account, API token):",
        "",
        "1. In the export venv: `qai-hub configure --api_token <your token>`",
        "2. `python tools/aihub_profile.py --device \"Snapdragon X Elite CRD\"` -- compiles both ONNX models for "
        "the NPU, runs a profile job on a real device and prints per-model latency and the job links.",
    ]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:lines.index("## Reference: Snapdragon NPU (published by Qualcomm AI Hub)")]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
