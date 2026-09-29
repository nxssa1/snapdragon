# MudraGuru benchmark

Generated 2026-09-29 21:24 with `python benchmark.py --frames 200 --coach 5`.

- **CPU:** AMD Ryzen AI 7 350 w/ Radeon 860M
- **GPU:** AMD Radeon(TM)  860M Graphics
- **OS:** Windows 10.0.26200 (AMD64)
- **Python:** 3.11.9
- **onnxruntime:** 1.24.4 ['DmlExecutionProvider', 'CPUExecutionProvider']
- **Input:** AI Hub sample hand in a 1280x720 frame, 200 timed frames after 10 warm-up frames

Median milliseconds per frame. FPS is the processing rate (camera excluded).

| Backend | Execution provider | Mode | Detector | Landmark | Pre/post | Rules | Total | p90 total | FPS | Hand found |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| onnx | onnx:Dml | tracking | 0.00 | 6.47 | 0.30 | 0.57 | 7.39 | 8.35 | 135 | 200/200 |
| onnx | onnx:Dml | detect every frame | 7.16 | 5.78 | 0.31 | 0.51 | 13.79 | 15.99 | 71 | 200/200 |
| onnx | onnx:CPU | tracking | 0.00 | 9.03 | 0.39 | 0.48 | 9.82 | 15.70 | 88 | 200/200 |
| onnx | onnx:CPU | detect every frame | 11.51 | 9.35 | 0.35 | 0.45 | 21.56 | 41.15 | 35 | 200/200 |
| mediapipe | mediapipe | built-in tracking | - | - | - | 0.60 | 18.32 | 19.76 | 54 | 200/200 |

**Coaching line** (Qwen3-0.6B, transformers, CPU, 4 threads): median 4.6 s per new sentence over 5 generations (runs on a background thread; cached per mudra + correction).

## Reference: Snapdragon NPU (published by Qualcomm AI Hub)

Same two models, float, ONNX Runtime with the QNN execution provider on the Hexagon NPU (qai-hub-models 0.63.0 `mediapipe_hand/perf.yaml`):

| Device | Detector ms | Landmark ms | Both |
|---|---:|---:|---:|
| Snapdragon X Elite CRD | 0.861 | 1.106 | 1.967 |
| Snapdragon X2 Elite CRD | 0.475 | 0.685 | 1.160 |

## Getting Snapdragon NPU numbers yourself

**On a Snapdragon X Elite / X Plus laptop** (Windows on ARM):

1. Install ARM64 Python 3.11+ (python.org "Windows ARM64" installer) and clone this repo.
2. `python -m venv .venv` then `.venv\Scripts\pip install numpy opencv-python onnxruntime-qnn`
   (mediapipe has no Windows ARM64 wheel, so add `--no-mediapipe` below).
3. `.venv\Scripts\python benchmark.py --eps qnn cpu --no-mediapipe` -- `qnn` runs on the Hexagon NPU through `QnnHtp.dll`, which ships inside onnxruntime-qnn.
4. Run the app with `python app.py --ep qnn --no-coach`; the FPS overlay shows `onnx:QNN`.

**Without a Snapdragon machine, on Qualcomm AI Hub cloud devices** (free account, API token):

1. In the export venv: `qai-hub configure --api_token <your token>`
2. `python tools/aihub_profile.py --device "Snapdragon X Elite CRD"` -- compiles both ONNX models for the NPU, runs a profile job on a real device and prints per-model latency and the job links.
