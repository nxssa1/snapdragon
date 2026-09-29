# MudraGuru

An offline webcam tutor for Bharatanatyam hasta mudras. It finds your hand, reads 21 landmarks,
works out which mudra you are holding, scores it, and tells you which finger to fix.
Nothing leaves the laptop.

```
webcam ──> hand detector + 21 landmarks ──> geometric features ──> rule table ──> score + correction ──> UI
           (MediaPipe now; Qualcomm AI Hub      (finger bend, thumb     (mudras.py)
            MediaPipe-Hand ONNX next)            angle, spread, touches)
```

## Run

```
py -3.11 -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python app.py                   # detect mode
.venv\Scripts\python app.py --mode practice   # practice mode
```

Keys: **M** switch detect/practice · **N / P** next/previous target · **D** show raw features ·
**C** save the current features as a labelled sample (to `logs/feature_samples.csv`) · **Q / Esc** quit.

Other flags: `--camera 1`, `--width 640 --height 480`, `--hold 3`, `--threshold 0.6`, `--no-mirror`,
`--benchmark 200` (headless timing of camera + landmarks, no window).

## Mudras recognised

Pataka, Tripataka, Ardhapataka, Kartarimukha, Mayura, Ardhachandra, Alapadma, Mushti, Shikhara, Hamsasya.

## How it works

- `mudraguru/tracker.py` — landmark backends behind one interface (`process(frame) -> [Hand]`).
- `mudraguru/features.py` — rotation-, scale- and mirror-invariant features: how straight each finger is
  (0–1), thumb angle to the palm axis, spread between neighbouring fingers, thumb-tip to fingertip distances.
- `mudraguru/mudras.py` — **the rule table**. Each mudra is a set of target ranges for those features. Each
  range gives a soft match; the score is their weighted geometric mean, and the worst-matching range becomes
  the correction ("Straighten your ring finger"). To tune, press **D** in the app, look at the numbers while
  holding the mudra, and widen/move the range.
- `mudraguru/practice.py` — practice mode: target mudra, hold for 2 s above 70% to pass, every attempt logged
  to `logs/practice_log.csv`, per-mudra record shown on screen.

## Tests

`python -m tests.test_mudras` builds synthetic hands for each mudra (random rotation, scale, mirroring, noise)
and checks each one is recognised, clearly ahead of the runner-up, and that corrections name the right finger.

## Performance (Lenovo IdeaPad Slim 5, Ryzen AI 7 350, CPU only)

1280×720 webcam via Media Foundation: ~25 FPS end to end, ~27 ms per frame for MediaPipe landmarks on CPU.

## Roadmap

1. Qualcomm AI Hub MediaPipe-Hand ONNX backend (execution provider selectable: QNN on Snapdragon, CPU/DirectML here).
2. Small LLM (Qwen3) for a one-line coaching tip.
3. NPU vs CPU benchmark script.
