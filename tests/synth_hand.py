"""A tiny kinematic hand model that produces 21 landmarks for a described pose.

Only used by the tests, to check the rule table without a camera.
Frame: y up along the palm, x to the thumb side, z out of the palm (fingers curl toward +z).
"""
import numpy as np

MCP = {"index": (0.30, 1.00), "middle": (0.10, 1.05), "ring": (-0.10, 1.00), "pinky": (-0.30, 0.90)}
BONES = {"index": (0.42, 0.25, 0.20), "middle": (0.47, 0.29, 0.22),
         "ring": (0.44, 0.27, 0.21), "pinky": (0.34, 0.20, 0.18)}
START = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
Z = np.array([0.0, 0.0, 1.0])


def _chain(base, direction, bones, flex_deg):
    pts, p, theta = [], np.asarray(base, float), 0.0
    for length, flex in zip(bones, flex_deg):
        theta += np.radians(flex)
        p = p + length * (np.cos(theta) * direction + np.sin(theta) * Z)
        pts.append(p)
    return pts


def make_hand(spread=None, flex=None, thumb_dir=20.0, thumb_flex=(5, 5), thumb_to=None):
    """spread: finger -> in-plane angle in degrees (+ toward thumb).
    flex: finger -> (mcp, pip, dip) flexion in degrees.
    thumb_dir: thumb angle from the palm axis (+ outward, - across the palm).
    thumb_to: finger name whose tip the thumb tip should touch."""
    spread = {"index": 4, "middle": 0, "ring": -4, "pinky": -8, **(spread or {})}
    flex = {f: (0, 0, 0) for f in MCP} | (flex or {})
    lm = np.zeros((21, 3))
    for f, (x, y) in MCP.items():
        s = np.radians(spread[f])
        d = np.array([np.sin(s), np.cos(s), 0.0])
        base = np.array([x, y, 0.0])
        lm[START[f]] = base
        lm[START[f] + 1:START[f] + 4] = _chain(base, d, BONES[f], flex[f])

    lm[1] = (0.25, 0.20, 0.0)
    a = np.radians(thumb_dir)
    d = np.array([np.sin(a), np.cos(a), 0.0])
    lm[2:5] = _chain(lm[1], d, (0.35, 0.30, 0.25), (0, *thumb_flex))
    if thumb_to:
        tip = lm[START[thumb_to] + 3] + (0, 0, 0.05)
        lm[4] = tip
        lm[3] = (lm[2] + tip) / 2 + (0, 0, 0.08)
    return lm


def randomize(lm, rng, noise=0.01):
    """Random 3D rotation, scale, shift and jitter -- features must not care."""
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    out = lm @ r.T * rng.uniform(50, 300) + rng.uniform(-500, 500, size=3)
    if rng.random() < 0.5:
        out[:, 0] *= -1  # mirror: left hand
    return out + rng.normal(scale=noise * np.linalg.norm(out[9] - out[0]), size=out.shape)


CURL = (80, 95, 60)
POSES = {
    "Pataka": make_hand(spread={"index": 2, "ring": -2, "pinky": -4}, thumb_dir=15),
    "Tripataka": make_hand(spread={"index": 2}, flex={"ring": (70, 90, 50)}, thumb_dir=15),
    "Ardhapataka": make_hand(spread={"index": 2}, flex={"ring": CURL, "pinky": CURL}, thumb_dir=20),
    "Kartarimukha": make_hand(spread={"index": 18, "middle": -12}, flex={"ring": CURL, "pinky": CURL},
                              thumb_to="ring"),
    "Mayura": make_hand(flex={"ring": (45, 55, 25)}, thumb_to="ring"),
    "Ardhachandra": make_hand(spread={"index": 2, "ring": -2, "pinky": -4}, thumb_dir=80),
    "Alapadma": make_hand(spread={"index": 25, "middle": 5, "ring": -18, "pinky": -40}, thumb_dir=75),
    "Mushti": make_hand(flex={f: CURL for f in MCP}, thumb_dir=-65, thumb_flex=(30, 20)),
    "Shikhara": make_hand(flex={f: CURL for f in MCP}, thumb_dir=5),
    "Hamsasya": make_hand(spread={"middle": 0, "ring": -10, "pinky": -22}, flex={"index": (40, 60, 30)},
                          thumb_to="index"),
}
