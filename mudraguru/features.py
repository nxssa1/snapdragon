"""Geometric features computed from 21 hand landmarks.

Landmark layout (MediaPipe / Qualcomm AI Hub MediaPipe-Hand):
    0 wrist
    thumb 1-4 (CMC, MCP, IP, TIP)
    index 5-8, middle 9-12, ring 13-16, pinky 17-20 (MCP, PIP, DIP, TIP)

All features are invariant to position, scale, rotation and left/right hand,
so the same rule table works for either hand and any camera distance.
"""
import numpy as np

FINGER_JOINTS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
LONG_FINGERS = ("index", "middle", "ring", "pinky")
TIPS = {name: joints[-1] for name, joints in FINGER_JOINTS.items()}

# Skeleton edges, used for drawing.
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


def _angle(u, v):
    """Angle between two vectors in degrees."""
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-9:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v) / denom, -1.0, 1.0))))


def compute_features(lm):
    """Return a dict of named features for a (21, 3) landmark array.

    Coordinates must share one scale on all axes (e.g. x*width, y*height, z*width).

    ext_<finger>   0 (fully bent) .. 1 (straight), from the summed joint bend angles
    thumb_angle    degrees between the thumb (MCP->tip) and the palm axis (wrist->middle MCP);
                   small = thumb along the palm/pointing up, large = thumb out to the side or across
    spread_im/mr/rp  degrees between neighbouring fingers (index-middle, middle-ring, ring-pinky)
    touch_<finger> distance thumb tip -> fingertip, in palm lengths (wrist->middle MCP)
    """
    lm = np.asarray(lm, dtype=np.float64)
    palm_axis = lm[9] - lm[0]
    palm = max(np.linalg.norm(palm_axis), 1e-6)
    f = {}

    for name in LONG_FINGERS:
        mcp, pip, dip, tip = FINGER_JOINTS[name]
        bend = (_angle(lm[mcp] - lm[0], lm[pip] - lm[mcp])
                + _angle(lm[pip] - lm[mcp], lm[dip] - lm[pip])
                + _angle(lm[dip] - lm[pip], lm[tip] - lm[dip]))
        f[f"ext_{name}"] = float(np.clip((200.0 - bend) / 160.0, 0.0, 1.0))

    _, t_mcp, t_ip, t_tip = FINGER_JOINTS["thumb"]
    t_bend = (_angle(lm[t_mcp] - lm[1], lm[t_ip] - lm[t_mcp])
              + _angle(lm[t_ip] - lm[t_mcp], lm[t_tip] - lm[t_ip]))
    f["ext_thumb"] = float(np.clip((130.0 - t_bend) / 100.0, 0.0, 1.0))
    f["thumb_angle"] = _angle(lm[t_tip] - lm[t_mcp], palm_axis)

    def direction(name):
        mcp, _, _, tip = FINGER_JOINTS[name]
        return lm[tip] - lm[mcp]

    f["spread_im"] = _angle(direction("index"), direction("middle"))
    f["spread_mr"] = _angle(direction("middle"), direction("ring"))
    f["spread_rp"] = _angle(direction("ring"), direction("pinky"))

    for name in LONG_FINGERS:
        f[f"touch_{name}"] = float(np.linalg.norm(lm[t_tip] - lm[TIPS[name]]) / palm)
    return f


class FeatureSmoother:
    """Exponential moving average over feature dicts, to steady the readout."""

    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.state = None

    def reset(self):
        self.state = None

    def __call__(self, feats):
        if self.state is None:
            self.state = dict(feats)
        else:
            a = self.alpha
            self.state = {k: a * v + (1 - a) * self.state[k] for k, v in feats.items()}
        return dict(self.state)
