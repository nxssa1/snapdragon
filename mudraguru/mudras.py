"""Rule table for single-hand (asamyuta) hasta mudras, and the scorer.

Each mudra lists target ranges for features from features.py:
    "feature": (low, high)            weight 1
    "feature": (low, high, weight)

Scoring: every constraint gives a match in (0, 1] -- 1 inside the range, falling off
as a Gaussian of (distance outside the range / tolerance). The mudra score is the
weighted geometric mean, so one badly-wrong finger pulls the score down hard.
The constraint with the lowest match becomes the correction hint.

To tune: run the app with the debug overlay (key D) and move the ranges below.
"""
import math
from dataclasses import dataclass

STRAIGHT = (0.70, 1.0)
BENT = (0.0, 0.45)
FIST = (0.0, 0.35)
TOGETHER = (0.0, 12.0)
APART = (15.0, 60.0)
TOUCHING = (0.0, 0.25)

# Per-feature tolerance (how far outside the range counts as one "sigma") and hint text.
# hints = (value too low, value too high)
FEATURE_INFO = {
    "ext_thumb": (0.15, ("Straighten your thumb", "Bend your thumb in")),
    "ext_index": (0.15, ("Straighten your index finger", "Bend your index finger")),
    "ext_middle": (0.15, ("Straighten your middle finger", "Bend your middle finger")),
    "ext_ring": (0.15, ("Straighten your ring finger", "Bend your ring finger")),
    "ext_pinky": (0.15, ("Straighten your little finger", "Bend your little finger")),
    "thumb_angle": (10.0, ("Move your thumb out, away from the palm",
                           "Bring your thumb in along the palm")),
    "spread_im": (5.0, ("Spread index and middle fingers apart",
                        "Bring index and middle fingers together")),
    "spread_mr": (5.0, ("Spread middle and ring fingers apart",
                        "Bring middle and ring fingers together")),
    "spread_rp": (5.0, ("Spread ring and little fingers apart",
                        "Bring ring and little fingers together")),
    "touch_index": (0.10, ("Move thumb away from index fingertip",
                           "Touch your thumb tip to your index fingertip")),
    "touch_middle": (0.10, ("Move thumb away from middle fingertip",
                            "Touch your thumb tip to your middle fingertip")),
    "touch_ring": (0.10, ("Move thumb away from ring fingertip",
                          "Touch your thumb tip to your ring fingertip")),
    "touch_pinky": (0.10, ("Move thumb away from little fingertip",
                           "Touch your thumb tip to your little fingertip")),
}

MUDRAS = {
    "Pataka": {
        "meaning": "flag",
        "description": "All four fingers straight and held together; thumb bent in against the side of the palm.",
        "rules": {
            "ext_index": STRAIGHT, "ext_middle": STRAIGHT, "ext_ring": STRAIGHT, "ext_pinky": STRAIGHT,
            "spread_im": TOGETHER, "spread_mr": TOGETHER, "spread_rp": TOGETHER,
            "thumb_angle": (0.0, 40.0),
        },
    },
    "Tripataka": {
        "meaning": "three parts of the flag",
        "description": "Pataka with the ring finger bent down.",
        "rules": {
            "ext_index": STRAIGHT, "ext_middle": STRAIGHT, "ext_ring": BENT, "ext_pinky": STRAIGHT,
            "spread_im": TOGETHER, "thumb_angle": (0.0, 45.0),
            "touch_ring": (0.40, 5.0),  # thumb is not holding the ring finger (that is Mayura)
        },
    },
    "Ardhapataka": {
        "meaning": "half flag",
        "description": "Tripataka with the little finger also bent: index and middle straight and together.",
        "rules": {
            "ext_index": STRAIGHT, "ext_middle": STRAIGHT, "ext_ring": BENT, "ext_pinky": BENT,
            "spread_im": TOGETHER, "thumb_angle": (0.0, 50.0),
        },
    },
    "Kartarimukha": {
        "meaning": "scissors",
        "description": "Index and middle straight and spread apart like scissors; ring and little bent, held by the thumb.",
        "rules": {
            "ext_index": STRAIGHT, "ext_middle": STRAIGHT, "ext_ring": BENT, "ext_pinky": BENT,
            "spread_im": (18.0, 70.0, 1.5), "touch_ring": (0.0, 0.40, 0.5),
        },
    },
    "Mayura": {
        "meaning": "peacock",
        "description": "Ring fingertip touches the thumb tip; the other fingers stay straight.",
        "rules": {
            "ext_index": STRAIGHT, "ext_middle": STRAIGHT, "ext_pinky": STRAIGHT,
            "touch_ring": (*TOUCHING, 2.0), "touch_index": (0.4, 5.0),
        },
    },
    "Ardhachandra": {
        "meaning": "half moon",
        "description": "Pataka with the thumb stretched straight out to the side.",
        "rules": {
            "ext_index": STRAIGHT, "ext_middle": STRAIGHT, "ext_ring": STRAIGHT, "ext_pinky": STRAIGHT,
            "spread_im": TOGETHER, "spread_mr": TOGETHER, "spread_rp": TOGETHER,
            "ext_thumb": (0.6, 1.0), "thumb_angle": (55.0, 130.0, 1.5),
        },
    },
    "Alapadma": {
        "meaning": "full-bloomed lotus",
        "description": "All fingers straight and fanned wide apart, thumb open.",
        "rules": {
            "ext_index": STRAIGHT, "ext_middle": STRAIGHT, "ext_ring": STRAIGHT, "ext_pinky": STRAIGHT,
            "spread_im": APART, "spread_mr": APART, "spread_rp": APART,
            "thumb_angle": (45.0, 130.0),
        },
    },
    "Mushti": {
        "meaning": "fist",
        "description": "All four fingers curled into the palm; thumb folded across the fingers.",
        "rules": {
            "ext_index": FIST, "ext_middle": FIST, "ext_ring": FIST, "ext_pinky": FIST,
            "thumb_angle": (50.0, 180.0, 1.5),
        },
    },
    "Shikhara": {
        "meaning": "peak",
        "description": "Mushti with the thumb raised straight up.",
        "rules": {
            "ext_index": FIST, "ext_middle": FIST, "ext_ring": FIST, "ext_pinky": FIST,
            "ext_thumb": (0.6, 1.0), "thumb_angle": (0.0, 35.0, 1.5),
        },
    },
    "Hamsasya": {
        "meaning": "swan's beak",
        "description": "Thumb tip and index tip joined; middle, ring and little fingers straight and slightly apart.",
        "rules": {
            "touch_index": (*TOUCHING, 2.0),
            "ext_middle": STRAIGHT, "ext_ring": STRAIGHT, "ext_pinky": STRAIGHT,
            "spread_mr": (6.0, 45.0), "spread_rp": (6.0, 45.0),
        },
    },
}

MIN_MATCH = 1e-3


@dataclass
class Correction:
    feature: str
    match: float
    hint: str


@dataclass
class MudraScore:
    name: str
    score: float
    corrections: list  # Correction, worst first


def _constraint_match(value, lo, hi, tol):
    if lo <= value <= hi:
        return 1.0, None
    d = (lo - value) if value < lo else (value - hi)
    return max(math.exp(-0.5 * (d / tol) ** 2), MIN_MATCH), ("low" if value < lo else "high")


def score_mudra(feats, name):
    rules = MUDRAS[name]["rules"]
    log_sum, w_sum, corrections = 0.0, 0.0, []
    for feat, spec in rules.items():
        lo, hi = spec[0], spec[1]
        w = spec[2] if len(spec) > 2 else 1.0
        tol, (hint_low, hint_high) = FEATURE_INFO[feat]
        match, side = _constraint_match(feats[feat], lo, hi, tol)
        log_sum += w * math.log(match)
        w_sum += w
        if side is not None:
            corrections.append(Correction(feat, match, hint_low if side == "low" else hint_high))
    corrections.sort(key=lambda c: c.match)
    return MudraScore(name, math.exp(log_sum / w_sum), corrections)


def classify(feats):
    """All mudras scored, best first."""
    return sorted((score_mudra(feats, n) for n in MUDRAS), key=lambda s: s.score, reverse=True)
