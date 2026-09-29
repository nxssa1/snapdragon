"""Checks the rule table against synthetic hands.  Run: python -m tests.test_mudras"""
import numpy as np

from mudraguru.features import compute_features
from mudraguru.mudras import MUDRAS, classify, score_mudra
from tests.synth_hand import CURL, POSES, make_hand, randomize


def test_every_mudra_has_a_pose():
    assert set(POSES) == set(MUDRAS)


def test_each_pose_is_recognised_under_rotation_scale_mirror_noise():
    rng = np.random.default_rng(0)
    failures = []
    for name, lm in POSES.items():
        for _ in range(25):
            ranked = classify(compute_features(randomize(lm, rng)))
            if ranked[0].name != name or ranked[0].score < 0.7:
                failures.append((name, ranked[0].name, round(ranked[0].score, 2),
                                 round(score_mudra(compute_features(lm), name).score, 2)))
    assert not failures, failures[:10]


def test_margin_between_best_and_second():
    for name, lm in POSES.items():
        ranked = classify(compute_features(lm))
        assert ranked[0].score - ranked[1].score > 0.3, (name, [(r.name, round(r.score, 2)) for r in ranked[:3]])


def test_correction_names_the_wrong_finger():
    # Pataka attempt with the ring finger bent -> should ask to straighten the ring finger.
    lm = make_hand(spread={"index": 2, "ring": -2, "pinky": -4}, flex={"ring": CURL}, thumb_dir=15)
    s = score_mudra(compute_features(lm), "Pataka")
    assert s.corrections[0].hint == "Straighten your ring finger", s.corrections
    # Ardhachandra attempt with thumb tucked -> should ask to move the thumb out.
    s = score_mudra(compute_features(POSES["Pataka"]), "Ardhachandra")
    assert "thumb out" in s.corrections[0].hint, s.corrections


if __name__ == "__main__":
    for fn in [v for k, v in dict(globals()).items() if k.startswith("test_")]:
        fn()
        print("ok ", fn.__name__)
