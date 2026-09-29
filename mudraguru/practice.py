"""Practice mode: one target mudra at a time, hold it for a couple of seconds to pass.

Every attempt (passed or skipped) is appended to a CSV so accuracy per mudra
accumulates across sessions.
"""
import csv
import os
import time
from collections import defaultdict

LOG_FIELDS = ["timestamp", "mudra", "result", "seconds", "mean_score", "best_score"]


class PracticeSession:
    def __init__(self, names, log_path, hold_seconds=2.0, threshold=0.7, celebrate_seconds=1.5):
        self.names = list(names)
        self.log_path = log_path
        self.hold_seconds = hold_seconds
        self.threshold = threshold
        self.celebrate_seconds = celebrate_seconds
        self.index = 0
        self.stats = defaultdict(lambda: {"attempts": 0, "passed": 0, "seconds": []})
        self._load_log()
        self._start_attempt()

    @property
    def target(self):
        return self.names[self.index]

    def _start_attempt(self, now=None):
        self.attempt_start = now or time.monotonic()
        self.hold_start = None
        self.scores = []
        self.celebrate_until = 0.0

    def update(self, ranked, now=None):
        """Feed one frame's ranked MudraScores (or None when no hand is visible).

        Returns (target_score_or_None, hold_progress 0..1, passed_this_frame)."""
        now = now or time.monotonic()
        if now < self.celebrate_until:
            return None, 1.0, False
        if self.celebrate_until:  # celebration over -> next mudra
            self._advance(1, now)
        if not ranked:
            self.hold_start = None
            return None, 0.0, False

        target_score = next(s for s in ranked if s.name == self.target)
        self.scores.append(target_score.score)
        holding = ranked[0].name == self.target and target_score.score >= self.threshold
        if not holding:
            self.hold_start = None
            return target_score, 0.0, False
        if self.hold_start is None:
            self.hold_start = now
        progress = min((now - self.hold_start) / self.hold_seconds, 1.0)
        if progress >= 1.0:
            self._record("passed", now)
            self.celebrate_until = now + self.celebrate_seconds
            return target_score, 1.0, True
        return target_score, progress, False

    @property
    def celebrating(self):
        return time.monotonic() < self.celebrate_until

    def next(self):
        self._skip_and_move(1)

    def prev(self):
        self._skip_and_move(-1)

    def _skip_and_move(self, step):
        now = time.monotonic()
        if self.scores and not self.celebrate_until:
            self._record("skipped", now)
        self._advance(step, now)

    def _advance(self, step, now):
        self.index = (self.index + step) % len(self.names)
        self._start_attempt(now)

    def _record(self, result, now):
        seconds = now - self.attempt_start
        st = self.stats[self.target]
        st["attempts"] += 1
        if result == "passed":
            st["passed"] += 1
            st["seconds"].append(seconds)
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mudra": self.target,
            "result": result,
            "seconds": f"{seconds:.1f}",
            "mean_score": f"{sum(self.scores) / len(self.scores):.3f}" if self.scores else "",
            "best_score": f"{max(self.scores):.3f}" if self.scores else "",
        }
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        new = not os.path.exists(self.log_path)
        with open(self.log_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=LOG_FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)

    def _load_log(self):
        if not os.path.exists(self.log_path):
            return
        with open(self.log_path, newline="") as fh:
            for row in csv.DictReader(fh):
                st = self.stats[row["mudra"]]
                st["attempts"] += 1
                if row["result"] == "passed":
                    st["passed"] += 1
                    st["seconds"].append(float(row["seconds"]))

    def summary(self, name):
        st = self.stats[name]
        if not st["attempts"]:
            return "not tried"
        pct = 100 * st["passed"] / st["attempts"]
        avg = f", avg {sum(st['seconds']) / len(st['seconds']):.1f}s" if st["seconds"] else ""
        return f"{st['passed']}/{st['attempts']} ({pct:.0f}%){avg}"
