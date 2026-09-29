"""Guru's coaching line: a small local LLM turns (mudra, correction) into one friendly sentence.

Runs on a background thread so the video loop never waits. Lines are cached per
(mudra, correction) pair -- in memory and in logs/coach_cache.json -- so each pair is only
generated once; when the student is not asking for anything new, the worker pre-generates
the remaining pairs so later sessions are instant.

Model: Qwen3-0.6B (Apache-2.0) via transformers on CPU. If transformers/torch or the model
folder are missing, the coach reports "unavailable" and the app just shows the plain hint.
"""
import json
import os
import threading

from mudraguru.mudras import FEATURE_INFO, MUDRAS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MODEL = os.path.join(ROOT, "models", "Qwen3-0.6B")
DEFAULT_CACHE = os.path.join(ROOT, "logs", "coach_cache.json")

SYSTEM = ("You are a warm Bharatanatyam teacher. Rewrite the given correction as ONE short, encouraging "
          "sentence (under 18 words) for your student. Keep the exact instruction; you may add an image or "
          "the mudra's meaning. No greeting, no quotes.")

# Few-shot examples: a 0.6B model follows the pattern far better than a description of it.
EXAMPLES = [
    ("Tripataka (three parts of the flag)", "Bend your ring finger",
     "Lovely flag - now fold just your ring finger down to make Tripataka."),
    ("Alapadma (full-bloomed lotus)", "Spread ring and little fingers apart",
     "Let your lotus bloom: open the ring and little fingers wider apart."),
    ("Shikhara (peak)", "Straighten your thumb",
     "Raise your thumb tall and straight, like the peak of a temple."),
]


# Largest value each feature can take, by name prefix: a range ending there has no "too high" hint.
FEATURE_MAX = {"ext_": 1.0, "thumb_angle": 180.0, "spread_": 180.0, "touch_": 5.0}


def all_pairs():
    """Every (mudra, hint) the rule table can produce."""
    pairs = []
    for name, m in MUDRAS.items():
        for feat, spec in m["rules"].items():
            lo, hi = spec[0], spec[1]
            low_hint, high_hint = FEATURE_INFO[feat][1]
            top = next(v for prefix, v in FEATURE_MAX.items() if feat.startswith(prefix))
            if lo > 0:
                pairs.append((name, low_hint))
            if hi < top:
                pairs.append((name, high_hint))
    return pairs


class Coach:
    def __init__(self, model_dir=DEFAULT_MODEL, cache_path=DEFAULT_CACHE, threads=4, prewarm=True):
        self.model_dir = model_dir
        self.cache_path = cache_path
        self.threads = threads
        self.prewarm = prewarm
        self.status = "loading"  # loading | ready | unavailable
        self.cache = self._load_cache()
        self._wanted = None
        self._cv = threading.Condition()
        self._stop = False
        self._thread = threading.Thread(target=self._run, name="coach", daemon=True)
        self._thread.start()

    # ---- public, called from the video loop (never blocks) ---------------------------------
    def line(self, mudra, hint):
        """Cached sentence for this pair, or None (and a background request) if not ready yet."""
        key = self._key(mudra, hint)
        if key in self.cache:
            return self.cache[key]
        if self.status != "unavailable":
            with self._cv:
                self._wanted = (mudra, hint)
                self._cv.notify()
        return None

    def close(self):
        with self._cv:
            self._stop = True
            self._cv.notify()

    # ---- worker -----------------------------------------------------------------------------
    @staticmethod
    def _key(mudra, hint):
        return f"{mudra}|{hint}"

    def _load_cache(self):
        try:
            with open(self.cache_path, encoding="utf-8") as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.cache, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, self.cache_path)

    def _load_model(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.set_num_threads(self.threads)  # leave the rest of the CPU for the video loop
        self.tok = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_dir, dtype=torch.float32)
        self.model.eval()
        self.torch = torch

    def generate(self, mudra, hint):
        messages = [{"role": "system", "content": SYSTEM}]
        for ex_mudra, ex_hint, ex_line in EXAMPLES:
            messages += [{"role": "user", "content": f"Mudra: {ex_mudra}\nCorrection: {ex_hint}"},
                         {"role": "assistant", "content": ex_line}]
        messages.append({"role": "user",
                         "content": f"Mudra: {mudra} ({MUDRAS[mudra]['meaning']})\nCorrection: {hint}"})
        prompt = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
        inputs = self.tok(prompt, return_tensors="pt")
        with self.torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=40, do_sample=False)
        text = self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return self._clean(text, hint)

    @staticmethod
    def _clean(text, hint):
        text = text.replace("<think>", "").replace("</think>", "").strip().strip('"').strip()
        line = text.splitlines()[0].strip() if text else ""
        if not line:
            return hint + "."
        if len(line) > 140:
            line = line[:140].rsplit(" ", 1)[0] + "..."
        return line

    def _next_job(self):
        with self._cv:
            while True:
                if self._stop:
                    return None
                if self._wanted is not None:
                    job, self._wanted = self._wanted, None
                    if self._key(*job) not in self.cache:
                        return job
                    continue
                if self.prewarm:
                    todo = [p for p in all_pairs() if self._key(*p) not in self.cache]
                    if todo:
                        return todo[0]
                self._cv.wait()

    def _run(self):
        if not os.path.isdir(self.model_dir):
            self.status = "unavailable"
            return
        try:
            self._load_model()
        except Exception as exc:  # missing torch/transformers, bad files...
            print(f"[MudraGuru] coach unavailable: {exc}")
            self.status = "unavailable"
            return
        self.status = "ready"
        while (job := self._next_job()) is not None:
            try:
                self.cache[self._key(*job)] = self.generate(*job)
                self._save_cache()
            except Exception as exc:
                print(f"[MudraGuru] coach failed on {job}: {exc}")
                self.cache[self._key(*job)] = job[1] + "."
