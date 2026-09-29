"""Download Qwen3-0.6B (Apache-2.0, ~1.5 GB) into models/Qwen3-0.6B for the coaching line."""
import os

from huggingface_hub import snapshot_download

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    print(snapshot_download("Qwen/Qwen3-0.6B", local_dir=os.path.join(ROOT, "models", "Qwen3-0.6B")))
