"""Export Qualcomm AI Hub's MediaPipe-Hand-Detection (detector + landmark) to ONNX, locally.

`qai-hub-models export mediapipe_hand` compiles on AI Hub and needs an API token; this script
instead loads the same PyTorch modules from qai_hub_models and runs torch.onnx.export, so no
token is needed. The palm anchors are saved next to the models for the post-processing.

Run in the export venv (it needs torch + qai-hub-models, the app does not):
    .venv-export\\Scripts\\python tools\\export_hand_onnx.py
"""
import os

import numpy as np
import torch
from qai_hub_models.models.mediapipe_hand.model import HandDetector, HandLandmarkDetector

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def export(model, name, output_names):
    model.eval()
    dummy = torch.rand(1, 3, 256, 256)
    path = os.path.join(OUT, f"{name}.onnx")
    torch.onnx.export(model, (dummy,), path, input_names=["image"], output_names=output_names,
                      opset_version=17, dynamo=False)
    print("wrote", path, f"{os.path.getsize(path) / 1e6:.1f} MB")


def main():
    os.makedirs(OUT, exist_ok=True)
    detector = HandDetector.from_pretrained()  # include_postprocessing=False: raw boxes + scores
    export(detector, "hand_detector", ["box_coords", "box_scores"])
    np.save(os.path.join(OUT, "anchors_palm.npy"), detector.detector.anchors.numpy())
    export(HandLandmarkDetector.from_pretrained(), "hand_landmark_detector", ["scores", "lr", "landmarks"])


if __name__ == "__main__":
    main()
