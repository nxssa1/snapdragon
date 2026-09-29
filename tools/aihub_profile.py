"""Profile MudraGuru's two hand models on a real Snapdragon device in Qualcomm AI Hub.

Needs a (free) AI Hub account:  pip install qai-hub  and  qai-hub configure --api_token <token>
    python tools/aihub_profile.py --device "Snapdragon X Elite CRD"
    python tools/aihub_profile.py --list-devices
"""
import argparse
import os

import qai_hub as hub

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = ["hand_detector", "hand_landmark_detector"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="Snapdragon X Elite CRD")
    ap.add_argument("--runtime", default="onnx", choices=["onnx", "qnn_context_binary", "qnn_dlc"],
                    help="onnx = ONNX Runtime + QNN EP, what the app uses on Snapdragon")
    ap.add_argument("--list-devices", action="store_true")
    args = ap.parse_args()

    if args.list_devices:
        for d in hub.get_devices():
            print(d.name, "|", d.os, "|", ", ".join(a for a in d.attributes if a.startswith("chipset")))
        return

    device = hub.Device(args.device)
    jobs = []
    for name in MODELS:
        compile_job = hub.submit_compile_job(
            model=os.path.join(ROOT, "models", f"{name}.onnx"),
            device=device,
            input_specs={"image": (1, 3, 256, 256)},
            options=f"--target_runtime {args.runtime}",
            name=f"mudraguru-{name}",
        )
        profile_job = hub.submit_profile_job(model=compile_job.get_target_model(), device=device,
                                             name=f"mudraguru-{name}")
        jobs.append((name, compile_job, profile_job))

    total = 0.0
    for name, compile_job, profile_job in jobs:
        profile = profile_job.download_profile()
        summary = profile["execution_summary"]
        ms = summary["estimated_inference_time"] / 1000.0  # reported in microseconds
        total += ms
        print(f"{name:<24} {ms:7.3f} ms   compile {compile_job.url}   profile {profile_job.url}")
    print(f"{'both models':<24} {total:7.3f} ms on {args.device} ({args.runtime})")


if __name__ == "__main__":
    main()
