"""Evaluate the existing scan 40 pair without training."""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import torch

from eval_dtu import evaluate
from notebooks.lab import ROOT
import dtu


def git(*args):
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
        return result.stdout if result.returncode == 0 else "unknown"
    except FileNotFoundError:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--off-checkpoint", type=Path, default=ROOT / "checkpoint/scan40_pair/off/checkpoint_40000.pt")
    parser.add_argument("--on-checkpoint", type=Path, default=ROOT / "checkpoint/scan40_pair/on/checkpoint_40000.pt")
    parser.add_argument("--step", type=int, default=40000)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dtu/scan40")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/results/scan40.json")
    parser.add_argument("--training-record", type=Path)
    parser.add_argument("--check-only", action="store_true", help="Check checkpoint steps without rendering or training.")
    args = parser.parse_args()
    if min(args.step, args.scale) < 1:
        parser.error("The step and scale must be positive.")
    output = args.output
    paths = {"off": args.off_checkpoint, "on": args.on_checkpoint}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("step") != args.step:
            raise ValueError(f"Expected step {args.step}: {path}")
        del saved
    protocol = None
    if args.training_record is not None:
        protocol = json.loads(args.training_record.read_text(encoding="utf-8"))
        if protocol["steps"] != args.step or protocol["training_views"] != [25, 22, 28] or protocol["seed"] != 42:
            raise ValueError("The training record does not match the requested controlled pair.")
        if Path(protocol["scene"]).resolve() != args.data_dir.resolve():
            raise ValueError("The training record and evaluation use different scenes.")
        for name, path in paths.items():
            if Path(protocol["checkpoints"][name]).resolve() != path.resolve():
                raise ValueError(f"The training record names a different {name} checkpoint.")
    if args.check_only:
        print(f"Both checkpoints are at step {args.step}. No rendering or training started.")
        return
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores, hashes, logs = {}, {}, {}
    for name, path in paths.items():
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        stream = io.StringIO()
        print(f"Evaluate {name} at scale {args.scale} on {device}.", flush=True)
        with contextlib.redirect_stdout(stream):
            _, scores[name] = evaluate(path, args.data_dir, args.scale, 512, device,
                                       return_per_view=True)
        logs[name] = stream.getvalue()
        if set(scores[name]) != set(dtu.TEST_IDS):
            raise ValueError("Expected one score for each of the 25 test views.")
        print(logs[name], flush=True)
    record = {
        "scene": args.data_dir.name, "step": args.step, "evaluation_scale": args.scale,
        "device": device, "evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {name: str(path.resolve()) for name, path in paths.items()},
        "checkpoint_sha256": hashes, "evaluation_revision": git("rev-parse", "HEAD").strip(),
        "evaluation_status": git("status", "--short"), "evaluation_local_diff": git("diff", "--", "*.py"),
        "training_provenance": "Unknown. Existing checkpoints do not store training settings or code revision.",
        "required_training_views": [25, 22, 28], "required_seed": 42,
        "required_final_depth_weights": {"off": 0.0, "on": 0.1},
        "score_precision": "Full precision per-view scores from eval_dtu.py; display values are rounded.",
    }
    if protocol is not None:
        record["training_provenance"] = "Recorded by compare_depth.py before this training pair."
        record["training_record"] = protocol
    rows = [{"view": i, "off_db": scores["off"][i], "on_db": scores["on"][i],
             "on_minus_off_db": scores["on"][i]-scores["off"][i]} for i in dtu.TEST_IDS]
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, log in logs.items():
        (output.parent / f"{output.stem}_{name}.log").write_text(log, encoding="utf-8")
    means = {name: sum(scores[name].values()) / len(dtu.TEST_IDS) for name in ("off", "on")}
    output.write_text(json.dumps({"record": record, "per_view": rows, "means_db": means,
                                 "on_minus_off_db": means["on"] - means["off"]}, indent=2), encoding="utf-8")
    print(f"Off {means['off']:.3f} dB | on {means['on']:.3f} dB | on minus off {means['on']-means['off']:+.3f} dB")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
