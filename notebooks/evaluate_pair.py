"""Evaluate a scan 40 pair from compare_depth.py without training.

With no arguments, it evaluates checkpoint/scan40_pair_new at the final step, checks it
against that folder's protocol.json, and writes notebooks/results/scan40.json for notebook 2.
"""

import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
from datetime import datetime, timezone

import torch

from eval_dtu import EVAL_SCALE, evaluate
from notebooks.lab import ROOT
from train import NUM_STEPS
from provenance import git
import dtu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair-dir", type=Path, default=ROOT / "checkpoint/scan40_pair_new",
                        help="Output folder of compare_depth.py, with off/, on/, and protocol.json.")
    parser.add_argument("--step", type=int, default=NUM_STEPS)
    parser.add_argument("--off-checkpoint", type=Path, help="Default: PAIR_DIR/off/checkpoint_STEP.pt")
    parser.add_argument("--on-checkpoint", type=Path, help="Default: PAIR_DIR/on/checkpoint_STEP.pt")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dtu/scan40")
    parser.add_argument("--scale", type=int, default=EVAL_SCALE)
    parser.add_argument("--output", type=Path, default=ROOT / "notebooks/results/scan40.json")
    parser.add_argument("--training-record", type=Path, help="Default: PAIR_DIR/protocol.json, if present.")
    parser.add_argument("--check-only", action="store_true", help="Check checkpoint steps without rendering or training.")
    args = parser.parse_args()
    if min(args.step, args.scale) < 1:
        parser.error("The step and scale must be positive.")
    output = args.output
    paths = {name: getattr(args, f"{name}_checkpoint") or args.pair_dir / name / f"checkpoint_{args.step}.pt"
             for name in ("off", "on")}
    if args.training_record is None and (args.pair_dir / "protocol.json").is_file():
        args.training_record = args.pair_dir / "protocol.json"
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
        "checkpoint_sha256": hashes, "evaluation_revision": git("rev-parse", "HEAD"),
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
