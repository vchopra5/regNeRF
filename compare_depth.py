"""Compare depth smoothing on and off on DTU scan 40. Canny is off in both runs."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import torch
import dtu
import train as training
from eval_dtu import evaluate
from train import DEPTH_DECAY_STEPS, NUM_STEPS, SCALE, SEED, add_render_settings, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_render_settings(parser)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint/scan40_pair"))
    args = parser.parse_args()
    scene = Path("data/dtu/scan40")
    depth_on_weight = 0.1
    conditions = [("on", depth_on_weight), ("off", 0.0)]
    print(f"Train two models on {scene.name}, views 25, 22, and 28. Use seed {SEED} and {NUM_STEPS} steps.")
    print(f"Run depth on first. Its fine depth weight goes from {args.depth_weight_start} "
          f"to {depth_on_weight} by step {DEPTH_DECAY_STEPS}. Then run depth off at zero.")
    print(f"Score the fine RGB output at scale {SCALE}. For each test photo, select the object pixels")
    print("with its mask. Compute PSNR from their RGB mean squared error. Average the 25 scores.")
    print("A positive on-minus-off difference means better foreground colour prediction on this scan.")
    print("This metric alone cannot establish more accurate depth. Use the sphere depth maps for that test.")
    if not args.run:
        print("Pass --run to start both training runs.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def git(*arguments):
        try:
            result = subprocess.run(["git", *arguments], capture_output=True, text=True,
                                    encoding="utf-8", errors="replace")
            return result.stdout.strip() if result.returncode == 0 else "unknown"
        except FileNotFoundError:
            return "unknown"
    protocol = dict(scene=str(scene), started_utc=datetime.now(timezone.utc).isoformat(),
                    code_revision=git("rev-parse", "HEAD"), local_status=git("status", "--short"),
                    local_diff=git("diff", "HEAD", "--", "*.py"), seed=SEED, steps=NUM_STEPS,
                    training_views=list(dtu.TRAIN_IDS[:3]), test_views=list(dtu.TEST_IDS),
                    training_scale=SCALE, automatic_evaluation_scale=SCALE,
                    device="cuda" if torch.cuda.is_available() else "cpu", optimizer="Adam",
                    background=args.background or "black",
                    density_noise_std=1.0 if args.density_noise_std is None else args.density_noise_std,
                    fine_weights={"on": depth_on_weight, "off": 0.0},
                    initial_fine_weights={"on": args.depth_weight_start, "off": 0.0},
                    coarse_depth_weights={"on": 0.1, "off": 0.0},
                    config={k: getattr(training, k) for k in ("BATCH_SIZE", "DEPTH_RAYS", "PATCH_SIZE",
                            "LR_INIT", "LR_FINAL", "LR_DELAY_STEPS", "LR_DELAY_MULT", "BOUND_STEPS",
                            "INITIAL_RANGE_FRACTION", "DEPTH_DECAY_STEPS")},
                    code_sha256={name: hashlib.sha256(Path(name).read_bytes()).hexdigest()
                                 for name in ("train.py", "model.py", "data.py", "dtu.py", "compare_depth.py", "eval_dtu.py")},
                    checkpoints={name: str(args.output_dir / name / f"checkpoint_{NUM_STEPS}.pt") for name, _ in conditions})
    validation = scene / "validation.json"
    if validation.is_file():
        protocol["data_validation"] = json.loads(validation.read_text())
    (args.output_dir / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    scores = {}
    for name, weight in conditions:
        folder = args.output_dir / name
        train(3, scene, folder, depth_weight=weight, background=args.background,
              density_noise_std=args.density_noise_std, depth_weight_start=args.depth_weight_start, seed=SEED)
        scores[name] = evaluate(folder / "mipnerf_final.pt", scene, SCALE, 512,
                                "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Off {scores['off']:.3f} dB | on {scores['on']:.3f} dB | "
          f"on minus off {scores['on'] - scores['off']:+.3f} dB")


if __name__ == "__main__":
    main()
