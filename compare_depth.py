"""Explain the scan 40 experiment. Pass --run to train and score both models."""

import argparse
from pathlib import Path

import torch
from eval_dtu import evaluate
from train import DEPTH_DECAY_STEPS, DEPTH_WEIGHT_START, NUM_STEPS, SCALE, SEED, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint/scan40_pair"))
    args = parser.parse_args()
    scene = Path("data/dtu/scan40")
    conditions = [("off", 0.0), ("on", 0.1)]
    print(f"Train two models on {scene.name}, views 25, 22, and 28. Use seed {SEED} and {NUM_STEPS} steps.")
    print(f"Depth off stays at zero. Depth on decays linearly from {DEPTH_WEIGHT_START} "
          f"to {conditions[1][1]} by step {DEPTH_DECAY_STEPS}.")
    print(f"Score the fine RGB output at scale {SCALE}. For each test photo, select the object pixels")
    print("with its mask. Compute PSNR from their RGB mean squared error. Average the 25 scores.")
    print("A positive on-minus-off difference means better foreground colour prediction on this scan.")
    print("This metric alone cannot establish more accurate depth. Use the sphere depth maps for that test.")
    if not args.run:
        print("Pass --run to start both training runs.")
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores = {}
    for name, weight in conditions:
        folder = args.output_dir / name
        train(3, scene, folder, depth_weight=weight)
        scores[name] = evaluate(folder / "mipnerf_final.pt", scene, SCALE, 512,
                                "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Off {scores['off']:.3f} dB | on {scores['on']:.3f} dB | "
          f"on minus off {scores['on'] - scores['off']:+.3f} dB")


if __name__ == "__main__":
    main()
