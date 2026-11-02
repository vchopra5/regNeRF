"""Explain the scan 114 experiment. Pass --run to train and score both models."""

import argparse
from pathlib import Path

import torch
from eval_dtu import evaluate
from train import NUM_STEPS, SCALE, SEED, train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("checkpoint/scan114_pair"))
    args = parser.parse_args()
    scene = Path("data/dtu/scan114")
    conditions = [("off", 0.0), ("on", 0.02)]
    print(f"Train two models on scan 114, views 25, 22, and 28. Use seed {SEED} and {NUM_STEPS} steps.")
    print("Change only the weight of the depth smoothness loss on unseen camera patches (0 or 0.02).")
    print(f"Score the fine RGB output at scale {SCALE}. For each test photo, select the object pixels")
    print("with its mask. Compute PSNR from their RGB mean squared error. Average the 25 scores.")
    print("A positive on-minus-off difference means better foreground colour prediction on this scan.")
    print("This metric alone cannot establish more accurate depth. Use the sphere depth maps for that test.")
    if not args.run:
        print("Pass --run to start both training runs.")
        return
    if args.output_dir.exists():
        parser.error("Choose a new --output-dir to preserve previous experiments.")
    args.output_dir.mkdir(parents=True)
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
