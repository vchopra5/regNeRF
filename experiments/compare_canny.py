"""Compare standard depth smoothing with Canny-masked smoothing on a synthetic scene."""

import argparse
from pathlib import Path

from train import NUM_STEPS, train


def train_pair(scene, output_dir, steps, scale):
    """Use the same three training views and depth weights in both runs."""
    for folder, use_canny in (("regnerf", False), ("canny", True)):
        train(
            num_views=3,
            data_dir=scene,
            checkpoint_dir=output_dir / folder,
            depth_weight=0.1,
            depth_weight_start=0.1,
            virtual_canny=use_canny,
            regnerf_density=True,
            stop_after=steps,
            scale=scale,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True,
                        help="Synthetic scene folder with transforms_train.json and RGB images.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Write the regnerf and canny checkpoints here.")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--scale", type=int, default=1, help="Image downsampling factor.")
    parser.add_argument("--run", action="store_true", help="Start both training runs.")
    args = parser.parse_args()
    if not 1 <= args.steps <= NUM_STEPS or args.scale < 1:
        parser.error(f"Use 1 to {NUM_STEPS} steps and a positive scale.")
    print(f"Train on three views of {args.scene} for {args.steps} steps at scale {args.scale}.")
    print("Both runs smooth depth on sampled camera patches with a constant weight of 0.1.")
    print("The canny run skips a pixel pair if Canny detects an edge at either pixel.")
    if args.run:
        if not (args.scene / "transforms_train.json").is_file():
            parser.error("The scene needs a transforms_train.json file.")
        train_pair(args.scene, args.output_dir, args.steps, args.scale)
    else:
        print("Pass --run to start training.")


if __name__ == "__main__":
    main()
