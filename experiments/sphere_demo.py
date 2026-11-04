"""Compare sphere depth weights on held-out test views."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from data import get_rays, load_data
from generate_sphere import generate, scan40_rotations
from model import load_model, render_rays
from train import NUM_STEPS, SCALE, SEED, add_depth_start_option, train


def prepare_views(split, indices, scene="toy_sphere"):
    images, poses, intrinsics = load_data(scene, split, scale=SCALE)
    h, w = images.shape[1:3]
    views = []
    for i in indices:
        rays = get_rays(h, w, intrinsics[i], poses[i])
        views.append(([r.reshape(-1, r.shape[-1]) for r in rays], images[i].reshape(-1, 3)))
    return views


@torch.inference_mode()
def score(model, views, *, sphere_foreground=False):
    device = next(model.parameters()).device
    bounds = model.near, model.far
    model.near, model.far = 2., 6.
    was_training = model.training
    model.eval()
    try:
        scores = []
        for rays, target in views:
            if sphere_foreground:
                origins, directions, _ = rays
                # Pixels whose camera rays intersect the unit sphere at the origin.
                hit = ((origins * directions).sum(-1).square() >=
                       directions.square().sum(-1) * (origins.square().sum(-1) - 1))
                rays, target = [field[hit] for field in rays], target[hit]
            mse = (render_rays(model, rays, 512, device).cpu() - target).square().mean()
            scores.append(float(-10 * mse.log10()))
        return float(np.mean(scores))
    finally:
        model.near, model.far = bounds
        model.train(was_training)


TEST_STEP = 5000


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_depth_start_option(parser, default=None)
    parser.add_argument("--dtu-views", "--nearby-views", dest="dtu_views", action="store_true",
                        help="Use the three training and 25 test camera angles from DTU scan 40.")
    parser.add_argument("--varied-distances", action="store_true",
                        help="Use training camera distances of 3.6, 4.0, and 4.4 scene units.")
    parser.add_argument("--stop-after", type=int,
                        help="Last training step. Keep the full learning rate schedule.")
    args = parser.parse_args()
    test_step = (args.stop_after or TEST_STEP) if args.dtu_views else TEST_STEP
    if args.stop_after is None:
        args.stop_after = test_step
    fine_weight = 0.1
    start_weight = fine_weight if args.depth_weight_start is None else args.depth_weight_start
    if not 1 <= args.stop_after <= NUM_STEPS:
        parser.error(f"--stop-after must be between 1 and {NUM_STEPS}")
    output = Path("checkpoint") / ("sphere_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True)
    scene = Path("toy_sphere")
    if args.dtu_views:
        scene = output / "sphere_dtu_views"
        rotations = scan40_rotations(Path("data/dtu/scan40"))
        generate(scene, train_distances=(3.6, 4.0, 4.4), rotations=rotations, radius=1.0,
                 image_size=(1600, 1200))
    elif args.varied_distances:
        scene = output / "sphere_varied_distances"
        generate(scene, train_distances=(3.6, 4.0, 4.4))
    if args.dtu_views:
        validation = prepare_views("val", range(len(rotations["val"])), scene)
    else:
        selected = np.random.default_rng(SEED).choice(12, 3, replace=False)
        validation = prepare_views("train", sorted(set(range(12)) - set(selected)), scene)
    steps = {10, 25, 50, 75, 100, 150, 200, 300, 400}
    steps.update(range(500, NUM_STEPS + 1, 500))
    steps = {s for s in steps if s <= args.stop_after} | {args.stop_after}
    curves = {}
    for name, weight in (("off", 0.), ("on", fine_weight)):
        folder = output / name
        curves[name] = {}

        def measure(step, model):
            if step in steps:
                curves[name][step] = score(model, validation)
                torch.save(model.state_dict(), folder / f"demo_{step}.pt")
                print(name, step, curves[name][step], flush=True)

        train(3, scene, folder, weight, stop_after=args.stop_after, after_step=measure,
              regnerf_density=True, use_all_synthetic_poses=True, depth_weight_start=start_weight)

    with (output / "validation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "off_db", "on_db", "on_minus_off_db"])
        for step in sorted(steps):
            off, on = (curves[key][step] for key in ("off", "on"))
            writer.writerow([step, off, on, on - off])
    if args.stop_after < test_step:
        print(f"Stopped before step {test_step}. Validation results: {output}")
        return
    test = prepare_views("test", range(len(rotations["test"]) if args.dtu_views else 4), scene)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for name in ("off", "on"):
        model = load_model(output / name / f"demo_{test_step}.pt", device)
        print(f"Depth {name} at step {test_step} | test {score(model, test):.3f} dB")
        if args.dtu_views:
            print(f"Mean foreground PSNR (25 test views): {score(model, test, sphere_foreground=True):.3f} dB")
    print("Results:", output)


if __name__ == "__main__":
    main()
