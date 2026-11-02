"""Compare sphere depth weights and find a stopping point on validation views."""

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from data import get_rays, load_data
from model import MipNeRF
from train import NUM_STEPS, SEED, train


def prepare_views(split, indices):
    images, poses, focal = load_data("toy_sphere", split, scale=4)
    h, w = images.shape[1:3]
    views = []
    for i in indices:
        rays = get_rays(h, w, focal, poses[i])
        views.append(([r.reshape(-1, r.shape[-1]) for r in rays], images[i].reshape(-1, 3)))
    return views


@torch.inference_mode()
def score(model, views):
    device = next(model.parameters()).device
    bounds = model.near, model.far
    model.near, model.far = 2., 6.
    was_training = model.training
    model.eval()
    try:
        scores = []
        for rays, target in views:
            predictions = [model(*(r.to(device) for r in batch))[1].cpu()
                           for batch in zip(*(r.split(512) for r in rays))]
            mse = (torch.cat(predictions) - target).square().mean()
            scores.append(float(-10 * mse.log10()))
        return float(np.mean(scores))
    finally:
        model.near, model.far = bounds
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stop-after", type=int, default=NUM_STEPS,
                        help="Last training step. Keep the full learning rate schedule.")
    args = parser.parse_args()
    if not 1 <= args.stop_after <= NUM_STEPS:
        parser.error(f"--stop-after must be between 1 and {NUM_STEPS}")
    output = Path("checkpoint") / ("sphere_demo_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True)
    selected = np.random.default_rng(SEED).choice(12, 3, replace=False)
    validation = prepare_views("train", sorted(set(range(12)) - set(selected)))
    steps = {10, 25, 50, 75, 100, 150, 200, 300, 400}
    steps.update(range(500, NUM_STEPS + 1, 500))
    steps = {s for s in steps if s <= args.stop_after} | {args.stop_after}
    curves = {}
    for name, weight in (("off", 0.), ("on", 0.1)):
        folder = output / name
        curves[name] = {}

        def measure(step, model):
            if step in steps:
                curves[name][step] = score(model, validation)
                torch.save(model.state_dict(), folder / f"demo_{step}.pt")
                print(name, step, curves[name][step], flush=True)

        train(3, "toy_sphere", folder, weight, stop_after=args.stop_after, after_step=measure)

    with (output / "validation.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["step", "off_db", "on_db", "on_minus_off_db"])
        for step in sorted(steps):
            off, on = (curves[key][step] for key in ("off", "on"))
            writer.writerow([step, off, on, on - off])
    best = max(sorted(steps), key=lambda s: curves["on"][s] - curves["off"][s])
    test = prepare_views("test", range(4))
    model = MipNeRF().to("cuda" if torch.cuda.is_available() else "cpu")
    for name in ("off", "on"):
        model.load_state_dict(torch.load(output / name / f"demo_{best}.pt", weights_only=True,
                                         map_location="cpu"))
        print("Test", name, best, score(model, test))
    print("Results:", output)


if __name__ == "__main__":
    main()
