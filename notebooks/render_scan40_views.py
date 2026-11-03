"""Save the scan 40 images that notebook 2 shows. Do not train.

Render one held-out view (colour and depth) from the depth-off and depth-on checkpoints. The notebook
reads these files, so the lab does not need the checkpoints. It does not save DTU photos: the notebook
reads those from data/dtu/scan40, which students download with prepare_dtu.py.

  python -m notebooks.render_scan40_views
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dtu
from data import get_rays
from model import load_model, render_rays
from provenance import file_hashes, git
from train import NUM_STEPS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "results" / "scan40"
def depth_image(depth, low, high):
    """Grey depth map: light is near, dark is far. Both models use the same range."""
    shade = np.clip((high - depth) / (high - low), 0, 1)
    return Image.fromarray(np.rint(255 * shade).astype("uint8"))


@torch.inference_mode()
def render(checkpoint, scene, view, scale, device):
    model = load_model(checkpoint, device, near=dtu.BOUNDS[0], far=dtu.BOUNDS[1])
    with np.load(scene / "cameras.npz") as cameras, Image.open(scene / "image" / f"{view:06}.png") as source:
        h, w, intrinsics, pose = dtu.camera(cameras, view, source.size, scale)
    rays = [field.reshape(h * w, -1) for field in get_rays(h, w, intrinsics, pose)]
    rgb, depth = render_rays(model, rays, 4096, device, return_depth=True)
    rgb = (rgb.reshape(h, w, 3).clamp(0, 1) * 255).round().byte().cpu().numpy()
    return Image.fromarray(rgb), depth.reshape(h, w).cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pair-dir", type=Path, default=ROOT / "checkpoint/scan40_pair_new",
                        help="Folder with off/ and on/ from compare_depth.py.")
    parser.add_argument("--step", type=int, default=NUM_STEPS)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data/dtu/scan40")
    parser.add_argument("--view", type=int, default=45)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.view not in dtu.TEST_IDS:
        parser.error(f"Use a held-out test view: {dtu.TEST_IDS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)

    renders = {}
    for name in ("off", "on"):
        checkpoint = args.pair_dir / name / f"checkpoint_{args.step}.pt"
        if torch.load(checkpoint, map_location="cpu", weights_only=True).get("step") != args.step:
            raise ValueError(f"Expected step {args.step}: {checkpoint}")
        renders[name] = (checkpoint, *render(checkpoint, args.data_dir, args.view, args.scale, device))
    # One depth range for both models: the 2nd to 98th percentile of their combined depth.
    combined = np.concatenate([depth.ravel() for _, _, depth in renders.values()])
    low, high = (float(v) for v in np.percentile(combined, [2, 98]))
    for name, (_, rgb, depth) in renders.items():
        rgb.save(args.output / f"view{args.view}_rgb_{name}.png")
        depth_image(depth, low, high).save(args.output / f"view{args.view}_depth_{name}.png")

    record = dict(view=args.view, scale=args.scale, step=args.step, depth_range=[low, high],
                  checkpoints={name: str(path) for name, (path, _, _) in renders.items()},
                  rendered_utc=datetime.now(timezone.utc).isoformat(), code_revision=git("rev-parse", "HEAD"),
                  code_sha256=file_hashes(("model.py", "data.py", "dtu.py", "notebooks/render_scan40_views.py")))
    (args.output / "render_record.json").write_text(json.dumps(record, indent=2))
    print(f"Saved view {args.view} renders to {args.output}")


if __name__ == "__main__":
    main()
