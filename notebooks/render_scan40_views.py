"""Save the scan 40 renders that notebook 2 shows. Do not train.

Render every held-out test view (colour and depth) from the depth-off and depth-on checkpoints, at
a small size (scale 4, 400 x 300) as JPEG. The notebook reads these files, so the lab needs no
checkpoints and no GPU, and students can switch between views at once. It saves no DTU photos:
the notebook reads bundled copies from notebooks/results/scan40_reference.

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
from provenance import file_hashes, git, relative
from train import NUM_STEPS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "results" / "scan40"
def depth_image(depth, low, high):
    """Grey depth map: light is near, dark is far. Both models use the same range."""
    shade = np.clip((high - depth) / (high - low), 0, 1)
    return Image.fromarray(np.rint(255 * shade).astype("uint8"))


@torch.inference_mode()
def render(model, scene, view, scale, device):
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
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--conditions", nargs="+", metavar="NAME=CHECKPOINT",
                        help="Render these checkpoints instead of PAIR_DIR/off and PAIR_DIR/on. "
                             "Names appear in the file names, e.g. view1_rgb_NAME.jpg.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.output.mkdir(parents=True, exist_ok=True)

    checkpoints, renders = {}, {}
    if args.conditions:
        for item in args.conditions:
            name, _, path = item.partition("=")
            if not name or not path:
                parser.error(f"Use NAME=CHECKPOINT, not {item!r}.")
            checkpoints[name] = Path(path)
    else:
        checkpoints = {name: args.pair_dir / name / f"checkpoint_{args.step}.pt" for name in ("off", "on")}
    for name in checkpoints:
        if torch.load(checkpoints[name], map_location="cpu", weights_only=True).get("step") != args.step:
            raise ValueError(f"Expected step {args.step}: {checkpoints[name]}")
        model = load_model(checkpoints[name], device, near=dtu.BOUNDS[0], far=dtu.BOUNDS[1])
        renders[name] = {view: render(model, args.data_dir, view, args.scale, device) for view in dtu.TEST_IDS}
        print(f"Rendered {len(dtu.TEST_IDS)} views with {name}", flush=True)

    depth_ranges = {}
    for view in dtu.TEST_IDS:
        # One depth range per view for both models: the 2nd to 98th percentile of their combined depth.
        combined = np.concatenate([renders[name][view][1].ravel() for name in renders])
        low, high = (float(v) for v in np.percentile(combined, [2, 98]))
        depth_ranges[view] = [low, high]
        for name in renders:
            rgb, depth = renders[name][view]
            rgb.save(args.output / f"view{view}_rgb_{name}.jpg", quality=90)
            depth_image(depth, low, high).save(args.output / f"view{view}_depth_{name}.jpg", quality=90)

    record = dict(views=list(dtu.TEST_IDS), scale=args.scale, step=args.step, depth_ranges=depth_ranges,
                  checkpoints={name: relative(path) for name, path in checkpoints.items()},
                  rendered_utc=datetime.now(timezone.utc).isoformat(), code_revision=git("rev-parse", "HEAD"),
                  code_sha256=file_hashes(("model.py", "data.py", "dtu.py", "notebooks/render_scan40_views.py")))
    (args.output / "render_record.json").write_text(json.dumps(record, indent=2))
    print(f"Saved {len(dtu.TEST_IDS)} views to {args.output}")


if __name__ == "__main__":
    main()
