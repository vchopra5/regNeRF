"""Score a DTU checkpoint near object-mask edges. Do not train.

DTU scan 40 has object masks but no reference depth in this repo. The mask outline is a true
occlusion boundary, so it locates where depth should change sharply. For each test view:

- foreground PSNR: all mask pixels (the same definition as eval_dtu.py, as a cross-check)
- boundary-band PSNR: mask pixels within BAND px of the mask outline
- interior PSNR: mask pixels at least INTERIOR px from the outline
- depth-edge alignment: take the strongest TOP_FRACTION of predicted-depth gradients inside the
  mask plus a BAND px margin. Report the fraction within BAND px of the outline and their mean
  distance to it. The arch's own creases also make depth edges, so this is a proxy, not a depth
  error.

Distances are Chebyshev distances (repeated 3 x 3 dilation), capped at DISTANCE_CAP px.
All sizes are in pixels at the evaluation scale.

  python eval_edges.py --checkpoint PATH [--views 1 2 ...]
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import dtu
from data import get_rays
from eval_dtu import EVAL_SCALE
from model import load_model, render_rays
from train import sobel_magnitude

BAND = 3
INTERIOR = 8
TOP_FRACTION = 0.05
DISTANCE_CAP = 20


def outline_distance(mask, cap=DISTANCE_CAP):
    """Chebyshev distance in pixels from each pixel to the mask outline, capped at cap."""
    m = mask[None, None].float()
    inner = m * (1 - F.max_pool2d(1 - m, 3, 1, 1))  # mask pixels with only mask neighbours
    outline = (m - inner).bool()[0, 0]             # mask pixels next to background
    distance = torch.full(mask.shape, float(cap), device=mask.device)
    reached = outline.clone()
    distance[reached] = 0
    front = reached[None, None].float()
    for step in range(1, cap):
        front = F.max_pool2d(front, 3, 1, 1)
        new = front[0, 0].bool() & ~reached
        distance[new] = step
        reached |= new
    return distance


def psnr(prediction, target):
    mse = (prediction - target).square().mean(dtype=torch.float64).item()
    return -10 * math.log10(mse) if mse else math.inf


@torch.inference_mode()
def score_view(model, scene, cameras, view, scale, device):
    with Image.open(scene / "image" / f"{view:06}.png") as photo:
        size = photo.size
        h, w, intrinsics, pose = dtu.camera(cameras, view, size, scale)
        target = photo.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
    with Image.open(scene / "mask" / f"{view:03}.png") as source:
        mask = np.asarray(source.convert("L").resize((w, h), Image.Resampling.NEAREST)) > 127
    target = torch.from_numpy(np.array(target)).to(device).float() / 255
    mask = torch.from_numpy(mask).to(device)
    rays = [field.reshape(h * w, -1) for field in get_rays(h, w, intrinsics, pose)]
    rgb, depth = render_rays(model, rays, 4096, device, return_depth=True)
    rgb, depth = rgb.reshape(h, w, 3), depth.reshape(h, w)

    distance = outline_distance(mask)
    band = mask & (distance <= BAND)
    interior = mask & (distance >= INTERIOR)
    # Depth gradient magnitude, same blur + Sobel as the regularizer. Valid output is 2 px smaller per side.
    gradient = F.pad(sobel_magnitude(depth[None, ..., None].expand(-1, -1, -1, 3)), (2, 2, 2, 2))[0]
    margin = F.max_pool2d(mask[None, None].float(), 2 * BAND + 1, 1, BAND)[0, 0].bool()
    region = margin.clone()
    region[:2], region[-2:], region[:, :2], region[:, -2:] = False, False, False, False
    values = gradient[region]
    count = max(1, int(TOP_FRACTION * values.numel()))
    threshold = values.topk(count).values.min()
    strong = region & (gradient >= threshold)
    near = distance[strong] <= BAND
    return dict(
        view=view,
        foreground_psnr=psnr(rgb[mask], target[mask]),
        boundary_band_psnr=psnr(rgb[band], target[band]),
        interior_psnr=psnr(rgb[interior], target[interior]),
        strong_depth_edges_near_outline=float(near.float().mean()),
        strong_depth_edge_distance_px=float(distance[strong].mean()),
        pixels=dict(foreground=int(mask.sum()), band=int(band.sum()), interior=int(interior.sum()),
                    strong_edges=int(strong.sum())),
    )


def evaluate_edges(checkpoint, scene, scale, device, views=dtu.TEST_IDS):
    model = load_model(checkpoint, device, near=dtu.BOUNDS[0], far=dtu.BOUNDS[1])
    print(f"Edge metrics | {checkpoint} | scale {scale}", flush=True)
    per_view = []
    with np.load(scene / "cameras.npz") as cameras:
        for view in views:
            result = score_view(model, scene, cameras, view, scale, device)
            per_view.append(result)
            print(f"View {view:02} | band {result['boundary_band_psnr']:.2f} dB | interior "
                  f"{result['interior_psnr']:.2f} dB | strong edges near outline "
                  f"{result['strong_depth_edges_near_outline']:.3f}", flush=True)
    keys = ("foreground_psnr", "boundary_band_psnr", "interior_psnr",
            "strong_depth_edges_near_outline", "strong_depth_edge_distance_px")
    means = {key: sum(r[key] for r in per_view) / len(per_view) for key in keys}
    return dict(checkpoint=str(checkpoint), scale=scale, settings=dict(
        band_px=BAND, interior_px=INTERIOR, top_fraction=TOP_FRACTION, distance_cap_px=DISTANCE_CAP),
        per_view=per_view, means=means)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dtu/scan40"))
    parser.add_argument("--scale", type=int, default=EVAL_SCALE)
    parser.add_argument("--views", type=int, nargs="+", default=list(dtu.TEST_IDS))
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result = evaluate_edges(args.checkpoint, args.data_dir, args.scale, device, tuple(args.views))
    print(json.dumps(result["means"], indent=2))
    if args.output:
        args.output.write_text(json.dumps(result, indent=2))
