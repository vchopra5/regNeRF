"""Compare uniform smoothing with the colour weight from Notebook 3."""
import argparse
import importlib.util
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageFilter

import train as training
from generate_sphere import orbit_rotation
from model import load_model, render_rays


def expanded(mask, size=5):
    return np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(size))) > 0


@torch.inference_mode()
def evaluate(scene, output, steps):
    spec = importlib.util.spec_from_file_location("pendant_scene", scene / "design.py")
    design = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {name: load_model(output / name / f"step_{steps}.pt", device,
                              near=2, far=6)
              for name in ("uniform", "edge")}
    rows = []
    # These are illustration views, selected before this pair is trained.
    for az, el in ((15, 5), (-15, 5), (0, 15)):
        tag = f"az{az}_el{el}_{steps}"
        rotation = orbit_rotation(az, el)
        _, rays, _, truth, ids = design.scene(rotation)
        reference = np.asarray(design.reference(rotation))
        foreground = ids >= 2
        boundary = expanded(foreground) & expanded(~foreground) & (ids > 0)
        flat = (ids == 1) & ~expanded(foreground, 15)
        fields = [ray.reshape(-1, ray.shape[-1]).to(device) for ray in rays]
        colours = [reference]
        depths = [truth]
        for name, model in models.items():
            rgb, depth = render_rays(model, fields, 2048, return_depth=True)
            rgb = rgb.cpu().numpy().reshape(reference.shape)
            depth = depth.cpu().numpy().reshape(ids.shape)
            mse = np.mean((rgb[foreground] - reference[foreground] / 255.) ** 2)
            row = dict(view=tag, mode=name, step=steps,
                       foreground_psnr=float(-10 * np.log10(mse)),
                       boundary_depth_rmse=float(np.sqrt(np.mean((depth[boundary]-truth[boundary])**2))),
                       flat_depth_rmse=float(np.sqrt(np.mean((depth[flat]-truth[flat])**2))))
            rows.append(row)
            colours.append(np.rint(rgb.clip(0, 1)*255).astype(np.uint8))
            depths.append(depth)
            np.savez_compressed(output / f"{tag}_{name}.npz", rgb=rgb, depth=depth)
        board = Image.new("RGB", (1240, 800), "#f7f5ef")
        draw = ImageDraw.Draw(board)
        title = ImageFont.truetype("arial.ttf", 25)
        font = ImageFont.truetype("arial.ttf", 18)
        draw.text((20, 12), f"Colour edge weighting | {steps:,} steps | view ({az}, {el})", font=title, fill="#203039")
        draw.text((20, 49), "Three training photos. Same seed and depth weight. Illustration views; one seed.", font=font, fill="#46545b")
        for i, (label, rgb, depth) in enumerate(zip(("Reference", "Edge weighting OFF", "Edge weighting ON"), colours, depths)):
            x = 10 + i * 410
            draw.text((x+5, 85), label, font=font, fill="#203039")
            board.paste(Image.fromarray(rgb), (x, 115))
            shade = np.rint((np.nan_to_num(depth, posinf=6).clip(2, 6)-2)/4*255).astype(np.uint8)
            board.paste(Image.fromarray(shade).convert("RGB"), (x, 445))
        draw.text((20, 420), "Depth: black = 2, white = 6 scene units. All columns use the same scale.", font=font, fill="#46545b")
        draw.text((20, 763), "OFF keeps uniform depth smoothing. ON multiplies each pair by exp(-10 * mean absolute RGB difference).", font=font, fill="#46545b")
        board.save(output / f"{tag}.png")
        # Fixed central crop; the same box is used for each condition.
        crop = Image.new("RGB", (1200, 420), "#f7f5ef")
        cd = ImageDraw.Draw(crop)
        for i, (label, rgb) in enumerate(zip(("Reference", "Edge weighting OFF", "Edge weighting ON"), colours)):
            cd.text((i*400+10, 8), label, font=font, fill="#203039")
            crop.paste(Image.fromarray(rgb).crop((120, 90, 280, 245)).resize((400, 388), Image.Resampling.NEAREST), (i*400, 32))
        crop.save(output / f"{tag}_crop.png")
    (output / f"scores_{steps}.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    scene = Path("checkpoint/ginkgo_20261102_162816")
    output = args.output_dir
    if args.evaluate_only:
        evaluate(scene, output, args.steps)
        return
    output.mkdir(parents=True, exist_ok=False)
    for source in ("train.py", "model.py", "compare_soft_edges.py"):
        shutil.copy2(source, output / source)
    record = dict(scene=str(scene), seed=42, steps=args.steps, scale=1,
                  depth_weight=0.1, depth_weight_start=0.1, stratified=True,
                  train_view_ids=[25, 22, 28],
                  edge_rule="exp(-10 * mean(abs(RGB_i-RGB_j)))",
                  guide="Detached predicted RGB from virtual patch; evaluation mode",
                  views=[[15, 5], [-15, 5], [0, 15]],
                  selection="Exploratory illustration. Views set before training this pair.",
                  revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  local_changes=subprocess.check_output(["git", "diff"], text=True),
                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    (output / "record.json").write_text(json.dumps(record, indent=2))
    training.PRINT_EVERY = 250
    torch.set_num_threads(4)
    start = time.time()
    for name in ("uniform", "edge"):
        def save(step, model):
            if step in {1000, 3000, 5000, args.steps}:
                torch.save(model.state_dict(), output / name / f"step_{step}.pt")
            if step % 250 == 0:
                print(f"{name}: {step} steps; total {time.time()-start:.0f} seconds", flush=True)
        training.train(3, scene, output / name, 0.1, stop_after=args.steps,
                       after_step=save, depth_weight_start=0.1, regnerf_density=True,
                       virtual_edge_weight=name == "edge", scale=1, seed=42)
    for step in sorted({s for s in (1000, 3000, 5000, args.steps) if s <= args.steps}):
        evaluate(scene, output, step)


if __name__ == "__main__":
    main()
