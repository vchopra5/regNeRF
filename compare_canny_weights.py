"""Compare uniform depth smoothing with Canny-based depth weights."""
import argparse
import importlib.util
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import train as training
from generate_sphere import orbit_rotation
from model import load_model, render_rays
from provenance import file_hashes, git


VIEWS = ((15, 5), (-15, 5), (0, 15))
MODES = ("regnerf", "canny_weight")
METRICS = ("boundary_depth_rmse", "flat_depth_rmse", "foreground_psnr")


def expand(mask, size):
    return np.asarray(Image.fromarray(mask).filter(ImageFilter.MaxFilter(size))) > 0


def regions(truth, ids):
    foreground = ids >= 2
    finite = np.isfinite(truth)
    boundary = expand(foreground, 5) & expand(~foreground, 5) & finite
    flat = (ids == 1) & ~expand(ids != 1, 15) & finite
    if not boundary.any() or not flat.any():
        raise ValueError("The depth regions must contain pixels.")
    return foreground, boundary, flat


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def make_figure(output, tag, reference, truth, boundary, flat, results, rows, steps, edge_label):
    """Use the same depth scale and error scale for both models."""
    board = Image.new("RGB", (1240, 1240), "#faf9f5")
    draw = ImageDraw.Draw(board)
    title = ImageFont.truetype("arial.ttf", 27)
    font = ImageFont.truetype("arial.ttf", 20)
    small = ImageFont.truetype("arial.ttf", 17)
    draw.text((20, 14), f"Pendant depth | {tag} | {steps:,} steps | seed 42", font=title, fill="#172d37")
    draw.text((20, 52), "Same training settings. Only the depth loss changes. Lower depth RMSE is better.", font=font, fill="#334955")
    colours = [reference] + [r[0] for r in results]
    depths = [truth] + [r[1] for r in results]
    for i, (label, rgb, depth) in enumerate(zip(("Reference", "RegNeRF smoothing", edge_label), colours, depths)):
        x = 10 + 410 * i
        draw.text((x, 91), label, font=font, fill="#172d37")
        board.paste(Image.fromarray(rgb), (x, 120))
        shade = np.rint((np.nan_to_num(depth, posinf=5).clip(3, 5) - 3) / 2 * 255).astype(np.uint8)
        board.paste(Image.fromarray(shade).convert("RGB"), (x, 460))
        if i == 0:
            masks = np.full((*truth.shape, 3), 238, dtype=np.uint8)
            masks[flat] = [64, 132, 173]
            masks[boundary] = [225, 105, 28]
            board.paste(Image.fromarray(masks), (x, 800))
        else:
            # Error map: white is zero; dark red is at least 0.5 scene units.
            error = np.abs(depth - truth)
            value = np.nan_to_num(error, posinf=0).clip(0, 0.5) / 0.5
            rgb_error = 255 + value[..., None] * (np.array([145, 20, 25]) - 255)
            rgb_error[~np.isfinite(truth)] = [220, 220, 220]
            board.paste(Image.fromarray(np.rint(rgb_error).astype(np.uint8)), (x, 800))
    draw.text((20, 430), "Depth: black = 3, white = 5 scene units. One scale for all columns.", font=small, fill="#334955")
    draw.text((20, 770), "Left: orange = boundary; blue = flat panel. Right: absolute depth error, white = 0, dark red = 0.5 or more.", font=small, fill="#334955")
    for j, row in enumerate(rows):
        label = "RegNeRF" if row["mode"] == "regnerf" else edge_label
        draw.text((20, 1115 + j * 28), f"{label}: boundary RMSE {row['boundary_depth_rmse']:.4f} | flat RMSE {row['flat_depth_rmse']:.4f} scene units", font=font, fill="#172d37")
    change = {key: 100 * (rows[1][key] / rows[0][key] - 1) for key in METRICS[:2]}
    draw.text((20, 1178), f"Error change: boundary {change[METRICS[0]]:+.1f}% | flat panel {change[METRICS[1]]:+.1f}%. Negative means improvement.", font=font, fill="#172d37")
    draw.text((20, 1210), "One synthetic scene and one seed. These fixed views were used in earlier pendant studies.", font=small, fill="#334955")
    board.save(output / f"{tag}_comparison.png")


@torch.inference_mode()
def evaluate(scene, output, steps):
    record = json.loads((output / "record.json").read_text())
    modes = tuple(record.get("modes", MODES))
    edge_label = "Canny + L1 edge loss" if record.get("l1_edge_loss") else "Canny-based weights"
    spec = importlib.util.spec_from_file_location("pendant_scene", scene / "design.py")
    design = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    models = {mode: load_model(output / mode / f"step_{steps}.pt", device, near=2, far=6)
              for mode in modes}
    rows = []
    for az, el in VIEWS:
        tag = f"az{az}_el{el}"
        rotation = orbit_rotation(az, el)
        _, rays, _, truth, ids = design.scene(rotation)
        reference = np.asarray(design.reference(rotation))
        foreground, boundary, flat = regions(truth, ids)
        fields = [ray.reshape(-1, ray.shape[-1]).to(device) for ray in rays]
        results, view_rows = [], []
        np.savez_compressed(output / f"{tag}_reference.npz", rgb=reference, depth=truth,
                            ids=ids, boundary=boundary, flat=flat)
        for mode, model in models.items():
            rgb, depth = render_rays(model, fields, 2048, return_depth=True)
            rgb = rgb.cpu().numpy().reshape(reference.shape)
            depth = depth.cpu().numpy().reshape(truth.shape)
            if not np.isfinite(depth).all() or not np.isfinite(rgb).all():
                raise ValueError("The model returned a nonfinite value.")
            mse = np.mean((rgb[foreground] - reference[foreground] / 255.) ** 2)
            row = dict(view=[az, el], mode=mode, step=steps,
                       checkpoint=str(output / mode / f"step_{steps}.pt"),
                       foreground_psnr=float(-10 * np.log10(mse)),
                       boundary_pixels=int(boundary.sum()), flat_pixels=int(flat.sum()))
            for name, mask in (("boundary", boundary), ("flat", flat)):
                error = depth[mask] - truth[mask]
                row[f"{name}_depth_rmse"] = float(np.sqrt(np.mean(error**2)))
                row[f"{name}_depth_mae"] = float(np.mean(np.abs(error)))
                row[f"{name}_depth_bias"] = float(np.mean(error))
            rows.append(row)
            view_rows.append(row)
            results.append((np.rint(rgb.clip(0, 1) * 255).astype(np.uint8), depth))
            np.savez_compressed(output / f"{tag}_{mode}_{steps}.npz", rgb=rgb, depth=depth)
            print(json.dumps(row), flush=True)
        make_figure(output, tag, reference, truth, boundary, flat, results, view_rows, steps, edge_label)
    means = {mode: {key: float(np.mean([r[key] for r in rows if r["mode"] == mode]))
                    for key in METRICS} for mode in modes}
    summary = dict(step=steps, aggregation="Arithmetic mean of three per-view scores",
                   depth_units="Scene units; camera-axis depth", means=means,
                   canny_minus_regnerf={key: means[modes[1]][key] - means[modes[0]][key]
                                       for key in METRICS},
                   depth_error_change_percent={key: 100 * (means[modes[1]][key] / means[modes[0]][key] - 1)
                                               for key in METRICS[:2]})
    write_json(output / f"scores_{steps}.json", rows)
    write_json(output / f"summary_{steps}.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=Path("checkpoint/ginkgo_20261102_162816"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--l1", action="store_true", help="Add the L1 depth-edge shortfall loss. Require CUDA.")
    parser.add_argument("--reuse-regnerf", type=Path,
                        help="Reuse a completed RegNeRF run from this comparison script.")
    args = parser.parse_args()
    if not 1 <= args.steps <= training.NUM_STEPS:
        parser.error("Use a step count from 1 to 20000.")
    torch.set_num_threads(4)
    if args.evaluate_only:
        evaluate(args.scene, args.output_dir, args.steps)
        return
    print(f"Compare uniform and Canny-based depth weights at {args.steps} steps.", flush=True)
    if not args.run:
        print("Pass --run with a new output folder to train both models.")
        return
    if args.l1 and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU training is disabled.")
    modes = ("regnerf", "canny_l1" if args.l1 else "canny_weight")
    metadata = json.loads((args.scene / "transforms_train.json").read_text())
    ids = [frame.get("dtu_view_id") for frame in metadata["frames"]]
    if ids != [25, 22, 28]:
        raise ValueError("Use the three pendant views 25, 22, and 28.")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sources = ("train.py", "model.py", "data.py", "dtu.py", "generate_sphere.py",
               "canny_regularizer.py", "compare_canny_weights.py", "provenance.py")
    snapshot = args.output_dir / "source"
    snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot / source)
    shutil.copy2(args.scene / "design.py", snapshot / "pendant_design.py")
    input_files = [str(args.scene / "transforms_train.json"), str(args.scene / "design.py")]
    input_files += [str(args.scene / (frame["file_path"] + ".png")) for frame in metadata["frames"]]
    record = dict(scene=str(args.scene), seed=42, steps=args.steps, scale=1, eval_scale=1,
                  modes=modes, l1_edge_loss=args.l1, edge_depth_margin=0.1,
                  edge_loss_coefficient=1.0 if args.l1 else 0.0,
                  require_cuda=args.l1,
                  train_view_ids=ids, depth_weight=0.1, depth_weight_start=0.1,
                  coarse_depth_weight=0.1, stratified=True, regnerf_density=True,
                  optimizer="Adam", learning_rate_schedule="Unchanged 20000-step schedule",
                  batch_size=training.BATCH_SIZE, depth_rays=training.DEPTH_RAYS,
                  patch_size=training.PATCH_SIZE, bounds=[2, 6], bound_steps=training.BOUND_STEPS,
                  canny=dict(sigma=1.2, gaussian_radius=4, sobel_divisor=8,
                             threshold=0.1, floor=0.05, halo=5),
                  guide="Detached rendered RGB; evaluation mode; no test images",
                  views=VIEWS, selection="Fixed before training; used in earlier pendant studies",
                  boundary="Two-pixel band on each side of pendant/chain silhouette; finite truth only",
                  flat="Back panel; exclude seven pixels around all non-panel pixels",
                  revision=git("rev-parse", "HEAD"), local_changes=git("diff"),
                  git_status=git("status", "--short"), source_sha256=file_hashes(sources),
                  input_sha256=file_hashes(input_files), torch_version=torch.__version__,
                  device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    if args.reuse_regnerf:
        previous = json.loads((args.reuse_regnerf / "record.json").read_text())
        for key in ("scene", "seed", "steps", "scale", "train_view_ids", "depth_weight",
                    "depth_weight_start", "coarse_depth_weight", "stratified", "regnerf_density",
                    "batch_size", "depth_rays", "patch_size", "bounds", "bound_steps", "input_sha256"):
            if previous[key] != record[key]:
                raise ValueError(f"The saved baseline has a different setting: {key}")
        for source in ("model.py", "data.py", "dtu.py", "generate_sphere.py"):
            if previous["source_sha256"][source] != record["source_sha256"][source]:
                raise ValueError(f"The saved baseline has different code: {source}")
        checkpoint = args.reuse_regnerf / "regnerf" / f"step_{args.steps}.pt"
        (args.output_dir / "regnerf").mkdir()
        shutil.copy2(checkpoint, args.output_dir / "regnerf" / checkpoint.name)
        shutil.copytree(args.reuse_regnerf / "source", args.output_dir / "baseline_source")
        record["baseline_source"] = str(checkpoint)
        record["baseline_record"] = previous
    write_json(args.output_dir / "record.json", record)
    training.PRINT_EVERY = 250
    start = time.monotonic()
    for mode in modes:
        if mode == "regnerf" and args.reuse_regnerf:
            print("Use the completed RegNeRF model from the same protocol.", flush=True)
            continue
        def save(step, model):
            if step in {4000, args.steps}:
                torch.save(model.state_dict(), args.output_dir / mode / f"step_{step}.pt")
            if step % 250 == 0:
                print(f"{mode}: {step}/{args.steps}; total {time.monotonic()-start:.0f} seconds", flush=True)
        training.train(3, args.scene, args.output_dir / mode, 0.1,
                       stop_after=args.steps, after_step=save, depth_weight_start=0.1,
                       regnerf_density=True, virtual_canny_weight=mode == "canny_weight",
                       virtual_canny_l1=mode == "canny_l1", edge_depth_margin=0.1,
                       require_cuda=args.l1, scale=1, seed=42, stratified=True)
    record["training_seconds"] = time.monotonic() - start
    write_json(args.output_dir / "record.json", record)
    evaluate(args.scene, args.output_dir, args.steps)


if __name__ == "__main__":
    main()
