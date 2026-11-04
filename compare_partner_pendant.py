"""Compare the pushed edge loss with the two saved pendant models."""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SAVED = ROOT / "checkpoint/canny_l1_pendant_8k_20261103"
SCENE = ROOT / "checkpoint/ginkgo_20261102_162816"
REVISION = "92d882b6f5106c78fccb2c67691b6eb80652a358"
MODES = ("regnerf", "canny_l1", "partner")
LABELS = ("RegNeRF smoothing", "Canny + L1 margin", "Partner L1 smoothing")
VIEWS = ((15, 5), (-15, 5), (0, 15))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    source.mkdir()
    names = ("train.py", "model.py", "data.py", "dtu.py", "generate_sphere.py",
             "provenance.py", "eval_dtu.py", "eval_edges.py")
    for name in names:
        data = subprocess.check_output(["git", "show", f"{REVISION}:{name}"], cwd=ROOT)
        (source / name).write_bytes(data)
    # Rendering, rays, and the scene must match the earlier comparison.
    old_record = json.loads((SAVED / "record.json").read_text())
    for name in ("model.py", "data.py", "dtu.py", "generate_sphere.py"):
        if (source / name).read_text(encoding="utf-8") != (SAVED / "source" / name).read_text(encoding="utf-8"):
            raise ValueError(f"The saved comparison uses different code: {name}")
    for name, expected in old_record["input_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"The scene changed: {name}")
    shutil.copy2(__file__, source / Path(__file__).name)
    record = dict(revision=REVISION, source_sha256={n: sha(source / n) for n in names},
                  scene=str(SCENE.relative_to(ROOT)), seed=42, steps=8000,
                  train_views=[25, 22, 28], scale=1, eval_scale=1, stratified=True,
                  coarse_depth_weight=0.1, fine_depth_weight=0.1, initial_depth_weight=0.1,
                  edge_loss_weight=0.1, edge_scale_quantile=0.9, edge_halo=2,
                  weight_selection="Repository example; fixed before training; no tuning",
                  cuda_required=True, device=torch.cuda.get_device_name(0),
                  adaptation="Move loaded training images to CUDA for edge-scale calculation; loss code unchanged",
                  previous_record=old_record, views=VIEWS,
                  checkpoints={mode: str((SAVED / mode / "step_8000.pt").relative_to(ROOT))
                               for mode in MODES[:2]})
    record["checkpoints"]["partner"] = str((output / "partner/step_8000.pt").relative_to(ROOT))
    record["saved_checkpoint_sha256"] = {mode: sha(ROOT / record["checkpoints"][mode]) for mode in MODES[:2]}
    write_json(output / "record.json", record)


def train_partner(output):
    # Import only the committed training code, not the working copy.
    sys.path.insert(0, str(output / "source"))
    import train as training

    assert Path(training.__file__).resolve() == (output / "source/train.py").resolve()
    original_load = training.load_data

    def load_on_gpu(*args, **kwargs):
        images, poses, intrinsics = original_load(*args, **kwargs)
        return images.cuda(), poses, intrinsics

    training.load_data = load_on_gpu
    training.PRINT_EVERY = 500
    start = time.monotonic()

    def save(step, model):
        if not next(model.parameters()).is_cuda:
            raise RuntimeError("The model must train on CUDA.")
        if step in (4000, 8000):
            torch.save(model.state_dict(), output / "partner" / f"step_{step}.pt")
        if step % 500 == 0:
            print(f"Partner: {step}/8000; {time.monotonic()-start:.0f} seconds", flush=True)

    training.train(3, SCENE, output / "partner", depth_weight=0.1,
                   depth_weight_start=0.1, stop_after=8000, after_step=save,
                   regnerf_density=True, scale=1, seed=42, stratified=True,
                   edge_loss_weight=0.1)
    record = json.loads((output / "record.json").read_text())
    record["training_seconds"] = time.monotonic() - start
    write_json(output / "record.json", record)


def figure(output, tag, reference, results, rows):
    board = Image.new("RGB", (1650, 1260), "#faf9f5")
    draw = ImageDraw.Draw(board)
    title = ImageFont.truetype("arial.ttf", 28)
    font = ImageFont.truetype("arial.ttf", 20)
    small = ImageFont.truetype("arial.ttf", 18)
    draw.text((15, 12), f"Pendant | {tag} | 8,000 steps | seed 42 | same three photos", font=title, fill="#172d37")
    draw.text((15, 52), "Lower depth RMSE is better. All models use the same evaluation regions and scales.", font=font, fill="#334955")
    truth = reference["depth"]
    columns = [(reference["rgb"], truth)] + results
    for i, (label, (rgb, depth)) in enumerate(zip(("Reference",) + LABELS, columns)):
        x = 10 + i * 410
        draw.text((x, 90), label, font=font, fill="#172d37")
        board.paste(Image.fromarray(rgb), (x, 120))
        gray = np.rint((np.nan_to_num(depth, posinf=5).clip(3, 5) - 3) * 127.5).astype(np.uint8)
        board.paste(Image.fromarray(gray).convert("RGB"), (x, 460))
        if i == 0:
            error_rgb = np.full((*truth.shape, 3), 238, dtype=np.uint8)
            error_rgb[reference["flat"]] = [64, 132, 173]
            error_rgb[reference["boundary"]] = [225, 105, 28]
        else:
            error = np.nan_to_num(np.abs(depth - truth), posinf=0).clip(0, 0.5) / 0.5
            error_rgb = np.rint(255 + error[..., None] * (np.array([145, 20, 25]) - 255)).astype(np.uint8)
            error_rgb[~np.isfinite(truth)] = [220, 220, 220]
        board.paste(Image.fromarray(error_rgb), (x, 800))
    draw.text((15, 430), "Depth: black = 3, white = 5 scene units.", font=small, fill="#334955")
    draw.text((15, 770), "Regions: orange = boundary; blue = flat panel. Absolute depth error: white = 0; dark red = 0.5 or more.", font=small, fill="#334955")
    for i, (label, row) in enumerate(zip(LABELS, rows)):
        draw.text((15, 1115 + i * 28), f"{label}: boundary RMSE {row['boundary_depth_rmse']:.4f} | flat RMSE {row['flat_depth_rmse']:.4f}", font=font, fill="#172d37")
    winner = min(rows, key=lambda r: r["boundary_depth_rmse"])["mode"]
    draw.text((15, 1205), f"Lowest boundary RMSE in this view: {LABELS[MODES.index(winner)]}. One scene and one seed.", font=font, fill="#172d37")
    board.save(output / f"{tag}_comparison.png")


@torch.inference_mode()
def evaluate(output):
    sys.path.insert(0, str(output / "source"))
    from model import load_model, render_rays
    from generate_sphere import orbit_rotation

    spec = importlib.util.spec_from_file_location("pendant_scene", SCENE / "design.py")
    design = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design)
    model = load_model(output / "partner/step_8000.pt", "cuda", near=2, far=6)
    rows = []
    for az, el in VIEWS:
        tag = f"az{az}_el{el}"
        reference = dict(np.load(SAVED / f"{tag}_reference.npz"))
        _, rays, _, truth, ids = design.scene(orbit_rotation(az, el))
        assert np.array_equal(truth, reference["depth"])
        fields = [ray.reshape(-1, ray.shape[-1]).cuda() for ray in rays]
        rgb, depth = render_rays(model, fields, 2048, return_depth=True)
        rgb = rgb.cpu().numpy().reshape(reference["rgb"].shape)
        depth = depth.cpu().numpy().reshape(truth.shape)
        np.savez_compressed(output / f"{tag}_partner_8000.npz", rgb=rgb, depth=depth)
        results, view_rows = [], []
        for mode in MODES:
            source = output if mode == "partner" else SAVED
            result = np.load(source / f"{tag}_{mode}_8000.npz")
            rgb, depth = result["rgb"], result["depth"]
            assert np.isfinite(rgb).all() and np.isfinite(depth).all()
            foreground = ids >= 2
            mse = np.mean((rgb[foreground] - reference["rgb"][foreground] / 255.)**2)
            row = dict(view=[az, el], mode=mode, foreground_psnr=float(-10 * np.log10(mse)))
            for region in ("boundary", "flat"):
                mask = reference[region]
                error = depth[mask] - truth[mask]
                row[f"{region}_depth_rmse"] = float(np.sqrt(np.mean(error**2)))
                row[f"{region}_depth_mae"] = float(np.mean(np.abs(error)))
            rows.append(row)
            view_rows.append(row)
            results.append((np.rint(rgb.clip(0, 1) * 255).astype(np.uint8), depth))
            print(json.dumps(row), flush=True)
        figure(output, tag, reference, results, view_rows)
    metrics = ("boundary_depth_rmse", "flat_depth_rmse", "foreground_psnr")
    means = {mode: {key: float(np.mean([r[key] for r in rows if r["mode"] == mode]))
                    for key in metrics} for mode in MODES}
    summary = dict(means=means, aggregation="Arithmetic mean of three per-view scores",
                   partner_minus_regnerf={k: means["partner"][k] - means["regnerf"][k] for k in metrics},
                   partner_minus_canny_l1={k: means["partner"][k] - means["canny_l1"][k] for k in metrics})
    write_json(output / "scores_8000.json", rows)
    write_json(output / "summary_8000.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. CPU fallback is disabled.")
    torch.set_num_threads(4)
    output = args.output_dir.resolve()
    if not args.evaluate_only:
        prepare(output)
        train_partner(output)
    evaluate(output)


if __name__ == "__main__":
    main()
