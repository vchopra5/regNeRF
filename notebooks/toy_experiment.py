"""Train one condition and cache its predictions for the two-sphere lab."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parent / "assets" / "two_spheres"
TOY_SEED = 42
# Repeat seeds for the saved runs. Images are saved for TOY_SEED only; scores for every seed.
TOY_SEEDS = (42, 142, 242, 342, 442)
CAMERAS = {"Front view": (15, 5), "Lower view": (15, -15)}


def saved_asset_names():
    """List the saved files used by the lesson and optional training."""
    names = ["reference.png", "reference.npz", "transforms_train.json"]
    names += [f"input_{i}.png" for i in range(3)]
    for count in (3, 9, 20):
        names += [f"coverage_500/{count}/{name}" for name in ("heldout.png", "inputs.png", "result.json")]
    names.append("coverage_500/3/prediction.npz")
    for condition in ("off", "on", "off_100", "on_100", "off_1000", "on_1000"):
        names += [f"{condition}/{name}" for name in ("rgb.png", "prediction.npz", "result.json")]
        names += [f"lower/{condition}/{name}" for name in ("rgb.png", "prediction.npz", "result.json")]
    names += ["lower/reference.png", "lower/reference.npz"]
    names += [f"seeds/{name}" for name in seed_record_names()]
    return names


def seed_record_names():
    """List the score records saved for every repeat seed."""
    names = []
    for seed in TOY_SEEDS:
        for count in (3, 9, 20):
            names.append(f"{seed}/coverage_500/{count}/result.json")
        for condition in ("off", "on", "off_100", "on_100", "off_1000", "on_1000"):
            names += [f"{seed}/{condition}/result.json", f"{seed}/lower/{condition}/result.json"]
    return names


def ensure_assets():
    """Unpack the small repository archive. Do not download or train."""
    import hashlib
    from zipfile import ZipFile

    bundle = Path(__file__).with_name("toy_saved_results.zip")
    if not bundle.is_file():
        raise FileNotFoundError(f"Missing {bundle.name}. Get the complete repository, including this archive.")
    base = ASSETS.resolve()
    with ZipFile(bundle) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest["version"] != 1 or set(manifest["files"]) != set(saved_asset_names()):
            raise ValueError("The toy archive has an unsupported manifest.")
        for name, checksum in manifest["files"].items():
            target = (base / name).resolve()
            if not target.is_relative_to(base):
                raise ValueError(f"The toy archive path is outside the data folder: {name}")
            payload = target.read_bytes() if target.is_file() else archive.read(name)
            if hashlib.sha256(payload).hexdigest() != checksum:
                raise ValueError(f"Saved toy data failed its checksum: {name}")
            if not target.is_file():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(payload)
    return ASSETS


def prepare_scene(output=ASSETS, extra_angles=()):
    """Build the sparse inputs, any extra camera views, and the held-out reference."""
    from generate_sphere import orbit_rotation
    from two_spheres import FOCAL, W, scene

    output.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, (angle, elevation) in enumerate([(-35, 12), (0, 22), (40, -8), *extra_angles]):
        pose, _, rgb, _, _ = scene(orbit_rotation(angle, elevation))
        name = f"input_{i}"
        Image.fromarray(rgb).save(output / (name + ".png"))
        frames.append(dict(file_path=name, transform_matrix=pose.tolist()))
    meta = dict(frames=frames, camera_angle_x=2 * np.arctan(W / (2 * FOCAL)))
    (output / "transforms_train.json").write_text(json.dumps(meta, indent=2))
    _, _, rgb, depth, ids = scene(orbit_rotation(15, 5))
    Image.fromarray(rgb[::2, ::2]).save(output / "reference.png")
    np.savez_compressed(output / "reference.npz", depth=depth[::2, ::2].astype("float32"), ids=ids[::2, ::2])


def run_condition(folder, steps, weight, training_scene=ASSETS, stratified=True, seed=TOY_SEED):
    import torch
    import train as training
    from provenance import file_hashes, git

    start = perf_counter()
    views = len(json.loads((training_scene / "transforms_train.json").read_text())["frames"])
    sources = ("train.py", "model.py", "data.py", "two_spheres.py", "notebooks/toy_experiment.py", "provenance.py")
    provenance = dict(code_revision=git("rev-parse", "HEAD"), local_changes=git("diff", "HEAD", "--", *sources),
                      code_sha256=file_hashes(sources),
                      seed=seed, stratified=stratified, evaluation_scale=2,
                      heldout_camera=dict(angle=15, elevation=5),
                      coarse_weight=0.1 if weight else 0.0, views=views,
                      training_transforms=json.loads((training_scene / "transforms_train.json").read_text()),
                      checkpoint=str((folder / "mipnerf_final.pt").resolve()))
    training.train(views, training_scene, folder, depth_weight=weight, depth_weight_start=weight,
          stop_after=steps, scale=2, regnerf_density=True, seed=seed, stratified=stratified)
    train_seconds = perf_counter() - start
    result = dict(steps=steps, fine_weight=weight, training_seconds=train_seconds,
                  device="cuda" if torch.cuda.is_available() else "cpu", **provenance)
    render_condition_view(folder / "mipnerf_final.pt", folder, CAMERAS["Front view"], result)
    render_condition_view(folder / "mipnerf_final.pt", folder / "lower", CAMERAS["Lower view"], result)
    result = json.loads((folder / "result.json").read_text())
    result["total_seconds"] = perf_counter() - start
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    return result


def render_condition_view(checkpoint, folder, camera, training_result):
    """Render a fixed camera from a checkpoint. Do not train or select settings."""
    import torch
    from generate_sphere import orbit_rotation
    from model import load_model, render_rays
    from two_spheres import scene

    start = perf_counter()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _, rays, target, truth, ids = scene(orbit_rotation(*camera))
    target, truth, ids = target[::2, ::2], truth[::2, ::2].astype("float32"), ids[::2, ::2]
    hit = ids > 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(checkpoint, device)
    fields = [r[::2, ::2].reshape(-1, r.shape[-1]) for r in rays]
    with torch.inference_mode():
        rgb, depth = (part.cpu().numpy() for part in render_rays(model, fields, 512, device, return_depth=True))
    depth = depth.reshape(hit.shape)
    mae = float(np.abs(depth[hit] - truth[hit]).mean())
    rgb = rgb.reshape(*hit.shape, 3)
    psnr = float(-10 * np.log10(np.mean((rgb[hit] - target[hit] / 255.) ** 2)))
    Image.fromarray(np.rint(rgb.clip(0, 1) * 255).astype("uint8")).save(folder / "rgb.png")
    np.savez_compressed(folder / "prediction.npz", depth=depth)
    result = dict(training_result, depth_mae=mae, foreground_psnr=psnr,
                  heldout_camera=dict(angle=camera[0], elevation=camera[1]),
                  render_seconds=perf_counter()-start)
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    return result
