"""Train one condition and cache its predictions for the two-sphere lab."""
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

ASSETS = Path(__file__).resolve().parent / "assets" / "two_spheres"


def prepare_scene(output=ASSETS, extra_angles=()):
    """Build the sparse inputs, any extra camera views, and the held-out reference."""
    from generate_sphere import orbit_rotation
    from two_spheres_canny import FOCAL, W, scene

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


def run_condition(folder, steps, weight, training_scene=ASSETS):
    import torch
    from generate_sphere import orbit_rotation
    from model import MipNeRF
    from train import train
    from two_spheres_canny import scene

    start = perf_counter()
    views = len(json.loads((training_scene / "transforms_train.json").read_text())["frames"])
    train(views, training_scene, folder, depth_weight=weight, depth_weight_start=weight,
          stop_after=steps, scale=2, regnerf_density=True)
    train_seconds = perf_counter() - start
    _, rays, _, _, _ = scene(orbit_rotation(15, 5))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MipNeRF().to(device).eval()
    model.load_state_dict(torch.load(folder / "mipnerf_final.pt", map_location=device, weights_only=True))
    fields = [r[::2, ::2].reshape(-1, r.shape[-1]).to(device).split(512) for r in rays]
    with torch.inference_mode():
        batches = [model(*r, return_depth=True)[1:] for r in zip(*fields)]
    rgb, depth = (torch.cat(items).cpu().numpy() for items in zip(*batches))
    with np.load(ASSETS / "reference.npz") as ref:
        hit = ref["ids"] > 0
        depth = depth.reshape(hit.shape)
        mae = float(np.abs(depth[hit] - ref["depth"][hit]).mean())
    rgb = rgb.reshape(*hit.shape, 3)
    target = np.asarray(Image.open(ASSETS / "reference.png")) / 255.
    psnr = float(-10 * np.log10(np.mean((rgb[hit] - target[hit]) ** 2)))
    Image.fromarray(np.rint(rgb.clip(0, 1) * 255).astype("uint8")).save(folder / "rgb.png")
    np.savez_compressed(folder / "prediction.npz", depth=depth)
    result = dict(steps=steps, fine_weight=weight, depth_mae=mae, foreground_psnr=psnr,
                  training_seconds=train_seconds, total_seconds=perf_counter()-start, device=device.type)
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    return result
