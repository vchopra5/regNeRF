import torch
import json
import numpy as np

from pathlib import Path
from PIL import Image


def get_rays(H, W, focal, pose):
    x, y = torch.meshgrid(
        torch.arange(W, dtype=pose.dtype, device=pose.device),
        torch.arange(H, dtype=pose.dtype, device=pose.device),
        indexing="xy"
    )

    directions = torch.stack([
        (x + 0.5 - W / 2) / focal,
        -(y + 0.5 - H / 2)/ focal,
        -torch.ones_like(x)
    ], dim=-1)

    rays_d = directions@pose[:3, :3].T

    rays_o = pose[:3, 3].expand_as(rays_d)

    radii = torch.full((H, W, 1), 2 / (focal*np.sqrt(12)), dtype=pose.dtype, device=pose.device)

    return rays_o, rays_d, radii


def load_data(scene_dir, split="train", scale=1):
    scene_dir = Path(scene_dir)

    with open(scene_dir / f"transforms_{split}.json") as f:
        meta = json.load(f)

    frames = meta["frames"]

    images = []
    poses = []

    for frame in frames:
        path = scene_dir / f"{frame['file_path']}.png"

        with Image.open(path) as image:
            w, h = image.size
            image = image.convert("RGBA")
            image = image.resize(
                (w // scale, h // scale),
                Image.Resampling.LANCZOS
            )

        images.append(np.asarray(image))

        poses.append(frame["transform_matrix"])

    images = np.stack(images)
    images = images.astype(np.float32) / 255.0

    rgb = images[..., :3]
    alpha = images[..., 3:4]

    images = rgb * alpha + (1.0 - alpha)

    poses = np.asarray(poses, dtype=np.float32)

    H, W = images.shape[1:3]

    focal = 0.5*W/np.tan(
        0.5*meta["camera_angle_x"]
    )

    return (
        torch.from_numpy(images),
        torch.from_numpy(poses),
        float(focal)
    )

