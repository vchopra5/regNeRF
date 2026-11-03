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

    return pixel_rays(H, W, focal, pose, x, y)


def pixel_rays(H, W, focal, pose, x, y):
    fx, fy, cx, cy, skew = (focal if isinstance(focal, tuple)
                             else (focal, focal, W / 2, H / 2, 0.0))
    y_cam = (y + 0.5 - cy) / fy
    directions = torch.stack([
        (x + 0.5 - cx - skew * y_cam) / fx,
        -y_cam,
        -torch.ones_like(x)
    ], dim=-1)

    rays_d = directions @ pose[..., :3, :3].transpose(-1, -2)

    rays_o = pose[..., :3, 3].unsqueeze(-2).expand_as(rays_d)

    radii = torch.full_like(rays_d[..., :1], 2 / (fx*np.sqrt(12)))

    return rays_o, rays_d, radii


def load_data(scene_dir, split="train", scale=1):
    """Return images, camera-to-world poses, and one set of intrinsics per view."""
    scene_dir = Path(scene_dir)
    if (scene_dir / "cameras.npz").is_file():
        from dtu import load_views
        return load_views(scene_dir, split, scale)

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
        [float(focal)] * len(frames)
    )

