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


def sample_intervals(num_rays, near=2.0, far=6.0, N=64, device=None, dtype=torch.float32):
    t_vals = torch.linspace(near, far, N+1, device=device, dtype=dtype)
    return t_vals.expand(num_rays, -1)


def conical_frustum_to_gaussian(rays_o, rays_d, t_vals, radii):

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    radii = radii.reshape(-1, 1)

    t0= t_vals[..., :-1]
    t1 = t_vals[..., 1:]

    t_mu= (t0 + t1) / 2
    t_delta = (t1 - t0)/ 2

    denom  = 3*(t_mu**2) + (t_delta**2)

    mean_t = t_mu + (2*t_mu*t_delta**2) / denom

    var_t = (
        t_delta**2 / 3
        - (4 / 15)
        * (t_delta**4 * (12 * t_mu**2 - t_delta**2))
        /denom**2
    )

    var_r = radii**2 * (
        t_mu**2 / 4
        + 5 * t_delta**2 / 12
        - (4/15)*t_delta**4/denom
    )

    means = rays_o[:, None, :] + rays_d[:, None, :]*mean_t[..., None]

    d_sq = rays_d**2
    d_mag_sq = d_sq.sum(dim=-1, keepdim=True).clamp_min(1e-10)

    cov_diag = (
        var_t[..., None]*d_sq[:, None, :]
        + var_r[..., None]*(1 - d_sq / d_mag_sq)[:, None, :]
    )

    return means, cov_diag


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

