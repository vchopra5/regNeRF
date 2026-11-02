
import argparse
import math
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F

from data import get_rays, load_data, pixel_rays
from model import MipNeRF
import numpy as np
import dtu

# Settings
DATA_DIR = "nerf_synthetic/chair"
SCALE = 4
BATCH_SIZE = 1024
DEPTH_RAYS = 1024
PATCH_SIZE = 8
RAYS_PER_CHUNK = 4096
NUM_STEPS = 20_000
SEED = 42
LR_INIT = 2e-3
LR_FINAL = 2e-5
LR_DELAY_STEPS = 512
LR_DELAY_MULT = 0.01
BOUND_STEPS = 256
INITIAL_RANGE_FRACTION = 0.5
PRINT_EVERY = 1_000
SAVE_EVERY = 5_000
DEPTH_WEIGHT = 0.1
DEPTH_WEIGHT_START = 400.0
DEPTH_DECAY_STEPS = 512


def depth_patch_loss(depth, opacity):
    """Weight the right and lower depth differences by the anchor pixel's opacity."""
    anchor = depth[..., :-1, :-1]
    dx = anchor - depth[..., :-1, 1:]
    dy = anchor - depth[..., 1:, :-1]
    weight = opacity[..., :-1, :-1].detach()
    return (weight * (dx.square() + dy.square())).mean()


def scheduled_depth_weight(step, final_weight):
    if final_weight == 0:
        return 0.0
    progress = min(max(step / DEPTH_DECAY_STEPS, 0.0), 1.0)
    return DEPTH_WEIGHT_START * (1 - progress) + final_weight * progress


def training_bounds(step, near, far):
    progress = min(step / BOUND_STEPS, 1.0)
    margin = (far - near) * (1 - INITIAL_RANGE_FRACTION) * (1 - progress) / 2
    return near + margin, far - margin


def learning_rate(step):
    progress = min(step / NUM_STEPS, 1.0)
    delay = math.sin(math.pi / 2 * min(step / LR_DELAY_STEPS, 1.0))
    multiplier = LR_DELAY_MULT + (1 - LR_DELAY_MULT) * delay
    return multiplier * LR_INIT * (LR_FINAL / LR_INIT) ** progress


@contextmanager
def training_matmul_precision():
    if not torch.cuda.is_available():
        yield
        return
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("high")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@training_matmul_precision()
def train(num_views=3, data_dir=DATA_DIR, checkpoint_dir=Path("checkpoint"), depth_weight=DEPTH_WEIGHT,
          stop_after=None, after_step=None, background=None, density_noise_std=None):
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)


    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # 1. Load the dataset
    images, poses, focal = load_data(
        data_dir, split="train", scale=SCALE
    )

    N, H, W, _ = images.shape
    if not 3 <= num_views <= N:
        raise ValueError(f"num_views must be between 3 and {N}")

    # Sparse generation

    is_dtu = (Path(data_dir) / "cameras.npz").is_file()
    train_indices = np.arange(num_views) if is_dtu else rng.choice(N, size=num_views, replace=False)

    poses = poses.to(device)
    # Training cameras are fixed. Cache only the selected views.
    rgb_views = {}
    for index in train_indices:
        intrinsics = focal[index] if is_dtu else focal
        fields = get_rays(H, W, intrinsics, poses[index])
        rgb_views[index] = (images[index].reshape(-1, 3).to(device),
                            tuple(field.reshape(-1, field.shape[-1]) for field in fields))
    if depth_weight != 0:
        depth_rng = torch.Generator(device=device).manual_seed(SEED + 1)
        # In case of RegNeRF, calculate camera position to add for depth loss
        sampling_poses = dtu.load_rig_poses(Path(data_dir)).to(device) if is_dtu else poses[train_indices]
        origins = sampling_poses[:, :3, 3]
        forward = -sampling_poses[:, :3, 2]
        box_min = origins.amin(dim=0)
        box_max = origins.amax(dim=0)
        up_ref = F.normalize(sampling_poses[:, :3, 1].mean(dim=0), dim=0)

        eye = torch.eye(3, device=poses.device, dtype=poses.dtype)
        P = eye - forward[:, :, None] * forward[:, None, :]

        A = P.reshape(-1, 3)
        b = (P @ origins.unsqueeze(-1)).reshape(-1)

        focus = torch.linalg.lstsq(A, b).solution

    print("Images:", images.shape)
    print("Focal:", focal)

    # 2. Initialize model and optimizer
    if background is None:
        background = "black" if is_dtu else "white"
    if density_noise_std is None:
        density_noise_std = 1.0 if is_dtu else 0.0
    near, far = dtu.BOUNDS if is_dtu else (2.0, 6.0)
    model = MipNeRF(near=near, far=far, background=background,
                    density_noise_std=density_noise_std, regnerf_density=is_dtu).to(device)
    print(f"Background: {background} | Training density noise std: {density_noise_std:g}")
    full_near, full_far = model.near, model.far
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate(0), fused=device.type == "cuda"
    )


    model.train()

    # 3. Training loop
    last_step = NUM_STEPS if stop_after is None else stop_after
    if not 1 <= last_step <= NUM_STEPS:
        raise ValueError(f"stop_after must be between 1 and {NUM_STEPS}")
    for step in range(1, last_step + 1):
        model.near, model.far = training_bounds(step, full_near, full_far)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate(step)

        # Choose a random training image
        image_idx = int(rng.choice(train_indices))

        ray_intrinsics = focal[image_idx] if is_dtu else focal
        targets, fields = rgb_views[image_idx]
        indices = torch.randint(H * W, (BATCH_SIZE,)).to(device, non_blocking=True)
        batch_rgb = targets[indices]
        rgb_rays = tuple(field[indices] for field in fields)

        # Accumulate the full batch gradient before clipping and updating the model.
        optimizer.zero_grad()
        loss = torch.zeros((), device=device)
        fine_mse = torch.zeros_like(loss)
        for start in range(0, BATCH_SIZE, RAYS_PER_CHUNK):
            chunk = slice(start, start + RAYS_PER_CHUNK)
            coarse_rgb, fine_rgb = model(*(ray[chunk] for ray in rgb_rays))
            coarse_loss = (coarse_rgb - batch_rgb[chunk]).square().sum() / BATCH_SIZE
            fine_loss = (fine_rgb - batch_rgb[chunk]).square().sum() / BATCH_SIZE
            rgb_loss = 0.1 * coarse_loss + fine_loss
            rgb_loss.backward()
            loss += rgb_loss.detach()
            fine_mse += fine_loss.detach() / 3

        current_weight = scheduled_depth_weight(step, depth_weight)
        if depth_weight != 0:
            # Sample one virtual camera and one contiguous pixel patch per patch index.
            patch_area = PATCH_SIZE ** 2
            patch_count = DEPTH_RAYS // patch_area
            positions = box_min + torch.rand(patch_count, 3, device=device, generator=depth_rng) * (box_max - box_min)
            back = F.normalize(positions - focus, dim=-1)
            right = F.normalize(torch.linalg.cross(up_ref.expand_as(back), back), dim=-1)
            up = torch.linalg.cross(back, right)
            rotations = torch.stack((right, up, back), dim=-1)
            patch_poses = torch.cat((rotations, positions.unsqueeze(-1)), dim=-1)

            offsets = torch.arange(patch_area, device=device)
            x = torch.randint(W - PATCH_SIZE + 1, (patch_count, 1), device=device, generator=depth_rng) + offsets % PATCH_SIZE
            y = torch.randint(H - PATCH_SIZE + 1, (patch_count, 1), device=device, generator=depth_rng) + offsets // PATCH_SIZE
            patch_rays = pixel_rays(H, W, ray_intrinsics, patch_poses, x, y)

            patches_per_chunk = RAYS_PER_CHUNK // patch_area
            for start in range(0, patch_count, patches_per_chunk):
                chunk = slice(start, start + patches_per_chunk)
                rays = [ray[chunk].flatten(0, 1) for ray in patch_rays]
                levels = model(*rays, return_depth_levels=True, generator=depth_rng)
                chunk_fraction = rays[0].shape[0] / DEPTH_RAYS
                depth_loss = loss.new_zeros(())
                for (depth, opacity), weight in zip(levels, (0.1, current_weight)):
                    depth = depth.reshape(-1, PATCH_SIZE, PATCH_SIZE)
                    depth_loss += weight * depth_patch_loss(depth, opacity.reshape_as(depth))
                depth_loss = depth_loss * chunk_fraction
                depth_loss.backward()
                loss += depth_loss.detach()

        torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        if after_step is not None:
            after_step(step, model)

        # 7. Print progress
        if step % PRINT_EVERY == 0:
            psnr = -10.0 * torch.log10(
                fine_mse.clamp_min(1e-10)
            )

            print(
                f"Step {step}/{NUM_STEPS} | "
                f"Loss: {loss.item():.6f} | "
                f"Depth weight: {current_weight:.4g} | "
                f"PSNR: {psnr.item():.2f} dB"
            )

        # 8. Save model checkpoints
        if step % SAVE_EVERY == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, checkpoint_dir / f"checkpoint_{step}.pt")

    # Save final model
    torch.save(model.state_dict(), checkpoint_dir / "mipnerf_final.pt")

    print("Training complete!")


def add_render_settings(parser):
    parser.add_argument("--background", choices=("black", "white"),
                        help="Background for unoccupied rays. DTU uses black. Other scenes use white.")
    parser.add_argument("--density-noise-std", type=float,
                        help="Training noise on raw density. DTU uses 1. Other scenes use 0.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_render_settings(parser)
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--num-views", type=int, default=9)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--depth-weight", type=float, default=DEPTH_WEIGHT,
                        help="Final depth loss weight after decay. Zero disables the loss.")
    parser.add_argument("--stop-after", type=int, help="Stop early and keep the original learning rate schedule.")
    train(**vars(parser.parse_args()))
