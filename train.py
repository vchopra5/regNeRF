
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from data import load_data, get_rays
from model import MipNeRF
import numpy as np
import dtu

# Settings
DATA_DIR = "nerf_synthetic/chair"
SCALE = 4
BATCH_SIZE = 512
NUM_STEPS = 44_000
SEED = 42
LR_INIT = 2e-3
LR_FINAL = 2e-5
LR_DELAY_STEPS = 512
LR_DELAY_MULT = 0.01
BOUND_STEPS = 256
INITIAL_RANGE_FRACTION = 0.5
PRINT_EVERY = 100
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


def train(num_views=3, data_dir=DATA_DIR, checkpoint_dir=Path("checkpoint"), depth_weight=DEPTH_WEIGHT,
          stop_after=None, after_step=None):
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
    # In case of RegNeRF, calculate camera position to add for depth loss
    train_poses = poses[train_indices]
    origins = train_poses[:, :3, 3]
    forward = -train_poses[:, :3, 2]
    box_min = origins.amin(dim=0)
    box_max = origins.amax(dim=0)
    up_ref = F.normalize(train_poses[:, :3, 1].mean(dim=0), dim=0)

    eye = torch.eye(3, device=poses.device, dtype=poses.dtype)
    P = eye - forward[:, :, None] * forward[:, None, :]

    A = P.reshape(-1, 3)
    b = (P @ origins.unsqueeze(-1)).reshape(-1)

    focus = torch.linalg.lstsq(A, b).solution


    print("Images:", images.shape)
    print("Focal:", focal)

    # 2. Initialize model and optimizer
    model = (MipNeRF(near=dtu.BOUNDS[0], far=dtu.BOUNDS[1]) if is_dtu else MipNeRF()).to(device)
    full_near, full_far = model.near, model.far
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate(0)
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

        pose = poses[image_idx]
        ray_intrinsics = focal[image_idx] if is_dtu else focal

        # Generate all rays for this camera
        rays_o, rays_d, radii = get_rays(
            H, W, ray_intrinsics, pose
        )

        # Flatten image into pixels and rays
        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        radii = radii.reshape(-1, 1)

        target_rgb = images[image_idx].reshape(-1, 3)

        # Choose a random batch of pixels
        indices = torch.randint(
            H * W, (BATCH_SIZE,)
        )

        batch_o = rays_o[indices.to(device)]
        batch_d = rays_d[indices.to(device)]
        batch_radii = radii[indices.to(device)]
        batch_rgb = target_rgb[indices].to(device)

        # 4. Forward pass
        coarse_rgb, fine_rgb = model(
            batch_o, batch_d, batch_radii
        )

        # In case of RegNeRF (sparse losses), calculate new camera angles for depth loss

        position = box_min + torch.rand_like(box_min) * (box_max - box_min)
        back = F.normalize(position - focus, dim=0)
        right = F.normalize(torch.linalg.cross(up_ref, back), dim=0)
        up = torch.linalg.cross(back, right)

        unseen_pose = eye.new_zeros((4, 4))
        unseen_pose[:3, :3] = torch.stack((right, up, back), dim=1)
        unseen_pose[:3, 3] = position
        unseen_pose[3, 3] = 1

        patch_size = 8
        rays_o, rays_d, radii = get_rays(H, W, ray_intrinsics, unseen_pose)

        y = torch.randint(0, H - patch_size + 1, ()).item()
        x = torch.randint(0, W - patch_size + 1, ()).item()

        patch_o = rays_o[y:y + patch_size, x:x+patch_size].reshape(-1, 3)
        patch_d = rays_d[y:y + patch_size, x:x+patch_size].reshape(-1, 3)
        patch_radii = radii[y:y+patch_size, x:x+patch_size].reshape(-1, 1)

        _, _, depth, opacity = model(patch_o, patch_d, patch_radii, return_depth_and_opacity=True)
        depth_patch = depth.reshape(patch_size, patch_size)
        depth_loss = depth_patch_loss(depth_patch, opacity.reshape_as(depth_patch))

        # 5. Compute losses
        coarse_loss = F.mse_loss(coarse_rgb, batch_rgb)
        fine_loss = F.mse_loss(fine_rgb, batch_rgb)

        current_weight = scheduled_depth_weight(step, depth_weight)
        loss = 0.1 * coarse_loss + fine_loss + current_weight * depth_loss

        # 6. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(model.parameters(), 0.1)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        if after_step is not None:
            after_step(step, model)

        # 7. Print progress
        if step % PRINT_EVERY == 0:
            psnr = -10.0 * torch.log10(
                fine_loss.detach().clamp_min(1e-10)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=DATA_DIR)
    parser.add_argument("--num-views", type=int, default=9)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoint"))
    parser.add_argument("--depth-weight", type=float, default=DEPTH_WEIGHT,
                        help="Final depth loss weight after decay. Zero disables the loss.")
    parser.add_argument("--stop-after", type=int, help="Stop early and keep the original learning rate schedule.")
    train(**vars(parser.parse_args()))
