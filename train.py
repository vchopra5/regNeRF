
import torch
import torch.nn.functional as F

from data import load_data, get_rays
from model import MipNeRF


# Settings
DATA_DIR = "nerf_synthetic/lego"
SCALE = 2
BATCH_SIZE = 256
NUM_STEPS = 30_000
LEARNING_RATE = 5e-4
PRINT_EVERY = 100
SAVE_EVERY = 5_000


def train():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Using device:", device)

    # 1. Load the dataset
    images, poses, focal = load_data(
        DATA_DIR, split="train", scale=SCALE
    )

    N, H, W, _ = images.shape

    poses = poses.to(device)

    print("Images:", images.shape)
    print("Focal:", focal)

    # 2. Initialize model and optimizer
    model = MipNeRF().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    model.train()

    # 3. Training loop
    for step in range(1, NUM_STEPS + 1):

        # Choose a random training image
        image_idx = torch.randint(N, (1,)).item()

        pose = poses[image_idx]

        # Generate all rays for this camera
        rays_o, rays_d, radii = get_rays(
            H, W, focal, pose
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

        # 5. Compute losses
        coarse_loss = F.mse_loss(coarse_rgb, batch_rgb)
        fine_loss = F.mse_loss(fine_rgb, batch_rgb)

        loss = 0.1 * coarse_loss + fine_loss

        # 6. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7. Print progress
        if step % PRINT_EVERY == 0:
            psnr = -10.0 * torch.log10(
                fine_loss.detach().clamp_min(1e-10)
            )

            print(
                f"Step {step}/{NUM_STEPS} | "
                f"Loss: {loss.item():.6f} | "
                f"PSNR: {psnr.item():.2f} dB"
            )

        # 8. Save model checkpoints
        if step % SAVE_EVERY == 0:
            torch.save({
                "step": step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, f"checkpoint_{step}.pt")

    # Save final model
    torch.save(model.state_dict(), "mipnerf_final.pt")

    print("Training complete!")


if __name__ == "__main__":
    train()
