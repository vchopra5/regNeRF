"""Render one dataset camera view to a PNG image."""

import argparse
import json
import math
from pathlib import Path

import torch
from PIL import Image

from data import get_rays
from model import MipNeRF


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Use an integer greater than zero.")
    return value


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoint/mipnerf_final.pt"))
    parser.add_argument("--data-dir", type=Path, default=Path("nerf_synthetic/lego"))
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--image-index", type=int, default=0)
    parser.add_argument("--scale", type=positive_int, default=2,
                        help="Divide the image width and height by this value.")
    parser.add_argument("--batch-size", type=positive_int, default=256)
    parser.add_argument("--output", type=Path, default=Path("renders/result.png"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def load_camera(data_dir, split, image_index, scale):
    with (data_dir / f"transforms_{split}.json").open() as source:
        metadata = json.load(source)
    frames = metadata["frames"]
    if not 0 <= image_index < len(frames):
        raise ValueError(f"Image index must be between 0 and {len(frames) - 1}.")

    frame = frames[image_index]
    image_path = data_dir / frame["file_path"]
    if not image_path.suffix:
        image_path = image_path.with_suffix(".png")
    # Read only the selected image header to get the camera dimensions.
    with Image.open(image_path) as source:
        width, height = source.size
    width, height = width // scale, height // scale
    if min(width, height) < 1:
        raise ValueError("The scale is too large for this image.")

    focal = 0.5 * width / math.tan(0.5 * metadata["camera_angle_x"])
    pose = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
    return height, width, focal, pose


@torch.inference_mode()
def render_image(model, height, width, focal, pose, batch_size, device):
    model.eval()
    # Keep the full ray arrays on the CPU. Transfer only the current batch.
    rays_o, rays_d, radii = get_rays(height, width, focal, pose.cpu())
    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    radii = radii.reshape(-1, 1)
    pixels = torch.empty((height * width, 3), dtype=torch.float32)
    num_batches = (len(pixels) + batch_size - 1) // batch_size

    for batch_index, start in enumerate(range(0, len(pixels), batch_size), 1):
        end = min(start + batch_size, len(pixels))
        _, fine_rgb = model(
            rays_o[start:end].to(device),
            rays_d[start:end].to(device),
            radii[start:end].to(device),
        )
        pixels[start:end] = fine_rgb.cpu()
        if batch_index % 100 == 0 or batch_index == num_batches:
            print(f"Rendered {end}/{len(pixels)} pixels", flush=True)

    if not torch.isfinite(pixels).all():
        raise ValueError("The model produced non-finite pixel values.")
    return (pixels.reshape(height, width, 3).clamp(0, 1)
            .mul(255).round().to(torch.uint8).numpy())


def main():
    args = parse_args()
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    # train.py saves either a state dictionary or a training checkpoint.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model = MipNeRF()
    model.load_state_dict(state_dict)
    model.to(device)

    height, width, focal, pose = load_camera(
        args.data_dir, args.split, args.image_index, args.scale
    )
    print(f"Rendering {width} x {height} pixels on {device}", flush=True)
    image = render_image(model, height, width, focal, pose, args.batch_size, device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output, format="PNG")
    print(f"Saved PNG: {args.output.resolve()}")


if __name__ == "__main__":
    main()
