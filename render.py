"""Render one dataset camera view to a PNG image."""

import argparse
import json
import math
from pathlib import Path

import torch
import numpy as np
import dtu
from PIL import Image

from data import get_rays
from model import load_model, render_rays


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
    if (data_dir / "cameras.npz").is_file():
        ids = dtu.view_ids(split)
        if not 0 <= image_index < len(ids):
            raise ValueError(f"Image index must be between 0 and {len(ids) - 1}.")
        index = ids[image_index]
        with np.load(data_dir / "cameras.npz") as archive, Image.open(
                data_dir / "image" / f"{index:06}.png") as source:
            return dtu.camera(archive, index, source.size, scale)
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
    # Keep the full ray arrays on the CPU. render_rays transfers one batch at a time.
    rays = [ray.reshape(height * width, -1) for ray in get_rays(height, width, focal, pose.cpu())]
    pixels = render_rays(model.eval(), rays, batch_size, device).cpu()
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

    bounds = dtu.BOUNDS if (args.data_dir / "cameras.npz").is_file() else (2.0, 6.0)
    model = load_model(args.checkpoint, device, near=bounds[0], far=bounds[1])

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
