"""Report foreground RGB PSNR for each held-out DTU view and their mean."""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import dtu
from data import get_rays
from model import MipNeRF
from render import positive_int


@torch.inference_mode()
def evaluate(checkpoint, scene, scale, batch_size, device):
    model = MipNeRF(near=dtu.BOUNDS[0], far=dtu.BOUNDS[1])
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved.get("model", saved))
    model.to(device).eval()
    scores = []
    print(f"Checkpoint {checkpoint} | scene {scene} | scale {scale}", flush=True)
    with np.load(scene / "cameras.npz") as cameras:
        for index in dtu.TEST_IDS:
            with Image.open(scene / "image" / f"{index:06}.png") as photo:
                size = photo.size
                h, w, intrinsics, pose = dtu.camera(cameras, index, size, scale)
                rgb = photo.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
            with Image.open(scene / "mask" / f"{index:03}.png") as source:
                if source.size != size:
                    raise ValueError(f"Mask size differs from photo size for view {index}.")
                mask = np.asarray(source.convert("L").resize((w, h), Image.Resampling.NEAREST)) > 127
            count = int(mask.sum())
            if not count:
                raise ValueError(f"Empty foreground mask for view {index}.")
            target = torch.from_numpy(np.asarray(rgb)[mask]).to(device).float() / 255
            rays = [field[mask] for field in get_rays(h, w, intrinsics, pose)]
            squared_error = torch.zeros((), device=device, dtype=torch.float64)
            for start in range(0, count, batch_size):
                batch = slice(start, start + batch_size)
                _, prediction = model(*(r[batch].to(device) for r in rays))
                squared_error += (prediction - target[batch]).square().sum(dtype=torch.float64)
            mse = squared_error.item() / (count * 3)
            if not math.isfinite(mse):
                raise ValueError(f"Non-finite RGB error for view {index}.")
            score = -10 * math.log10(mse) if mse else math.inf
            scores.append(score)
            print(f"View {index:02} | {count} foreground pixels | {score:.3f} dB", flush=True)
    mean = sum(scores) / len(scores)
    print(f"Mean foreground PSNR ({len(scores)} test views): {mean:.3f} dB", flush=True)
    return mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dtu/scan40"))
    parser.add_argument("--scale", type=positive_int, default=2)
    parser.add_argument("--batch-size", type=positive_int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.data_dir, args.scale, args.batch_size, args.device)
