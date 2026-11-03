"""Render one held-out view for the sphere notebook."""
import json

import numpy as np
import torch
from PIL import Image

from data import get_rays
from model import MipNeRF


@torch.inference_mode()
def render_view(scene, view_id, checkpoints, scale=1, *, profile_cache=None):
    metadata = json.loads((scene / "transforms_test.json").read_text())
    frame = next(frame for frame in metadata["frames"] if frame["dtu_view_id"] == view_id)
    with Image.open(scene / (frame["file_path"] + ".png")) as image:
        reference = image.resize((image.width // scale, image.height // scale), Image.Resampling.LANCZOS)
    w, h = reference.size
    focal = 0.5 * w / np.tan(0.5 * metadata["camera_angle_x"])
    pose = torch.tensor(frame["transform_matrix"], dtype=torch.float32)
    rays = get_rays(h, w, focal, pose)
    origin, direction, _ = rays
    # The nearest intersection with the unit sphere gives reference depth.
    a = direction.square().sum(-1)
    b = (origin * direction).sum(-1)
    discriminant = b.square() - a * (origin.square().sum(-1) - 1)
    foreground = discriminant >= 0
    truth = (-b - discriminant.clamp_min(0).sqrt()) / a
    colours = [reference]
    depths = [truth]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batches = [field.reshape(-1, field.shape[-1]).to(device).split(512) for field in rays]
    for checkpoint in checkpoints:
        model = MipNeRF(near=2, far=6).to(device).eval()
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
        chunks = [model(*batch, return_depth=True)[1:] for batch in zip(*batches)]
        rgb, depth = (torch.cat(parts).cpu() for parts in zip(*chunks))
        colours.append(Image.fromarray((rgb.reshape(h, w, 3).clamp(0, 1).numpy()*255).round().astype("uint8")))
        depths.append(depth.reshape(h, w))

    if profile_cache is not None:
        np.savez_compressed(profile_cache, truth=truth.numpy(), ids=foreground.numpy(),
                            predictions=np.stack([d.numpy() for d in depths[1:]]))
    # Use the same depth scale and reference foreground mask in every panel.
    panels = Image.new("RGB", (w * len(colours), h * 2), "white")
    for i, (colour, depth) in enumerate(zip(colours, depths)):
        shade = (255 * (4 - depth).clamp(0, 1)).round().byte().numpy()
        depth_image = np.repeat(shade[:, :, None], 3, axis=2)
        depth_image[~foreground.numpy()] = [49, 60, 69]
        panels.paste(colour, (i * w, 0))
        panels.paste(Image.fromarray(depth_image), (i * w, h))
    rows = []
    mask = foreground.numpy()
    target = np.asarray(reference, dtype=float)[mask] / 255
    for path, image, depth in zip(checkpoints, colours[1:], depths[1:]):
        predicted = np.asarray(image, dtype=float)[mask] / 255
        mse = np.mean((predicted - target) ** 2)
        rows.append({"Checkpoint": path.parent.name,
                     "Displayed foreground PSNR (dB)": round(-10 * np.log10(mse), 3) if mse else float("inf"),
                     "Surface depth MAE": round(float((depth - truth)[foreground].abs().mean()), 4)})
    return panels, rows
