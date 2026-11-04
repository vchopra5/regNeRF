"""Move a camera around the two scored Notebook 3 models. Do not train."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import dtu
from data import get_rays
from model import load_model, render_rays
from notebooks.lab import ROOT
from notebooks.live_camera import _digest, moved_camera
from notebooks.toy_figures import Board, TITLE, SMALL


def local_checkpoints():
    folder = ROOT / "notebooks/models"
    manifest = json.loads((folder / "manifest.json").read_text())
    record = json.loads((ROOT / "notebooks/results/partner_stone/results.json").read_text())
    paths = {}
    for mode, key in (("regnerf", "extension_regnerf"), ("partner", "edge_l1")):
        entry = manifest[key]
        if (entry["source"].replace("\\", "/") != record["checkpoints"][mode].replace("\\", "/")
                or entry["step"] != record["settings"]["steps"]
                or entry["source_checkpoint_sha256"] != record["checkpoint_sha256"][mode]):
            raise ValueError("The camera model does not match the scored Notebook 3 checkpoint.")
        path = folder / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Restore the model file: {path.name}")
        if _digest(str(path), path.stat().st_mtime_ns) != entry["sha256"]:
            raise ValueError(f"The model file changed: {path.name}")
        paths[mode] = path
    return paths, record


def rotation(axis, degrees):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    angle = np.deg2rad(degrees)
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


def orbit_camera(horizontal, vertical):
    """Orbit the scene origin at camera 25's fixed distance and focal length."""
    if not np.isfinite([horizontal, vertical]).all() or not (-180 <= horizontal <= 180 and -90 <= vertical <= 90):
        raise ValueError("Use horizontal angles from -180 to 180 and vertical angles from -90 to 90.")
    h, w, intrinsics, pose = moved_camera(ROOT / "notebooks/results/scan40_reference", 0)
    base = pose.numpy()
    yaw = rotation(base[:3, 1], horizontal)
    pitch = rotation(yaw @ base[:3, 0], vertical)
    turn = torch.tensor(pitch @ yaw, dtype=pose.dtype)
    moved = pose.clone()
    moved[:3, :3] = turn @ pose[:3, :3]
    moved[:3, 3] = turn @ pose[:3, 3]
    return h, w, intrinsics, moved


@lru_cache(maxsize=32)
@torch.inference_mode()
def _render(path, modified, horizontal, vertical, device):
    model = load_model(path, device, near=dtu.BOUNDS[0], far=dtu.BOUNDS[1])
    h, w, intrinsics, pose = orbit_camera(horizontal, vertical)
    rays = [r.reshape(h * w, -1) for r in get_rays(h, w, intrinsics, pose)]
    rgb, depth = render_rays(model, rays, 512, device, return_depth=True)
    rgb = rgb.cpu().numpy().reshape(h, w, 3)
    depth = depth.cpu().numpy().reshape(h, w)
    if not np.isfinite(rgb).all() or not np.isfinite(depth).all():
        raise ValueError("The model produced an invalid colour or depth value.")
    return rgb, depth


def render_comparison(horizontal, vertical, *, device=None):
    """Return colour and depth boards with fixed start views above moved views."""
    orbit_camera(horizontal, vertical)
    paths, record = local_checkpoints()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    low, high = record["render_record"]["depth_range"]
    boards = {}
    pictures = {}
    for mode, path in paths.items():
        for angles in ((0, 0), (horizontal, vertical)):
            pictures[mode, angles] = _render(str(path), path.stat().st_mtime_ns, *angles, device)
    for kind in ("rgb", "depth"):
        board = Board(640, 586)
        for col, (mode, label) in enumerate((("regnerf", "RegNeRF"), ("partner", "RegNeRF + edge L1"))):
            x = col * 324
            board.text((x, 4), label, size=TITLE)
            for row, angles in enumerate(((0, 0), (horizontal, vertical))):
                y = 34 + row * 262
                label = "Start: camera 25" if row == 0 else f"Turn {horizontal:g} deg; tilt {vertical:g} deg"
                board.text((x, y), label, size=SMALL)
                rgb, depth = pictures[mode, angles]
                values = rgb.clip(0, 1) if kind == "rgb" else np.clip((depth - low) / (high - low), 0, 1)
                pixels = np.rint(values * 255).astype(np.uint8)
                board.paste(Image.fromarray(pixels), (x, y + 22), (300, 225))
        caption = ("Same camera for both models. Top: start. Bottom: your view." if kind == "rgb"
                   else f"Depth: dark = {low:.3f}; light = {high:.3f}. Same scale as the saved views.")
        board.text((0, 566), caption, size=SMALL)
        boards[kind] = board.image
    return boards, device, record["settings"]["steps"]
