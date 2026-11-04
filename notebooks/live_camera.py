"""Render a moved camera from the shared scan 40 models. Do not train."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

import dtu
from model import load_model
from notebooks.lab import ROOT
from notebooks.toy_figures import Board, TITLE, SMALL, MUTED
from render import render_image


def local_checkpoints():
    """Check the shared files and their source against the score record."""
    folder = ROOT / "notebooks/models"
    manifest = json.loads((folder / "manifest.json").read_text())
    record = json.loads((ROOT / "notebooks/results/scan40.json").read_text())["record"]
    paths = {}
    for name, key in (("off", "off"), ("on", "regnerf")):
        entry = manifest[key]
        if entry["source"] != record["checkpoints"][name] or entry["step"] != record["step"]:
            raise ValueError(f"The shared {key} model does not refer to the scored checkpoint.")
        path = folder / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Restore the saved model: {path}")
        if _digest(str(path), path.stat().st_mtime_ns) != entry["sha256"]:
            raise ValueError(f"This model does not match its saved file hash: {path}")
        paths[name] = path
    return paths, record["step"]


@lru_cache(maxsize=4)
def _digest(path, modified):
    with open(path, "rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def moved_camera(scene, angle, scale=16):
    """Rotate camera 25 about its right axis through the scene origin."""
    bundled = scene / "cameras.json"
    if bundled.is_file():
        camera = json.loads(bundled.read_text(encoding="utf-8"))["live_camera"]
        if scale != camera["scale"]:
            raise ValueError("The bundled live camera uses scale 16.")
        h, w = camera["height"], camera["width"]
        intrinsics = tuple(camera["intrinsics"])
        pose = torch.tensor(camera["pose"], dtype=torch.float32)
    else:
        with np.load(scene / "cameras.npz") as archive, Image.open(scene / "image/000025.png") as source:
            h, w, intrinsics, pose = dtu.camera(archive, 25, source.size, scale)
    axis = pose[:3, 0].numpy().astype(float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    theta = np.deg2rad(angle)
    rotation = torch.tensor(np.eye(3) + np.sin(theta) * cross
                            + (1 - np.cos(theta)) * (cross @ cross), dtype=pose.dtype)
    moved = pose.clone()
    moved[:3, :3] = rotation @ pose[:3, :3]
    moved[:3, 3] = rotation @ pose[:3, 3]
    return h, w, intrinsics, moved


@lru_cache(maxsize=32)
def _render(path, modified, scene, angle, device):
    # Keep only the small output in memory. Release the model after each render.
    model = load_model(path, device, near=dtu.BOUNDS[0], far=dtu.BOUNDS[1])
    h, w, intrinsics, pose = moved_camera(Path(scene), angle)
    return Image.fromarray(render_image(model, h, w, intrinsics, pose, 512, device))


def render_comparison(scene, angle, *, device=None):
    """Show the start and moved views at the same size for both models."""
    if not 0 <= angle <= 20:
        raise ValueError("Use a camera angle from 0 to 20 degrees.")
    paths, step = local_checkpoints()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pictures = {}
    for name, path in paths.items():
        for position in (0, angle):
            pictures[name, position] = _render(str(path), path.stat().st_mtime_ns,
                                               str(scene), position, device)
    board = Board(640, 558)
    for col, name in enumerate(("off", "on")):
        x = col * 324
        board.text((x, 4), f"Smoothing {name}", size=TITLE)
        for row, position in enumerate((0, angle)):
            y = 34 + row * 262
            board.text((x, y), "Start: training camera 25" if row == 0 else f"Moved: {angle:g} degrees",
                       size=SMALL, fill=MUTED)
            board.paste(pictures[name, position], (x, y + 22), (300, 225))
    return board.image, device, step
