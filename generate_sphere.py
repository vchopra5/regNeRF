"""Create the toy sphere images, camera poses, and reference depth maps."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def generate(output):
    size = 256
    angle = np.deg2rad(36)
    focal = 0.5 * size / np.tan(angle / 2)
    y, x = np.indices((size, size)) + 0.5
    camera_rays = np.stack(((x - size / 2) / focal,
                            -(y - size / 2) / focal, -np.ones_like(x)), axis=-1)
    colours = np.array([[42, 107, 174], [246, 186, 60], [206, 72, 64]])
    light = np.array([1., -1., 2.]) / np.sqrt(6)
    for split, angles in {
        "train": [(-45, 5), (0, 25), (45, 10)],
        "test": [(22, 15)],
    }.items():
        (output / split).mkdir(parents=True, exist_ok=True)
        frames = []
        for i, (azimuth, elevation) in enumerate(angles):
            az, el = np.deg2rad([azimuth, elevation])
            back = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
            right = np.array([-np.sin(az), np.cos(az), 0.])
            pose = np.eye(4)
            pose[:3, :3] = np.column_stack((right, np.cross(back, right), back))
            origin = pose[:3, 3] = 4 * back
            rays = camera_rays @ pose[:3, :3].T

            # Solve |origin + t * ray|^2 = 1. Use the nearer intersection.
            a = (rays * rays).sum(axis=-1)
            b = rays @ origin
            disc = b*b - a * (origin @ origin - 1)
            hit = disc >= 0
            depth = np.zeros((size, size), dtype=np.float32)
            depth[hit] = (-b[hit] - np.sqrt(disc[hit])) / a[hit]
            normal = origin + depth[hit, None] * rays[hit]
            band = np.digitize(normal[:, 2], [-1/3, 1/3])
            shade = 0.65 + 0.35 * np.maximum(normal @ light, 0)
            image = np.full((size, size, 3), 255, dtype=np.uint8)
            image[hit] = np.rint(colours[band] * shade[:, None]).astype(np.uint8)

            name = f"{split}/r_{i}"
            Image.fromarray(image).save(output / f"{name}.png")
            np.save(output / f"{name}_depth.npy", depth)
            frames.append({"file_path": name, "transform_matrix": pose.tolist()})
        metadata = {"camera_angle_x": float(angle), "frames": frames}
        (output / f"transforms_{split}.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent / "toy_sphere")
