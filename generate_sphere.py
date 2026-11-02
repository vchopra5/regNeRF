"""Create the toy sphere images, camera poses, and reference depth maps."""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def scan40_rotations(scene):
    import dtu

    rig = dtu.load_rig_poses(scene).numpy()[:, :3, :3].astype(np.float64)
    # Express the rig in a sphere frame where camera 25 is level and upright.
    sphere_basis = np.eye(3)[:, [1, 2, 0]]
    rotations = (sphere_basis @ rig[dtu.TRAIN_IDS[0]].T) @ rig
    return {split: [(i, rotations[i]) for i in ids] for split, ids in (
        ("train", dtu.TRAIN_IDS[:3]), ("val", dtu.TRAIN_IDS[3:] + dtu.EXCLUDED),
        ("test", dtu.TEST_IDS))}


def orbit_rotation(azimuth, elevation):
    az, el = np.deg2rad([azimuth, elevation])
    back = np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.])
    return np.column_stack((right, np.cross(back, right), back))


def generate(output, train_distances=(4.0,), x_radius=1.0, views=None, radius=1.0,
             image_size=(256, 256), *, rotations=None):
    if not np.isfinite(x_radius) or x_radius <= 0:
        raise ValueError("The x radius must be positive and finite.")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("The radius must be positive and finite.")
    axes = radius * np.array([x_radius, 1.0, 1.0])
    width, height = image_size
    # Keep the vertical framing when the image becomes wider.
    focal = 0.5 * height / np.tan(np.deg2rad(36) / 2)
    angle = 2 * np.arctan(0.5 * width / focal)
    y, x = np.indices((height, width)) + 0.5
    camera_rays = np.stack(((x - width / 2) / focal,
                            -(y - height / 2) / focal, -np.ones_like(x)), axis=-1)
    colours = np.array([[42, 107, 174], [246, 186, 60], [206, 72, 64]])
    light = np.array([1., -1., 2.]) / np.sqrt(6)
    if views is None and rotations is None:
        views = {
            "train": [
                (-45, 5), (0, 25), (45, 10),
                (75, -20), (105, 40), (135, 0), (165, -30),
                (-165, 55), (-135, 15), (-105, -10), (-75, 35), (-20, -40),
            ],
            "test": [(22, 15), (90, 10), (180, -15), (-120, 30)],
        }
    if rotations is None:
        rotations = {split: [(None, orbit_rotation(*angle)) for angle in angles]
                     for split, angles in views.items()}
    for split, cameras in rotations.items():
        (output / split).mkdir(parents=True, exist_ok=True)
        frames = []
        for i, (camera_id, rotation) in enumerate(cameras):
            pose = np.eye(4)
            pose[:3, :3] = rotation
            distance = train_distances[i % len(train_distances)] if split == "train" else 4.0
            origin = pose[:3, 3] = distance * rotation[:, 2]
            rays = camera_rays @ pose[:3, :3].T

            # Intersect the unit sphere in scaled coordinates. Keep t in camera units.
            scaled_rays = rays / axes
            scaled_origin = origin / axes
            a = (scaled_rays * scaled_rays).sum(axis=-1)
            b = scaled_rays @ scaled_origin
            disc = b*b - a * (scaled_origin @ scaled_origin - 1)
            hit = disc >= 0
            depth = np.zeros((height, width), dtype=np.float32)
            depth[hit] = (-b[hit] - np.sqrt(disc[hit])) / a[hit]
            surface = origin + depth[hit, None] * rays[hit]
            band = np.digitize(surface[:, 2] / radius, [-1/3, 1/3])
            normal = surface / axes**2
            normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
            shade = 0.65 + 0.35 * np.maximum(normal @ light, 0)
            image = np.full((height, width, 3), 255, dtype=np.uint8)
            image[hit] = np.rint(colours[band] * shade[:, None]).astype(np.uint8)

            name = f"{split}/r_{i}"
            Image.fromarray(image).save(output / f"{name}.png")
            np.save(output / f"{name}_depth.npy", depth)
            frame = {"file_path": name, "transform_matrix": pose.tolist()}
            if camera_id is not None:
                frame["dtu_view_id"] = camera_id
            frames.append(frame)
        metadata = {"camera_angle_x": float(angle), "frames": frames}
        (output / f"transforms_{split}.json").write_text(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    generate(Path(__file__).resolve().parent / "toy_sphere")
