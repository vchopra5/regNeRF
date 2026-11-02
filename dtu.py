"""Camera calibration and view selection for an IDR-format DTU scan."""

import numpy as np
import torch
from PIL import Image


TRAIN_IDS = (25, 22, 28, 40, 44, 48, 0, 8, 13)
EXCLUDED = (3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 36, 37, 38, 39)
TEST_IDS = tuple(i for i in range(49) if i not in TRAIN_IDS + EXCLUDED)
BOUNDS = (0.5, 3.5)


def view_ids(split):
    if split == "train":
        return TRAIN_IDS
    if split == "test":
        return TEST_IDS
    raise ValueError("Use the train or test split for DTU.")


def _calibration(archive, index):
    projection = (archive[f"world_mat_{index}"].astype(np.float64)
                  @ archive[f"scale_mat_{index}"])[:3]
    m = projection[:, :3]
    # M = K R. QR on its reversed transpose gives the RQ decomposition.
    q, r = np.linalg.qr(m[::-1].T)
    k, rotation = r.T[::-1, ::-1], q.T[::-1]
    signs = np.diag(np.sign(np.diag(k)))
    k, rotation = k @ signs, signs @ rotation
    k /= k[2, 2]
    pose = np.eye(4)
    pose[:3, 3] = np.linalg.solve(m, -projection[:, 3])
    pose[:3, :3] = rotation.T @ np.diag([1., -1., -1.])
    return k, torch.tensor(pose, dtype=torch.float32)


def load_rig_poses(scene):
    with np.load(scene / "cameras.npz") as archive:
        return torch.stack([_calibration(archive, i)[1] for i in range(49)])


def camera(archive, index, size, scale):
    width, height = size
    w, h = width // scale, height // scale
    if min(w, h) < 1:
        raise ValueError("The scale is too large for this image.")
    k, pose = _calibration(archive, index)
    sx, sy = w / width, h / height
    # DTU indexes pixel centres at integers. get_rays uses half-integers.
    intrinsics = (sx*k[0, 0], sy*k[1, 1], sx*(k[0, 2]+0.5),
                  sy*(k[1, 2]+0.5), sx*k[0, 1])
    return h, w, intrinsics, pose


def load_views(scene, split, scale):
    images, poses, intrinsics = [], [], []
    with np.load(scene / "cameras.npz") as archive:
        for index in view_ids(split):
            with Image.open(scene / "image" / f"{index:06}.png") as image:
                h, w, calibration, pose = camera(archive, index, image.size, scale)
                rgb = image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS)
            images.append(np.asarray(rgb))
            poses.append(pose)
            intrinsics.append(calibration)
    return torch.from_numpy(np.stack(images)).float().div_(255), torch.stack(poses), intrinsics
