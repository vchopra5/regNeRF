"""Ray-trace the two-sphere toy scene with reference depth and object IDs."""
import numpy as np
import torch

from data import get_rays


H, W = 300, 400
FOCAL = 0.5 * H / np.tan(np.deg2rad(18))
SPHERES = [((0.55, -0.35, 0), 0.65), ((-0.7, 0.42, 0), 0.78)]


def scene(rotation, distance=4, *, sideways_shift=0.0):
    pose = torch.eye(4)
    pose[:3, :3] = torch.tensor(rotation)
    pose[:3, 3] = distance * pose[:3, 2] + sideways_shift * pose[:3, 0]
    rays = get_rays(H, W, FOCAL, pose)
    o, d = (r.numpy() for r in rays[:2])
    a = (d*d).sum(-1)
    depth = np.full((H, W), np.inf)
    ids = np.zeros((H, W), dtype=np.uint8)
    rgb = np.full((H, W, 3), 255, dtype=np.uint8)
    for i, (center, radius) in enumerate(SPHERES, 1):
        v = o - center
        b = (v*d).sum(-1)
        disc = b*b - a*((v*v).sum(-1)-radius**2)
        t = (-b - np.sqrt(disc.clip(0))) / a
        hit = (disc >= 0) & (t > 0) & (t < depth)
        depth[hit], ids[hit] = t[hit], i
        n = (v[hit] + t[hit, None]*d[hit])/radius
        shade = .65 + .35*(n @ np.array([1.,-1.,2.])/np.sqrt(6)).clip(0)
        color = np.where((abs(n[:, 2:3]) < .30), [244,192,71], [199,60,46]) if i == 1 else np.array([50,142,183])
        rgb[hit] = np.rint(shade[:,None]*color)
    return pose, rays, rgb, depth, ids
