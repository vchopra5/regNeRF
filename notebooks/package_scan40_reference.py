"""Package small reference photos and camera data. Do not train."""
import hashlib
import json

import numpy as np
from PIL import Image

import dtu
from notebooks.lab import ROOT


def main():
    source = ROOT / "data/dtu/scan40"
    output = ROOT / "notebooks/results/scan40_reference"
    (output / "image").mkdir(parents=True, exist_ok=True)
    sizes = {}
    for view in sorted(set(dtu.TRAIN_IDS[:3] + dtu.TEST_IDS)):
        with Image.open(source / "image" / f"{view:06}.png") as photo:
            sizes[str(view)] = list(photo.size)
            preview = photo.convert("RGB")
            preview.thumbnail((400, 300), Image.Resampling.LANCZOS)
            preview.save(output / "image" / f"{view:06}.png")
    with np.load(source / "cameras.npz") as cameras:
        h, w, intrinsics, pose = dtu.camera(cameras, 25, sizes["25"], 16)
    record = dict(scene="DTU scan 40", train_ids=dtu.TRAIN_IDS, test_ids=dtu.TEST_IDS,
                  rig_poses=dtu.load_rig_poses(source).tolist(),
                  live_camera=dict(view=25, scale=16, height=h, width=w,
                                   intrinsics=intrinsics, pose=pose.tolist()),
                  original_photo_sizes=sizes,
                  camera_sha256=hashlib.sha256((source / "cameras.npz").read_bytes()).hexdigest(),
                  note="Reduced photos are for display only. Saved PSNR uses the original evaluation protocol.")
    (output / "cameras.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Packaged {len(sizes)} reference photos and all camera poses.")


if __name__ == "__main__":
    main()
