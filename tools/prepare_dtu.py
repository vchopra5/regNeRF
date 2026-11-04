"""Check IDR-format DTU data. Download scan 40 only when --download is given."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

import dtu


DATASET = "turandai/gaussian-surfels-dtu"
REVISION = "c70b6188f0eb1bac67843971e15ef97d213c6c09"
CAMERA_SHA256 = "af1ba75cbb137e5e9c83036a8db6da463365c6f54079ec31d8b8a162779fad78"


def validate(scene):
    scene = Path(scene)
    required = [scene / "cameras.npz"]
    required += [scene / "image" / f"{i:06}.png" for i in range(49)]
    required += [scene / "mask" / f"{i:03}.png" for i in dtu.TEST_IDS]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing DTU files:\n" + "\n".join(missing))
    sizes = {}
    with np.load(scene / "cameras.npz", allow_pickle=False) as cameras:
        for i in range(49):
            for prefix in ("world_mat", "scale_mat"):
                matrix = cameras[f"{prefix}_{i}"]
                if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
                    raise ValueError(f"Invalid camera matrix: {prefix}_{i}")
            with Image.open(scene / "image" / f"{i:06}.png") as image:
                sizes[i] = image.size
                image.verify()
            h, w, intrinsics, pose = dtu.camera(cameras, i, sizes[i], 2)
            if not np.isfinite(intrinsics).all() or not np.isfinite(pose.numpy()).all() or min(h, w) < 1:
                raise ValueError(f"Invalid calibration for view {i}")
    if len(set(sizes.values())) != 1:
        raise ValueError("The scan images must have the same dimensions.")
    for i in dtu.TEST_IDS:
        with Image.open(scene / "mask" / f"{i:03}.png") as mask:
            if mask.size != sizes[i]:
                raise ValueError(f"Mask size differs from image size for view {i}")
            if not (np.asarray(mask.convert("L")) > 127).any():
                raise ValueError(f"Empty foreground mask for view {i}")
    return dict(scene=scene.name, image_count=49, image_size=list(sizes[0]),
                training_views=list(dtu.TRAIN_IDS[:3]), test_views=list(dtu.TEST_IDS),
                foreground_mask_count=len(dtu.TEST_IDS),
                camera_sha256=hashlib.sha256((scene / "cameras.npz").read_bytes()).hexdigest(),
                file_sha256={p.relative_to(scene).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in required})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/dtu/scan40"))
    parser.add_argument("--download", action="store_true", help="Download only scan-40 images, masks, and cameras.")
    args = parser.parse_args()
    source = None
    if args.download:
        if args.data_dir.name != "scan40":
            parser.error("The pinned download is for scan40. Use a destination ending in scan40.")
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=DATASET, repo_type="dataset", revision=REVISION,
                          allow_patterns=["scan40/cameras.npz", "scan40/image/*.png", "scan40/mask/*.png"],
                          local_dir=args.data_dir.parent, token=False)
        source = dict(repository=DATASET, revision=REVISION, source_type="community mirror of IDR-format DTU")
    report = validate(args.data_dir)
    if source is not None:
        if report["camera_sha256"] != CAMERA_SHA256:
            raise ValueError("The downloaded scan-40 camera checksum differs from the verified mirror.")
        report["source"] = source
        (args.data_dir / "download_source.json").write_text(json.dumps(source, indent=2), encoding="utf-8")
    elif (args.data_dir / "download_source.json").is_file():
        report["source"] = json.loads((args.data_dir / "download_source.json").read_text())
    (args.data_dir / "validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Validated {report['scene']}: 49 images, 49 calibrated cameras, {len(dtu.TEST_IDS)} test masks.")
    print(f"Image size: {report['image_size']}. Training views: {report['training_views']}.")
    print(f"Saved {args.data_dir / 'validation.json'}. No training started.")


if __name__ == "__main__":
    main()
