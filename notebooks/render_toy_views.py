"""Prepare the lower-camera saved results from existing model checkpoints."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from notebooks.toy_experiment import ASSETS, CAMERAS, TOY_SEED, render_condition_view


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=Path("checkpoint/toy_two_spheres_v1"))
    args = parser.parse_args()
    from generate_sphere import orbit_rotation
    from two_spheres import scene
    output = ASSETS / "lower"
    output.mkdir(parents=True, exist_ok=True)
    _, _, rgb, depth, ids = scene(orbit_rotation(*CAMERAS["Lower view"]))
    Image.fromarray(rgb[::2, ::2]).save(output / "reference.png")
    np.savez_compressed(output / "reference.npz", depth=depth[::2, ::2].astype("float32"), ids=ids[::2, ::2])
    for name in ("off", "on", "off_100", "on_100", "off_1000", "on_1000"):
        record = json.loads((ASSETS / name / "result.json").read_text())
        record["seed"] = TOY_SEED
        result = render_condition_view(args.checkpoint_root / name / "mipnerf_final.pt", output / name,
                                       CAMERAS["Lower view"], record)
        print(name, result["foreground_psnr"], result["depth_mae"], flush=True)


if __name__ == "__main__":
    main()
