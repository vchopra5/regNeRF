"""Train the saved toy-lesson runs with the current code and copy the results into the assets.

One seed trains 9 models (about 7 minutes on a GPU). The first seed in --seeds also saves
images; it must be TOY_SEED. Every seed saves its score records under assets/seeds/<seed>/.
Checkpoints go to --checkpoint-root, which Git ignores. Then run
`python -m notebooks.package_toy_results` to rebuild the archive.

Examples:
  python -m notebooks.build_toy_results --checkpoint-root checkpoint/toy_v3 --seeds 42 142 242 342 442
  python -m notebooks.build_toy_results --checkpoint-root checkpoint/toy_v3_extra --seeds 142 --scores-only
"""
import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from notebooks.toy_experiment import ASSETS, TOY_SEED, ensure_assets, prepare_scene, run_condition

# Section 1: the first 3 are the lesson's training views. The 9- and 20-photo runs add views.
COVERAGE_ANGLES = [(-35, 12), (0, 22), (40, -8), (-20, -8), (10, -8), (-35, 7), (40, 7), (-5, 22), (25, 22),
                   (-35, -8), (-5, -8), (25, -8), (-20, 7), (-5, 7), (10, 7), (25, 7), (-35, 22), (-20, 22),
                   (10, 22), (40, 22)]
COVERAGE_STEPS = 500
# Section 6: smoothing off and on at three step counts.
CONDITION_STEPS = (100, 300, 1000)
CONDITION_WEIGHT = 0.1


def copy_files(source, target, names):
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        shutil.copyfile(source / name, target / name)


def input_grid(scene, count, columns=5, cell=(128, 98)):
    """Thumbnail grid of the training photos, as shown in section 1."""
    rows = -(-count // columns)
    grid = Image.new("RGB", (cell[0] * min(count, columns), cell[1] * rows), "white")
    for i in range(count):
        with Image.open(scene / f"input_{i}.png") as photo:
            photo = photo.convert("RGB").resize((cell[0], round(cell[0] * photo.height / photo.width)),
                                                Image.Resampling.LANCZOS)
        grid.paste(photo, (cell[0] * (i % columns), cell[1] * (i // columns)))
    return grid


def build_seed(root, seed, images):
    """Train all 9 runs for one seed. Save scores for the seed; save images only if requested."""
    records = ASSETS / "seeds" / str(seed)
    files = ("rgb.png", "prediction.npz", "result.json") if images else ("result.json",)
    for steps in CONDITION_STEPS:
        for prefix, weight in (("off", 0.0), ("on", CONDITION_WEIGHT)):
            name = prefix if steps == 300 else f"{prefix}_{steps}"
            run_condition(root / name, steps, weight, seed=seed)
            for camera in ("", "lower"):
                copy_files(root / name / camera, records / camera / name, ("result.json",))
                if images:
                    copy_files(root / name / camera, ASSETS / camera / name, files)
            print(seed, name, json.loads((records / name / "result.json").read_text())["foreground_psnr"], flush=True)

    for count in (3, 9, 20):
        scene = root / f"coverage_scene_{count}"
        prepare_scene(scene, extra_angles=COVERAGE_ANGLES[3:count])
        folder = root / f"coverage_{count}"
        result = run_condition(folder, COVERAGE_STEPS, 0.0, training_scene=scene, seed=seed)
        result.update(views=count, angles=[list(a) for a in COVERAGE_ANGLES[:count]])
        record = records / "coverage_500" / str(count)
        record.mkdir(parents=True, exist_ok=True)
        (record / "result.json").write_text(json.dumps(result, indent=2))
        if images:
            target = ASSETS / "coverage_500" / str(count)
            target.mkdir(parents=True, exist_ok=True)
            (target / "result.json").write_text(json.dumps(result, indent=2))
            shutil.copyfile(folder / "rgb.png", target / "heldout.png")
            if count == 3:
                shutil.copyfile(folder / "prediction.npz", target / "prediction.npz")
            input_grid(scene, count).save(target / "inputs.png")
        print(seed, "coverage", count, result["foreground_psnr"], result["depth_mae"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint-root", type=Path, required=True, help="New folder for the models.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[TOY_SEED])
    parser.add_argument("--scores-only", action="store_true", help="Do not replace the saved images.")
    args = parser.parse_args()
    if not args.scores_only and args.seeds[0] != TOY_SEED:
        parser.error(f"The first seed saves the images, so it must be {TOY_SEED}. Or pass --scores-only.")
    if not (ASSETS / "transforms_train.json").is_file():
        ensure_assets()
    args.checkpoint_root.mkdir(parents=True, exist_ok=False)
    for i, seed in enumerate(args.seeds):
        build_seed(args.checkpoint_root / f"seed_{seed}", seed, images=not args.scores_only and i == 0)
    print("Done. Now run: python -m notebooks.package_toy_results")


if __name__ == "__main__":
    main()
