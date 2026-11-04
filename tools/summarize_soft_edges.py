"""Make a comparison figure with the measured result beside the renders."""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from generate_sphere import orbit_rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    folder, steps = args.folder, args.steps
    record = json.loads((folder / "record.json").read_text())
    spec = importlib.util.spec_from_file_location("pendant_scene", Path(record["scene"]) / "design.py")
    design = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(design)
    rotation = orbit_rotation(15, 5)
    truth = design.scene(rotation)[3]
    colours = [np.asarray(design.reference(rotation))]
    depths = [truth]
    for mode in ("uniform", "edge"):
        result = np.load(folder / f"az15_el5_{steps}_{mode}.npz")
        colours.append(np.rint(result["rgb"].clip(0, 1)*255).astype(np.uint8))
        depths.append(result["depth"])
    scores = json.loads((folder / f"scores_{steps}.json").read_text())
    metrics = ("foreground_psnr", "boundary_depth_rmse", "flat_depth_rmse")
    means = {mode: {key: float(np.mean([r[key] for r in scores if r["mode"] == mode]))
                    for key in metrics} for mode in ("uniform", "edge")}
    result = dict(step=steps, means=means,
                  on_minus_off={key: means["edge"][key]-means["uniform"][key] for key in metrics})
    (folder / f"summary_{steps}.json").write_text(json.dumps(result, indent=2))
    board = Image.new("RGB", (1200, 970), "#f7f5ef")
    draw = ImageDraw.Draw(board)
    title = ImageFont.truetype("arial.ttf", 29)
    font = ImageFont.truetype("arial.ttf", 19)
    small = ImageFont.truetype("arial.ttf", 17)
    draw.text((20, 15), "Does a colour edge weight preserve depth boundaries?", font=title, fill="#203039")
    draw.text((20, 56), f"Gold pendant | three input views | {steps:,} steps | same seed and depth weight", font=font, fill="#46545b")
    box = (120, 80, 280, 235)
    for i, (label, rgb, depth) in enumerate(zip(("Reference", "Edge weight OFF", "Edge weight ON"), colours, depths)):
        x = i*400+10
        draw.text((x+5, 97), label, font=font, fill="#203039")
        board.paste(Image.fromarray(rgb).crop(box).resize((380, 368), Image.Resampling.NEAREST), (x, 128))
        shade = np.rint((np.nan_to_num(depth, posinf=5).clip(3, 5)-3)/2*255).astype(np.uint8)
        board.paste(Image.fromarray(shade).convert("RGB").crop(box).resize((380, 368), Image.Resampling.NEAREST), (x, 525))
    draw.text((20, 500), "Depth: black = 3, white = 5 scene units. One scale for all columns.", font=small, fill="#46545b")
    boundary_change = 100 * (means["edge"]["boundary_depth_rmse"] / means["uniform"]["boundary_depth_rmse"]-1)
    flat_change = 100 * (means["edge"]["flat_depth_rmse"] / means["uniform"]["flat_depth_rmse"]-1)
    psnr_change = result["on_minus_off"]["foreground_psnr"]
    draw.text((20, 903), f"Mean changes across 3 views: boundary depth error {boundary_change:+.1f}% | flat depth error {flat_change:+.1f}% | RGB {psnr_change:+.2f} dB", font=small, fill="#203039")
    draw.text((20, 935), "Exploratory result: one synthetic scene and one seed. These are illustration views, not an independent test.", font=small, fill="#46545b")
    board.save(folder / f"submission_{steps}.png")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
