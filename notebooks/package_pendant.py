"""Package the saved pendant evidence. Do not train or render a model."""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from notebooks.lab import ROOT
from notebooks.toy_figures import Board, TITLE, SMALL


def main():
    earlier = ROOT / "checkpoint/canny_l1_pendant_8k_20261103"
    partner = ROOT / "checkpoint/partner_pendant_8k_20261103_run2"
    output = ROOT / "notebooks/results/pendant"
    output.mkdir(parents=True, exist_ok=True)
    scores = [r for r in json.loads((partner / "scores_8000.json").read_text()) if r["mode"] in ("regnerf", "partner")]
    original_summary = json.loads((partner / "summary_8000.json").read_text())
    summary = {"means": {m: original_summary["means"][m] for m in ("regnerf", "partner")},
               "partner_minus_regnerf": original_summary["partner_minus_regnerf"],
               "aggregation": original_summary["aggregation"]}
    run = json.loads((partner / "record.json").read_text())
    previous = run["previous_record"]
    modes = ("regnerf", "partner")
    labels = ("Reference", "RegNeRF", "RegNeRF + edge L1")
    for az, el in ((15, 5), (-15, 5), (0, 15)):
        tag = f"az{az}_el{el}"
        reference = np.load(earlier / f"{tag}_reference.npz")
        arrays = [reference] + [np.load((partner if m == "partner" else earlier) / f"{tag}_{m}_8000.npz") for m in modes]
        truth = reference["depth"]
        for kind in ("rgb", "depth", "error"):
            board = Board(806, 282)
            for i, (label, array) in enumerate(zip(labels, arrays)):
                x = 8 + i * 266
                board.text((x, 8), label, size=TITLE)
                if kind == "rgb":
                    rgb = array["rgb"] if i == 0 else np.rint(array["rgb"].clip(0, 1) * 255).astype(np.uint8)
                elif kind == "depth":
                    rgb = np.rint((np.nan_to_num(array["depth"], posinf=5).clip(3, 5)-3)*127.5).astype(np.uint8)
                elif i == 0:
                    rgb = np.full((*truth.shape, 3), 238, dtype=np.uint8)
                    rgb[reference["flat"]] = [64, 132, 173]
                    rgb[reference["boundary"]] = [225, 105, 28]
                else:
                    error = np.nan_to_num(np.abs(array["depth"]-truth), posinf=0).clip(0, .5)/.5
                    rgb = np.rint(255 + error[..., None]*(np.array([145, 20, 25])-255)).astype(np.uint8)
                    rgb[~np.isfinite(truth)] = 220
                board.paste(Image.fromarray(rgb), (x, 36), (256, 192))
            captions = {"rgb": ["Saved colour renders. Same camera and image size for both models."],
                        "depth": ["Depth in scene units: black = 3; white = 5.", "Values outside this range are clipped for display only."],
                        "error": ["Left: orange = boundary; blue = flat panel. Grey = no reference depth.", "Absolute depth error: white = 0; dark red = 0.5 scene units or more."]}[kind]
            for line, caption in enumerate(captions):
                board.text((8, 239 + line * 18), caption, size=SMALL)
            board.image.save(output / f"{tag}_{kind}.png")
    record = dict(summary=summary, per_view=scores, views=run["views"],
                  settings={key: run[key] for key in ("seed", "steps", "train_views", "scale", "eval_scale",
                      "stratified", "coarse_depth_weight", "fine_depth_weight", "initial_depth_weight",
                      "edge_loss_weight", "edge_scale_quantile", "edge_halo", "weight_selection", "device", "adaptation")},
                  checkpoints={m: run["checkpoints"][m] for m in modes}, partner_revision=run["revision"],
                  source_sha256=run["source_sha256"],
                  input_sha256=previous["input_sha256"],
                  baseline_source_sha256=previous["baseline_record"]["source_sha256"],
                  selection="Three fixed views used in earlier pendant studies. One scene and one seed.",
                  boundary=previous["boundary"], flat=previous["flat"],
                  source_records=[str((earlier / "record.json").relative_to(ROOT)), str((partner / "record.json").relative_to(ROOT))])
    (output / "results.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Saved nine panels and the score record in {output}")


if __name__ == "__main__":
    main()
