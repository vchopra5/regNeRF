"""Compare RegNeRF with RegNeRF + edge regularizer on DTU scan 40.

The edge regularizer adds an edge-aware L1 depth term on unseen-camera patches.
It finds colour edges itself (blur + Sobel on the rendered patch, no Canny).
The baseline is the saved RegNeRF run. Only the edge term changes.

Stages (each needs --output-dir):
  --pilot  Train 5000-step runs for each candidate lambda. Score them and the baseline's
           step-5000 checkpoint on validation views only. Choose lambda.
  --run    Train the 20000-step edge run with the chosen lambda.
  --score  Score the edge run and the baseline on the 25 test views.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

import dtu
import train as training
from eval_dtu import EVAL_SCALE, evaluate
from provenance import file_hashes, git, relative
from train import NUM_STEPS, SCALE, SEED, train

SCENE = Path("data/dtu/scan40")
BASELINE_DIR = Path("checkpoint/scan40_pair_new/on")
DEPTH_WEIGHT = 0.1
DEPTH_WEIGHT_START = 400.0
PILOT_STEPS = 5000
PILOT_LAMBDAS = (0.01, 0.1, 1.0)
# Training-split views not used by the 3-view runs. They are not test views.
VALIDATION_VIEWS = tuple(dtu.TRAIN_IDS[3:6])
SOURCES = ("train.py", "model.py", "data.py", "dtu.py", "eval_dtu.py", "eval_edges.py",
           "compare_edge.py", "provenance.py")


def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def record(**extra):
    return dict(scene=str(SCENE), utc=datetime.now(timezone.utc).isoformat(),
                code_revision=git("rev-parse", "HEAD"), local_status=git("status", "--short"),
                local_diff=git("diff", "HEAD", "--", "*.py"),
                code_sha256=file_hashes([s for s in SOURCES if Path(s).is_file()]),
                seed=SEED, training_views=list(dtu.TRAIN_IDS[:3]), training_scale=SCALE,
                evaluation_scale=EVAL_SCALE, device=device(), stratified=True,
                fine_depth_weight=DEPTH_WEIGHT, initial_fine_depth_weight=DEPTH_WEIGHT_START,
                coarse_depth_weight=0.1, edge_scale_quantile=training.EDGE_SCALE_QUANTILE,
                edge_halo=training.EDGE_HALO, baseline_dir=str(BASELINE_DIR),
                baseline_protocol=json.loads((BASELINE_DIR.parent / "protocol.json").read_text()),
                **extra)


def train_edge(folder, weight, steps):
    train(3, SCENE, folder, depth_weight=DEPTH_WEIGHT, depth_weight_start=DEPTH_WEIGHT_START,
          seed=SEED, stratified=True, stop_after=steps, edge_loss_weight=weight)


def pilot(output):
    out = output / "pilot"
    scores = {}
    baseline = BASELINE_DIR / f"checkpoint_{PILOT_STEPS}.pt"
    mean, per_view = evaluate(baseline, SCENE, EVAL_SCALE, 512, device(), return_per_view=True,
                              views=VALIDATION_VIEWS)
    scores["0"] = dict(checkpoint=relative(baseline), mean_db=mean, per_view_db=per_view)
    for weight in PILOT_LAMBDAS:
        folder = out / f"lambda_{weight:g}"
        if not (folder / f"checkpoint_{PILOT_STEPS}.pt").is_file():
            train_edge(folder, weight, PILOT_STEPS)
        checkpoint = folder / f"checkpoint_{PILOT_STEPS}.pt"
        mean, per_view = evaluate(checkpoint, SCENE, EVAL_SCALE, 512, device(),
                                  return_per_view=True, views=VALIDATION_VIEWS)
        scores[f"{weight:g}"] = dict(checkpoint=relative(checkpoint), mean_db=mean, per_view_db=per_view)
    # Best mean validation PSNR among the candidates. A tie (within 0.01 dB) goes to the smaller lambda.
    best = max(PILOT_LAMBDAS, key=lambda w: (round(scores[f"{w:g}"]["mean_db"], 2), -w))
    result = record(stage="pilot", steps=PILOT_STEPS, validation_views=list(VALIDATION_VIEWS),
                    candidates=list(PILOT_LAMBDAS), scores=scores, chosen_lambda=best,
                    rule="highest mean validation foreground PSNR; tie within 0.01 dB -> smaller lambda; "
                         "lambda 0 is the RegNeRF baseline for reference only")
    (output / "pilot.json").write_text(json.dumps(result, indent=2))
    for key, value in scores.items():
        print(f"lambda {key:>5} | validation mean {value['mean_db']:.3f} dB")
    print(f"Chosen lambda: {best:g}")


def chosen_lambda(output, given):
    if given is not None:
        return given
    path = output / "pilot.json"
    if not path.is_file():
        raise SystemExit("Run --pilot first, or pass --edge-loss-weight.")
    return json.loads(path.read_text())["chosen_lambda"]


def run(output, weight):
    folder = output / "edge"
    (output / "protocol.json").write_text(json.dumps(
        record(stage="run", steps=NUM_STEPS, edge_loss_weight=weight,
               checkpoints=dict(regnerf=str(BASELINE_DIR / f"checkpoint_{NUM_STEPS}.pt"),
                                edge=str(folder / f"checkpoint_{NUM_STEPS}.pt"))), indent=2))
    train_edge(folder, weight, NUM_STEPS)


def save_edge_map(path):
    """Save the soft edge map e of training view 25, with the same scale rule as training."""
    import numpy as np
    from PIL import Image
    from data import load_data

    images, _, _ = load_data(SCENE, split="train", scale=SCALE)
    magnitudes = [training.sobel_magnitude(images[i : i + 1]) for i in range(3)]
    scale = torch.quantile(torch.cat([m.flatten() for m in magnitudes]),
                           training.EDGE_SCALE_QUANTILE).item()
    edges = (1 - torch.exp(-magnitudes[0][0] / scale)).numpy()
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(255 * (1 - edges)).astype("uint8")).save(path)  # dark = strong edge
    return scale


def score(output, results):
    from eval_edges import evaluate_edges

    protocol = json.loads((output / "protocol.json").read_text())
    checkpoints = {name: Path(path) for name, path in protocol["checkpoints"].items()}
    per_model = {name: evaluate_edges(path, SCENE, EVAL_SCALE, device()) for name, path in checkpoints.items()}
    psnr = {name: evaluate(path, SCENE, EVAL_SCALE, 512, device(), return_per_view=True)
            for name, path in checkpoints.items()}
    means = {name: value[0] for name, value in psnr.items()}
    per_view = [dict(view=v, regnerf_db=psnr["regnerf"][1][v], edge_db=psnr["edge"][1][v],
                     edge_minus_regnerf_db=psnr["edge"][1][v] - psnr["regnerf"][1][v])
                for v in dtu.TEST_IDS]
    pilot_path = output / "pilot.json"
    summary = dict(
        record={k: v for k, v in protocol.items() if k != "baseline_protocol"}
        | dict(stage="score", score_utc=datetime.now(timezone.utc).isoformat(),
               score_revision=git("rev-parse", "HEAD"), score_local_status=git("status", "--short"),
               checkpoints={k: relative(v) for k, v in checkpoints.items()}),
        pilot=json.loads(pilot_path.read_text()) if pilot_path.is_file() else None,
        per_view=per_view, means_db=means, edge_minus_regnerf_db=means["edge"] - means["regnerf"],
        edges=per_model,
        edge_map=dict(view=dtu.TRAIN_IDS[0], scale=SCALE,
                      edge_scale=save_edge_map(results.parent / "scan40_edge" / "edge_map_view25.png")))
    if summary["pilot"]:
        summary["pilot"].pop("baseline_protocol", None)
        summary["pilot"].pop("local_diff", None)
    results.parent.mkdir(parents=True, exist_ok=True)
    results.write_text(json.dumps(summary, indent=2))
    print(f"RegNeRF {means['regnerf']:.3f} dB | + edge {means['edge']:.3f} dB | "
          f"difference {summary['edge_minus_regnerf_db']:+.3f} dB")
    print(f"Saved {results}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, help="New folder for this comparison.")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--edge-loss-weight", type=float, help="Override the pilot's chosen lambda.")
    parser.add_argument("--results", type=Path, default=Path("notebooks/results/scan40_edge.json"))
    args = parser.parse_args()
    print(f"Baseline: RegNeRF, {BASELINE_DIR}, seed {SEED}, views 25, 22, 28, {NUM_STEPS} steps.")
    print(f"Edge run: the same settings plus lambda x edge-aware L1 depth term on the fine level.")
    print(f"Pilot: {PILOT_STEPS} steps, lambda in {PILOT_LAMBDAS}, scored on validation views "
          f"{VALIDATION_VIEWS} only.")
    print(f"Score: foreground PSNR on the 25 test views at scale {EVAL_SCALE}, plus mask-edge metrics.")
    if not (args.pilot or args.run or args.score):
        print("Pass --pilot, --run, or --score with --output-dir.")
        return
    if args.output_dir is None:
        parser.error("Pass --output-dir.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.pilot:
        pilot(args.output_dir)
    if args.run:
        run(args.output_dir, chosen_lambda(args.output_dir, args.edge_loss_weight))
    if args.score:
        score(args.output_dir, args.results)


if __name__ == "__main__":
    main()
