"""Train and score Canny-based weights and uniform smoothing on DTU scan 40."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import torch
import canny_regularizer
import dtu
import train as training
from provenance import file_hashes, git


SOURCES = ("experiments/compare_canny_dtu.py", "train.py", "model.py", "data.py", "dtu.py",
           "tools/eval_dtu.py", "render.py", "provenance.py", "canny_regularizer.py")
MODES = ("canny_weight", "regnerf")


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def worker(output):
    record = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    start = time.monotonic()
    torch.set_num_threads(4)
    training.PRINT_EVERY = 250
    try:
        for mode in MODES:
            def progress(step, model):
                if step == 1 or step % 250 == 0:
                    elapsed = time.monotonic() - start
                    write_json(output / "status.json", dict(
                        state="training", mode=mode, step=step, steps=training.NUM_STEPS,
                        elapsed_seconds=elapsed, updated_utc=datetime.now(timezone.utc).isoformat()))
                    print(f"{mode}: {step}/{training.NUM_STEPS}; elapsed {elapsed:.0f} seconds", flush=True)
            training.train(3, Path(record["scene"]), output / mode,
                           depth_weight=0.1, depth_weight_start=400.0,
                           stop_after=training.NUM_STEPS, after_step=progress,
                           background="black", density_noise_std=1.0, regnerf_density=True,
                           scale=4, seed=42, stratified=True,
                           virtual_canny_weight=mode == "canny_weight")
        scores = {}
        for mode in MODES:
            write_json(output / "status.json", dict(state="evaluation", mode=mode))
            score_path = output / f"{mode}_scores.json"
            command = [sys.executable, "-u", str(Path(__file__).with_name("eval_dtu.py")),
                       "--checkpoint", record["checkpoints"][mode], "--data-dir", record["scene"],
                       "--scale", "2", "--device", "cuda", "--output-json", str(score_path)]
            subprocess.run(command, check=True)
            scores[mode] = json.loads(score_path.read_text(encoding="utf-8"))
        means = {mode: scores[mode]["mean_foreground_psnr"] for mode in MODES}
        difference = means["canny_weight"] - means["regnerf"]
        summary = dict(means_db=means, canny_minus_regnerf_db=difference,
                       scores=scores, steps=training.NUM_STEPS, evaluation_scale=2,
                       elapsed_seconds=time.monotonic() - start,
                       scope="Single-scan investigation of RegNeRF Table 3. The paper uses a different setup and a benchmark aggregate.",
                       geometry="No scan 40 reference depth is present. Boundary and flat-surface depth errors are not measured.")
        write_json(output / "summary.json", summary)
        write_json(output / "status.json", dict(state="complete", **summary))
        print(json.dumps(summary, indent=2), flush=True)
    except BaseException:
        write_json(output / "status.json", dict(state="failed", error=traceback.format_exc()))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if args.worker:
        worker(output)
        return
    print("Train Canny-based weights and uniform smoothing on scan 40.")
    print("Use views 25, 22, 28; seed 42; 20000 steps; training scale 4; test scale 2.")
    print("Both use fine depth weight 400 to 0.1 over 512 steps, and coarse weight 0.1.")
    if not args.run:
        print("Pass --run and a new output folder to start.")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("The planned GPU run needs CUDA.")
    scene = Path("data/dtu/scan40").resolve()
    if not (scene / "cameras.npz").is_file():
        raise FileNotFoundError(scene / "cameras.npz")
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "source"
    snapshot.mkdir()
    for source in SOURCES:
        shutil.copy2(source, snapshot / Path(source).name)
    inputs = [str(Path("data/dtu/scan40/cameras.npz"))]
    inputs += [str(Path("data/dtu/scan40/image") / f"{i:06}.png") for i in dtu.TRAIN_IDS[:3]]
    record = dict(scene=str(scene), seed=42, steps=training.NUM_STEPS,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  training_views=list(dtu.TRAIN_IDS[:3]), test_views=list(dtu.TEST_IDS),
                  training_scale=4, evaluation_scale=2, stratified=True,
                  depth_weight=0.1, depth_weight_start=400.0, coarse_depth_weight=0.1,
                  background="black", density_noise_std=1.0, regnerf_density=True,
                  optimizer="Adam", device=torch.cuda.get_device_name(0),
                  torch_version=torch.__version__, python=sys.executable,
                  code_revision=git("rev-parse", "HEAD"), local_status=git("status", "--short"),
                  local_diff=git("diff", "HEAD"), source_sha256=file_hashes(SOURCES),
                  training_input_sha256=file_hashes(inputs),
                  config={key: getattr(training, key) for key in (
                      "BATCH_SIZE", "DEPTH_RAYS", "PATCH_SIZE", "RAYS_PER_CHUNK", "LR_INIT",
                      "LR_FINAL", "LR_DELAY_STEPS", "LR_DELAY_MULT", "BOUND_STEPS",
                      "INITIAL_RANGE_FRACTION", "DEPTH_DECAY_STEPS")},
                  bounds=list(dtu.BOUNDS), depth_sampling_seed=43,
                  canny=dict(sigma=1.2, radius=4, sobel_divisor=8, threshold=0.1, floor=0.05, halo=5),
                  guide="Detached rendered RGB in evaluation mode; no test photos or masks",
                  checkpoints={mode: str(output / mode / "checkpoint_20000.pt") for mode in MODES})
    write_json(output / "protocol.json", record)
    subprocess.run([sys.executable, "-u", str(snapshot / "compare_canny_dtu.py"),
                    "--output-dir", str(output), "--worker"], check=True)


if __name__ == "__main__":
    main()
