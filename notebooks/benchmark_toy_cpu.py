"""Estimate toy training time from a short CPU run. Run only when requested."""
import argparse
import json
import os
from pathlib import Path
import platform
from tempfile import mkdtemp
from time import perf_counter
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--target", type=int, default=300)
    args = parser.parse_args()
    if args.steps < 5 or args.threads < 1 or args.target < args.steps:
        parser.error("Use at least 5 sample steps, at least 1 thread, and a target no smaller than the sample.")

    # Change the device only in this separate benchmark process.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import numpy as np
    import torch
    import train as training
    from notebooks.toy_experiment import ensure_assets, run_condition

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    ensure_assets()
    root = Path(__file__).resolve().parents[1]
    parent = root / "checkpoint" / "toy_cpu_timing"
    parent.mkdir(parents=True, exist_ok=True)
    folder = Path(mkdtemp(prefix="run_", dir=parent))
    timestamps = []
    real_train = training.train

    def timed_train(*positional, **keywords):
        def after_step(step, model):
            timestamps.append(perf_counter())
            print(f"CPU timing: step {step}/{args.steps}", flush=True)
        return real_train(*positional, **keywords, after_step=after_step)

    with patch.object(torch.cuda, "is_available", return_value=False), patch.object(training, "train", timed_train):
        result = run_condition(folder, args.steps, .05)
    assert result["device"] == "cpu"
    # Discard the first two updates. Keep initialization and rendering in the total.
    intervals = np.diff(timestamps)[1:]
    median = float(np.median(intervals))
    extra = args.target - args.steps
    estimate = result["total_seconds"] + extra * median
    report = dict(sample_steps=args.steps, target_steps=args.target, threads=args.threads,
                  interop_threads=1, device="cpu", processor=platform.processor(),
                  python=platform.python_version(), torch=torch.__version__, fine_weight=.05,
                  coarse_weight=.1, views=3, scale=2, seed=result["seed"],
                  code_revision=result["code_revision"], code_sha256=result["code_sha256"],
                  sampled_step_seconds=intervals.tolist(), median_step_seconds=median,
                  measured_training_seconds=result["training_seconds"],
                  measured_render_and_score_seconds=result["total_seconds"]-result["training_seconds"],
                  measured_total_seconds=result["total_seconds"], estimated_total_seconds=estimate,
                  estimated_minutes=estimate/60,
                  note="Extrapolated after two warm-up updates. Includes initialization, rendering, and scoring. Hardware and load can change the time.")
    (folder / "timing.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    destination = root / "notebooks" / "results" / "toy_cpu_timing.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
