"""Measure notebook 2 camera rendering on CPU and GPU. Do not train."""

import json
import platform
from statistics import median
from time import perf_counter

import torch

from notebooks import live_camera
from notebooks.lab import ROOT
from provenance import file_hashes, git


def main():
    cpu = platform.processor()
    if platform.system() == "Windows":
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                            r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    report = dict(
        cpu=cpu, cpu_threads=torch.get_num_threads(),
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        torch=torch.__version__, python=platform.python_version(),
        image_size=[100, 75], chunk_size=512, repeats=3,
        revision=git("rev-parse", "HEAD"),
        code_sha256=file_hashes(("notebooks/live_camera.py", "notebooks/benchmark_camera.py",
                                "model.py", "render.py", "data.py", "dtu.py")),
        model_manifest=json.loads((ROOT / "notebooks/models/manifest.json").read_text()),
        note="Wall time includes file checks, model loading, rendering, and figure assembly. "
             "Excludes Python imports and browser display. First sample includes device startup. "
             "First request renders four images; new angle reuses the two start images; "
             "repeat reuses all four images. No training. OS file cache is not cleared.",
        devices={},
    )
    for device in ("cpu", "cuda"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        samples = {name: [] for name in ("first_request", "new_angle", "repeat_angle")}
        for trial in range(3):
            live_camera._render.cache_clear()
            live_camera._digest.cache_clear()
            for name, angle in (("first_request", 10), ("new_angle", 20), ("repeat_angle", 20)):
                start = perf_counter()
                image, used, _ = live_camera.render_comparison(ROOT / "data/dtu/scan40", angle, device=device)
                if device == "cuda":
                    torch.cuda.synchronize()
                elapsed = perf_counter() - start
                assert used == device and image.size == (1280, 1116)
                samples[name].append(elapsed)
                print(f"{device}, trial {trial + 1}, {name}: {elapsed:.3f} s", flush=True)
        report["devices"][device] = dict(seconds=samples, median_seconds={k: median(v) for k, v in samples.items()})
    destination = ROOT / "notebooks/results/camera_timing.json"
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {destination}", flush=True)


if __name__ == "__main__":
    main()
