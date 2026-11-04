"""Check the lessons in a copy with no training outputs or downloaded data."""
import argparse
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


def worker(live_camera=False):
    class NoTraining(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] == "train" or (not live_camera and fullname.split(".")[0] == "torch"):
                raise ImportError(f"Disabled for the fresh-copy check: {fullname}")

    sys.meta_path.insert(0, NoTraining())
    root = Path.cwd()
    assert not (root / "checkpoint").exists() and not (root / "data").exists()
    assert not (root / "notebooks/assets/two_spheres").exists()
    started = time.perf_counter()
    if live_camera:
        import torch
        torch.set_num_threads(4)
    counts = {}
    for name in ("01_toy", "02_paper", "03_extension"):
        spec = importlib.util.spec_from_file_location(name, root / "notebooks" / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.app.run()
        count = 1
        if name == "01_toy":
            for camera in ("Front view", "Lower view"):
                for step in (100, 300, 1000):
                    module.app.run(defs={"camera_view": SimpleNamespace(value=camera),
                                         "stage": SimpleNamespace(value=step)})
                    count += 1
        elif name == "02_paper":
            from notebooks.scan40_assets import TEST_IDS
            for view in TEST_IDS:
                module.app.run(defs={"view_choice": SimpleNamespace(value=view)})
                count += 1
            # The optional live render must show a useful message without PyTorch.
            outputs, _ = module.app.run(defs={"camera_settings": SimpleNamespace(value={"angle": 10})})
            if live_camera:
                assert "Rendered on" in "\n".join(getattr(o, "text", "") for o in outputs)
            count += 1
        else:
            outputs, _ = module.app.run(defs={"orbit_settings": SimpleNamespace(
                value={"horizontal": 20, "vertical": 10})})
            if live_camera:
                assert "Rendered in" in "\n".join(getattr(o, "text", "") for o in outputs)
            count += 1
            for guide in ("Aligned edge", "Texture edge", "Missing colour edge"):
                for penalty in ("L1", "L2"):
                    for strength in (0, 2, 8):
                        module.app.run(defs={"case": SimpleNamespace(value=guide),
                                             "penalty": SimpleNamespace(value=penalty),
                                             "strength": SimpleNamespace(value=strength)})
                        count += 1
            for camera in (f"view{row['view']}" for row in json.loads(
                    (root / "notebooks/results/partner_stone/results.json").read_text())["per_view"]):
                module.app.run(defs={"view_choice": SimpleNamespace(value=camera)})
                count += 1
            for connected in (False, True):
                for strength in (0, 2, 8):
                    module.app.run(defs={"connected": SimpleNamespace(value=connected),
                                         "pair_strength": SimpleNamespace(value=strength)})
                    count += 1
            for span in (1, 3, 5, 7):
                module.app.run(defs={"ramp_span": SimpleNamespace(value=span)})
                count += 1
        counts[name] = count
        print(f"{name}: {count} saved-data checks passed", flush=True)
    assert ("torch" in sys.modules) == live_camera and "train" not in sys.modules
    assert not (root / "checkpoint").exists() and not (root / "data").exists()
    report = dict(checks=counts, seconds=time.perf_counter()-started,
                  torch_imported=live_camera, training_imported=False,
                  downloaded_data_present=False, training_output_present=False)
    (root / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--live-camera", action="store_true",
                        help="Also render both notebook camera panels from the bundled models.")
    args = parser.parse_args()
    if args.worker:
        worker(args.live_camera)
        return
    root = Path(__file__).resolve().parents[1]
    exports = root / "notebooks/exports"
    exports.mkdir(exist_ok=True)
    clean = Path(tempfile.mkdtemp(prefix="fresh_notebooks_", dir=exports))
    # Copy only files distributed with the repository, including pending new files.
    # A downloaded ZIP has no .git folder; use the same exclusions in that case.
    if (root / ".git").exists():
        names = subprocess.check_output(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
        ).decode("utf-8").split("\0")
        files = [root / name for name in sorted(set(names)) if name and (root / name).is_file()]
    else:
        excluded = {".git", ".venv", ".lab_runtime", "__pycache__", "__marimo__", "exports",
                    "checkpoint", "data", "renders", "two_spheres"}
        files = []
        for folder, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d not in excluded and not d.startswith("nerf_synthetic")]
            files.extend(Path(folder) / name for name in names)
    for source in files:
        destination = clean / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(clean), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    print(f"Fresh copy: {clean}", flush=True)
    subprocess.run([sys.executable, "-u", "-m", "notebooks.verify_saved_notebooks", "--worker",
                    *(["--live-camera"] if args.live_camera else [])],
                   cwd=clean, env=env, check=True)


if __name__ == "__main__":
    main()
