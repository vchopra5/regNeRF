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


def worker():
    class NoTraining(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in ("torch", "train"):
                raise ImportError(f"Disabled for the fresh-copy check: {fullname}")

    sys.meta_path.insert(0, NoTraining())
    root = Path.cwd()
    assert not (root / "checkpoint").exists() and not (root / "data").exists()
    assert not (root / "notebooks/assets/two_spheres").exists()
    started = time.perf_counter()
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
            module.app.run(defs={"camera_settings": SimpleNamespace(value={"angle": 10})})
            count += 1
        else:
            module.app.run(defs={"orbit_settings": SimpleNamespace(
                value={"horizontal": 20, "vertical": 10})})
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
    assert "torch" not in sys.modules and "train" not in sys.modules
    assert not (root / "checkpoint").exists() and not (root / "data").exists()
    report = dict(checks=counts, seconds=time.perf_counter()-started,
                  torch_imported=False, training_imported=False,
                  downloaded_data_present=False, training_output_present=False)
    (root / "verification.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker()
        return
    root = Path(__file__).resolve().parents[1]
    exports = root / "notebooks/exports"
    exports.mkdir(exist_ok=True)
    clean = Path(tempfile.mkdtemp(prefix="fresh_notebooks_", dir=exports))
    shutil.copytree(root / "notebooks", clean / "notebooks",
                    ignore=shutil.ignore_patterns("exports", "__pycache__", "__marimo__", "assets"))
    assets = clean / "notebooks/assets"
    assets.mkdir()
    shutil.copy2(root / "notebooks/assets/regnerf_figure_1.png", assets)
    for name in ("model.py", "data.py", "dtu.py", "render.py"):
        shutil.copy2(root / name, clean / name)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(clean), OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_NUM_THREADS="1")
    print(f"Fresh copy: {clean}", flush=True)
    subprocess.run([sys.executable, "-u", "-m", "notebooks.verify_saved_notebooks", "--worker"],
                   cwd=clean, env=env, check=True)


if __name__ == "__main__":
    main()
