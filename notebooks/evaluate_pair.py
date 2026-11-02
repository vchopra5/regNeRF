"""Evaluate the existing scan 40 pair without training."""

import contextlib
import hashlib
import io
import json
import subprocess
from datetime import datetime, timezone

import torch

from eval_dtu import evaluate
from notebooks.lab import ROOT
import dtu


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def main():
    output = ROOT / "notebooks/results/scan40.json"
    paths = {name: ROOT / f"checkpoint/scan40_pair/{name}/checkpoint_40000.pt"
             for name in ("off", "on")}
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved.get("step") != 40000:
            raise ValueError(f"Expected step 40000: {path}")
        del saved
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores, hashes, logs = {}, {}, {}
    for name, path in paths.items():
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        stream = io.StringIO()
        print(f"Evaluate {name} at scale 2 on {device}.", flush=True)
        with contextlib.redirect_stdout(stream):
            _, scores[name] = evaluate(path, ROOT / "data/dtu/scan40", 2, 512, device,
                                       return_per_view=True)
        logs[name] = stream.getvalue()
        if set(scores[name]) != set(dtu.TEST_IDS):
            raise ValueError("Expected one score for each of the 25 test views.")
        print(logs[name], flush=True)
    record = {
        "scene": "scan40", "step": 40000, "evaluation_scale": 2,
        "device": device, "evaluation_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": {name: str(path.relative_to(ROOT)) for name, path in paths.items()},
        "checkpoint_sha256": hashes, "evaluation_revision": git("rev-parse", "HEAD").strip(),
        "evaluation_status": git("status", "--short"), "evaluation_local_diff": git("diff", "--", "*.py"),
        "training_provenance": "Unknown. Existing checkpoints do not store training settings or code revision.",
        "required_training_views": [25, 22, 28], "required_seed": 42,
        "required_final_depth_weights": {"off": 0.0, "on": 0.1},
        "score_precision": "Full precision per-view scores from eval_dtu.py; display values are rounded.",
    }
    rows = [{"view": i, "off_db": scores["off"][i], "on_db": scores["on"][i],
             "on_minus_off_db": scores["on"][i]-scores["off"][i]} for i in dtu.TEST_IDS]
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, log in logs.items():
        (output.parent / f"scan40_{name}.log").write_text(log, encoding="utf-8")
    output.write_text(json.dumps({"record": record, "per_view": rows}, indent=2), encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
