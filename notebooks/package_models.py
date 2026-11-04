"""Copy trained scan 40 models into notebooks/models for the notebooks. Do not train.

Each file holds the model weights only (no optimizer state), about 2.5 MB. The manifest records
the source checkpoint, step, SHA-256, and the training record, so readers can trace each model.

  python -m notebooks.package_models regnerf=checkpoint/scan40_pair_new/on/checkpoint_20000.pt
  python -m notebooks.package_models edge=checkpoint/scan40_edge/edge/checkpoint_20000.pt
"""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from provenance import git, relative

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "notebooks" / "models"
MANIFEST = MODELS / "manifest.json"


def training_record(checkpoint):
    """Find the protocol.json written before training, next to or above the run folder."""
    for folder in (checkpoint.parent, checkpoint.parent.parent):
        path = folder / "protocol.json"
        if path.is_file():
            record = json.loads(path.read_text())
            record.pop("local_diff", None)
            record.pop("baseline_protocol", None)
            return relative(path), record
    return None, None


def package(name, checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = saved.get("model", saved)
    step = saved.get("step")
    MODELS.mkdir(parents=True, exist_ok=True)
    target = MODELS / f"scan40_{name}.pt"
    torch.save(weights, target)
    reloaded = torch.load(target, map_location="cpu", weights_only=True)
    if any(isinstance(v, torch.Tensor) and not torch.equal(v, reloaded[k]) for k, v in weights.items()):
        raise ValueError(f"Saved weights differ from {checkpoint}")
    record_path, record = training_record(checkpoint)
    return target, dict(file=target.name, source=relative(checkpoint), step=step,
                        source_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                        sha256=hashlib.sha256(target.read_bytes()).hexdigest(), bytes=target.stat().st_size,
                        packaged_utc=datetime.now(timezone.utc).isoformat(),
                        packaged_revision=git("rev-parse", "HEAD"),
                        training_record=record_path, training=record)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", metavar="NAME=CHECKPOINT")
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    for item in args.models:
        name, _, path = item.partition("=")
        if not name or not path:
            parser.error(f"Use NAME=CHECKPOINT, not {item!r}.")
        target, entry = package(name, Path(path))
        manifest[name] = entry
        print(f"{name}: {relative(target)} | step {entry['step']} | {entry['bytes'] / 1e6:.2f} MB")
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    print(f"Updated {relative(MANIFEST)}")


if __name__ == "__main__":
    main()
