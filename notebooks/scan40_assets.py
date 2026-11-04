"""Read the bundled scan 40 camera record without PyTorch or downloaded data."""
import json
import numpy as np
from notebooks.lab import ROOT

SCENE = ROOT / "notebooks/results/scan40_reference"
RECORD = json.loads((SCENE / "cameras.json").read_text(encoding="utf-8"))
TRAIN_IDS = tuple(RECORD["train_ids"])
TEST_IDS = tuple(RECORD["test_ids"])


def load_rig_poses(scene=SCENE):
    record = json.loads((scene / "cameras.json").read_text(encoding="utf-8"))
    return np.asarray(record["rig_poses"], dtype=np.float32)
