# Sparse-view depth experiments

Both experiments train on three RGB views. Each pair changes one setting.

| Experiment | First run | Second run |
| --- | --- | --- |
| Depth smoothing on or off | RGB loss and depth smoothing | RGB loss only |
| Canny on or off | RGB loss and depth smoothing | Same loss, with depth smoothing skipped at Canny edges |

## 1. Does depth smoothing help?

Run the DTU scan 40 comparison from the project root.

Follow the [DTU data and training guide](docs/dtu_test.md) to download scan 40, check its files, train a new pair, and save the evaluation record. It also contains a task for another agent.

With uv, run `uv sync` once, then put `uv run` in front of each `python` command.

Read the settings before training:

```sh
python compare_depth.py
```

The script trains the depth-on model first. It reports mean foreground PSNR across 25 test views for each model. A higher score means a closer match to the reference RGB images. Canny is off in both runs.

**Result** (scan 40, seed 42, 20,000 steps, coarse jitter, scale 2): depth off 12.36 dB, depth on 18.42 dB, on minus off **+6.06 dB**. Depth smoothing scored higher in 24 of 25 test views. This is one scan, not the paper's benchmark aggregate. Notebook 2 shows the per-view scores and renders; `notebooks/results/scan40.json` holds the full record.

## 2. Does masking edges help?

For example, run the comparison on the local pendant scene.

```sh
python compare_canny.py --scene checkpoint/ginkgo_20261102_162816 --output-dir checkpoint/pendant_pair_new --steps 5000 --run
```

You can supply another synthetic scene with training images and a `transforms_train.json` file. Set `--scale` to the image downsampling factor. The pendant uses 1.

Both models use the same depth weights and sample patches from unseen cameras. The Canny run renders RGB for each sampled patch. It skips a horizontal or vertical depth difference if either pixel touches a detected edge. RGB supervision stays active. The script saves the models in the `regnerf` and `canny` subfolders.

Omit `--run` from either command to read the settings. Use a new output folder to keep earlier checkpoints.

## Toy notebook

Open the interactive lesson from a fresh checkout:

```sh
python -m pip install -r requirements-notebooks.txt
python -m notebooks.open_toy
```

The small saved-results archive and figures are included. Opening the lesson does not need checkpoints, DTU data, or training. Install PyTorch only for optional new training. See [the notebook guide](notebooks/README.md) for details.
