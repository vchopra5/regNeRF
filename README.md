# Sparse-view depth experiments

Both experiments train on three RGB views of DTU scan 40. Each pair changes one setting.

| Experiment | First run | Second run |
| --- | --- | --- |
| Depth smoothing on or off | RGB loss and depth smoothing | RGB loss only |
| Edge regularizer on or off | RGB loss and depth smoothing | Same loss, plus an edge-aware L1 depth term |

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

## 2. Does an edge regularizer sharpen depth?

Notebook 3 compares RegNeRF with RegNeRF plus an edge-aware L1 depth term on DTU scan 40. The term is cheap at colour edges. It finds them with a blur and a Sobel gradient on the rendered patch, with no Canny thresholds and no learned parameters. The baseline is the saved RegNeRF model from section 1. Only the new term changes.

Read the settings first:

```sh
python compare_edge.py
```

Then run the three stages. Each one writes to the same new output folder:

```sh
python compare_edge.py --output-dir checkpoint/scan40_edge --pilot   # 3 x 5000 steps, about 55 min
python compare_edge.py --output-dir checkpoint/scan40_edge --run     # 20000 steps, about 70 min
python compare_edge.py --output-dir checkpoint/scan40_edge --score   # about 40 min
python -m notebooks.render_scan40_views --output notebooks/results/scan40_edge \
  --conditions regnerf=checkpoint/scan40_pair_new/on/checkpoint_20000.pt \
  edge=checkpoint/scan40_edge/edge/checkpoint_20000.pt
```

The pilot chooses the weight λ from validation views 40, 44, and 48 only. These are training-split views that the 3-view models do not use. The score stage writes `notebooks/results/scan40_edge.json` with the 25 test-view PSNR values and the mask-edge metrics from `eval_edges.py`. Times are for an RTX 4070 Laptop GPU.

To use the term on another scene, call the trainer directly. Train the same command with `--edge-loss-weight 0` as the baseline, so the pair changes one setting:

```sh
python train.py --data-dir PATH/TO/SCENE --num-views 3 --checkpoint-dir checkpoint/NEW_FOLDER --edge-loss-weight 0.1
```

The scene needs training images and a `transforms_train.json` file, or a DTU `cameras.npz`. The edge term needs the depth loss (`--depth-weight` above 0, the default 0.1) and cannot be combined with `--virtual-canny`. The edge scale comes from the scene's own training photos. Choose the weight on views that are not test views.

`compare_canny.py` and `train.py --virtual-canny` remain for the older Canny mask. That comparison has no saved result.

## Toy notebook

Open the interactive lesson from a fresh checkout:

```sh
python -m pip install -r requirements-notebooks.txt
python -m notebooks.open_toy
```

The small saved-results archive and figures are included. Opening the lesson does not need checkpoints, DTU data, or training. Install PyTorch only for optional new training. See [the notebook guide](notebooks/README.md) for details.
