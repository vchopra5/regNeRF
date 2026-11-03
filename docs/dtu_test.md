# DTU test-data setup and training

Use this guide from the repository root. It covers download, data checks, a controlled training pair, and evaluation. You can also give the task at the end to another agent. These steps do not use marimo.

The current experiment uses **DTU scan 40**, the stone arch in RegNeRF Figure 3. Scan 114 is Buddha. Keep its files and results separate. This project makes a **single-scan investigation of RegNeRF Table 3**. It does not reproduce the paper's benchmark aggregate or full training setup.

## 1. Set up Python

Use Python 3.10 or later in a separate environment. Activate that environment before you run the commands. Confirm that `python --version` prints the intended version.

If you need a new environment, run `python -m venv .venv`. Activate it with `.venv\Scripts\Activate.ps1` in PowerShell, or `source .venv/bin/activate` in a POSIX shell. The repository ignores `.venv`.

Install the training packages:

```sh
python -m pip install torch numpy Pillow huggingface_hub
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

Use the GPU when CUDA is available. Otherwise, the trainer uses the CPU. The toy notebook's runtime estimate does not apply to DTU training.

## 2. Get scan 40

The loader needs **IDR-format DTU data**. The [IDR data convention](https://github.com/lioryariv/idr/blob/main/DATA_CONVENTION.md) defines the image folders, masks, projection matrices, and normalization matrices. A directory called `dtu_test` from another project is usable only if it has this format.

For a small download, use the [scan-40 community mirror](https://huggingface.co/datasets/turandai/gaussian-surfels-dtu/tree/main/scan40):

```sh
python prepare_dtu.py --download
```

This command downloads only 49 images, 49 masks, and `cameras.npz`. It uses the fixed source revision `c70b6188f0eb1bac67843971e15ef97d213c6c09`. It excludes the large normal-map folder and all other scans. The required download is about 124 MB. It writes the files under `data/dtu/scan40`, then checks them. It does not start training. The download uses Hugging Face's [file-filtering API](https://huggingface.co/docs/huggingface_hub/guides/download#filter-files-to-download).

The verified scan-40 camera file has this SHA-256:

```text
af1ba75cbb137e5e9c83036a8db6da463365c6f54079ec31d8b8a162779fad78
```

For the original source, follow the [IDR download instructions](https://github.com/lioryariv/idr#data). Their [download script](https://github.com/lioryariv/idr/blob/main/data/download_data.sh) links to the full 15-scan archive. Extract that archive outside the repository. Copy only its `scan40` directory to `data/dtu/scan40`. Use the fixed-camera file `cameras.npz`. Do not substitute `cameras_linear_init.npz`, raw MVS camera files, or resized images from another data release.

The final layout must be:

```text
data/dtu/scan40/
    cameras.npz
    image/
        000000.png
        ...
        000048.png
    mask/
        000.png
        ...
        048.png
```

Keep the zero-based view IDs. Images use six digits; masks use three. Each camera file must contain `world_mat_0` through `world_mat_48` and `scale_mat_0` through `scale_mat_48`. Each matrix is 4 × 4. Keep image pixels and calibration from the same release. The verified mirror images are 1600 × 1200 pixels.

If the files are already present, run only the check:

```sh
python prepare_dtu.py
```

The check reads all 49 images and calibrations. It checks the 25 masks needed for evaluation, including their dimensions and nonempty foregrounds. It saves file hashes and the view lists in `data/dtu/scan40/validation.json`. A download also saves `download_source.json`. Keep these records with the experiment.

## 3. Read the controlled protocol

```sh
python compare_depth.py
```

This command prints the settings and stops. It does not train.

The current settings are:

| Setting | Value |
| --- | --- |
| Training views | 25, 22, 28 |
| Random seed | 42 for both models |
| Steps | 20,000 for both models |
| Training scale | 4, which gives 400 × 300 images for this release |
| Optimizer | Adam; the schedule is in `train.py` |
| Depth on | Fine weight decreases from 400 to 0.1 by step 512; coarse weight is 0.1 |
| Depth off | Both depth terms are disabled; no depth patches are sampled |
| Background and density noise | Black background; density noise standard deviation 1 |
| Final evaluation scale | 2 for both models, which gives 800 × 600 images |

Keep the optimizer, step count, RGB ray sampling, bounds schedule, resolution, and seed the same. Change only the depth weight. Depth patches use a separate random generator, so they do not change RGB sampling.

`dtu.py` is the source of truth for the split. It defines nine candidate training views; `num_views=3` selects the first three. It defines these 25 test views:

```text
1, 2, 9, 10, 11, 12, 14, 15, 23, 24, 26, 27, 29,
30, 31, 32, 33, 34, 35, 41, 42, 43, 45, 46, 47
```

Test images and masks must not guide training or parameter selection. Masks select foreground pixels during evaluation. RGB training does not read them. Camera calibration is also used to sample virtual cameras.

## 4. Train in a new output directory

Choose the directory name before you start. Use a name that does not already contain a training pair. For example:

```sh
python compare_depth.py --output-dir checkpoint/scan40_pair_new --run
```

The command trains depth on, then depth off. It saves `protocol.json` before training. That record includes the code revision, local changes, code hashes, data-validation record, seed, weights, training settings, and expected checkpoint paths.

Each condition saves periodic checkpoints and `mipnerf_final.pt`. At the current 20,000-step setting, use:

```text
checkpoint/scan40_pair_new/off/checkpoint_20000.pt
checkpoint/scan40_pair_new/on/checkpoint_20000.pt
```

Reusing an output directory replaces checkpoints with matching names. A partial directory is not proof of a completed pair. Confirm that both step-20000 checkpoints exist. The current script starts training from the beginning; it does not resume a partial run.

The comparison command also prints scores at scale 2, the same scale as the final evaluation. It scores `mipnerf_final.pt`, which holds the same weights as `checkpoint_20000.pt`. Use the evaluation below for the final record, because it also saves per-view scores and hashes.

`--run` needs `--output-dir`. This stops a run from replacing the older pair in `checkpoint/scan40_pair`.

## 5. Evaluate both checkpoints at scale 2

`notebooks.evaluate_pair` reads `checkpoint/scan40_pair_new` by default: the step-20000 checkpoints in `off/` and `on/`, and `protocol.json`. Give `--pair-dir` and `--step` for another pair.

Check the stored step numbers and the training record first:

```sh
python -m notebooks.evaluate_pair --check-only
```

Run `eval_dtu.py` for each saved model:

```sh
python eval_dtu.py --checkpoint checkpoint/scan40_pair_new/off/checkpoint_20000.pt --scale 2
python eval_dtu.py --checkpoint checkpoint/scan40_pair_new/on/checkpoint_20000.pt --scale 2
```

To save full-precision per-view scores, both means, the signed difference, checkpoint hashes, and the training record in one file, run:

```sh
python -m notebooks.evaluate_pair
```

It writes `notebooks/results/scan40.json`, which notebook 2 reads, and two text logs beside it. It evaluates at scale 2. It does not train. For an automated run, the combined command can replace the two separate evaluation calls because it calls the same `eval_dtu.evaluate` function for each checkpoint.

The report uses the arithmetic mean of **25 per-view foreground RGB PSNR values**. It does not use training-batch PSNR or pooled error over all test pixels. Report `on minus off` in dB. Report a tie or decline as measured. RGB PSNR alone does not establish better geometry. A geometry claim needs separate reference depth or geometry and an appropriate error measure.

Then save the renders that notebook 2 shows: all 25 test views from both models, colour and depth, at scale 4 (about 4 minutes):

```sh
python -m notebooks.render_scan40_views
```

It writes `notebooks/results/scan40/` with a record of the checkpoints it used. It does not train. It saves no DTU photos; notebook 2 reads those from `data/dtu/scan40`.

Earlier results came from an older step-40000 pair with no training record. Do not treat a 20,000-step run as a reproduction of that run.

## 6. Task for another agent

Copy this task and replace the output directory if it already exists:

```text
Read AGENTS.md and docs/dtu_test.md. Work from the repository root.
Set up PyTorch, NumPy, Pillow, and huggingface_hub. Use CUDA when available.
Get IDR-format DTU scan 40 with python prepare_dtu.py --download, or validate
an existing copy with python prepare_dtu.py. Do not download other scans.
Run python compare_depth.py to read the protocol. Keep the current settings:
views 25, 22, 28; seed 42; 20,000 steps; training scale 4; Adam; the same RGB
sampling and bounds schedule. Change only the depth weight. Do not use test
photos or masks to select parameters. Use separate depth sampling randomness.
Train both conditions in a new checkpoint/scan40_pair_new directory with
compare_depth.py --run. Preserve the protocol.json file. Check that both
checkpoint_20000.pt files exist and contain step 20000. Evaluate both at
scale 2 with python -m notebooks.evaluate_pair, then save the notebook images
with python -m notebooks.render_scan40_views.
Return the code revision and local changes, data source and hashes, seed,
weights, settings, checkpoint paths and hashes, all 25 per-view foreground
PSNR scores for each model, both arithmetic means, and signed on-minus-off
difference. Report a tie or decline plainly. Describe this as a single-scan
investigation of RegNeRF Table 3. Keep scan 114 results separate. Do not claim
better geometry from RGB PSNR alone. Do not commit data, weights, or logs.
```

## Repository size

`data/`, `checkpoint/`, and `renders/` are ignored by Git. Keep dataset downloads, camera caches, model weights, training records, and evaluation outputs there. Commit code, this guide, and small reviewed summary records only. The toy lesson has its own small saved-results archive; DTU images are not part of it.
