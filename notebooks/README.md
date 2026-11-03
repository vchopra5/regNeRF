# Depth smoothness lab

Use the notebooks in order. Allow one hour for the team review.

| File | Time | Task |
| --- | --- | --- |
| `01_toy.py` | 15 min, plus optional training | Inspect the sphere, move a depth patch, then compare models trained with and without the patch penalty. |
| `02_paper.py` | 25 min | Inspect scan 40 and compare with RegNeRF Table 3. |
| `03_extension.py` | 15 min | Change the penalty at colour edges. Measure boundary and flat-surface errors. |
| Discussion | 5 min | State the result, its limit, and the next test. |

Each notebook has a question, a prediction, evidence, and discussion prompts.
The first notebook uses saved two-sphere results. Every saved run was trained with 5 seeds (42, 142, 242, 342, 442) and coarse jitter. The images show seed 42; the scores cover all 5 seeds. The camera and shape sections use the same three-photo baseline at 500 steps. The final experiment compares smoothing off and on at 100, 300, and 1,000 steps. Select the lower or front camera in the final section. Each view has its own reference and scores. The lower view shows the curved stripe more clearly. It was chosen after inspecting the images, so it is an additional illustration rather than an independent test.

The lab uses saved results only, so nothing trains. After the lab, open the optional panel at the end of the notebook and select **Run comparison** to train your own comparison. The default is a fine depth weight of 0.05 at 300 steps. The slider covers 0 to 0.10. Positive fine weights also enable the coarse weight of 0.10; zero disables both depth terms. Submission reuses matching results or trains the requested condition and any missing baseline. New runs save checkpoints, RGB, depth, scores, and code records under `checkpoint/toy_two_spheres_student_v3`. Every custom run passes seed 42 explicitly to the trainer. Model initialization, view sampling, and RGB ray sampling reset to this seed. A separate generator uses seed 43 for depth patches. Cache reuse requires the same seed, weight, step count, and coarse jitter setting. One training run saves renders for both cameras, so changing camera reuses the same model. The third notebook solves a small depth problem.

**Open these figures if you cannot run Notebook 1.**

1. Compare [3, 9, and 20 input photos at 500 steps](toy_figures/coverage_500.png).
2. [Move the camera](toy_figures/camera_shift.png), then inspect [all three surface strips](toy_figures/shape_presets.png).
3. Compare [simulated one-dimensional fits](toy_figures/curve_settings.png) with smoothing 0, 4, and 256.
4. Compare [one moved pixel](toy_figures/one_pixel.png) with a [moved whole patch](toy_figures/whole_patch.png), both at 0.10 units. The loss values and four neighbour contributions appear in each figure.
5. Compare [saved training at all three step counts](toy_figures/training_stages.png) (seed 42). Over 5 seeds, smoothing lowers the depth MAE at 300 steps by about 0.02, or 11–12% (in 4 of 5 seeds in the front view and 5 of 5 in the lower view). At 1,000 steps it lowers the depth MAE by about 0.006, in 4 of 5 seeds in each view. At 100 steps it makes no real difference. Its effect on foreground RGB PSNR varies widely: at 300 steps the mean change is +0.60 dB (front) and +0.84 dB (lower), and single seeds range from -0.45 to +2.27 dB. Seed 42 is the least favourable seed at 300 steps.

The paper notebook reads saved images and an evaluation record. It shows a clear
missing-result message until that record exists. Opening a notebook does not start training.

## Open the notebooks

The camera-motion slider redraws the stripe and camera diagram from cached depth arrays in under one second per change.

Use Python 3.10 or later. Run these commands from the repository root in a Python environment:

```sh
python -m pip install -r requirements-notebooks.txt
python -m notebooks.open_toy
```

The launcher sets the project import path. It uses the Python environment in which you run it. Add `--view` to open the reading view. The editor keeps the code editable.

The saved lesson needs marimo, NumPy, and Pillow. Install PyTorch in the same environment only if you want to use optional training:

```sh
python -m pip install torch
```

The trainer uses CUDA when a compatible GPU and PyTorch installation are available. Otherwise, it uses the CPU. For 300 steps, allow about 16 minutes per model with four CPU threads. This is an estimate from a short timing test.

Open the other notebooks with `python -m marimo edit notebooks/02_paper.py` or `python -m marimo edit notebooks/03_extension.py`, with the repository root on `PYTHONPATH`. Their data requirements are separate from the toy lesson.

## Files included for the toy lesson

The repository includes `toy_saved_results.zip`, a 1.53 MB archive with 54 small saved files. It contains the reference, three training photos, depth arrays, RGB predictions, and scores required by this lesson. The notebook checks the files and unpacks missing files into the ignored `notebooks/assets/two_spheres` folder on first use. It does not download data or train a model when opened.

The paper figure and static fallback figures are also included. No model checkpoints, DTU dataset, local package folder, HTML export, or custom training history is needed. A fresh checkout has six saved comparison scores for each camera. Custom scores appear after you submit your own run.

To update the small archive after an intentional change to the saved evidence, run `python -m notebooks.package_toy_results`. Do not add the full local assets or checkpoint folders to Git.

## Refresh the toy preview

```sh
python -m notebooks.export_toy_figures
python -m marimo export html notebooks/01_toy.py --no-include-code -o notebooks/exports/01_toy.html -f
```

Keep the repository root on `PYTHONPATH` when you call marimo directly. In PowerShell, use `$env:PYTHONPATH = "$PWD"`. In a POSIX shell, use `export PYTHONPATH="$PWD"`.

To make a static HTML preview, run the command above. Git does not track it, so it exists only on your machine. The preview shows the default lesson and contains a static comparison gallery. Its controls do not calculate or train. Use the Python source in marimo for live controls and editable code. Saved controls usually take less than one second after the first load. On the laptop GPU used for the saved runs (RTX 4070 Laptop), 300-step training took about 23 seconds with smoothing off and 44 seconds with it on, plus about 2 seconds for rendering and scoring. Older records from a different setup show 8–11 seconds. The CPU estimate is about 16 minutes per model at 300 steps with four threads and smoothing on. A 12-step CPU test measured 3.05 seconds per step, plus about 17 seconds per camera for rendering and scoring. Extrapolation gives 15.8 minutes. Hardware and other work can change the time. See [the timing record](results/toy_cpu_timing.json).

To measure this on another CPU, run `python -m notebooks.benchmark_toy_cpu --steps 12 --threads 4 --target 300`. This command starts a short training test in a new ignored output folder and updates the timing record. It does not change the saved lesson results.

## Prepare the paper evidence

Notebook 2 reads the model results from `notebooks/results/`, which is in Git, and the DTU photos from `data/dtu/scan40`, which is not. **Before the lab, each student runs `uv run python prepare_dtu.py --download`** (about 120 MB, under a minute, no account needed). The checkpoints are not needed.

Prepare the results once, before the lab:

```sh
uv run python prepare_dtu.py --download
uv run python compare_depth.py --output-dir checkpoint/scan40_pair_new --run
uv run python -m notebooks.evaluate_pair
uv run python -m notebooks.render_scan40_views
```

Training the pair takes about 1 to 1.5 hours on a laptop GPU: about 46 minutes for the depth-on model and less for the depth-off model. Evaluation and rendering take about 10 minutes. `compare_depth.py` records the code revision and settings in `protocol.json` before training. `evaluate_pair` checks the pair against that record and writes `notebooks/results/scan40.json`. `render_scan40_views` writes the view-45 renders to `notebooks/results/scan40/`. Missing files and photos show a warning with the command that makes them.

Open the notebook with `env PYTHONPATH=$PWD uv run marimo run notebooks/02_paper.py` (fish) or `PYTHONPATH=$PWD uv run marimo run notebooks/02_paper.py` (bash, zsh).

## Sources and limits

Use [RegNeRF, Table 3](https://www.cvlibs.net/publications/Niemeyer2022CVPR.pdf)
for the published reference values. The scan 40 comparison is a single-scan
investigation with a different setup. It is not the paper benchmark aggregate.
The extension is a controlled depth-signal experiment. It does not yet measure
the effect of an edge penalty in a trained NeRF.

The controls follow [marimo's input API](https://docs.marimo.io/api/inputs/).
The source code is visible in each notebook for the team to inspect.

To retrain every saved run with the current code, run `python -m notebooks.build_toy_results --checkpoint-root checkpoint/NEW_FOLDER --seeds 42 142 242 342 442`, then `python -m notebooks.package_toy_results`. One seed takes about 7 minutes on the GPU above. Check and update the result text in `01_toy.py` and this file afterwards. Students use the archive and do not need the checkpoints.
