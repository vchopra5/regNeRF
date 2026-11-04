# Neural graphics

Teodor Cirstoiu and Vishal Chopra

## Study question

Does encouraging better geometry improve novel view rendering when we only have a few images?

We test this question with depth smoothing from three training views. We then test whether a penalty that uses colour edges preserves depth boundaries.

## Papers we read

- NeRF (2020)
- mip-NeRF (2021)
- RegNeRF (2022)
- NeuS (2021)
- NeuS2 (2023)

The notebooks focus on RegNeRF. Our implementation uses a mip-NeRF model with RegNeRF depth smoothing and sample-space annealing. NeuS and NeuS2 informed our reading, but are not part of the experiments.

## Why the question is interesting

A model can match its training images and still learn the wrong shape. The errors may only appear from a new angle. Depth smoothing can reduce shape error, but it can also smooth real boundaries. We measure image quality and depth error separately because an improvement in one does not guarantee an improvement in the other.

## Toy case

Two spheres. One striped sphere with another blue sphere behind it. At 300 steps, smoothing reduces mean depth MAE by about 11–12%. It also improves mean foreground RGB PSNR: **+0.60 dB in the front view and +0.84 dB in the lower view**, averaged over five seeds. Depth and image quality both improve.

Each pair uses the same training settings, with smoothing off or on. The figures show seed 42; the means cover seeds 42, 142, 242, 342, and 442. The lower camera was selected after image inspection and serves as an additional illustration.

## Paper result to reproduce

The sphere test shows better depth and RGB scores. We next test RGB quality on DTU scan 40, the stone arch in RegNeRF Figure 3. **Depth smoothing increases mean foreground RGB PSNR by 6.06 dB.** It improves the score in 24 of 25 test views.

| Depth off | Depth on | On minus off |
| --- | --- | --- |
| 12.36 dB | 18.42 dB | **+6.06 dB** |

Both models train on views **25, 22, and 28**, with seed 42 and 20,000 steps. We keep the optimizer, RGB ray sampling, resolution, bounds schedule, and coarse jitter (`stratified=True`) fixed. Only the depth penalty changes. Off uses zero. On uses a coarse weight of 0.1 and a fine weight that falls from 400 to 0.1. Depth patches use a separate random generator and are skipped when smoothing is off.

Both models retain sample-space annealing and density noise. Thus, the comparison measures the effect of depth smoothing within this setup. The depth-off model is not plain mip-NeRF. All results in this pair use coarse jitter.

We score both models at step 20,000 and evaluation scale 2. We report the **arithmetic mean of 25 per-view foreground RGB PSNR values**, using the test IDs from `dtu.py`. Test photos and masks do not enter training or parameter selection. These are test-view scores, not training-batch scores.

This is a **single-scan investigation of RegNeRF Table 3**. The paper reports a benchmark aggregate with a different training setup. Here, scan 40 tests image quality; the synthetic scenes provide known depth for geometry tests.

The [evaluation record](../notebooks/results/scan40.json) stores checkpoint paths, code revision, local changes, seed, depth weights, jitter, step count, scale, and per-view scores. Moved-camera renders have no reference photo and do not change the official result.

## Extension

Does a penalty that uses colour edges improve the stone arch? We trained the same method twice. **The added edge L1 term increases mean foreground RGB PSNR in both runs: +0.38 dB and +0.23 dB, an average of +0.31 dB.**

| Run | RegNeRF | RegNeRF + edge L1 | Edge L1 minus RegNeRF | Better in |
| --- | --- | --- | --- | --- |
| First (RTX 5090) | 18.37 dB | 18.76 dB | **+0.38 dB** | 20 of 25 views |
| Second (RTX 4070) | 18.42 dB | 18.65 dB | **+0.23 dB** | 13 of 25 views |
| **Average** | 18.40 dB | 18.70 dB | **+0.31 dB** | |

Notebook 3 starts with three small experiments. Increase smoothing between two pixels, then disconnect them. Spread a depth step over more pixels and compare its L1 and L2 costs. Then change the colour guide to test an aligned edge, a texture edge, and a missing colour edge. These calculations have known depth. They show how smoothing can remove a real boundary and how colour weights change the fit.

Teodor's regularizer keeps RegNeRF's L2 depth penalty and adds a weighted L1 penalty on fine depth. A blur and Sobel filters detect colour gradients in rendered patches. Strong colour gradients reduce the added penalty. The colour weights receive no gradients. The original L2 penalty remains active at the boundary.

Both models train on DTU scan 40, views **25, 22, and 28**, with seed 42 and 20,000 steps. They use Adam, coarse jitter, training scale 4, and the same sampling and bounds schedule. The fine L2 weight falls from 400 to 0.1 over 512 steps. Coarse L2 stays at 0.1. The extension adds fine L1 with coefficient λ = 0.1. The first run fixed λ before training. The second run confirmed it with a pilot: three 5,000-step runs with λ = 0.01, 0.1, and 1, scored on validation views 40, 44, and 48. λ = 0.1 scored highest; λ = 1 lost 2.2 dB. Test photos and masks do not enter training or select this coefficient.

We evaluate both checkpoints at step 20,000 and scale 2. The table reports the **arithmetic mean of 25 per-view foreground RGB PSNR values**. Each run has its own RegNeRF baseline. The first run reused an earlier RegNeRF run (18.37 dB) after checks of its settings, source, and training inputs. The second run used Notebook 2's RegNeRF model (18.42 dB). Compare each edge model with its own baseline. GPU arithmetic is not exactly repeatable, so two runs with the same seed differ.

Select any of the 25 test views to inspect the arch opening, pillar edges, and top block. Each view shows both models and their RGB scores. The mean score uses all 25 test views. The depth panels use one shared scale: dark means near and light means far. The left reference panel is a colour photo.

The added term reduces the vertical mark inside the opening. The depth boundary remains blurred, and both models distort the top block. **No reference depth is available for this scan, so we cannot measure boundary or flat-surface depth error.** The small simulations' depth scores do not apply to the arch. Single views do not repeat: the two runs agree on the sign of the change in only 10 of 25 views. The second run's mask-outline metrics show a colour gain spread over the whole object and no sharper depth edges. This is one scan and two runs with one seed, not the paper's benchmark aggregate. See the [first-run record](../notebooks/results/partner_stone/results.json) and the [second-run record](../notebooks/results/scan40_edge.json) for checkpoint paths, source hashes, settings, the pilot, and all 25 scores. Earlier pendant results are separate from this notebook.

Next, compare added uniform L1 with added weighted L1 at the same coefficient. Repeat with more seeds and use reference depth to measure errors at true boundaries and on flat surfaces.

## Interactive visualization

Use the controls to compare models, camera views, RGB images, depth maps, and errors. In the toy case, move a depth patch and inspect the change in its penalty. In the paper notebook, select a test view or optionally render a moved camera. In the extension, connect or disconnect two pixels, spread a depth step, and change the colour guide. Then compare saved RGB and depth for the stone arch. Use the two camera sliders to look around it from a new position, then select **Render view**. Try turn −52°, tilt 24°: there, both RegNeRF models break into floaters, and both edge-L1 models keep a solid block. We chose this angle after inspecting renders, so it is an illustration, not a measurement.

## Path through the notebooks

The three notebooks form one investigation. They are interactive **marimo notebooks stored as Python files**, not Jupyter `.ipynb` files. Each notebook has a question, controls, saved evidence, and discussion prompts.

| Order | Notebook | What to change and inspect | Time |
| --- | --- | --- | --- |
| 1 | [Toy case](../notebooks/01_toy.py) | Change camera coverage, move a depth patch, and compare smoothing off and on. Inspect RGB and known depth at 100, 300, and 1,000 training steps. | 15 min |
| 2 | [Paper result](../notebooks/02_paper.py) | Switch among 25 scan 40 test views. Compare images, depth, and per-view RGB scores. Optionally render a moved camera. | 25 min |
| 3 | [Extension](../notebooks/03_extension.py) | Test neighbour smoothing, L1/L2 costs, and colour guides. See how the edge weight was chosen. Inspect all 25 test views, turn the camera around the arch, and compare two training runs. | 30 min |
| 4 | Team discussion | State what improved, what did not, and which test should come next. | 5 min |

**Total lesson time: about 75 minutes**, plus environment setup. Opening the notebooks does not start training. The saved lessons need no GPU, DTU download, or local training outputs.

## Environment and setup

Use Python **3.11 or later**. Run commands from the repository root. Create and activate an environment first:

```sh
python -m venv .venv
```

In Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

On macOS or Linux:

```sh
source .venv/bin/activate
```

Install the required notebook packages: PyTorch, marimo, NumPy, and Pillow.

```sh
python -m pip install -r requirements-notebooks.txt
python -m notebooks.open_notebook --notebook 1
```

Close the notebook server with `Ctrl+C`. Then open the next notebook:

```sh
python -m notebooks.open_notebook --notebook 2
python -m notebooks.open_notebook --notebook 3
```

Run one launcher command at a time. The launcher opens the interactive reading view by default. Add `--edit` to open the code editor. Use the live notebooks for controls; static HTML exports do not calculate new results.

PyTorch is required for the notebook environment and is installed by the command above. Check whether it can use CUDA:

```sh
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

These optional notebook tasks use CUDA when available and otherwise use the CPU. The toy archive, scan 40 reference images, saved comparisons, and 50 extension panels are included. Notebook 3 reads its panels and 25 test scores from `notebooks/results/partner_stone/`, and the second run from `notebooks/results/scan40_edge.json`. The small experiments recalculate depth fits. The saved-view selector loads images. The camera panel renders new RGB and depth images only when you select **Render view**. It uses the two scored models in `notebooks/models/` and does not train.

If you already use `uv`, `uv sync` installs the full project environment, including PyTorch. Use `uv run python` in place of `python` for the commands above.

## Runtime and optional reproduction

| Task | Expected time | Basis |
| --- | --- | --- |
| Read all three lessons and discuss | About 75 min | Suggested path above; excludes package installation. |
| Optional toy pair, 300 steps per model | About 1–2 min on GPU; allow about 32 min on CPU | RTX 4070 Laptop: 23 s off and 44 s on, plus rendering. CPU estimate uses a short four-thread timing test. |
| Notebook 2 moved-camera render | About 1 s for the first GPU request; 14 s on CPU | Median measurements on RTX 5090 and Ryzen 9 7900X. A new angle takes about 0.46 s and 7.1 s, respectively. |
| Rebuild all five seeds of toy evidence | About 35 min on GPU | About 7 min per seed on the RTX 4070 Laptop. |
| Train and score the scan 40 pair | About 80 min on GPU | RTX 4070 Laptop: about 46 min on, 26 min off, and 8 min for scoring. Allow about 12 min more for a separate evaluation and saved renders. |

Notebook 3 also has a full horizontal turn and a vertical turn from −90° to 90°. Both models use the same camera. Repeated angles reuse their images. Each render shows its elapsed time. There are no reference photos at these new positions, so the 25-view scores stay unchanged.

Hardware and other work can change these times. Package-download time is separate. We have no verified total runtime for rebuilding all three studies, including the stone-arch extension. Full retraining is not part of the 75-minute lesson.

For new DTU training, follow the [data and training guide](dtu_test.md). Read the protocol with `python -m experiments.compare_depth` first. Training requires `--run` and a new `--output-dir`. Keep earlier checkpoints. Use `python -m tools.eval_dtu --checkpoint PATH` for each model at the same evaluation scale.

The [notebook guide](../notebooks/README.md) gives commands to rebuild saved evidence. The saved lessons do not need its DTU download steps; use those steps only to prepare new training data. Retraining the extension requires the scan 40 data and the baseline checkpoint recorded in `notebooks/results/partner_stone/results.json`. The second run used `python -m experiments.compare_edge --output-dir checkpoint/scan40_edge --pilot --run --score`, which took 2 hours 20 minutes on the RTX 4070 Laptop GPU, including the pilot. The saved Notebook 3 lesson needs neither.

## Division of work

The table follows the repository commit history. Both members contributed to the investigation.

| Team member | Work |
| --- | --- |
| **Vishal Chopra** | Initial model, data loading, training, and rendering code; RegNeRF adaptation and depth-loss fixes; sphere generation; initial toy notebook and controlled DTU training guide; earlier Canny-mask experiment. |
| **Teodor Cirstoiu** | Current edge-detection regularizer and comparison code; coarse jitter; five-seed toy rerun; toy and paper notebook revisions; scan 40 evaluation results, 25-view selector, and shared model weights; `uv` environment setup. |
| **Both** | Toy lesson development, refinement of the test procedure, and notebook documentation. |
