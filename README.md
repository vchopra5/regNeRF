# Neural graphics

Teodor Cirstoiu and Vishal Chopra

## Study question

Does encouraging better geometry improve novel view rendering when we only have a few images?

We test this question with depth smoothing from three training views. We then test whether a penalty that uses colour edges preserves depth boundaries.

## Papers we read

- NeRF (2020)
- mip-NeRF (2021)
- RegNeRF (2022)

The notebooks focus on RegNeRF. Our implementation uses a mip-NeRF model with RegNeRF depth smoothing and sample-space annealing.

## Why the question is interesting

A model can match its training images and still learn the wrong shape. The errors may only appear from a new angle. Depth smoothing can reduce shape error, but it can also smooth real boundaries. We measure image quality and depth error separately because an improvement in one does not guarantee an improvement in the other.

## Toy case

Two spheres. One striped sphere with another blue sphere behind it. At 300 steps, smoothing reduces mean depth MAE by about 11–12%. It also improves mean foreground RGB PSNR: **+0.60 dB in the front view and +0.84 dB in the lower view**, averaged over five seeds. Depth and image quality both improve.

## Paper result to reproduce

The sphere test shows better depth and RGB scores. We next test RGB quality on DTU scan 40, the stone arch in RegNeRF Figure 3. **Depth smoothing increases mean foreground RGB PSNR by 6.06 dB.** It improves the score in 24 of 25 test views.

| Depth off | Depth on | On minus off |
| --- | --- | --- |
| 12.36 dB | 18.42 dB | **+6.06 dB** |

Both models train on views **25, 22, and 28**, with seed 42 and 20,000 steps. We keep the optimizer, RGB ray sampling, resolution, bounds schedule, and coarse jitter (`stratified=True`) fixed. Only the depth penalty changes. Off uses zero. On uses a coarse weight of 0.1 and a fine weight that falls from 400 to 0.1. Depth patches use a separate random generator and are skipped when smoothing is off.

Both models retain sample-space annealing and density noise. Thus, the comparison measures the effect of depth smoothing within this setup. The depth-off model is not plain mip-NeRF. All results in this pair use coarse jitter.

We score both models at step 20,000 and evaluation scale 2. We report the **arithmetic mean of 25 per-view foreground RGB PSNR values**, using the test IDs from `dtu.py`. Test photos and masks do not enter training or parameter selection. These are test-view scores, not training-batch scores.

This is a **single-scan investigation of RegNeRF Table 3**. The paper reports a benchmark aggregate with a different training setup. Here, scan 40 tests image quality; the synthetic scenes provide known depth for geometry tests.

## Extension

Our regularizer keeps the original L2 depth penalty and adds a weighted L1 penalty on fine depth. A blur and Sobel filters detect colour gradients in rendered patches. Strong colour gradients reduce the added penalty. The colour weights receive no gradients.

We test a synthetic pendant with known depth. Both models use three training views, seed 42, 8,000 steps, coarse jitter, and coarse and fine L2 weights of 0.1. The added L1 coefficient is 0.1, fixed before training. The completed baseline was reused after checks of its settings, source, and scene files.

We measure depth RMSE in a two-pixel band on both sides of true pendant and chain boundaries. We also measure the flat back panel, excluding seven pixels around non-panel pixels. The scores below are arithmetic means of three per-view scores.

| Measure | RegNeRF | RegNeRF + edge L1 | Change |
| --- | --- | --- | --- |
| Boundary depth RMSE, scene units | 0.6057 | 0.5053 | 16.6% lower |
| Flat-panel depth RMSE, scene units | 0.3350 | 0.1776 | 47.0% lower |
| Foreground RGB PSNR | 20.10 dB | 19.65 dB | **−0.45 dB** |

Depth error decreases, but image quality decreases by this RGB measure. The chain and cutouts still have errors. This is one scene and one seed. The three evaluation cameras were used in earlier studies, so this is an exploratory result. See the [extension record](notebooks/results/pendant/results.json).

Our next test would be to compare the uniform L1 with added weighted L1 at the same coefficient. This would separate the effect of colour weights from the effect of adding another penalty.

## Interactive visualization

Use the controls to compare models, camera views, RGB images, depth maps, and errors. In the toy case, move a depth patch and inspect the change in its penalty. In the paper notebook, select a test view or optionally render a moved camera. In the extension, change the colour guide and compare L1 and L2 fits before you inspect the saved pendant results.

## Path through the notebooks

The three notebooks form one investigation. They are interactive **marimo notebooks stored as Python files**, not Jupyter `.ipynb` files. Each notebook has a question, controls, saved evidence, and discussion prompts.

| Order | Notebook | What to change and inspect | Time |
| --- | --- | --- | --- |
| 1 | [Toy case](notebooks/01_toy.py) | Change camera coverage, move a depth patch, and compare smoothing off and on. Inspect RGB and known depth at 100, 300, and 1,000 training steps. | 15 min |
| 2 | [Paper result](notebooks/02_paper.py) | Switch among 25 scan 40 test views. Compare images, depth, and per-view RGB scores. Optionally render a moved camera. | 15 min |
| 3 | [Extension](notebooks/03_extension.py) | Compare L1 and L2 penalties in a small depth problem. Change the colour guide. Then inspect the trained pendant at three cameras. | 15 min |

**Total lesson time: about 45 minutes**, plus environment setup. The saved lessons need no GPU, DTU download, or local training outputs.

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
```

Open the first notebook.
```sh
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

These optional notebook tasks use CUDA when available and otherwise use the CPU. The toy archive, scan 40 reference images, saved comparisons, and pendant panels are included. The moved-camera panel uses the models in `notebooks/models/`.

If you already use `uv`, `uv sync` installs the full project environment, including PyTorch. Use `uv run python` in place of `python` for the commands above.

## Division of work

The table follows the repository commit history. Both members contributed to the investigation.

| Team member | Work |
| --- | --- |
| **Vishal Chopra** | Initial model, data loading, training, and rendering code; RegNeRF adaptation and depth-loss fixes; initial toy notebook and controlled DTU training guide; |
| **Teodor Cirstoiu** | Current edge-detection regularizer and comparison code; coarse jitter; five-seed toy rerun; toy and paper notebook revisions; scan 40 evaluation results, 25-view selector, and shared model weights; `uv` environment setup. |
| **Both** | Toy lesson development, refinement of the test procedure, and notebook documentation. |
