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

We extended the model tp add a weighted L1 depth penalty to RegNeRF. Colour edges reduce its weight; the original L2 penalty stays active. On DTU scan 40, it increases mean foreground RGB PSNR by **0.38 dB**, with gains in 20 of 25 test views.

| RegNeRF | RegNeRF + edge L1 | Edge L1 minus RegNeRF |
| --- | --- | --- |
| 18.37 dB | 18.76 dB | **+0.38 dB** |

Scores are average of 25 test-view foreground RGB PSNR values at step 20,000 and scale 2. This pair uses its own **18.37 dB baseline**.

The mark inside the opening is reduced, but depth edges remain blurred. No reference depth is available to measure shape error. This is a single-scan investigation of RegNeRF Table 3, with one seed. [Notebook 3](notebooks/03_extension.py) contains the small depth experiments and arch comparisons.

## Interactive visualization

Use the controls to compare models, camera views, RGB images, depth maps, and errors. In the toy case, move a depth patch and inspect the change in its penalty. In the paper notebook, select a test view or optionally render a moved camera. In the extension, connect or disconnect two pixels, spread a depth step, and change the colour guide. Then compare saved RGB and depth for the stone arch. Use the two camera sliders to look around it from a new position, then select **Render view**.

## Path through the notebooks

The three notebooks form one investigation. They are interactive **marimo notebooks stored as Python files**, not Jupyter `.ipynb` files. Each notebook has a question, controls, saved evidence, and discussion prompts.

| Order | Notebook | What to change and inspect | Time |
| --- | --- | --- | --- |
| 1 | [Toy case](notebooks/01_toy.py) | Change camera coverage, move a depth patch, and compare smoothing off and on. Inspect RGB and known depth at 100, 300, and 1,000 training steps. | 15 min |
| 2 | [Paper result](notebooks/02_paper.py) | Switch among 25 DTU scan 40 test views. Compare images, depth, and per-view RGB scores. Optionally render a moved camera. | 15 min |
| 3 | [Extension](notebooks/03_extension.py) | Test neighbour smoothing, L1/L2 costs, and colour guides. Inspect all 25 test views, then turn the camera around the arch and compare both models. | 15 min |

**Total lesson time: about 45 minutes**, plus environment setup. Opening the notebooks does not start training.


## Open the notebooks

Download and extract the full repository, or clone it. Open a terminal in the `regNerF` folder.
Use Python **3.11 or later**. Create and activate an environment:

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

The camera panels and optional toy training use CUDA when available and otherwise use the CPU.

## Division of work

The table follows the repository commit history. Both members contributed to the investigation.

| Team member | Work |
| --- | --- |
| **Vishal Chopra** | Initial model, data loading, training, and rendering code; RegNeRF adaptation and depth-loss fixes; DTU training guide; |
| **Teodor Cirstoiu** | Edge-detection regularizer and comparison code; coarse jitter; toy, paper and extension notebooks; |
| **Both** | Toy lesson development, refinement of the test procedure, and notebook documentation. |
