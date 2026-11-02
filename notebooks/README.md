# Depth smoothness lab

Use the notebooks in order. Allow one hour for the team review.

| File | Time | Task |
| --- | --- | --- |
| `01_toy.py` | 15 min | Isolate the smoothness penalty with known sphere depth. |
| `02_paper.py` | 25 min | Inspect scan 40 and compare with RegNeRF Table 3. |
| `03_extension.py` | 15 min | Change the penalty at colour edges. Measure boundary and flat-surface errors. |
| Discussion | 5 min | State the result, its limit, and the next test. |

Each notebook has a question, a prediction, evidence, and discussion prompts.
The first and third notebooks solve small depth problems. They do not train a NeRF.
The paper notebook reads saved images and an evaluation record. It shows a clear
missing-result message until that record exists. No notebook starts training.

## Open the notebooks

Run commands from the project root. Use Python with PyTorch, NumPy, and Pillow.
Install `marimo==0.25.2` in that environment if needed.

For the local package installation in this project, use PowerShell:

```powershell
$env:PYTHONPATH = "$PWD\.lab_runtime;$PWD"
& C:\Users\visha\miniconda3\python.exe -m marimo edit notebooks/01_toy.py
```

Open `02_paper.py` and `03_extension.py` in the same way.
Keep the project root on `PYTHONPATH`. The notebooks share `notebooks/lab.py`.

## Prepare the paper evidence

Before the review, evaluate the existing scan 40 pair:

```powershell
& C:\Users\visha\miniconda3\python.exe -m notebooks.evaluate_pair
```

The command uses the local GPU if available. It evaluates both step-40000
checkpoints at scale 2 and saves `notebooks/results/scan40.json` and raw logs.
It does not run training. It records missing training provenance as unknown.
The existing images have no complete provenance record. Treat them as inspection
material until the model pair is verified. A numerical evaluation alone cannot
recover the original seed, settings, or training code state.

For a new controlled training pair, first record the code revision and local
changes. Read the protocol with `python compare_depth.py`. Start training only
when requested. Use a new output directory to preserve existing models.
Evaluate both saved models at scale 2 with `eval_dtu.py`.

## Sources and limits

Use [RegNeRF, Table 3](https://www.cvlibs.net/publications/Niemeyer2022CVPR.pdf)
for the published reference values. The scan 40 comparison is a single-scan
investigation with a different setup. It is not the paper benchmark aggregate.
The extension is a controlled depth-signal experiment. It does not yet measure
the effect of an edge penalty in a trained NeRF.

The controls follow [marimo's input API](https://docs.marimo.io/api/inputs/).
The source code is visible in each notebook for the team to inspect.
