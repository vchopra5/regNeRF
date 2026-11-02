# Working instructions

Use ASD-STE100 for written instructions. Run commands from the project root in a Python environment with PyTorch, NumPy, and Pillow. Use the local GPU when available.

## Investigation

The project studies depth smoothness with sparse camera views. The toy sphere has known depth. The current paper comparison uses DTU scan 40, the stone arch in RegNeRF Figure 3. Earlier scan 114 results refer to Buddha. Keep those results separate. The proposed extension changes the smoothness penalty at image edges. Create notebooks only when the user requests them.

## Controlled experiments

- Run `python compare_depth.py` to read the protocol. Use `--run` only when a training experiment is requested. A rerun replaces checkpoints with matching names in the selected output directory. Use a new `--output-dir` to keep an earlier pair.
- Train both conditions on views 25, 22, and 28. Use the same seed, optimizer, number of steps, ray sampling, bounds schedule, and resolution. Change only the depth weight. Use a separate random generator for depth sampling. Skip depth patches when the depth weight is zero.
- Use the test IDs from `dtu.py`. Keep test photos and masks out of training and parameter selection. Compare checkpoints at the same step.
- Run `python eval_dtu.py --checkpoint PATH` for each saved model. Use the same evaluation scale in both runs. The default is 2. Record the checkpoint paths, code revision and local changes, seed, depth weights, step count, scale, per-view scores, both means, and the signed difference in dB.
- Report the arithmetic mean of 25 per-view foreground RGB PSNR values. Distinguish this result from training batch PSNR. Report a tie or a decrease as measured.
- Describe the result as a single-scan investigation of RegNeRF Table 3. The paper reports a benchmark aggregate and uses a different training setup. Use reference depth to assess geometry. For the edge extension, also measure errors at true depth boundaries and on flat surfaces.
