# Training experiments

Run these commands from the repository root. The notebooks use bundled results and do not need these runs.

| Module | Purpose |
| --- | --- |
| `experiments.compare_depth` | Compare depth smoothing off and on for DTU scan 40. |
| `experiments.compare_edge` | Select an added edge L1 weight on validation views, then train and score it. |
| `experiments.compare_partner_dtu` | Reproduce the Notebook 3 edge L1 run from its recorded source revision. |
| `experiments.compare_canny` | Compare the earlier binary Canny mask. |
| `experiments.compare_canny_dtu` | Compare Canny-based depth weights on scan 40. |
| `experiments.compare_canny_weights` | Compare gradient weights and the optional L1 margin on the pendant. |
| `experiments.compare_partner_pendant` | Compare the added edge L1 term on the pendant. |
| `experiments.compare_soft_edges` | Compare gradual colour weights on the pendant. |
| `experiments.sphere_demo` | Train and inspect the original sphere scene. |

Read each command's options before use:

```sh
python -m experiments.compare_depth
python -m experiments.compare_depth --help
```

These two commands do not train. Some other commands start training when run without a flag. Use `--help` to inspect them first.

Keep the pendant and scan 40 results separate. Use new output folders for new experiments. See the [DTU guide](../docs/dtu_test.md) for the controlled scan 40 procedure. Historical source paths in saved result records refer to the files as they existed when those models trained.
