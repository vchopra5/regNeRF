# Data and evaluation tools

Run commands from the repository root. None of these tools is needed to open the bundled notebooks.

```sh
python -m tools.prepare_dtu --help
python -m tools.eval_dtu --help
python -m tools.eval_edges --help
```

- `prepare_dtu` downloads or validates DTU scan 40 for new experiments.
- `download_dtu_reference` reads selected files from the official DTU archive.
- `eval_dtu` scores foreground RGB on the test views.
- `eval_edges` measures image and mask-edge proxies. These are not reference-depth errors.
- `render_partner_comparison` renders the two saved Notebook 3 models at all 25 test cameras.
- `render_canny_comparison` renders the earlier Canny comparison.
- `summarize_soft_edges` summarizes the synthetic pendant comparison.

The rendering and summary tools require the local experiment outputs named in their source. A repository download includes the notebook panels and small camera models, but not full training checkpoints or datasets.
