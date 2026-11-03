import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="RegNeRF on DTU scan 40")


@app.cell
def _():
    import json

    import marimo as mo
    import numpy as np
    from PIL import Image

    import dtu
    from notebooks.lab import PAPER, ROOT

    results = ROOT / "notebooks" / "results"
    images = results / "scan40"
    scan = ROOT / "data" / "dtu" / "scan40" / "image"

    def photo(index, width=400):
        """A small copy of one DTU photo, made in memory. The photos stay out of Git."""
        with Image.open(scan / f"{index:06}.png") as source:
            source = source.convert("RGB")
            return source.resize((width, round(source.height * width / source.width)), Image.Resampling.LANCZOS)

    def no_photos():
        return mo.callout(
            mo.md("**The DTU photos are not downloaded yet.** They are not in Git. Run this once "
                  "(about 120 MB, under a minute):\n\n```sh\nuv run python prepare_dtu.py --download\n```"),
            kind="warn",
        )

    def result(text, kind="info", title="What this shows"):
        return mo.callout(mo.md(text), kind=kind, title=title)

    def stat(value, label, caption=None):
        return mo.stat(value, label=label, caption=caption, bordered=True)

    def missing(what, command):
        return mo.callout(
            mo.md(f"**{what} not saved yet.** Run this before the lab:\n\n```sh\n{command}\n```"),
            kind="warn",
        )

    return PAPER, dtu, images, json, missing, mo, no_photos, np, photo, result, results, scan, stat


@app.cell
def _(PAPER, mo):
    mo.md(
        f"""
        # Depth smoothing on real photos
        Notebook 1 showed that depth smoothing improves the shape of two spheres. Does it also help
        on photographs of a real object?

        **The test.** [DTU](https://roboimagedata.compute.dtu.dk/) scan 40, the stone arch in
        [RegNeRF Figure 3]({PAPER}). Two models train on the same 3 photos with the same settings.
        One adds RegNeRF's depth smoothing. Both are scored on 25 photos they never saw.

        **How to use this page.** Everything uses saved results, so nothing trains. Read each
        section, then the box beside it. Allow about 25 minutes.

        **Prediction.** If notebook 1 transfers, smoothing should improve the colour seen from new
        cameras. This can fail: colour PSNR does not measure the shape directly.
        """
    )
    return


@app.cell
def _(PAPER, mo, result):
    _rows = [
        {"Paper condition": "Full RegNeRF", "PSNR (dB)": 18.89},
        {"Paper condition": "Without appearance regularization", "PSNR (dB)": 18.34},
        {"Paper condition": "Without geometry regularization", "PSNR (dB)": 14.34},
        {"Paper condition": "Without sample-space annealing", "PSNR (dB)": 10.17},
    ]
    mo.vstack(
        [
            mo.md(
                f"""
                ---
                ## 1. What the paper reports
                RegNeRF's ablation ([Table 3]({PAPER})) removes one part at a time. These are the
                paper's values, averaged over many DTU scans, not measurements from this repo.
                """
            ),
            mo.hstack(
                [
                    mo.ui.table(_rows, selection=None, show_column_summaries=False),
                    result(
                        "Removing the geometry (depth) regularization costs **4.55 dB**. Removing the "
                        "appearance loss costs only 0.55 dB.\n\n"
                        "This repo has no appearance loss, so **both of our models lack it**. Our pair "
                        "therefore matches no row pair exactly. It asks a narrower question: on one scan, "
                        "with this code, does adding depth smoothing help?",
                        title="How to read it",
                    ),
                ],
                widths=[1, 1],
                align="center",
                gap=1.5,
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _(dtu, mo, no_photos, photo, scan):
    _views = dtu.TRAIN_IDS[:3]
    _body = (
        mo.hstack(
            [mo.image(photo(i), width=300, caption=f"View {i}") for i in _views],
            justify="start",
            gap=1,
            wrap=True,
        )
        if all((scan / f"{i:06}.png").is_file() for i in _views)
        else no_photos()
    )
    mo.vstack(
        [
            mo.md(
                f"""
                ---
                ## 2. The three training photos
                Both models see only these views ({", ".join(map(str, _views))}). **Watch** which parts
                of the arch each photo shows, and which parts no photo shows. The 25 test photos and
                their masks are never used for training or for choosing settings.
                """
            ),
            _body,
        ]
    )
    return


@app.cell
def _(images, json, missing, mo, no_photos, photo, result, scan):
    _record_file = images / "render_record.json"
    if _record_file.is_file():
        _record = json.loads(_record_file.read_text())
        _view = _record["view"]

        def _image(name, caption):
            return mo.image(str(images / f"view{_view}_{name}.png"), width=330, caption=caption)

        _reference = (mo.image(photo(_view), width=330, caption="Real photo")
                      if (scan / f"{_view:06}.png").is_file() else no_photos())
        _body = mo.vstack(
            [
                mo.hstack(
                    [_reference, _image("rgb_off", "Smoothing off"), _image("rgb_on", "Smoothing on")],
                    widths="equal",
                    gap=0.5,
                ),
                mo.hstack(
                    [mo.md(""), _image("depth_off", "Depth, smoothing off"),
                     _image("depth_on", "Depth, smoothing on")],
                    widths="equal",
                    gap=0.5,
                ),
                mo.md(
                    f"<small>Depth: light is near, dark is far. Both depth images use the same range, "
                    f"{_record['depth_range'][0]:.2f} to {_record['depth_range'][1]:.2f} scene units. "
                    f"Step {_record['step']}, scale {_record['scale']}.</small>"
                ),
                result(
                    "**Describe only what you can see.** Where does each model miss structure or put "
                    "the wrong colour? Where does the depth change suddenly without an edge in the photo? "
                    "One view is an illustration. The score in section 4 uses all 25 test views.",
                    kind="neutral",
                    title="Look and compare",
                ),
            ]
        )
        _view_text = f"held-out view {_view}"
    else:
        _body = missing("The view renders are", "uv run python -m notebooks.render_scan40_views")
        _view_text = "one held-out view"
    mo.vstack(
        [
            mo.md(
                f"""
                ---
                ## 3. One new camera
                Both models render {_view_text}, a camera they never saw. **Watch** the arch's opening and
                its edges in the colour images, then the depth images below them.
                """
            ),
            _body,
        ]
    )
    return


@app.cell
def _(dtu, json, missing, mo, np, result, results, stat):
    _file = results / "scan40.json"
    _command = "uv run python -m notebooks.evaluate_pair"
    if _file.is_file():
        _data = json.loads(_file.read_text())
        _rows = _data["per_view"]
        if tuple(r["view"] for r in _rows) != dtu.TEST_IDS:
            raise ValueError("The record must contain the 25 test views in protocol order.")
        _off = np.array([r["off_db"] for r in _rows])
        _on = np.array([r["on_db"] for r in _rows])
        _delta = _on - _off
        _better = int((_delta > 0).sum())
        _mean = float(_delta.mean())
        _effect = (f"**raised** the mean foreground PSNR by **{_mean:.2f} dB**" if _mean > 0 else
                   f"**lowered** the mean foreground PSNR by **{-_mean:.2f} dB**" if _mean < 0 else
                   "**did not change** the mean foreground PSNR")
        _record = _data["record"]
        _body = mo.vstack(
            [
                mo.hstack(
                    [
                        stat(f"{_off.mean():.2f} dB", "Smoothing off", "mean of 25 test views"),
                        stat(f"{_on.mean():.2f} dB", "Smoothing on", "mean of 25 test views"),
                        stat(f"{_mean:+.2f} dB", "On minus off", f"better in {_better} of 25 views"),
                    ],
                    widths="equal",
                    gap=1,
                ),
                result(
                    f"On this scan, depth smoothing {_effect}. It was better in **{_better} of 25** test views; the "
                    f"change per view ranges from {_delta.min():+.2f} to {_delta.max():+.2f} dB.\n\n"
                    "**Limits.** One scan and one seed. PSNR measures colour, not shape. Notebook 1 "
                    "showed that colour PSNR can vary a lot between seeds.",
                    kind="success" if _mean > 0 else "warn",
                ),
                mo.accordion(
                    {
                        "Every test view": mo.ui.table(
                            [{"View": r["view"], "Off (dB)": round(r["off_db"], 2), "On (dB)": round(r["on_db"], 2),
                              "On minus off (dB)": round(r["on_minus_off_db"], 2)} for r in _rows],
                            selection=None,
                            show_column_summaries=False,
                        ),
                        "Evaluation record": mo.md("```json\n" + json.dumps(_record, indent=2) + "\n```"),
                    }
                ),
            ]
        )
    else:
        _body = missing("The evaluation is", _command)
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 4. The score over 25 new cameras
                For each test photo, the score uses only the object's pixels (its mask), computes the
                colour error, and turns it into PSNR. The result is the mean of the 25 PSNR values.
                Higher is better. This is not the PSNR printed during training.
                """
            ),
            _body,
        ]
    )
    return


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md("---\n## Summary"),
            mo.md(
                """
                **Questions for discussion**
                - Does the evidence support better colour prediction on this scan? How large is the
                  change compared with the variation between seeds in notebook 1?
                - The paper reports +4.55 dB for geometry regularization. Why might one scan, with
                  this code and without the appearance loss, show a different size of effect?
                - What evidence would you need to claim that the *shape* improved?

                **Next.** Smoothing helps inside a surface, but a real depth edge should stay sharp.
                Notebook 3 tests a penalty that relaxes at image edges.
                """
            ),
            mo.accordion(
                {
                    "How the two models were trained": mo.md(
                        """
                        - **Same for both:** training views 25, 22, and 28; seed 42; 20,000 steps; Adam with
                          the same learning-rate schedule; training scale 4; the same ray sampling, coarse
                          jitter, and sample-space annealing. Depth patches use a separate random generator.
                        - **The only difference:** the depth weight. With smoothing on, the fine depth
                          weight falls from 400 to 0.1 by step 512, and the coarse depth weight is 0.1.
                          With smoothing off, both are 0 and no depth patches are rendered.
                        - **Scoring:** both models at the same step, at evaluation scale 2, on the 25 test
                          views from `dtu.py`.
                        - **Neither model** has RegNeRF's appearance loss.
                        - **Record:** `compare_depth.py` writes the code revision, settings, and checkpoint
                          paths to `protocol.json` before training. `evaluate_pair.py` checks the pair
                          against it and stores it in the evaluation record above.
                        """
                    ),
                    "Prepare the results (before the lab)": mo.md(
                        """
                        Run from the project root. Training takes about 1 to 1.5 hours on a laptop GPU.
                        Students need only the first command; the results are saved in Git.

                        ```sh
                        uv run python prepare_dtu.py --download
                        uv run python compare_depth.py --output-dir checkpoint/scan40_pair_new --run
                        uv run python -m notebooks.evaluate_pair
                        uv run python -m notebooks.render_scan40_views
                        ```
                        """
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
