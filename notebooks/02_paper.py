import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="RegNeRF on DTU scan 40")


@app.cell
def _():
    import json

    import marimo as mo
    import numpy as np
    from PIL import Image

    from notebooks import scan40_assets as dtu
    from notebooks.lab import PAPER, ROOT
    from notebooks.scan40_figures import camera_map, per_view_change, small, table3_chart, view_grid
    from notebooks.toy_figures import SCALE

    results = ROOT / "notebooks" / "results"
    images = results / "scan40"
    scene = dtu.SCENE
    downloaded = (scene / "cameras.json").is_file() and (scene / "image").is_dir()

    def show(image, caption=None):
        """Show a figure at its display size. Figures are drawn at SCALE times that size."""
        return mo.image(image, width=image.width // SCALE, caption=caption)

    def result(text, kind="info", title="What this shows"):
        return mo.callout(mo.md(text), kind=kind, title=title)

    def stat(value, label, caption=None):
        return mo.stat(value, label=label, caption=caption, bordered=True)

    def missing(what, command):
        return mo.callout(mo.md(f"**{what} not saved yet.** Run this before the lab:\n\n```sh\n{command}\n```"),
                          kind="warn")

    def no_photos():
        return mo.callout(
            mo.md("**The saved reference images are missing.** Restore "
                  "`notebooks/results/scan40_reference` from the complete repository. No training is needed."),
            kind="warn",
        )

    return (
        Image,
        PAPER,
        camera_map,
        downloaded,
        dtu,
        images,
        json,
        missing,
        mo,
        no_photos,
        np,
        per_view_change,
        result,
        results,
        scene,
        show,
        small,
        stat,
        table3_chart,
        view_grid,
    )


@app.cell
def _(PAPER, downloaded, mo, scene, small):
    _intro = mo.md(
        f"""
        **The question.** Notebook 1 showed that depth smoothing improves the shape of two spheres.
        Does it also help on photographs of a real object?

        **The test.** [DTU](https://roboimagedata.compute.dtu.dk/) scan 40, the stone arch in
        [RegNeRF Figure 3]({PAPER}). Two models train on the same 3 photos with the same settings.
        One adds RegNeRF's depth smoothing. Both are scored on 25 photos they never saw.

        **Prediction.** If notebook 1 transfers, smoothing should improve the colour seen from new
        cameras. This can fail: colour PSNR does not measure the shape directly.

        **How to use this page.** Compare the saved results, then render a moved camera from the saved
        models in section 3a. Nothing trains. Read each section, then the box beside it.
        Allow about 25 minutes.

        Small reference photos, camera data, model renders, and scores come with the notebook.
        No DTU download or prior training is needed. The reference previews are reduced for display;
        the saved PSNR values still use the original evaluation protocol.
        PyTorch is needed only if you choose the optional moved-camera render.

        1. What the paper reports
        2. The cameras and the three training photos
        3. New cameras
           - 3a. What changes when you move the camera?
        4. The score over 25 new cameras
        """
    )
    mo.vstack(
        [
            mo.md("# Depth smoothing on real photos"),
            mo.hstack(
                [
                    mo.image(small(scene / "image" / "000025.png", 400), width=380,
                             caption="Scan 40, training view 25")
                    if downloaded else mo.md(""),
                    _intro,
                ],
                widths=[1, 1.4],
                align="start",
                gap=2,
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _(PAPER, mo, result, show, table3_chart):
    _rows = [
        ("Full RegNeRF", 18.89),
        ("Without appearance regularization", 18.34),
        ("Without geometry regularization", 14.34),
        ("Without sample-space annealing", 10.17),
    ]
    mo.vstack(
        [
            mo.md(
                f"""
                ---
                ## 1. What the paper reports
                RegNeRF's ablation ([Table 3]({PAPER})) removes one part at a time. These are the
                paper's values, averaged over many DTU scans. They are not measurements from this repo.
                """
            ),
            mo.hstack(
                [
                    show(table3_chart(_rows)),
                    result(
                        "Removing the geometry (depth) regularization costs **4.55 dB**. Removing the "
                        "appearance loss costs only 0.55 dB.\n\n"
                        "This repo has no appearance loss, so **both of our models lack it**. Our pair "
                        "matches no two rows exactly. It asks a narrower question: on one scan, with this "
                        "code, does adding depth smoothing help?",
                        title="How to read it",
                    ),
                ],
                widths=[1.5, 1],
                align="center",
                gap=1.5,
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _(camera_map, downloaded, dtu, mo, no_photos, result, scene, show, small):
    _train = dtu.TRAIN_IDS[:3]
    if downloaded:
        _body = mo.vstack(
            [
                mo.hstack(
                    [
                        show(camera_map(dtu.load_rig_poses(scene), _train, dtu.TEST_IDS, 45)),
                        result(
                            "Each dot is one camera of the DTU rig, placed by the direction it looks from. "
                            "The 3 **training** cameras sit in a single row, near the middle height. The "
                            "25 **test** cameras spread above, below, and to both sides.\n\n"
                            "**Most test views look from directions that no training photo covers.** The "
                            "model must guess the shape there.",
                        ),
                    ],
                    widths=[1.6, 1],
                    align="center",
                    gap=1.5,
                    wrap=True,
                ),
                mo.hstack(
                    [mo.image(small(scene / "image" / f"{i:06}.png", 400), width=300, caption=f"Training view {i}")
                     for i in _train],
                    justify="start",
                    gap=1,
                    wrap=True,
                ),
            ]
        )
    else:
        _body = no_photos()
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 2. The cameras and the three training photos
                **Watch** where the training cameras are compared with the test cameras. Then look at
                the three photos: which parts of the arch does no photo show? The 25 test photos and
                their masks are never used for training or for choosing settings.
                """
            ),
            _body,
        ]
    )
    return


@app.cell
def _(dtu, mo):
    view_choice = mo.ui.dropdown({f"View {v}": v for v in dtu.TEST_IDS}, value="View 45", label="Test view")
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 3. New cameras
                Both models render every test camera, which neither saw in training. Start with view 45,
                the diamond in the camera map. **Watch** the opening under the top stone and the edges of the
                two pillars, first in colour, then in depth. Then **change** the view: compare a camera close
                to a training camera with one far from all three.
                """
            ),
            view_choice,
        ]
    )
    return (view_choice,)


@app.cell
def _(Image, downloaded, images, json, missing, mo, no_photos, result, scene, show, small, view_choice, view_grid):
    _record_file = images / "render_record.json"
    _view = view_choice.value
    if not _record_file.is_file():
        _body = missing("The test-view renders are", "uv run python -m notebooks.render_scan40_views")
    elif not downloaded:
        _body = no_photos()
    else:
        _record = json.loads(_record_file.read_text())

        def _open(name):
            return Image.open(images / f"view{_view}_{name}.jpg")

        _body = mo.vstack(
            [
                show(
                    view_grid(small(scene / "image" / f"{_view:06}.png", 600), _open("rgb_off"), _open("rgb_on"),
                              _open("depth_off"), _open("depth_on"), _record["depth_ranges"][str(_view)]),
                    caption=f"Test view {_view}, step {_record['step']}, rendered at scale {_record['scale']}. "
                    "Top: colour. Bottom: depth (light is near).",
                ),
                result(
                    "**Describe only what you can see.** Where does each model miss structure or put the wrong "
                    "colour? Where does the depth change suddenly without an edge in the photo? Each view is "
                    "an illustration; the score in section 4 uses all 25 test views.",
                    kind="neutral",
                    title="Look and compare",
                ),
            ]
        )
    _body
    return


@app.cell
def _(json, mo, results):
    camera_settings = (
        mo.md("{angle}")
        .batch(angle=mo.ui.slider(start=0, stop=20, step=2, value=10, show_value=True,
                                  label="Camera turn (degrees)", full_width=True))
        .form(submit_button_label="Render view")
    )
    _timing_path = results / "camera_timing.json"
    _timing_note = mo.md("")
    if _timing_path.is_file():
        _timing = json.loads(_timing_path.read_text())
        _rows = []
        for _device, _label in (("cpu", "CPU"), ("cuda", "GPU")):
            if _device in _timing["devices"]:
                _times = _timing["devices"][_device]["median_seconds"]
                _rows.append(f"| {_label} | {_times['first_request']:.1f} s | "
                             f"{_times['new_angle']:.1f} s | {_times['repeat_angle']:.2f} s |")
        _timing_note = mo.callout(mo.md(
            "Median times from three trials at 100 × 75 pixels per image.\n\n"
            "| Device | First render | New angle | Repeated angle |\n"
            "| --- | ---: | ---: | ---: |\n" + "\n".join(_rows) + "\n\n"
            "**First render:** both models at the start and moved positions (four images). "
            "**New angle:** two new images; the start images are reused. "
            "**Repeated angle:** all images are reused from memory.\n\n"
            f"Measured on {_timing['cpu']} with {_timing['cpu_threads']} CPU threads, "
            f"and {_timing['gpu']}. Includes model loading and figure assembly. "
            "Excludes notebook startup and browser display. Your computer and other running tasks "
            "can change these times."
        ), kind="neutral", title="How long does it take?")
    mo.vstack([
        mo.md("""
        ---
        ## 3a. What changes when you move the camera?
        Start at training camera 25. Turn the camera around the arch, away from the training row.
        Both models use the same camera. Their weights stay fixed.

        **Predict.** Which model will keep the arch opening and pillar edges more stable?
        **Change** the turn to 0, 10, then 20 degrees. Select **Render view** after each change.
        **Watch** the opening and both pillars. The top row stays at the start position.
        The bottom row shows your new view.

        The page computes new 100 × 75 images from the saved models. It does not train.
        It uses the GPU when available. Repeated settings reuse the render.
        **Same model pair.** This activity uses the shared models from `scan40_pair_new` at step 20,000,
        the pair used for the saved images and scores in sections 3 and 4. The model files come with
        the notebook. The page checks their file hashes before rendering.
        """),
        _timing_note,
        camera_settings,
    ])
    return (camera_settings,)


@app.cell
def _(camera_settings, downloaded, mo, no_photos, result, scene, show):
    if camera_settings.value is None:
        _body = result("Select a camera turn, then select **Render view**. No render starts when you open the page.",
                       kind="neutral", title="Try it")
    elif not downloaded:
        _body = no_photos()
    else:
        try:
            from notebooks.live_camera import render_comparison
            with mo.status.spinner(title="Rendering both saved models. No training."):
                _figure, _device, _step = render_comparison(scene, camera_settings.value["angle"])
            _body = mo.hstack([
                show(_figure, caption=f"Scored pair: scan40_pair_new, step {_step}. "
                     f"Rendered on {_device} at 100 × 75 pixels; enlarged for display."),
                result(
                    "The model has already learned a 3D scene. **Moving the camera asks it to draw that "
                    "scene from a different position.** It does not learn anything new.\n\n"
                    "Read the images like this:\n\n"
                    "- **Left column:** Model trained without depth smoothing.\n"
                    "- **Right column:** Model trained with depth smoothing.\n"
                    "- **Top row:** Original camera position.\n"
                    "- **Bottom row:** Camera position you selected.\n\n"
                    "At **0°**, the camera has not moved. Each bottom image should match the image above it.\n\n"
                    "At **10° or 20°**, you see the arch from a new angle. Look for mistakes that become "
                    "visible: a pillar may look broken, or a false surface may block the opening. "
                    "“False structure” means a surface the model predicts that should not be there.\n\n"
                    "The question is: **Which model keeps a more plausible arch as the camera moves?** "
                    "Smoothing may help, but you must check the images.\n\n"
                    "We have no real photo from these exact new positions. Thus, we can inspect visible "
                    "problems, but cannot calculate an accurate comparison with a reference photo. "
                    "The existing PSNR scores still come from the original 25 test photos.",
                    kind="neutral",
                ),
            ], widths=[2, 1], align="center", gap=1, wrap=True)
        except (FileNotFoundError, ValueError, ImportError) as _error:
            _body = result(
                f"**The live render is not ready.**\n\n```text\n{_error}\n```\n\n"
                "This activity needs `notebooks/models/scan40_off.pt`, "
                "`notebooks/models/scan40_regnerf.pt`, and their `manifest.json`. "
                "PyTorch is needed only for this optional render. Install it to use the shared models. "
                "The saved comparisons in sections 3 and 4 still work without it. No training is needed.",
                kind="warn", title="Saved models needed",
            )
    _body
    return


@app.cell
def _(dtu, json, missing, mo, np, per_view_change, result, results, show, stat, view_choice):
    _file = results / "scan40.json"
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
        _effect = (f"**raised** the mean colour score by **{_mean:.2f} dB**" if _mean > 0 else
                   f"**lowered** the mean colour score by **{-_mean:.2f} dB**" if _mean < 0 else
                   "**did not change** the mean colour score")
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
                show(per_view_change(dtu.TEST_IDS, _delta, highlight=view_choice.value),
                     caption=f"The view chosen in section 3 (view {view_choice.value}) has a dark label."),
                result(
                    f"On this scan, depth smoothing {_effect}. It was better in **{_better} of 25** test views; "
                    f"the change per view ranges from {_delta.min():+.2f} to {_delta.max():+.2f} dB.\n\n"
                    f"**Against the paper.** Smoothing on ({_on.mean():.2f} dB) compares with the row without "
                    f"appearance regularization (18.34 dB), because both lack only that loss. Smoothing off "
                    f"({_off.mean():.2f} dB) has no close row: the paper's row without geometry (14.34 dB) "
                    "keeps the appearance loss.\n\n"
                    "**Limits.** One scan and one seed. PSNR measures colour, not shape. Notebook 1 showed "
                    "that colour PSNR can vary a lot between seeds.",
                    kind="success" if _mean > 0 else "warn",
                ),
                mo.accordion(
                    {
                        "Every test view (table)": mo.ui.table(
                            [{"View": r["view"], "Off (dB)": round(r["off_db"], 2), "On (dB)": round(r["on_db"], 2),
                              "On minus off (dB)": round(r["on_minus_off_db"], 2)} for r in _rows],
                            selection=None,
                            show_column_summaries=False,
                        ),
                        "Evaluation record": mo.md("```json\n" + json.dumps(_data["record"], indent=2) + "\n```"),
                    }
                ),
            ]
        )
    else:
        _body = missing("The evaluation is", "uv run python -m notebooks.evaluate_pair")
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 4. The score over 25 new cameras
                For each test photo, the score uses only the object's pixels (its mask), measures the colour
                error, and turns it into PSNR. The result is the mean of the 25 PSNR values; higher is
                better. This is not the PSNR printed during training. **Watch** whether the bars agree in
                sign, or whether a few views decide the mean.
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
                - Does the evidence support better colour prediction on this scan? How large is the change
                  compared with the variation between seeds in notebook 1?
                - The paper reports +4.55 dB for geometry regularization. Compare both of our means with
                  the rows in section 1. Which of our two scores is further from the paper, and why?
                - Which test views changed most in section 4? Where are they in the camera map? Open them
                  in section 3: what does the smoothing-off model put in front of the camera?
                - What evidence would you need to claim that the *shape* improved?
                - In section 3a, what changed when you moved the camera? Which visible structures stayed stable?

                **Next.** Smoothing helps inside a surface, but a real depth edge should stay sharp. Notebook 3
                adds an L1 depth term that relaxes at colour edges, and compares it with this RegNeRF model.
                """
            ),
            mo.accordion(
                {
                    "How the two models were trained": mo.md(
                        """
                        - **Same for both:** training views 25, 22, and 28; seed 42; 20,000 steps; Adam with the
                          same learning-rate schedule; training scale 4; the same ray sampling, coarse jitter,
                          and sample-space annealing. Depth patches use a separate random generator.
                        - **The only difference:** the depth weight. With smoothing on, the fine depth weight
                          falls from 400 to 0.1 by step 512, and the coarse depth weight is 0.1. With smoothing
                          off, both are 0 and no depth patches are rendered.
                        - **Scoring:** both models at the same step, at evaluation scale 2, on the 25 test views
                          from `dtu.py`.
                        - **Neither model** has RegNeRF's appearance loss.
                        - **Record:** `experiments/compare_depth.py` writes the code revision, settings, and checkpoint paths
                          to `protocol.json` before training. `evaluate_pair.py` checks the pair against it and
                          stores it in the evaluation record.
                        """
                    ),
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
