import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="RegNeRF toy lesson")


@app.cell
def _():
    import json
    from pathlib import Path
    from tempfile import mkdtemp

    import marimo as mo
    import numpy as np
    from PIL import Image

    from notebooks.lab import solve_depth
    from notebooks.toy_experiment import TOY_SEED, ensure_assets
    from notebooks.toy_figures import (
        SCALE,
        STRIPE_POINT,
        comparison,
        crisp,
        curve_fit_figure,
        depth_profile,
        patch_experiment,
        patch_numbers,
        shear_patch,
        surface_inspector,
    )

    scene = ensure_assets()

    def show(image, caption=None):
        """Show a figure at its display size. Figures are drawn at SCALE times that size."""
        return mo.image(image, width=image.width // SCALE, caption=caption)

    def result(text, kind="info", title="What this shows"):
        return mo.callout(mo.md(text), kind=kind, title=title)

    def stat(value, label, caption=None):
        return mo.stat(value, label=label, caption=caption, bordered=True)

    return (
        Image,
        Path,
        STRIPE_POINT,
        TOY_SEED,
        comparison,
        crisp,
        curve_fit_figure,
        depth_profile,
        json,
        mkdtemp,
        mo,
        np,
        patch_experiment,
        patch_numbers,
        result,
        scene,
        shear_patch,
        show,
        solve_depth,
        stat,
        surface_inspector,
    )


@app.cell
def _(Image, json, np, scene):
    with np.load(scene / "reference.npz") as _ref:
        truth, ids = _ref["depth"], _ref["ids"]
    baseline_folder = scene / "coverage_500" / "3"
    with np.load(baseline_folder / "prediction.npz") as _model:
        baseline = _model["depth"]
    reference_rgb = np.asarray(Image.open(scene / "reference.png").convert("RGB"))
    _angle = json.loads((scene / "transforms_train.json").read_text())["camera_angle_x"]
    focal = truth.shape[1] / (2 * np.tan(_angle / 2))
    return baseline, focal, ids, reference_rgb, truth


@app.cell
def _(mo, scene):
    mo.vstack(
        [
            mo.md(
                """
                # Depth smoothing with three photos
                A hands-on look at the depth penalty in
                [RegNeRF (Niemeyer et al., CVPR 2022)](https://arxiv.org/abs/2112.00724).
                """
            ),
            mo.hstack(
                [
                    mo.image(
                        str(scene.parent / "regnerf_figure_1.png"),
                        width=400,
                        caption="RegNeRF Figure 1. Middle row: NeRF from 3 photos. Bottom row: RegNeRF.",
                        alt="Paper Figure 1: three input photos, poor NeRF views, and clearer RegNeRF views.",
                    ),
                    mo.md(
                        """
                        **The question.** Three photos can give a NeRF the right colours but the
                        wrong shape. RegNeRF adds a penalty that keeps depth smooth. Does that
                        penalty give a better picture from a new camera?

                        **The test scene.** Two spheres with known depth, so we can measure the
                        shape directly. Full RegNeRF also uses other parts. This page tests only
                        the depth penalty.

                        **How to use this page.** Each section has controls at the top. Change one
                        control, then read the box beside the figure. Nothing trains until you
                        select **Run comparison** in section 7. Allow about 15 minutes.

                        1. More photos
                        2. Wrong depth and camera motion
                        3. Where the model put the surface
                        4. Smoothing a noisy curve
                        5. What the depth loss measures
                        6. Training with and without smoothing
                        7. Train your own comparison
                        """
                    ),
                ],
                widths=[1, 1.15],
                align="start",
                gap=2,
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    photo_count = mo.ui.radio(
        ["3 photos", "9 photos", "20 photos"], value="3 photos", inline=True, label="Training photos"
    )
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 1. What do more photos change?
                **Change** the number of training photos. **Watch** the yellow stripe and the
                blue fragment at the left. Each model trained for 500 steps with smoothing off.
                The reference camera was not used for training.
                """
            ),
            photo_count,
        ]
    )
    return (photo_count,)


@app.cell
def _(crisp, json, mo, photo_count, result, scene, show, stat):
    _count = int(photo_count.value.split()[0])
    _folder = scene / "coverage_500" / str(_count)
    _score = json.loads((_folder / "result.json").read_text())
    if _count == 3:
        _inputs = mo.hstack(
            [mo.image(str(scene / f"input_{i}.png"), width=200) for i in range(3)], justify="start", gap=0.5
        )
    else:
        _inputs = mo.image(str(_folder / "inputs.png"), alt=f"All {_count} training photos.")
    mo.vstack(
        [
            mo.hstack(
                [
                    show(crisp(scene / "reference.png", 300), caption="Reference (held-out camera)"),
                    show(crisp(_folder / "heldout.png", 300), caption=f"Model from {_count} photos"),
                    mo.vstack(
                        [
                            mo.hstack(
                                [
                                    stat(f"{_score['foreground_psnr']:.2f} dB", "Colour match", "PSNR, higher is better"),
                                    stat(f"{_score['depth_mae']:.3f}", "Depth error", "MAE, lower is better"),
                                ],
                                widths="equal",
                            ),
                            result(
                                "With 3 photos, an extra blue fragment appears at the left. With 9 and "
                                "20 photos, the colours get cleaner (PSNR **19.94 → 20.37 → 21.03 dB**), "
                                "but the depth error does not fall (**0.166 → 0.176 → 0.193**).\n\n"
                                "**More photos improved the colour, not the shape.**"
                            ),
                        ]
                    ),
                ],
                widths=[1, 1, 1.15],
                align="start",
                gap=1,
                wrap=True,
            ),
            mo.md(f"**The {_count} training photos**"),
            _inputs,
        ]
    )
    return


@app.cell
def _(mo):
    camera_shift = mo.ui.slider(
        -0.8, 0.8, step=0.05, value=0, show_value=True, label="Move the camera sideways (scene units)"
    )
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 2. Why does wrong depth damage a new view?
                We keep the colours fixed and change only the depth. **Change** the camera
                position. **Watch** the white dot against the white line in the two close-ups.
                """
            ),
            camera_shift,
        ]
    )
    return (camera_shift,)


@app.cell
def _(STRIPE_POINT, baseline, camera_shift, focal, mo, reference_rgb, result, shear_patch, show, truth):
    _x, _y = STRIPE_POINT
    _offset = focal * camera_shift.value * (1 / truth[_y, _x] - 1 / baseline[_y, _x])
    if camera_shift.value == 0:
        _text = (
            "At zero shift, both patches line up, although the model depth is wrong "
            f"(true **{truth[_y, _x]:.2f}**, model **{baseline[_y, _x]:.2f}**). Move the slider."
        )
    else:
        _side = "right" if _offset > 0 else "left"
        _text = (
            f"With the model depth, the marked point lands **{abs(_offset):.1f} pixels too far {_side}**. "
            f"The model puts it at **{baseline[_y, _x]:.2f}** instead of **{truth[_y, _x]:.2f}**. "
            "A point that is too far away moves too little when the camera moves. Uneven depth also "
            "bends the grid."
        )
    mo.hstack(
        [
            show(shear_patch(reference_rgb, truth, baseline, focal, camera_shift.value)),
            result(_text + "\n\n**Wrong depth is hidden from the training cameras. It shows when the camera moves.**"),
        ],
        widths=[2.3, 1],
        align="center",
        gap=1,
        wrap=True,
    )
    return


@app.cell
def _(mo):
    region = mo.ui.radio(["Centre", "Near the rim", "Full width"], value="Centre", inline=True, label="Strip")
    turn = mo.ui.slider(-65, 65, step=5, value=25, show_value=True, label="Turn both surfaces (degrees)")
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 3. Where did the model put the surface?
                **Change** the strip: Centre, then Near the rim, then Full width. **Watch** the
                orange curve against the dark reference curve. Turn the surfaces to see the
                bumps from the side.
                """
            ),
            mo.hstack([region, turn], justify="start", gap=3, wrap=True),
        ]
    )
    return region, turn


@app.cell
def _(baseline, depth_profile, focal, ids, mo, np, region, result, show, stat, surface_inspector, truth, turn):
    _left, _right = {"Centre": (55, 75), "Near the rim": (90, 110), "Full width": (25, 110)}[region.value]
    _row = 75
    _true, _learned = truth[_row, _left : _right + 1], baseline[_row, _left : _right + 1]
    _limits = (min(_true.min(), _learned.min()) - 0.03, max(_true.max(), _learned.max()) + 0.03)
    _lesson = {
        "Centre": "The true centre is nearly flat. The model adds false bumps and puts this strip too far away.",
        "Near the rim": "The true depth rises steeply as the sphere curves away. "
        "A smooth surface does not need a flat depth curve.",
        "Full width": "The true surface forms a bowl. The model keeps an overall depth offset and adds local bumps.",
    }[region.value]
    mo.hstack(
        [
            mo.vstack(
                [
                    show(surface_inspector(truth, baseline, ids, focal, turn.value, _row, _left, _right)),
                    show(depth_profile(truth, baseline, ids == 1, _row, _left, _right, limits=_limits)),
                ]
            ),
            mo.vstack(
                [
                    stat(f"{np.mean(_learned - _true):+.3f}", "Mean depth error on this strip",
                         "scene units; positive = too far"),
                    result(_lesson + "\n\n**Some bumps are false. Some curvature is real. Smoothing must keep "
                           "the real shape.**"),
                ]
            ),
        ],
        widths=[2, 1],
        align="center",
        gap=1,
        wrap=True,
    )
    return


@app.cell
def _(mo):
    fit_weight = mo.ui.slider(steps=[0, 1, 4, 16, 64, 256], value=4, show_value=True, label="Smoothing strength")
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 4. Can smoothing remove false bumps?
                A simulated one-dimensional test: noisy readings of the true sphere curve.
                An exact fit follows every error. **Change** the smoothing to 0, 4, then 256.
                **Watch** the centre and both edges of the orange fit.
                """
            ),
            fit_weight,
        ]
    )
    return (fit_weight,)


@app.cell
def _(curve_fit_figure, fit_weight, mo, np, result, show, solve_depth, stat, truth):
    _true = truth[75, 27:109]
    _readings = _true + np.random.default_rng(42).normal(0, 0.06, len(_true))
    _fit = solve_depth(_readings, fit_weight.value)
    _before, _after = [np.sqrt(np.mean((curve - _true) ** 2)) for curve in (_readings, _fit)]
    if fit_weight.value == 0:
        _text, _kind = "With no smoothing, the fit follows every noisy reading.", "neutral"
    elif fit_weight.value <= 4:
        _text, _kind = "Moderate smoothing removes many false bumps and keeps the bowl shape.", "success"
    else:
        _text, _kind = "Strong smoothing flattens the edges. It removes real shape as well as noise.", "warn"
    mo.hstack(
        [
            show(curve_fit_figure(_true, _readings, _fit)),
            mo.vstack(
                [
                    mo.hstack(
                        [stat(f"{_before:.3f}", "Error, exact fit", "RMSE"),
                         stat(f"{_after:.3f}", "Error, your fit", "RMSE")],
                        widths="equal",
                    ),
                    result(_text + "\n\n**The strength matters: too little keeps noise, too much flattens "
                           "the shape.**", kind=_kind),
                ]
            ),
        ],
        widths=[2, 1],
        align="center",
        gap=1,
        wrap=True,
    )
    return


@app.cell
def _(mo):
    move = mo.ui.radio(["Marked pixel", "Whole patch"], value="Marked pixel", inline=True, label="Move")
    distance = mo.ui.slider(0, 0.30, step=0.01, value=0.10, show_value=True,
                            label="Distance away from the camera (scene units)")
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 5. What does the depth loss measure?
                This 8 x 8 depth patch comes from the sphere centre. **Change** the distance to
                move the marked pixel. **Watch** the loss change and the four orange pairs.
                Then select **Whole patch** and keep the same distance.
                """
            ),
            mo.hstack([move, distance], justify="start", gap=3, wrap=True),
        ]
    )
    return distance, move


@app.cell
def _(distance, mo, move, patch_experiment, patch_numbers, result, show, stat, truth):
    _patch = truth[71:79, 62:70]
    _whole = move.value == "Whole patch"
    _image, _count, _error = patch_experiment(_patch, distance.value, _whole)
    *_, _losses, _changed, _rows = patch_numbers(_patch, distance.value, _whole)
    _change = _losses[1] - _losses[0]
    if distance.value == 0:
        _text, _kind = "At zero distance, nothing moves and the loss is unchanged.", "neutral"
    elif _whole:
        _text, _kind = (
            "Every depth moved by the same amount. All neighbour differences stay the same, so the loss "
            "does not change, although the whole patch is now in the wrong place.",
            "warn",
        )
    else:
        _text, _kind = (
            f"The bump changes **{_count}** neighbour differences. Their squares add to the loss.",
            "info",
        )
    _table = mo.ui.table(
        [
            {
                "Pair": r["Pair"],
                "Depth difference": f"{r['Depth difference before']:+.4f} → {r['Depth difference after']:+.4f}",
                "Share of loss": f"{r['Loss share before']:.7f} → {r['Loss share after']:.7f}",
            }
            for r in _rows
        ],
        selection=None,
        show_column_summaries=False,
    )
    mo.vstack(
        [
            mo.hstack(
                [
                    show(_image),
                    mo.vstack(
                        [
                            stat(f"{_change:+.6f}", "Loss change", f"{_losses[0]:.6f} → {_losses[1]:.6f}"),
                            result(_text + "\n\n**Smoothing penalizes bumps. It cannot tell the true distance; "
                                   "the photos must do that.**", kind=_kind),
                        ]
                    ),
                ],
                widths=[2, 1],
                align="center",
                gap=1,
                wrap=True,
            ),
            mo.accordion(
                {
                    "The four pairs around the marked pixel": _table,
                    "The exact loss used by the trainer": mo.md(
                        r"""
                        For one 8 x 8 patch, the trainer uses the 49 pixels that have a right and a lower
                        neighbour:

                        $$L_{\mathrm{patch}}=\frac{1}{49}\sum_{i=0}^{6}\sum_{j=0}^{6}\operatorname{stopgrad}(A_{ij})
                        \left[(d_{ij}-d_{i,j+1})^2+(d_{ij}-d_{i+1,j})^2\right].$$

                        $A$ is the predicted opacity; here $A=1$. $d$ is the predicted depth,
                        $d=\sum_k w_k t_k/\max(\sum_k w_k,\epsilon)$, clipped to the ray bounds. The
                        trainer renders these patches from new, unseen cameras and averages over them.

                        With smoothing on, $L=L_{\mathrm{RGB}}+0.1\,L_{\mathrm{patch,coarse}}+\lambda\,
                        L_{\mathrm{patch,fine}}$. With smoothing off, both depth terms are off. The trainer
                        never sees the true depth. This page uses true depth only to explain and to score.
                        """
                    ),
                }
            ),
        ]
    )
    return


@app.cell
def _(mo):
    camera_view = mo.ui.radio(["Lower view", "Front view"], value="Lower view", inline=True, label="New camera")
    stage = mo.ui.slider(steps=[100, 300, 1000], value=300, show_value=True, label="Training steps")
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 6. Does training with smoothing improve the new view?
                Two saved models saw the same three photos. One also used depth smoothing
                (weight 0.10). **Change** the steps to 300, then 100, then 1,000. **Watch** the
                curved yellow stripe in the lower view. Then switch to the front view.
                """
            ),
            mo.hstack([camera_view, stage], justify="start", gap=3, wrap=True),
        ]
    )
    return camera_view, stage


@app.cell
def _(TOY_SEED, camera_view, json, scene):
    experiment_scene = scene / "lower" if camera_view.value == "Lower view" else scene
    saved = []
    for _steps in (100, 300, 1000):
        for _name in ("off", "on"):
            _folder = experiment_scene / (_name if _steps == 300 else f"{_name}_{_steps}")
            _record = json.loads((_folder / "result.json").read_text())
            _record.setdefault("seed", TOY_SEED)
            saved.append((_folder, _record))

    def score_row(result, source):
        return {
            "Source": source,
            "Steps": result["steps"],
            "Seed": result["seed"],
            "Fine weight": result["fine_weight"],
            "Colour PSNR (dB)": round(result["foreground_psnr"], 2),
            "Depth MAE": round(result["depth_mae"], 3),
        }

    return experiment_scene, saved, score_row


@app.cell
def _(camera_view, comparison, experiment_scene, mo, result, saved, show, stage, stat):
    _pair = [(p, r) for p, r in saved if r["steps"] == stage.value]
    _off, _on = [r for _, r in _pair]
    _delta = _on["foreground_psnr"] - _off["foreground_psnr"]
    _depth_delta = _on["depth_mae"] - _off["depth_mae"]
    _observation = {
        100: "Both images are still blurred. Smoothing slightly lowers the colour score and raises the depth error.",
        300: (
            "Without smoothing, most of the curved yellow stripe is missing. With smoothing, the stripe is "
            "visible but blurred."
            if camera_view.value == "Lower view"
            else "With smoothing, the yellow stripe is more distinct. The colour score rises and the depth "
            "error falls."
        ),
        1000: "Both stripes are clearer than at 300 steps. Smoothing improves both scores, but neither image "
        "matches the reference.",
    }[stage.value]
    mo.vstack(
        [
            show(
                comparison(experiment_scene, [("Smoothing off", _pair[0][0]), ("Smoothing on", _pair[1][0])]),
                caption=f"{camera_view.value}, {stage.value} steps. Top: colour. Bottom: depth.",
            ),
            mo.hstack(
                [
                    stat(f"{_on['foreground_psnr']:.2f} dB", "Colour match, smoothing on",
                         f"{_delta:+.2f} dB vs off ({_off['foreground_psnr']:.2f})"),
                    stat(f"{_on['depth_mae']:.3f}", "Depth error, smoothing on",
                         f"{_depth_delta:+.3f} vs off ({_off['depth_mae']:.3f})"),
                    result(
                        _observation + "\n\n**In these saved runs, smoothing helps at 300 and 1,000 steps, "
                        "but not at 100.**",
                        kind="success" if _delta > 0 else "warn",
                    ),
                ],
                widths=[1, 1, 2],
                align="center",
                gap=1,
                wrap=True,
            ),
        ]
    )
    return


@app.cell
def _(mo):
    settings = (
        mo.md("{weight}\n\n{steps}")
        .batch(
            weight=mo.ui.slider(start=0, stop=0.10, step=0.01, value=0.05, label="Fine depth weight",
                                show_value=True, full_width=True),
            steps=mo.ui.number(start=100, stop=1000, step=100, value=300, label="Training steps"),
        )
        .form(submit_button_label="Run comparison")
    )
    mo.vstack(
        [
            mo.md(
                """
                ---
                ## 7. Train your own comparison
                Pick a smoothing weight and a step count, then select **Run comparison**. The page
                trains your model and a model with smoothing off, unless a matching result is
                already saved. Weight 0 turns smoothing off. Every run uses seed 42.
                """
            ),
            mo.hstack(
                [
                    settings,
                    mo.callout(
                        mo.md(
                            "**Time.** About 10 seconds per model on a GPU. On a CPU, about 20 minutes per "
                            "model at 300 steps.\n\nThe result uses the camera you selected in section 6. "
                            "New training needs PyTorch."
                        ),
                        kind="neutral",
                    ),
                ],
                widths=[1.2, 1],
                align="start",
                gap=1.5,
                wrap=True,
            ),
        ]
    )
    return (settings,)


@app.cell
def _(
    Path,
    TOY_SEED,
    camera_view,
    comparison,
    experiment_scene,
    json,
    mkdtemp,
    mo,
    np,
    saved,
    scene,
    score_row,
    settings,
    show,
):
    from importlib.util import find_spec

    from notebooks.toy_experiment import run_condition

    _cache = scene.parents[2] / "checkpoint" / "toy_two_spheres_student_v3"
    _history = list(saved)
    for _base_file in sorted(_cache.glob("*/result.json")):
        _file = _base_file.parent / "lower" / "result.json" if camera_view.value == "Lower view" else _base_file
        if _file.is_file():
            _history.append((_file.parent, json.loads(_file.read_text())))

    def _scores():
        return mo.ui.table(
            [score_row(r, "Saved" if p.parent == experiment_scene else "Custom") for p, r in _history],
            selection=None,
            show_column_summaries=False,
        )

    def _matches(result, steps, weight):
        return (
            result["steps"] == steps
            and result.get("seed") == TOY_SEED
            and np.isclose(result["fine_weight"], weight, rtol=0, atol=1e-8)
        )

    _panel = mo.md("*Submit the form above to see your comparison here.*")
    if settings.value is not None:
        _steps, _weight = int(settings.value["steps"]), float(settings.value["weight"])
        _needs_training = any(not any(_matches(r, _steps, _w) for _, r in _history) for _w in (0.0, _weight))
        mo.stop(
            _needs_training and find_spec("torch") is None,
            mo.vstack(
                [
                    mo.callout(
                        mo.md("**New training needs PyTorch.** Install it in this environment, then restart "
                              "marimo. All saved results still work."),
                        kind="warn",
                    ),
                    _scores(),
                ]
            ),
        )
        _models, _sources = [], []
        for _w in dict.fromkeys([0.0, _weight]):
            _match = next(((p, r) for p, r in _history if _matches(r, _steps, _w)), None)
            if _match is None:
                _cache.mkdir(parents=True, exist_ok=True)
                _folder = Path(mkdtemp(prefix="run_", dir=_cache))
                with mo.status.spinner(title=f"Training {_steps} steps with fine weight {_w:g}"):
                    run_condition(_folder, _steps, _w)
                _folder = _folder / "lower" if camera_view.value == "Lower view" else _folder
                _history.append((_folder, json.loads((_folder / "result.json").read_text())))
                _sources.append("trained now")
            else:
                _folder = _match[0]
                _sources.append("reused a saved result")
            _models.append(("Smoothing off" if _w == 0 else f"Fine weight {_w:g}", _folder))
        _off = next(r for p, r in _history if p == _models[0][1])
        _custom = next(r for p, r in _history if p == _models[-1][1])
        _delta = _custom["foreground_psnr"] - _off["foreground_psnr"]
        _panel = mo.vstack(
            [
                show(
                    comparison(experiment_scene, _models),
                    caption=f"{camera_view.value}, {_steps} steps, seed {TOY_SEED}. Top: colour. Bottom: depth.",
                ),
                mo.callout(
                    mo.md(
                        f"Fine weight **{_weight:g}** at **{_steps}** steps: colour PSNR "
                        f"**{_custom['foreground_psnr']:.2f} dB** (**{_delta:+.2f} dB** vs smoothing off), depth "
                        f"MAE **{_custom['depth_mae']:.3f}**. Models: {'; '.join(_sources)}."
                    ),
                    kind="success" if _delta > 0 else "warn",
                    title="Your result",
                ),
            ]
        )
    mo.vstack(
        [
            _panel,
            mo.md(f"**All scores, {camera_view.value.lower()}.** Higher PSNR is better. Lower MAE is better."),
            _scores(),
            mo.accordion(
                {
                    "Experiment details": mo.md(
                        """
                        - Sections 2–3 use the front view of the three-photo model at 500 steps. The saved
                          pairs in section 6 compare fine depth weights 0 and 0.10 at the same step count.
                        - Each camera has its own reference and scores. The lower view was chosen after
                          looking at the images, so it is an extra illustration, not an independent test.
                        - A positive fine weight also turns on a coarse depth weight of 0.10. Weight 0 turns
                          off both depth terms.
                        - Every run uses seed 42 for model initialization and RGB ray sampling. Depth patches
                          use a separate generator with seed 43. All other training settings stay fixed.
                          A saved result is reused only at the same seed, weight, and step count.
                        - PSNR and depth MAE use the true foreground at evaluation scale 2, for the selected
                          held-out camera. They are not training-batch scores or paper benchmark scores.
                          Differences use unrounded values.
                        - The depth images hide the background, so false background surfaces show only in
                          colour. Section 2 moves saved depths with fixed colours. It does not render a new
                          NeRF view.
                        - The CPU time comes from a 12-step test with four threads (3.54 s per step, plus about
                          20 s per camera for scoring). It is an estimate, not a full measurement.
                        - The readings in section 4 use fixed random noise. True depth gives only their score.
                        """
                    )
                }
            ),
        ]
    )
    return


@app.cell
def _(mo, scene):
    _static = scene.parents[1] / "toy_figures"
    mo.vstack(
        [
            mo.md("---\n## Summary"),
            mo.callout(
                mo.md(
                    "Three photos can fix the colours and still leave the depth wrong. Smoothing can improve "
                    "a new view, but it cannot find the true distance; the photos must do that. Next, "
                    "notebook 2 tests the same penalty on photographs of a real object."
                ),
                kind="info",
                title="What we learned",
            ),
            mo.md(
                """
                **Questions for discussion**
                - Which saved step count contradicts "smoothing always helps"?
                - Why can a patch have zero loss change and still have the wrong depth?
                """
            ),
            mo.accordion(
                {
                    "Static figures (for the HTML preview, where the controls do not work)": mo.vstack(
                        [
                            mo.vstack([mo.md(f"**{title}**"), mo.image(str(_static / name), alt=title)])
                            for title, name in [
                                ("More photos: 3, 9, and 20 at 500 steps", "coverage_500.png"),
                                ("Camera shift: 0 and +0.8 units", "camera_shift.png"),
                                ("Surface: centre, near the rim, and full width", "shape_presets.png"),
                                ("Smoothing a noisy curve: 0, 4, and 256", "curve_settings.png"),
                                ("One pixel moved by 0.10 units", "one_pixel.png"),
                                ("Whole patch moved by 0.10 units", "whole_patch.png"),
                                ("Saved training: both cameras at 100, 300, and 1,000 steps", "training_stages.png"),
                            ]
                        ]
                    )
                }
            ),
        ]
    )
    return


if __name__ == "__main__":
    app.run()
