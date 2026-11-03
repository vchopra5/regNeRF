import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    from pathlib import Path
    from tempfile import mkdtemp
    import marimo as mo
    import numpy as np
    from PIL import Image
    from notebooks.toy_experiment import TOY_SEED, ensure_assets
    scene = ensure_assets()
    from notebooks.toy_figures import comparison, depth_profile, surface_inspector, shear_patch, STRIPE_POINT, curve_fit_figure, patch_experiment
    from notebooks.lab import solve_depth

    return (
        Image,
        Path,
        TOY_SEED,
        STRIPE_POINT,
        comparison,
        curve_fit_figure,
        depth_profile,
        json,
        mkdtemp,
        mo,
        np,
        patch_experiment,
        scene,
        shear_patch,
        solve_depth,
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
    mo.vstack([
        mo.md("""
        # RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs
        *Three photos constrain colour. Does depth smoothing improve the view from a new camera?*

        [Niemeyer et al. · CVPR 2022 · arXiv](https://arxiv.org/abs/2112.00724)
        """),
        mo.image(str(scene.parent / "regnerf_figure_1.png"), width=480,
                 alt="Paper Figure 1: three input photos, poor NeRF views, and clearer full RegNeRF views."),
        mo.md("""**Figure 1.** Compare the face in the middle and bottom rows. Full RegNeRF gives clearer new views from three photos. It uses more than depth smoothing. Here we test only the depth penalty on a small scene with known shape.

        Start with the photos, then inspect the shape behind the picture. Allow about 15 minutes. The figures use saved results or quick calculations. Training starts only when you submit the final form.

        **HTML preview:** controls are static. Use the comparison gallery at the end, or open this source in marimo for live controls and editable code.
        """),
    ])
    return


@app.cell
def _(mo):
    photo_count = mo.ui.radio(["3 photos", "9 photos", "20 photos"], value="3 photos", inline=True, label="Saved training inputs")
    mo.vstack([mo.md("""## 1. What do more photos change?
    Select 3, 9, then 20 photos. Each baseline trained for **500 steps with smoothing off**. Compare the yellow stripe with the reference. Look for the extra blue fragment at the left. The reference camera was not used for training."""), photo_count])
    return (photo_count,)


@app.cell
def _(json, mo, photo_count, scene):
    _count = int(photo_count.value.split()[0])
    _folder = scene / "coverage_500" / str(_count)
    _score = json.loads((_folder / "result.json").read_text())
    mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("**Reference · held-out camera**"), mo.image(str(scene / "reference.png"), width=300, alt="Reference from the fixed held-out camera.")]),
            mo.vstack([mo.md(f"**{_count}-photo baseline · {_score['foreground_psnr']:.2f} dB**"), mo.image(str(_folder / "heldout.png"), width=300, alt=f"Held-out prediction from {_count} photos at 500 steps.")]),
        ], justify="start", gap=2, wrap=True),
        mo.md(f"**What the model saw: all {_count} training photos**"),
        mo.image(str(_folder / "inputs.png"), width=500, alt=f"All {_count} training photos."),
        mo.md("""**Observation.** The 3-photo render contains an extra blue fragment. With 9 and 20 photos, the stripe is cleaner, but its edges remain blurred. Foreground RGB PSNR rises from **19.94 → 20.37 → 21.03 dB**. Depth error does not improve: **0.166 → 0.176 → 0.193 units**.

        More photos improve colour in these saved runs. That result does not establish the correct shape. Next, hold the colours fixed and test what wrong depth does when the camera moves.

        *Saved predictions · 500 steps · no training on selection.*"""),
    ])
    return


@app.cell
def _(mo):
    camera_shift = mo.ui.slider(-0.8, 0.8, step=0.05, value=0, show_value=True, label="Sideways camera shift (scene units)")
    mo.vstack([mo.md("""## 2. Why does wrong depth damage a new view?
    Start at zero shift. Move the camera right. Watch the marked stripe point against the white guide. The circle marks the start camera; the square marks the moving camera."""), camera_shift])
    return (camera_shift,)


@app.cell
def _(
    STRIPE_POINT,
    baseline,
    camera_shift,
    focal,
    mo,
    reference_rgb,
    shear_patch,
    truth,
):
    _x, _y = STRIPE_POINT
    _offset = focal * camera_shift.value * (1 / truth[_y, _x] - 1 / baseline[_y, _x])
    _observation = ("At zero shift, the two colour patches line up despite different depths." if camera_shift.value == 0 else
                    f"The marked point is **{abs(_offset):.1f} source pixels too {'far right' if _offset > 0 else 'far left'}** with the model depth.")
    mo.vstack([
        mo.image(shear_patch(reference_rgb, truth, baseline, focal, camera_shift.value), alt="Camera path and two projected stripe patches with a fixed white position guide."),
        mo.md(f"""**{_observation}** Its true depth is **{truth[_y, _x]:.2f}** units; the baseline puts it at **{baseline[_y, _x]:.2f}**. A point that is too far away moves too little when the camera shifts. Uneven depths also make the grid uneven.

        Both panels use the same colours. Only depth changes. The white guide marks where the point should be. Next, inspect the surface that caused the error.

        *Live calculation.*"""),
    ])
    return


@app.cell
def _(mo):
    region = mo.ui.radio(["Centre", "Near the rim", "Full width"], value="Centre", inline=True, label="Sphere strip")
    turn = mo.ui.slider(-65, 65, step=5, value=25, show_value=True, label="Rotate both surfaces (degrees)")
    mo.vstack([mo.md("""## 3. Where did the baseline put the surface?
    Inspect the red sphere. Orange marks the same strip on both surfaces. Select **Centre**, then **Near the rim**, then **Full width**. Use the curve to distinguish false bumps from real curvature."""), region, turn])
    return region, turn


@app.cell
def _(
    baseline,
    depth_profile,
    focal,
    ids,
    mo,
    np,
    region,
    surface_inspector,
    truth,
    turn,
):
    _left, _right = {"Centre": (55, 75), "Near the rim": (90, 110), "Full width": (25, 110)}[region.value]
    _row = 75
    _true, _learned = truth[_row, _left:_right+1], baseline[_row, _left:_right+1]
    _limits = (min(_true.min(), _learned.min())-.03, max(_true.max(), _learned.max())+.03)
    _lesson = {
        "Centre": "The true centre is nearly flat. The baseline adds false bumps and puts this strip too far away.",
        "Near the rim": "The true depth rises steeply as the sphere curves away. A smooth surface need not have a flat depth curve.",
        "Full width": "The true surface forms a bowl in this plot. The baseline keeps an overall depth offset and adds local bumps.",
    }[region.value]
    mo.vstack([
        mo.image(surface_inspector(truth, baseline, ids, focal, turn.value, _row, _left, _right), alt="True and baseline red sphere surfaces with the same strip marked in orange."),
        mo.image(depth_profile(truth, baseline, ids == 1, _row, _left, _right, limits=_limits), alt=f"True and baseline depth curves for {region.value.lower()}."),
        mo.md(f"""**{_lesson}** The mean depth error along this strip is **{np.mean(_learned-_true):+.3f} units**. Positive means too far away.

        Can smoothing remove the false bumps while preserving the bowl shape? Test that idea with simulated readings.

        *Saved prediction · live rotation and strip selection.*"""),
    ])
    return


@app.cell
def _(mo):
    fit_weight = mo.ui.slider(steps=[0, 1, 4, 16, 64, 256], value=4, show_value=True, label="Curve smoothing")
    mo.vstack([mo.md("""## 4. Can smoothing remove false bumps?
    **Simulated one-dimensional example.** Both panels use the same true red-sphere curve and the same noisy readings. Random errors move adjacent readings in different directions. An exact fit follows those errors and creates false bumps.

    Compare smoothing **0**, **4**, and **256**. Watch the centre and both sides. Moderate smoothing lets the fit pass between noisy readings. Too much smoothing pulls real depth changes towards a flat line."""), fit_weight])
    return (fit_weight,)


@app.cell
def _(curve_fit_figure, fit_weight, mo, np, solve_depth, truth):
    _true = truth[75, 27:109]
    _readings = _true + np.random.default_rng(42).normal(0, .06, len(_true))
    _fit = solve_depth(_readings, fit_weight.value)
    _before, _after = [np.sqrt(np.mean((curve-_true)**2)) for curve in (_readings, _fit)]
    _result = ("At zero, the fits coincide." if fit_weight.value == 0 else
               "Moderate smoothing removes many false bumps and keeps the bowl shape." if fit_weight.value <= 4 else
               "Stronger smoothing flattens the sides and changes the real shape.")
    mo.vstack([
        mo.image(curve_fit_figure(_true, _readings, _fit), alt="Simulated one-dimensional fits to identical noisy sphere readings on identical axes."),
        mo.md(f"""**{_result}** Depth RMSE: **{_before:.3f} → {_after:.3f} units**.

        Next, apply the same idea to a patch of depth values.

        *Live calculation · simulated readings.*"""),
    ])
    return


@app.cell
def _(mo):
    move = mo.ui.radio(["Marked pixel", "Whole patch"], value="Marked pixel", inline=True, label="Move")
    distance = mo.ui.slider(0, .30, step=.01, value=.10, show_value=True, label="Distance away from camera (scene units)")
    mo.vstack([mo.md("""## 5. What does the depth loss detect?
    This **8 × 8 patch** comes from the sphere centre. Move the marked interior pixel by **0.10 units**. Read the loss change at the top and follow the four orange pairs. Then select **Whole patch** without changing the distance."""), move, distance])
    return distance, move


@app.cell
def _(distance, mo, move, patch_experiment, truth):
    _whole = move.value == "Whole patch"
    _image, _count, _error = patch_experiment(truth[71:79, 62:70], distance.value, _whole)
    _observation = ("At zero, the profiles coincide and the loss is unchanged." if distance.value == 0 else
                    "Every depth moved by the same amount. All neighbour differences and the loss stay unchanged." if _whole else
                    "The bump changes four neighbour differences. Their squared differences increase the loss.")
    mo.vstack([
        mo.image(_image, alt="Depth patch with prominent original loss, changed loss, loss change, four neighbour contributions, and a depth profile."),
        mo.md(f"""**{_observation}** Changed neighbour pairs: **{_count}**. Mean absolute depth error: **{_error:.4f} units**.

        **Smoothing can discourage a bump. It cannot determine absolute distance.** Colour observations must still constrain the surface position.

        *Live calculation.*"""),
        mo.accordion({"Exact depth loss used by this trainer": mo.md(r"""
        For one 8 × 8 patch, the trainer uses 49 anchor pixels with a right and lower neighbour:

        $$L_{\mathrm{patch}}=\frac{1}{49}\sum_{i=0}^{6}\sum_{j=0}^{6}\operatorname{stopgrad}(A_{ij})[(d_{ij}-d_{i,j+1})^2+(d_{ij}-d_{i+1,j})^2].$$

        Here $A=1$. The trainer uses predicted opacity $A$ and predicted depth from new camera patches. It averages over the patches. Depth is the opacity-normalized weighted sample distance, $d=\sum_k w_k t_k/\max(\sum_k w_k,\epsilon)$, clipped to the ray bounds. The displayed patch calculation uses the same neighbour loss.

        With smoothing on, $L=L_{\mathrm{RGB}}+0.1L_{\mathrm{patch,coarse}}+\lambda L_{\mathrm{patch,fine}}$. With smoothing off, both depth terms are disabled. The trainer needs no true depth. True depth is used here for explanation and evaluation.
        """)}),
        mo.md("Now train with this penalty. Does it improve the picture from a new camera?"),
    ])
    return


@app.cell
def _(mo):
    camera_view = mo.ui.radio(["Lower view", "Front view"], value="Lower view", inline=True, label="New camera")
    stage = mo.ui.slider(steps=[100, 300, 1000], value=300, show_value=True, label="Saved training steps")
    settings = mo.md("{weight}\n\n{steps}").batch(
        weight=mo.ui.slider(start=0, stop=.10, step=.01, value=.05, label="Fine depth weight", show_value=True, include_input=False, full_width=True),
        steps=mo.ui.number(start=100, stop=1000, step=100, value=300, label="Training steps"),
    ).form(submit_button_label="Run comparison")
    mo.vstack([
        mo.md("""## 6. Does training with smoothing improve the new view?
        Both models see the same three photos. One also uses depth smoothing. In the lower view, look for the curved yellow stripe. Switch to the front view to compare. Does the clearer picture also have a better shape?"""),
        camera_view,
        mo.hstack([
            mo.vstack([mo.md("""### Saved predictions
            Start at 300 steps, then compare 100 and 1,000. Watch how the result changes as training continues.

            *Saved predictions · no training on selection.*"""), stage]),
            mo.vstack([mo.md("""### Optional new training
            Try a smoothing weight of **0.05 at 300 steps**. The recorded 300-step GPU runs took **8–11 seconds per model**. **CPU estimate: about 20 minutes per model with four threads.** A new comparison can train two models. Saved matches load immediately."""), settings,
                mo.md("Weight 0 turns smoothing off. Each run uses seed **42**. New training needs PyTorch. Training starts only when you select **Run comparison**.")]),
        ], widths=[1, 1], align="start", gap=2, wrap=True),
    ])
    return camera_view, settings, stage


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
        return {"Source": source, "Steps": result["steps"], "Seed": result["seed"], "Fine weight": result["fine_weight"],
                "Foreground RGB PSNR (dB)": round(result["foreground_psnr"], 2),
                "Surface depth MAE (units)": round(result["depth_mae"], 3)}

    return experiment_scene, saved, score_row


@app.cell
def _(camera_view, comparison, experiment_scene, mo, saved, stage):
    _pair = [(p, r) for p, r in saved if r["steps"] == stage.value]
    _off, _on = [r for _, r in _pair]
    _delta = _on["foreground_psnr"] - _off["foreground_psnr"]
    _observation = {100: "Both RGB images are still blurred. Smoothing slightly lowers the image score and increases depth error.",
                    300: ("Without smoothing, most of the curved yellow stripe is missing. With smoothing, the stripe is visible but blurred." if camera_view.value == "Lower view" else "With smoothing, the yellow stripe is more distinct. The image score improves and depth error falls."),
                    1000: "Both stripes are clearer than at 300 steps. Smoothing improves the scores, but the images still differ from the reference."}[stage.value]
    mo.vstack([
        mo.md("**RGB above · depth below · same depth scale in all three columns**"),
        mo.image(comparison(experiment_scene, [("Smoothing off", _pair[0][0]), ("Smoothing on", _pair[1][0])]), alt=f"{camera_view.value}: reference, smoothing off, and smoothing on at {stage.value} steps; RGB above depth."),
        mo.md(f"""**{_observation}** Foreground RGB PSNR: **{_off['foreground_psnr']:.2f} → {_on['foreground_psnr']:.2f} dB ({_delta:+.2f} dB)**. Surface depth MAE: **{_off['depth_mae']:.3f} → {_on['depth_mae']:.3f} units**.

        Higher PSNR means a closer colour match. Lower MAE means closer surface depth. These scores describe the spheres in the selected view.

        **Result:** smoothing helps at 300 and 1,000 steps in these saved runs. It does not help at 100 steps. The result can change with the view, training settings, or scene."""),
    ])
    return


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
):
    from importlib.util import find_spec
    from notebooks.toy_experiment import run_condition
    _cache = scene.parents[2] / "checkpoint" / "toy_two_spheres_student_v3"
    _history = list(saved)
    for _base_file in sorted(_cache.glob("*/result.json")):
        _file = _base_file.parent / "lower" / "result.json" if camera_view.value == "Lower view" else _base_file
        if _file.is_file():
            _history.append((_file.parent, json.loads(_file.read_text())))
    _panel = mo.md("Submit the optional form to compare a custom weight. The saved results below are ready now.")
    if settings.value is not None:
        _steps, _weight = int(settings.value["steps"]), float(settings.value["weight"])
        _needs_training = any(not any(r["steps"] == _steps and r.get("seed") == TOY_SEED and np.isclose(r["fine_weight"], _w, rtol=0, atol=1e-8)
                                     for _, r in _history) for _w in (0.0, _weight))
        mo.stop(_needs_training and find_spec("torch") is None, mo.vstack([
            mo.md("**New training needs PyTorch.** Install it in this Python environment with `python -m pip install torch`, then restart marimo. You can still explore every saved result."),
            mo.ui.table([score_row(r, "Saved" if p.parent == experiment_scene else "Custom") for p, r in _history], selection=None),
        ]))
        _models = []
        _sources = []
        for _w in dict.fromkeys([0.0, _weight]):
            _match = next(((p, r) for p, r in _history if r["steps"] == _steps and r.get("seed") == TOY_SEED and np.isclose(r["fine_weight"], _w, rtol=0, atol=1e-8)), None)
            if _match is None:
                _cache.mkdir(parents=True, exist_ok=True)
                _folder = Path(mkdtemp(prefix="run_", dir=_cache))
                with mo.status.spinner(title=f"Train {_steps} steps with fine weight {_w:g}"):
                    run_condition(_folder, _steps, _w)
                _folder = _folder / "lower" if camera_view.value == "Lower view" else _folder
                _result = json.loads((_folder / "result.json").read_text())
                _history.append((_folder, _result))
                _sources.append("Trained now")
            else:
                _folder, _result = _match
                _sources.append("Reused saved result")
            _models.append(("Smoothing off" if _w == 0 else f"Fine weight {_w:g}", _folder))
        _off = next(r for p, r in _history if p == _models[0][1])
        _custom = next(r for p, r in _history if p == _models[-1][1])
        _delta = _custom["foreground_psnr"] - _off["foreground_psnr"]
        _panel = mo.vstack([
            mo.md(f"**Custom comparison · {camera_view.value.lower()} · {_steps} steps · seed {TOY_SEED} · fine weight {_weight:g}.** {'; '.join(_sources)}. RGB score change from smoothing off: **{_delta:+.2f} dB**."),
            mo.image(comparison(experiment_scene, _models), alt=f"Custom comparison from the {camera_view.value.lower()}, with RGB above depth."),
            mo.md(f"Custom RGB PSNR: **{_custom['foreground_psnr']:.2f} dB**. Depth MAE: **{_custom['depth_mae']:.3f} units**. Compare the custom row with saved scores below."),
        ])
    mo.vstack([
        _panel,
        mo.accordion({"Experiment details": mo.md("""The camera and shape sections use the front view of the three-photo baseline at 500 steps. The final saved pairs compare depth weights 0 and 0.10 at matching step counts. Each camera has its own reference and scores. The lower view is an additional illustration chosen after inspecting the images; it is not an independent test of the method.

        Positive fine weights also enable a coarse depth weight of 0.10. Zero disables both depth terms. **Every custom run uses seed 42**, including the RGB sampling and model initialization. Depth sampling uses a separate generator with seed 43. All other training settings stay fixed. Submission reuses results only at the same seed, weight, and step count. Two fresh seed-42 runs matched exactly on this system. Different hardware or software can cause small numerical differences.

        RGB PSNR and depth MAE use the true foreground at evaluation scale 2. These are scores for the selected held-out toy camera, not training batches or the paper benchmark. Displayed scores are rounded; differences use the unrounded values. The depth display masks the background, so false background surfaces are visible only in RGB. The camera demonstration projects saved depths with fixed colours; it does not render a new NeRF view.

        Saved controls usually update in under one second after the first load. The CPU estimate uses a 12-step test with four threads and fine depth weight 0.05. The measured rate was 3.54 seconds per step, plus about 20 seconds per camera for rendering and scoring. Extrapolation gives 18 minutes for 300 steps, rounded to about 20 minutes. This is an estimate, not a full 300-step CPU measurement. CPU hardware, thread settings, and other work can change the time.

        The one-dimensional readings use fixed random noise. True depth supplies their error score, not the fitting data.
        """)}),
        mo.md(f"**Saved and custom scores · {camera_view.value.lower()} · seed {TOY_SEED}**"),
        mo.ui.table([score_row(r, "Saved" if p.parent == experiment_scene else "Custom") for p, r in _history], selection=None),
    ])
    return


@app.cell
def _(mo, scene):
    _static = scene.parents[1] / "toy_figures"
    mo.vstack([
        mo.accordion({"Static figures and runtime notes": mo.vstack([
            mo.md("""The HTML preview is static. Its controls do not calculate or train. Use these figures to compare fixed settings. Open `01_toy.py` in marimo to use the controls and edit the code. Each figure below uses the same calculations as the live lesson."""),
            *[mo.vstack([mo.md(f"**{title}**"), mo.image(str(_static / name), alt=title)]) for title, name in [
                ("Sparse inputs: 3, 9, and 20 photos at 500 steps", "coverage_500.png"),
                ("Camera shift: 0 and +0.8 units", "camera_shift.png"),
                ("Shape: Centre, Near the rim, and Full width", "shape_presets.png"),
                ("Simulated fits: smoothing 0, 4, and 256", "curve_settings.png"),
                ("One pixel moved by 0.10 units", "one_pixel.png"),
                ("Whole patch moved by 0.10 units", "whole_patch.png"),
                ("Saved training: both cameras at 100, 300, and 1,000 steps", "training_stages.png"),
            ]],
        ])}),
        mo.md("""Three photos can constrain colour while leaving depth wrong. Smoothing can improve a new view, but it cannot determine absolute distance. Next, test whether it also helps with photographs of a real object in notebook 2.

        **Outstanding questions:** Which saved step count contradicts “smoothing always helps”? Why can a patch have zero loss change and still have wrong depth?"""),
    ])
    return


if __name__ == "__main__":
    app.run()
