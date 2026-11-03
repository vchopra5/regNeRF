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
    from notebooks.toy_experiment import ASSETS as scene
    from notebooks.toy_figures import comparison, depth_profile, surface_inspector
    return Path, comparison, depth_profile, json, mkdtemp, mo, np, scene, surface_inspector


@app.cell
def _(mo, scene):
    mo.vstack([
        mo.md("""
        # RegNeRF: Regularizing Neural Radiance Fields for View Synthesis from Sparse Inputs
        *A visual lab on what three photos reveal about shape, and what a depth penalty can recover.*

        [Niemeyer et al. · CVPR 2022 · Read the paper on arXiv](https://arxiv.org/abs/2112.00724)
        """),
        mo.image(str(scene.parent/"regnerf_figure_1.png"), width=480,
                 alt="Paper Figure 1. Three input photos above the novel views produced by NeRF and full RegNeRF."),
        mo.md("*Figure 1.* Three input photos (top). Look at the face in the new views from NeRF (middle) and full RegNeRF (bottom).")
    ])
    return


@app.cell
def _(mo):
    photo_count = mo.ui.radio(["3 photos", "9 photos", "20 photos"], value="3 photos",
                              inline=True, label="Train with")
    mo.vstack([
        mo.md("""
        ---

        ## How many views does the baseline need?
        Keep the model fixed. Add photos and watch the yellow stripe from a new angle.
        """),
        photo_count,
    ])
    return (photo_count,)


@app.cell
def _(json, mo, photo_count, scene):
    count = int(photo_count.value.split()[0])
    folder = scene / "coverage_500" / str(count)
    score = json.loads((folder / "result.json").read_text())
    def render_panel(title, path, caption):
        picture = mo.image(str(path), width="100%", alt=title)
        return f"""<figure style="margin:0;min-width:0">
            <figcaption style="margin-bottom:.4rem"><strong>{title}</strong></figcaption>
            <div style="background:white;border:1px solid #a4adb64d;border-radius:6px;overflow:hidden;line-height:0">{picture.text}</div>
            <small>{caption}</small></figure>"""

    _reference = render_panel("Reference", scene/"reference.png", "Held-out camera")
    _prediction = render_panel("Prediction", folder/"heldout.png",
                               f"Foreground PSNR <b>{score['foreground_psnr']:.2f} dB</b>")
    _inputs = mo.image(str(folder/"inputs.png"), width=min(count, 5)*77,
                       alt=f"All {count} training views")
    mo.Html(f"""<section style="max-width:600px">
        <div style="display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px">{_reference}{_prediction}</div>
        <div style="margin-top:1rem;padding-top:.65rem;border-top:1px solid #a4adb64d">
            <strong>Training photos ({count})</strong>
            <div style="max-width:100%;line-height:0">{_inputs.text}</div>
        </div>
        <small>{score['steps']:,} steps · Smoothing off · Cached, under 1 s</small>
    </section>""")
    return


@app.cell
def _(mo):
    camera_shift = mo.ui.slider(-0.8, 0.8, step=0.05, value=0, show_value=True,
                                label="Sideways camera shift (scene units)")
    mo.vstack([mo.md("## Why does depth matter when the camera moves?\nStart at zero, then move the camera to the right. Watch where the yellow stripe crosses the grid. Both close-ups use exactly the same paint colours."), camera_shift])
    return (camera_shift,)


@app.cell
def _(camera_shift, json, mo, np, photo_count, scene):
    from PIL import Image as _Image
    from notebooks.toy_figures import STRIPE_POINT, shear_patch
    with np.load(scene/"reference.npz") as _ref:
        _truth = _ref["depth"]
    with np.load(scene/"coverage_500"/photo_count.value.split()[0]/"prediction.npz") as _model:
        _estimated = _model["depth"]
    _rgb = np.asarray(_Image.open(scene/"reference.png").convert("RGB"))
    _angle = json.loads((scene/"transforms_train.json").read_text())["camera_angle_x"]
    _focal = _truth.shape[1] / (2*np.tan(_angle/2))
    _shift = camera_shift.value
    _x, _y = STRIPE_POINT
    _offset = _focal*_shift*(1/_truth[_y,_x] - 1/_estimated[_y,_x])
    _result = (f"The marked spot on the yellow stripe is **{abs(_offset):.1f} pixels too "
               f"{'far right' if _offset > 0 else 'far left'}** in the model image."
               if _shift else "The two stripes line up from this camera position. Move the camera right to see the error.")
    mo.vstack([
        mo.image(shear_patch(_rgb,_truth,_estimated,_focal,_shift),alt="A camera slides sideways under a fixed point on the striped sphere. Two close-ups show how the same coloured patch moves with true and learned depths."),
        mo.md(f"{_result} The model places that spot at depth **{_estimated[_y,_x]:.2f}**. Its true depth is **{_truth[_y,_x]:.2f}**."),
        mo.md("**Wrong depth distorts the rendered image when you change the viewpoint.** Parts of an object end up in the wrong places, so its shape or texture can look wrong. Here you can see it in the texture: the yellow stripe bends and appears to slide across the red sphere."),
    ])
    return


@app.cell
def _(mo):
    mo.md("""## Where did the model put the sphere?
    The stripe bent when the camera moved. Look at the depth the baseline learned from three photos. The orange strip marks the same pixels on the true surface and on the model surface. Start at the centre, then follow the curve towards the rim.""")
    return


@app.cell
def _(mo):
    region = mo.ui.radio(["Centre", "Near the rim", "Full width"], value="Centre",
                         inline=True, label="Part of the sphere")
    turn = mo.ui.slider(-65, 65, step=5, value=25, show_value=True, label="Rotate both surfaces (°)")
    mo.vstack([region, turn])
    return region, turn


@app.cell
def _(np, scene):
    with np.load(scene.parent/"sphere/profile33.npz") as data:
        truth, ids = data["truth"], data["ids"]
        baseline = data["predictions"][0]
    return baseline, ids, truth


@app.cell
def _(baseline, depth_profile, ids, json, mo, np, region, scene, surface_inspector, truth, turn):
    _left, _right = {"Centre": (90,110), "Near the rim": (140,159), "Full width": (40,159)}[region.value]
    _row = 75
    _strip = np.s_[_row, _left:_right+1]
    _true, _learned = truth[_strip], baseline[_strip]
    _limits = (min(_true.min(), _learned.min())-.03, max(_true.max(), _learned.max())+.03)
    _fov = json.loads((scene.parent/"sphere/transforms_test.json").read_text())["camera_angle_x"]
    _focal = truth.shape[1] / (2*np.tan(_fov/2))
    _readout = (f"True depth ranges from **{_true.min():.3f} to {_true.max():.3f}** units. "
                f"The baseline predicts **{_learned.min():.3f} to {_learned.max():.3f}**.")
    _lesson = {"Centre": "The model puts the centre too far away and adds bumps to a nearly flat strip.",
               "Near the rim": "Here the black line climbs. Even on a smooth surface, depth can change quickly where the sphere curves away.",
               "Full width": "Both ends of the true profile are farther from the camera than the centre. The bottom of the curve contains the centre strip."}[region.value]
    mo.vstack([
        mo.image(surface_inspector(truth,baseline,ids,_focal,turn.value,_row,_left,_right), alt="True and learned surfaces with the same pixels marked in orange."),
        mo.image(depth_profile(truth,baseline,ids,_row,_left,_right,limits=_limits), alt="Depth along the marked row, from left to right."),
        mo.md(_readout + " " + _lesson),
        mo.md("*Saved predictions, 300 steps. Changing the strip or angle takes under a second.*"),
    ])
    return


@app.cell
def _(mo):
    fit_weight = mo.ui.slider(steps=[0, 1, 4, 16, 64, 256], value=4, show_value=True,
                              label="Curve smoothing")
    mo.vstack([mo.md("## Can we remove the bumps?\nLook near the centre of the left graph. The sphere is smooth, but one grey reading puts the surface too close and the next puts it farther away. The orange line fits both readings exactly, so it draws a bump that the sphere does not have. From a new camera angle, those two points would move by the wrong amounts, just like the stripe.\n\nIncrease the smoothing on the right. A sharp jump now costs more, so the fit can pass between the readings and still follow the curve."), fit_weight])
    return (fit_weight,)


@app.cell
def _(fit_weight, mo, np, truth):
    from notebooks.lab import solve_depth
    from notebooks.toy_figures import curve_fit_figure
    _true = truth[75, 43:157]
    _readings = _true + np.random.default_rng(42).normal(0, .06, len(_true))
    _fit = solve_depth(_readings, fit_weight.value)
    _before, _after = [np.sqrt(np.mean((curve-_true)**2)) for curve in (_readings, _fit)]
    _result = ("At zero, the two fits coincide." if fit_weight.value == 0 else
               "The fitted curve follows the bowl shape through the noisy readings." if fit_weight.value <= 4 else
               "The sides start to flatten as the penalty pulls neighbouring depths together.")
    mo.vstack([mo.image(curve_fit_figure(_true, _readings, _fit), alt="Two fits to the same noisy readings across the sphere, compared with the true depth on identical axes."),
               mo.md(f"{_result} Depth error (RMSE): **{_before:.3f}** units on the left and **{_after:.3f}** on the right."),
               mo.md("*This one-dimensional fit uses the simulated readings. The true curve supplies the error score. Each update takes under a second.*")])
    return


@app.cell
def _(mo):
    move = mo.ui.radio(["Marked pixel", "Whole patch"], value="Marked pixel", inline=True, label="Move")
    distance = mo.ui.slider(0, .30, step=.01, value=.10, show_value=True, label="Distance away from camera (scene units)")
    mo.vstack([mo.md("## What does one wrong depth cost?\nTake an 8 × 8 patch from the centre of the true sphere. Push the marked pixel away from the camera. Then push the whole patch by the same distance. Which change does a smoothness penalty detect?"), move, distance])
    return distance, move


@app.cell
def _(distance, mo, move, truth):
    from notebooks.toy_figures import patch_experiment
    _whole = move.value == "Whole patch"
    _image, _count, _error = patch_experiment(truth[71:79,96:104], distance.value, _whole)
    if distance.value == 0:
        _observation = "Both profiles coincide. Every pixel is at its true depth."
    elif _whole:
        _observation = "The patch is too far away. Its shape stays the same, so the smoothness loss stays the same."
    else:
        _observation = "The bump changes the distance to all four neighbours. Each change contributes to the loss."
    mo.vstack([mo.image(_image, alt="A marked pixel on an eight by eight patch and a side profile of its row. Both controls use the same depth scale."),
               mo.md(f"**{_count} neighbour differences changed.** Mean depth error across the patch is **{_error:.4f}** units. {_observation}"),
               mo.accordion({"How the trainer calculates this loss": mo.md(r"For each pixel with a right and lower neighbour, compute $A[(d-d_{\rm right})^2+(d-d_{\rm below})^2]$. Average the 49 sums. Here $A=1$ because the sphere is opaque. During training, $A$ is the predicted opacity. The trainer uses predicted depths from a new camera. It needs no true depth map.")}),
               mo.md("*Live calculation on a saved sphere patch. Each edit takes under a second.*")])
    return


@app.cell
def _(mo):
    stage = mo.ui.slider(steps=[100,300,1000],value=300,show_value=True,label="Saved training steps")
    settings = mo.md("{weight}\n\n{steps}").batch(
        weight=mo.ui.slider(start=0, stop=2, step=.01, value=.05, label="Fine depth weight",
                            show_value=True, include_input=False, full_width=True),
        steps=mo.ui.number(start=1, stop=20000, step=1, value=300, label="Training steps"),
    ).form(submit_button_label="Run comparison")
    _saved = mo.vstack([mo.md("### Explore saved runs\nCompare weights **0** and **0.1** at the same step. Each pair loads in under a second."), stage])
    _train = mo.vstack([mo.md("### Try your own weight\nTry **0.05** at 300 steps. Allow about 8 seconds on the lab GPU or 20 minutes on a four-thread CPU."),
                        settings, mo.accordion({"Depth weights": mo.md("Weight 0 disables smoothing. Positive weights use a coarse weight of 0.1.")})])
    mo.vstack([mo.md("## Does the penalty improve the new view?\nReturn to the stripe. Both models learn colour from the same three photos. One also learns from depth differences in patches it renders from new cameras. Compare the stripe first, then the surface below it."),
               mo.hstack([_saved, _train], widths=[1,1], align="start", gap=2, wrap=True)])
    return settings, stage


@app.cell
def _(json, scene):
    saved = []
    for _steps in (100,300,1000):
        for _name in ("off","on"):
            _folder = scene / (_name if _steps==300 else f"{_name}_{_steps}")
            saved.append((_folder,json.loads((_folder/"result.json").read_text())))
    def score_row(result, source):
        return {"Source":source,"Steps":result["steps"],"Fine weight":result["fine_weight"],
                "Foreground PSNR (dB) ↑":round(result["foreground_psnr"],2),
                "Surface depth MAE ↓":round(result["depth_mae"],3)}
    return saved, score_row


@app.cell
def _(mo, saved, scene, stage):
    _off,_on = [r for _,r in saved if r["steps"]==stage.value]
    _file = "comparison.png" if stage.value==300 else f"comparison_{stage.value}.png"
    mo.vstack([mo.image(str(scene/_file),alt="Reference, depth off, depth on. RGB above and depth below."),
               mo.md(f"At **{stage.value} steps**, smoothing {'reduces' if _on['depth_mae'] < _off['depth_mae'] else 'increases'} "
                     f"the mean surface depth error from **{_off['depth_mae']:.3f}** to **{_on['depth_mae']:.3f}** units. "
                     f"The image score (foreground PSNR) {'rises' if _on['foreground_psnr'] > _off['foreground_psnr'] else 'falls'} "
                     f"from **{_off['foreground_psnr']:.2f}** to **{_on['foreground_psnr']:.2f} dB**. Higher scores mean a closer colour match in this view.")])
    return


@app.cell
def _(Path, comparison, json, mkdtemp, mo, saved, scene, score_row, settings):
    # Check for a completed run before starting the trainer. Partial runs have no result file.
    from notebooks.toy_experiment import run_condition
    _cache = scene.parents[2]/"checkpoint"/"toy_two_spheres_student_v1"
    _history = list(saved)
    for _file in sorted(_cache.glob("*/result.json")):
        _history.append((_file.parent,json.loads(_file.read_text())))
    _panel = mo.md("")
    if settings.value is not None:
        _steps,_weight = int(settings.value["steps"]),float(settings.value["weight"])
        _folder = next((p for p,r in _history if (r["steps"],r["fine_weight"]) == (_steps,_weight)),None)
        _source = "Cached"
        if _folder is None:
            _cache.mkdir(parents=True,exist_ok=True)
            _folder = Path(mkdtemp(prefix="run_",dir=_cache))
            _result = run_condition(_folder,_steps,_weight)
            _history.append((_folder,_result))
            _source = "Trained now"
        _baseline = next((p for p,r in _history if (r["steps"],r["fine_weight"])==(_steps,0)),None)
        _models = ([("Depth off",_baseline)] if _baseline and _weight else []) + [(f"Fine weight {_weight:g}",_folder)]
        _hint = " Run weight 0 at this step count to add a baseline." if _baseline is None else ""
        _panel = mo.vstack([mo.md(f"**{_source} · {_steps} steps · fine depth weight {_weight:g}.**{_hint}"),
                            mo.image(comparison(scene,_models))])
    _history_table = mo.ui.table([score_row(r,"Saved" if p.parent==scene else p.name) for p,r in _history],selection=None)
    mo.vstack([_panel, mo.accordion({"Compare saved runs": _history_table})])
    return


if __name__ == "__main__":
    app.run()
