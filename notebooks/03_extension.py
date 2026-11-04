import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="When neighbours have different depths")


@app.cell
def _():
    import json

    import marimo as mo
    import numpy as np
    from PIL import Image

    from notebooks.edge_lesson import pair_figure, pipeline_figure, profile_experiment, ramp_figure
    from notebooks.lab import ROOT
    from notebooks.scan40_figures import per_view_change, small
    from notebooks.toy_figures import SCALE

    results = ROOT / "notebooks/results"
    evidence = results / "partner_stone"
    data = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
    means = data["means_db"]
    # Second training run of the same method (RTX 4070), with its weight pilot and edge metrics.
    _second_file = results / "scan40_edge.json"
    second = json.loads(_second_file.read_text(encoding="utf-8")) if _second_file.is_file() else None

    def show(image, caption=None):
        """Show a figure at its display size, at most 900 px wide."""
        return mo.image(image, width=min(image.width // SCALE, 900), caption=caption)

    def panel(tag, kind):
        return show(Image.open(evidence / f"{tag}_{kind}.png"))

    def result(text, kind="info", title="What this shows"):
        return mo.callout(mo.md(text), kind=kind, title=title)

    def stat(value, label, caption=None):
        return mo.stat(value, label=label, caption=caption, bordered=True)

    def missing(what):
        return mo.callout(mo.md(f"**{what} is not in this copy.** Restore `notebooks/results/scan40_edge.json` "
                                "from the complete repository. No training is needed."), kind="warn")

    return (
        ROOT,
        data,
        json,
        means,
        missing,
        mo,
        np,
        pair_figure,
        panel,
        per_view_change,
        pipeline_figure,
        profile_experiment,
        ramp_figure,
        result,
        second,
        show,
        small,
        stat,
    )


@app.cell
def _(ROOT, mo, small):
    _photo = ROOT / "notebooks/results/scan40_reference/image/000025.png"
    _intro = mo.md(
        """
        **The question.** Two pixels can sit next to each other in a photo but belong to different
        surfaces. RegNeRF's depth smoothing pulls their depths together. Notebook 2 showed that the
        smoothing helps on the stone arch. What does it do at the edge of an object, and can we fix it?

        **Our change.** Keep RegNeRF's smoothing and add a second depth term. It uses absolute (L1)
        differences and becomes weaker where the rendered colour changes. The colour edges come from
        a blur and a Sobel filter, the first two steps of the Canny detector. No thresholds and no
        learned parameters.

        **Prediction.** If the term works, the arch keeps its colour score or improves, and depth
        edges stay sharper. This can fail: stone texture is also a colour edge.

        **How to use this page.** Sections 1 to 4 are small experiments with known depth. Change the
        controls and read the box beside each figure. Sections 5 to 9 use saved results from the arch.
        Section 7 can render new camera views from the saved models. Nothing trains. Allow about
        35 minutes.

        1. Why does smoothing pull an edge together?
        2. Why does L2 blur a depth step?
        3. Can colour tell us where to stop smoothing?
        4. How does RegNeRF + edge L1 work?
        5. How was the weight chosen?
        6. Does it help the stone arch?
        7. Move around the arch
        8. The score over 25 test views
        9. Does the result repeat?
        """
    )
    mo.vstack([
        mo.md("# When neighbours have different depths"),
        mo.hstack(
            [mo.image(small(_photo, 400), width=380, caption="Scan 40, training view 25") if _photo.is_file()
             else mo.md(""), _intro],
            widths=[1, 1.4], align="start", gap=2, wrap=True,
        ),
    ])
    return


@app.cell
def _(mo):
    pair_strength = mo.ui.slider(0, 8, step=0.5, value=2, show_value=True, label="Smoothing strength")
    connected = mo.ui.checkbox(value=True, label="Smooth across the boundary")
    mo.vstack([
        mo.md(
            """
            ---
            ## 1. Why does smoothing pull an edge together?
            Two neighbouring pixels see different surfaces, at depths 2 and 4. **Change** the smoothing
            strength, then clear **Smooth across the boundary**. **Watch** the gap between the two dots.
            """
        ),
        mo.hstack([pair_strength, connected], justify="start", gap=3, wrap=True),
    ])
    return connected, pair_strength


@app.cell
def _(connected, mo, pair_figure, pair_strength, result, show, stat):
    _image, _gap = pair_figure(pair_strength.value, connected.value)
    _pulled = connected.value and pair_strength.value > 0
    mo.vstack([
        mo.hstack(
            [
                show(_image),
                mo.vstack([
                    mo.hstack([stat("2.00", "True depth gap"), stat(f"{_gap:.2f}", "Fitted gap")], widths="equal"),
                    result(("Smoothing pulls the two surfaces closer than they are. "
                            if _pulled else "The fitted depths match the two surfaces. ")
                           + "Without the link, each pixel keeps its own depth.\n\n"
                           "**Smoothing across a boundary moves both surfaces toward each other.**"),
                ]),
            ],
            widths=[2, 1], align="start", gap=1.5, wrap=True,
        ),
        mo.accordion({"What this experiment solves": mo.md(r"""
            Two exact depth observations, 2 and 4. Fit them with
            $(d_1-2)^2+(d_2-4)^2+\lambda w(d_1-d_2)^2$.
            The checkbox sets $w$ to 1 or 0. The fitted gap is $2/(1+2\lambda w)$.
            This small calculation balances the observed depths against the smoothing penalty.
            """)}),
    ])
    return


@app.cell
def _(mo):
    ramp_span = mo.ui.slider(steps=[1, 3, 5, 7], value=7, show_value=True,
                             label="Spread the change over this many pairs")
    mo.vstack([
        mo.md(
            """
            ---
            ## 2. Why does L2 blur a depth step?
            L1 adds the absolute depth differences. L2 adds their squares. **Change** how many pixel pairs
            share the same depth change. **Watch** the two costs below the plots.
            """
        ),
        ramp_span,
    ])
    return (ramp_span,)


@app.cell
def _(mo, ramp_figure, ramp_span, result, show):
    mo.vstack([
        mo.hstack(
            [
                show(ramp_figure(ramp_span.value)),
                result("Spreading the step over more pairs lowers the L2 cost. The L1 cost stays the same.\n\n"
                       "**L2 favours a gradual change. L1 allows a sharp step, but does not say where.**"),
            ],
            widths=[2, 1], align="start", gap=1.5, wrap=True,
        ),
        mo.accordion({"The two costs": mo.md(r"""
            For a total change $A$ over $k$ equal differences:
            L2 costs $k(A/k)^2=A^2/k$; L1 costs $k|A/k|=|A|$.
            Equal depths have zero cost for both. These figures compare the penalty alone,
            with fixed endpoints. The data loss is not included.
            """)}),
    ])
    return


@app.cell
def _(mo):
    penalty = mo.ui.radio(["L2", "L1"], value="L2", inline=True, label="Penalty")
    strength = mo.ui.slider(steps=[0, 0.1, 0.5, 1, 2, 4, 8], value=2, show_value=True, label="Strength")
    case = mo.ui.dropdown(["Aligned edge", "Texture edge", "Missing colour edge"],
                          value="Aligned edge", label="Colour guide")
    mo.vstack([
        mo.md(
            """
            ---
            ## 3. Can colour tell us where to stop smoothing?
            One line of pixels with a known depth step. The left plot smooths every pair equally. The right
            plot smooths less where the colour changes. **Change** the penalty and the colour guide.
            **Watch** the depth step and the flat surface. The observations stay the same.
            """
        ),
        mo.hstack([penalty, strength, case], justify="start", gap=3, wrap=True),
    ])
    return case, penalty, strength


@app.cell
def _(case, mo, penalty, profile_experiment, result, show, strength):
    _image, _rows = profile_experiment(case.value, penalty.value, strength.value)
    _delta = _rows[1]["Boundary RMSE"] - _rows[0]["Boundary RMSE"]
    _flat_delta = _rows[1]["Flat RMSE"] - _rows[0]["Flat RMSE"]
    _note = {
        "Aligned edge": "Here the colour change lines up with the depth step.",
        "Texture edge": "The extra colour stripe lies on a flat surface. It reduces smoothing where the depth "
                        "should stay flat.",
        "Missing colour edge": "The colour stays constant across the depth step. Both plots use the same weights "
                               "and give the same fit.",
    }[case.value]
    mo.vstack([
        mo.hstack(
            [
                show(_image),
                result(f"{_note}\n\nColour weights change the depth error (RMSE) by **{_delta:+.3f}** at the step "
                       f"and **{_flat_delta:+.3f}** on the flat surface. Negative means less error.\n\n"
                       "**Colour helps only where colour edges and depth edges agree.**"),
            ],
            widths=[2, 1], align="start", gap=1.5, wrap=True,
        ),
        mo.accordion({"Simulation details": mo.md(r"""
            Both panels minimize the same squared data error plus $\lambda\sum w\,\phi(\Delta d)$.
            Choose $\phi(z)=z^2$ or $|z|$. The observations and noise seed stay fixed.
            The simple teaching weight is $\exp(-10|\Delta\mathrm{colour}|)$.
            Same coefficient does not mean equal effective strength for L1 and L2.

            Boundary pixels are within 3.5 pixels of the true step. Flat pixels are at least
            12.5 pixels away. Reference depth is used only for scoring.
            This simulation replaces the smoothing penalty. RegNeRF + edge L1 below instead
            keeps the original L2 term and adds a weighted L1 term.
            """)}),
    ])
    return


@app.cell
def _(mo, pipeline_figure, result, show):
    mo.vstack([
        mo.md(
            """
            ---
            ## 4. How does RegNeRF + edge L1 work?
            We keep RegNeRF's L2 smoothing and add an L1 penalty on the fine depth. The new penalty
            becomes weaker where the model's rendered colour changes. **Watch** where the colour enters:
            only as a weight, never as a target.
            """
        ),
        mo.hstack(
            [
                show(pipeline_figure()),
                result("The added L1 term allows more depth change at colour edges. The original L2 term still "
                       "smooths those edges.\n\n"
                       "**A colour edge permits a depth step. It does not force one.**"),
            ],
            widths=[2, 1], align="center", gap=1.5, wrap=True,
        ),
        mo.accordion({"Equations and filter settings": mo.md(r"""
            $\Delta_xd$ and $\Delta_yd$ are right and lower depth differences.
            $a$ is anchor opacity, with gradients stopped.
            $$S_2=\operatorname{mean}[a((\Delta_xd)^2+(\Delta_yd)^2)]$$
            $$S_1=\operatorname{mean}[a(w_x|\Delta_xd|+w_y|\Delta_yd|)]$$
            $$L_{\mathrm{depth}}=0.1S_{2,\mathrm{coarse}}+\lambda_2(t)S_{2,\mathrm{fine}}+\lambda S_{1,\mathrm{fine}}.$$

            The fine L2 weight $\lambda_2(t)$ falls from 400 to 0.1 over the first 512 steps.
            The coarse L2 coefficient stays at 0.1. The edge weight $\lambda$ is 0.1 in both saved runs;
            section 5 explains why.

            Convert RGB to grey with coefficients 0.299, 0.587, 0.114.
            Apply a 3 × 3 Gaussian with sigma 1, then Sobel filters.
            For gradient magnitude $g$, use $w_{ij}=\exp(-\max(g_i,g_j)/s)$.
            The scale $s$ is the 90th percentile over the three training photos.
            A two-pixel halo supplies the filter's context.

            Guide rendering uses evaluation mode and no gradients. The weights are recomputed
            each step but receive no gradient through this loss. RGB supervision stays active.
            Virtual patches use a separate random generator. The code adds RGB and depth gradients,
            clips them, then updates the network with Adam.

            The loss is the edge-aware smoothness of self-supervised monocular depth (Godard et al., 2017).
            The new part is its use on RegNeRF's unseen-camera patches, with the rendered colour as the guide.
            """)}),
    ])
    return


@app.cell
def _(missing, mo, result, second):
    if second is None or not second.get("pilot"):
        _body = missing("The weight pilot")
    else:
        _pilot = second["pilot"]
        _scores = _pilot["scores"]
        _base = _scores["0"]["mean_db"]
        _rows = [{"λ": "0 (RegNeRF)" if key == "0" else key,
                  "Validation mean (dB)": round(value["mean_db"], 3),
                  "Change from RegNeRF (dB)": "" if key == "0" else f"{value['mean_db'] - _base:+.2f}",
                  **{f"View {view}": round(db, 2) for view, db in value["per_view_db"].items()}}
                 for key, value in _scores.items()]
        _views = ", ".join(str(v) for v in _pilot["validation_views"])
        _best = _pilot["chosen_lambda"]
        _body = mo.hstack(
            [
                mo.ui.table(_rows, selection=None, show_column_summaries=False),
                result(
                    f"The pilot chose **λ = {_best:g}**, the same value the first run fixed in advance. "
                    f"λ = 1 lost {_base - _scores['1']['mean_db']:.1f} dB; λ = 0.01 gained less than λ = 0.1.\n\n"
                    "**λ = 0.1 is a reasonable weight, confirmed without any test view.**",
                ),
            ],
            widths=[1.6, 1], align="start", gap=1.5, wrap=True,
        )
    mo.vstack([
        mo.md(
            """
            ---
            ## 5. How was the weight chosen?
            The edge term has one setting: its weight **λ** (also called alpha). Too small, and the term
            does nothing. Too large, and it overrides the colour and smoothing losses. The two saved runs
            set λ in two ways:

            - **First run (fixed).** λ = 0.1 was set before training, from the repository example. No result
              was used to choose it. This keeps the test views clean, but gives no evidence that 0.1 is a
              good value.
            - **Second run (pilot).** Before the main run, three short runs (5,000 steps each) tried
              λ = 0.01, 0.1, and 1. Each was scored on views 40, 44, and 48. These are in the DTU training
              split, but the 3-view models never train on them, and they are not test views. The rule was
              fixed in advance: highest mean validation PSNR; a tie within 0.01 dB goes to the smaller λ.
              The saved RegNeRF model at step 5,000 is the λ = 0 reference.

            **Watch** whether one value stands out.
            """
        ),
        _body,
    ])
    return


@app.cell
def _(data, mo):
    view_choice = mo.ui.dropdown(
        {f"View {row['view']}": f"view{row['view']}" for row in data["per_view"]},
        value="View 24", label="Saved test camera")
    mo.vstack([
        mo.md(
            """
            ---
            ## 6. Does it help the stone arch?
            Both models trained on the same three photos of DTU scan 40 (first run). These 25 saved
            comparisons open without rendering. **Change** the test view, then switch between **Depth**
            and **Colour**. **Watch** the opening, the pillar edges, and the top block.
            """
        ),
        view_choice,
    ])
    return (view_choice,)


@app.cell
def _(data, mo, panel, result, view_choice):
    _notes = {
        "view24": "The added term reduces the vertical mark inside the opening. Both depth maps still have uneven "
                  "surfaces.",
        "view10": "Both models distort the top block. RegNeRF + edge L1 has a lower colour score in this view.",
        "view43": "The vertical mark is reduced, but the depth boundary around the opening remains blurred.",
    }
    _score = next(row for row in data["per_view"] if f"view{row['view']}" == view_choice.value)
    mo.vstack([
        mo.ui.tabs({"Depth": panel(view_choice.value, "depth"), "Colour": panel(view_choice.value, "rgb")}),
        result(f"Foreground RGB PSNR: RegNeRF **{_score['regnerf_db']:.2f} dB**; "
               f"RegNeRF + edge L1 **{_score['partner_db']:.2f} dB**; "
               f"difference **{_score['partner_minus_regnerf_db']:+.2f} dB**.\n\n"
               + _notes.get(view_choice.value, "Compare the opening, pillar edges, and top block.")
               + " The left panel is a real photo. We have no reference depth for this scan, so a smoother "
               "depth map alone cannot tell us which shape is more accurate.",
               kind="neutral", title="Look and compare"),
    ])
    return


@app.cell
def _(ROOT, json, mo):
    orbit_settings = (
        mo.md("{horizontal}\n\n{vertical}")
        .batch(
            horizontal=mo.ui.slider(-180, 180, step=1, value=0, show_value=True,
                                    label="Turn left or right (degrees)", full_width=True),
            vertical=mo.ui.slider(-90, 90, step=1, value=0, show_value=True,
                                  label="Turn up or down (degrees)", full_width=True),
        )
        .form(submit_button_label="Render view")
    )
    _timing_path = ROOT / "notebooks/results/camera_timing.json"
    _timing = {}
    if _timing_path.is_file():
        _record = json.loads(_timing_path.read_text(encoding="utf-8"))
        _rows = []
        for _device, _label in (("cpu", "CPU"), ("cuda", "GPU")):
            if _device in _record["devices"]:
                _times = _record["devices"][_device]["median_seconds"]
                _rows.append(f"| {_label} | {_times['first_request']:.1f} s | "
                             f"{_times['new_angle']:.1f} s | {_times['repeat_angle']:.2f} s |")
        _timing = {"How long does a render take?": mo.md(
            "Notebook 2 measured these median times over three trials at 100 × 75 pixels per image. "
            "Notebook 3 uses different models and also builds depth images, so its times can differ.\n\n"
            "| Device | First render | New angle | Repeated angle |\n"
            "| --- | ---: | ---: | ---: |\n" + "\n".join(_rows) + "\n\n"
            "**First render:** start and moved positions for both models. **New angle:** the start images "
            "are reused. **Repeated angle:** all images are reused.\n\n"
            f"Measured on {_record['cpu']} with {_record['cpu_threads']} CPU threads, and {_record['gpu']}. "
            "Your computer and other running tasks can change the time.")}
    mo.vstack([
        mo.md(
            """
            ---
            ## 7. Move around the arch
            Turn the camera around the arch, or look from above or below. Select **Render view** to draw both
            first-run models from that position. The top row stays at camera 25; the bottom row follows your
            settings. Nothing trains: the page renders new 100 × 75 images from the saved weights.

            **Change** the angles: start at zero, then try a 20° turn and a 10° tilt.
            **Watch** the opening and the pillars.

            **Then try turn −52°, tilt 24°.** This view is far from all three training cameras. Both RegNeRF
            models in this project break into a cloud of floaters there. Both RegNeRF + edge L1 models, from two
            separate training runs, keep a solid block with clear outlines. We found this angle by inspecting
            renders, so it is an illustration, not a measurement.
            """
        ),
        mo.accordion(_timing) if _timing else mo.md(""),
        orbit_settings,
    ])
    return (orbit_settings,)


@app.cell
def _(mo, orbit_settings, result, show):
    if orbit_settings.value is None:
        _body = result("Choose the two angles, then select **Render view**. No render starts when you open the page.",
                       kind="neutral", title="Try it")
    else:
        try:
            import time as _time
            from notebooks.extension_camera import render_comparison as _render_comparison
            _start = _time.perf_counter()
            with mo.status.spinner(title="Drawing both models from your camera position. No training."):
                _boards, _device, _step = _render_comparison(**orbit_settings.value)
            _elapsed = _time.perf_counter() - _start
            _body = mo.vstack([
                mo.ui.tabs({"Colour": show(_boards["rgb"]), "Depth": show(_boards["depth"])}),
                mo.md(f"Rendered in **{_elapsed:.2f} s** on {'GPU' if _device == 'cuda' else 'CPU'}. "
                      f"Both models use their saved weights at step {_step:,}."),
                result("At zero degrees, each lower image matches the image above it. As you move, look for "
                       "surfaces that block the opening or break a pillar. Views far from the training cameras "
                       "expose the largest errors.\n\n"
                       "We have no reference photos at these new positions, so they get no PSNR score. "
                       "The measured comparison is the 25 test views in sections 6 and 8.",
                       kind="neutral", title="Look and compare"),
            ])
        except (ImportError, FileNotFoundError, ValueError, KeyError) as _error:
            _body = result(f"**The camera view could not load.**\n\n```text\n{_error}\n```\n\n"
                           "Install the notebook requirements and restore the model files in `notebooks/models/`. "
                           "The saved views in section 6 still work.", kind="warn", title="Saved models needed")
    _body
    return


@app.cell
def _(data, means, mo, np, per_view_change, result, show, stat, view_choice):
    _rows = data["per_view"]
    _delta = np.array([row["partner_minus_regnerf_db"] for row in _rows])
    _highlight = int(view_choice.value.removeprefix("view"))
    mo.vstack([
        mo.md(
            """
            ---
            ## 8. The score over 25 test views
            For each test photo, the score uses only the object's pixels (its mask), measures the colour
            error, and turns it into PSNR. The result is the mean of the 25 values; higher is better. It is
            not the PSNR printed during training. **Watch** whether the bars agree in sign, or whether a few
            views decide the mean.
            """
        ),
        mo.hstack([
            stat(f"{means['regnerf']:.2f} dB", "RegNeRF", "mean of 25 test views"),
            stat(f"{means['partner']:.2f} dB", "RegNeRF + edge L1", "mean of 25 test views"),
            stat(f"{data['difference_db']:+.2f} dB", "Difference", f"better in {data['wins']} of 25 views"),
        ], widths="equal", gap=1),
        mo.hstack(
            [
                show(per_view_change([row["view"] for row in _rows], _delta, highlight=_highlight,
                                     change="edge L1"),
                     caption=f"The view chosen in section 6 (view {_highlight}) has a dark label."),
                result(f"In the first run, the edge term raised the mean colour score by "
                       f"**{data['difference_db']:.2f} dB** and was better in **{data['wins']} of 25** views. "
                       f"The change per view ranges from {_delta.min():+.2f} to {_delta.max():+.2f} dB.\n\n"
                       "**Limits.** One scan, one run. PSNR measures colour, not shape. Section 9 checks "
                       "whether the gain repeats.",
                       kind="success" if data["difference_db"] > 0 else "warn"),
            ],
            widths=[2, 1], align="start", gap=1.5, wrap=True,
        ),
        mo.accordion({
            "All 25 test scores (table)": mo.ui.table(
                [{"View": row["view"], "RegNeRF (dB)": round(row["regnerf_db"], 3),
                  "+ edge L1 (dB)": round(row["partner_db"], 3),
                  "Difference (dB)": round(row["partner_minus_regnerf_db"], 3)} for row in _rows],
                selection=None, show_column_summaries=False),
        }),
    ])
    return


@app.cell
def _(data, means, missing, mo, np, result, second, stat):
    if second is None:
        _body = missing("The second run")
    else:
        _first = {row["view"]: row["partner_minus_regnerf_db"] for row in data["per_view"]}
        _again = {row["view"]: row["edge_minus_regnerf_db"] for row in second["per_view"]}
        _views = sorted(_first)
        _a = np.array([_first[v] for v in _views])
        _b = np.array([_again[v] for v in _views])
        _both_up = int(((_a > 0) & (_b > 0)).sum())
        _both_down = int(((_a < 0) & (_b < 0)).sum())
        _disagree = int(((_a > 0) != (_b > 0)).sum())
        _r = float(np.corrcoef(_a, _b)[0, 1])
        _second_means = second["means_db"]
        _average = (data["difference_db"] + second["edge_minus_regnerf_db"]) / 2
        _runs = [
            {"Run": "First run (RTX 5090, λ fixed)", "RegNeRF (dB)": round(means["regnerf"], 2),
             "+ edge L1 (dB)": round(means["partner"], 2), "Difference (dB)": f"{data['difference_db']:+.2f}",
             "Better in": f"{data['wins']} of 25"},
            {"Run": "Second run (RTX 4070, λ by pilot)", "RegNeRF (dB)": round(_second_means["regnerf"], 2),
             "+ edge L1 (dB)": round(_second_means["edge"], 2),
             "Difference (dB)": f"{second['edge_minus_regnerf_db']:+.2f}", "Better in": f"{int((_b > 0).sum())} of 25"},
        ]
        _edges = {name: value["means"] for name, value in second["edges"].items()}
        _metrics = [
            ("boundary_band_psnr", "Colour near the outline (dB)", "higher is better", 2),
            ("interior_psnr", "Colour inside the object (dB)", "higher is better", 2),
            ("strong_depth_edges_near_outline", "Strong depth edges within 3 px of the outline", "higher is better", 3),
            ("strong_depth_edge_distance_px", "Mean distance of strong depth edges (px)", "lower is better", 2),
        ]
        _metric_rows = [{"Metric": label, "RegNeRF": round(_edges["regnerf"][key], digits),
                         "+ edge L1": round(_edges["edge"][key], digits),
                         "Difference": f"{_edges['edge'][key] - _edges['regnerf'][key]:+.{digits}f}",
                         "Direction": direction} for key, label, direction, digits in _metrics]
        _band = _edges["edge"]["boundary_band_psnr"] - _edges["regnerf"]["boundary_band_psnr"]
        _inner = _edges["edge"]["interior_psnr"] - _edges["regnerf"]["interior_psnr"]
        _near = (_edges["edge"]["strong_depth_edges_near_outline"]
                 - _edges["regnerf"]["strong_depth_edges_near_outline"])
        _body = mo.vstack([
            mo.hstack([
                stat(f"{data['difference_db']:+.2f} dB", "First run", "edge L1 minus RegNeRF"),
                stat(f"{second['edge_minus_regnerf_db']:+.2f} dB", "Second run", "edge L1 minus RegNeRF"),
                stat(f"{_average:+.2f} dB", "Average", "of the two runs"),
            ], widths="equal", gap=1),
            mo.hstack(
                [
                    mo.ui.table(_runs, selection=None, show_column_summaries=False),
                    result(f"Both runs favour the edge term on the mean. **Single views do not repeat:** the runs "
                           f"agree in sign in only {25 - _disagree} of 25 views ({_both_up} better in both, "
                           f"{_both_down} worse in both), and the per-view changes are almost uncorrelated "
                           f"(r = {_r:.2f}).\n\n"
                           "**Trust the mean, not single views.**"),
                ],
                widths=[1.6, 1], align="start", gap=1.5, wrap=True,
            ),
            mo.md("#### Near the edge and away from it (second run)"),
            mo.hstack(
                [
                    mo.ui.table(_metric_rows, selection=None, show_column_summaries=False),
                    result(f"Colour improved about equally near the outline ({_band:+.2f} dB) and inside the "
                           f"object ({_inner:+.2f} dB). The share of strong depth edges next to the outline changed "
                           f"by {_near:+.3f}.\n\n"
                           "**The colour gain is spread over the object. No sign of sharper depth edges.**",
                           title="How to read it"),
                ],
                widths=[1.6, 1], align="start", gap=1.5, wrap=True,
            ),
            mo.accordion({
                "How the edge metrics are defined": mo.md(
                    "Each test view is split by its object mask, at evaluation scale 2. The mask outline is a true "
                    "occlusion boundary. **Colour near the outline:** mask pixels within 3 px of it. **Colour inside "
                    "the object:** mask pixels at least 8 px from it. **Strong depth edges:** the 5% of pixels with "
                    "the largest predicted-depth gradient, inside the mask plus a 3 px margin. The arch's own "
                    "corners also make depth edges, so the last two rows are a proxy, not a depth error. "
                    "`tools/eval_edges.py` computes them."),
                "What differs between the two runs": mo.md(
                    "Both edge models use the same code (revision 92d882b), λ = 0.1, seed 42, 20,000 steps, and "
                    "views 25, 22, and 28. The first run trained on an RTX 5090 with λ fixed in advance. Its RegNeRF "
                    "baseline is an earlier run from the same machine (revision 24aaafb). The second run trained on an RTX 4070 with λ from the pilot, and reused the "
                    "notebook 2 RegNeRF model as its baseline. GPU arithmetic is not exactly repeatable, so two runs "
                    "with the same seed still differ. The two RegNeRF models differ by "
                    f"{abs(means['regnerf'] - _second_means['regnerf']):.2f} dB in the mean."),
            }),
        ])
    mo.vstack([
        mo.md(
            """
            ---
            ## 9. Does the result repeat?
            We trained the same method a second time, on a different computer. If the gain is real, both
            runs should favour the edge term. **Watch** the two differences, then whether single views agree.
            """
        ),
        _body,
    ])
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("---\n## Summary"),
        mo.md(
            """
            **Questions for discussion**
            - In sections 2 and 3, which change keeps a step sharp: L1, the colour weight, or both?
            - Section 5 confirmed λ = 0.1 on validation views. Why must the test views stay out of that choice?
            - Both runs gave a positive mean, but single views disagree. What does that say about reading
              individual views in section 6?
            - In section 7, the −52° / 24° view looks very different. Why can one striking view not replace
              the 25-view score?
            - What evidence would you need to claim that the *shape* improved?

            **Next.** Train more seeds of both models. Add reference depth for scan 40 and measure depth error
            at true boundaries and on flat stone. Compare added uniform L1 with added weighted L1 at the same
            λ, to separate the two parts of the change.
            """
        ),
        mo.accordion({
            "How the models were trained": mo.md(
                """
                - **Same for all four models:** 20,000 steps, seed 42, training views 25, 22, and 28, scale 4
                  (400 × 300), Adam, coarse jitter, 1,024 RGB rays and 1,024 depth rays per step, 8 × 8 depth
                  patches. The fine L2 weight falls from 400 to 0.1 over 512 steps. The coarse L2 stays at 0.1.
                - **The only difference in each pair:** the edge models add the fine L1 term with λ = 0.1.
                - **First run:** RTX 5090, committed source 92d882b. Its edge scale was computed once on CUDA;
                  the loss formula is unchanged. λ fixed from the repository example. Its RegNeRF baseline is an
                  earlier run (revision 24aaafb), reused after checking its settings, source, and training inputs.
                - **Second run:** RTX 4070, revision 92d882b with a clean tree. λ chosen by the pilot in section 5.
                  Its RegNeRF baseline is the notebook 2 model.
                - **Test views:** test photos and masks never enter training or the choice of λ.
                - **Records:** the first run is in `notebooks/results/partner_stone/results.json` (checkpoint
                  paths, source hashes, settings). The second run, with its pilot and edge metrics, is in
                  `notebooks/results/scan40_edge.json`, written by `experiments/compare_edge.py`.
                """
            ),
            "Depth display and limits": mo.md(
                """
                The depth panels use one shared scale across both models and all 25 cameras. Dark means near;
                light means far. The range is the pooled foreground 2nd to 98th percentile. Values outside it are
                clipped for display. No foreground mask hides the predicted depth. The reference panel is a
                colour photo, not a depth map.

                We cannot measure boundary or flat-surface depth error without reference depth. The small
                simulations in sections 1 to 3 have known depth; their results do not transfer to this scan.
                This is a single-scan investigation of RegNeRF Table 3. The paper reports a benchmark aggregate
                and uses a different training setup.
                """
            ),
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
