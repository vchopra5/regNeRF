import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="When neighbours have different depths")


@app.cell
def _():
    import json
    import marimo as mo
    from PIL import Image
    from notebooks.lab import ROOT
    from notebooks.edge_lesson import pair_figure, ramp_figure, profile_experiment, pipeline_figure

    evidence = ROOT / "notebooks/results/partner_stone"
    data = json.loads((evidence / "results.json").read_text(encoding="utf-8"))
    means = data["means_db"]

    def show(image):
        return mo.image(image, width=min(image.width, 900))

    def panel(tag, kind):
        return show(Image.open(evidence / f"{tag}_{kind}.png"))

    def result(text):
        return mo.callout(mo.md(text), kind="info")

    return (ROOT, data, json, means, mo, pair_figure, panel, pipeline_figure,
            profile_experiment, ramp_figure, result, show)


@app.cell
def _(mo):
    mo.md("""
    # When neighbours have different depths
    Two pixels can sit next to each other in a photo but belong to different surfaces.
    Depth smoothing pulls their predicted depths together. What happens at the edge of an object?

    **How to use this page.** Work through the small experiments first. Change the controls,
    watch the figures, and read the result boxes. Then compare the saved arch results and,
    if you wish, render a new camera view in section 5. Nothing trains.

    The saved figures, scores, and models come with the notebook. No DTU download or prior
    training is needed. PyTorch is needed for the optional camera render.

    1. Why L2 blurs a depth step
    2. Whether colour can guide smoothing
    3. How the new penalty works
    4. Saved test views
    5. Move around the arch
    6. Scores across the test views

    Increase **Smoothing strength** and watch the two coloured dots move closer.
    Then clear **Smooth across the boundary**. The dots return to the true depths.
    """)
    return


@app.cell
def _(mo):
    pair_strength = mo.ui.slider(0, 8, step=0.5, value=2, show_value=True, label="Smoothing strength")
    connected = mo.ui.checkbox(value=True, label="Smooth across the boundary")
    mo.hstack([pair_strength, connected], wrap=True)
    return connected, pair_strength


@app.cell
def _(connected, mo, pair_figure, pair_strength, result, show):
    _image, _gap = pair_figure(pair_strength.value, connected.value)
    mo.vstack([
        show(_image),
        result(f"True depth gap: **2.00**. Fitted gap: **{_gap:.2f}** scene units. "
               + ("Smoothing pulls the two surfaces closer than they should be." if connected.value and pair_strength.value > 0
                  else "The fitted depths match the two surfaces.")),
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
        mo.md("""
        ## 1. Why does L2 blur a depth step?
        L1 adds the absolute depth differences. L2 adds their squares.
        Spread the same depth change over more pixels. Watch the two costs below the plots.
        """),
        ramp_span,
    ])
    return (ramp_span,)


@app.cell
def _(mo, ramp_figure, ramp_span, result, show):
    mo.vstack([
        show(ramp_figure(ramp_span.value)),
        result("**Spreading the step lowers the L2 cost. The L1 cost stays the same.** "
               "L2 therefore favours a gradual change here. L1 allows a sharp step, but does not locate it."),
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
    strength = mo.ui.slider(steps=[0, 0.1, 0.5, 1, 2, 4, 8], value=2,
                            show_value=True, label="Strength")
    case = mo.ui.dropdown(["Aligned edge", "Texture edge", "Missing colour edge"],
                          value="Aligned edge", label="Colour guide")
    mo.vstack([
        mo.md("""
        ## 2. Can colour tell us where to stop smoothing?
        Choose L1 or L2, then change the colour guide. The left plot smooths every pair equally.
        The right plot reduces smoothing where the colour changes. Compare the depth step and the flat surface.
        """),
        mo.hstack([penalty, strength, case], wrap=True),
        mo.md("*The controls recalculate this small depth fit. The observations stay the same.*"),
    ])
    return case, penalty, strength


@app.cell
def _(case, mo, penalty, profile_experiment, result, show, strength):
    _image, _rows = profile_experiment(case.value, penalty.value, strength.value)
    _delta = _rows[1]["Boundary RMSE"] - _rows[0]["Boundary RMSE"]
    _flat_delta = _rows[1]["Flat RMSE"] - _rows[0]["Flat RMSE"]
    _note = {
        "Aligned edge": "Here the colour change lines up with the depth step.",
        "Texture edge": "The extra colour stripe lies on a flat surface. It reduces smoothing where the depth should stay flat.",
        "Missing colour edge": "The colour stays constant across the depth step. Both plots use the same weights and give the same fit.",
    }[case.value]
    mo.vstack([
        show(_image),
        result(f"{_note} Using colour weights changes depth error (RMSE) by **{_delta:+.3f}** "
               f"at the step and **{_flat_delta:+.3f}** on the flat surface. A negative change means less error."),
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
        mo.md("""
        ## 3. How RegNeRF + edge L1 works
        We keep RegNeRF's L2 smoothing and add an L1 penalty to the fine depth output.
        The new penalty becomes weaker where the model's rendered colour changes.
        """),
        show(pipeline_figure()),
        result("**The added L1 term allows more depth change at colour edges.** "
               "The original L2 term still smooths those edges, and a colour change does not force a depth step."),
        mo.accordion({"Equations and filter settings": mo.md(r"""
            $\Delta_xd$ and $\Delta_yd$ are right and lower depth differences.
            $a$ is anchor opacity, with gradients stopped.
            $$S_2=\operatorname{mean}[a((\Delta_xd)^2+(\Delta_yd)^2)]$$
            $$S_1=\operatorname{mean}[a(w_x|\Delta_xd|+w_y|\Delta_yd|)]$$
            $$L_{\mathrm{depth}}=0.1S_{2,\mathrm{coarse}}+\lambda_2(t)S_{2,\mathrm{fine}}+0.1S_{1,\mathrm{fine}}.$$

            The fine L2 weight $\lambda_2(t)$ falls from 400 to 0.1 over the first 512 steps.
            The coarse L2 and added fine L1 coefficients stay at 0.1.

            Convert RGB to grey with coefficients 0.299, 0.587, 0.114.
            Apply a 3 × 3 Gaussian with sigma 1, then Sobel filters.
            For gradient magnitude $g$, use $w_{ij}=\exp(-\max(g_i,g_j)/s)$.
            The scale $s$ is the 90th percentile over the three training photos.
            A two-pixel halo supplies the filter's context.

            Guide rendering uses evaluation mode and no gradients. The weights are recomputed
            each step but receive no gradient through this loss. RGB supervision stays active.
            Virtual patches use a separate random generator. The code adds RGB and depth gradients,
            clips them, then updates the network with Adam.
            """)}),
    ])
    return


@app.cell
def _(data, mo):
    view_choice = mo.ui.dropdown(
        {f"View {row['view']}": f"view{row['view']}" for row in data["per_view"]},
        value="View 24", label="Saved test camera")
    mo.vstack([
        mo.md("""
        ## 4. Does it help the stone arch?
        Choose a test view, then switch between **Depth** and **Colour**.
        Look at the opening, pillar edges, and top block. Where do the models differ?

        Both models trained on the same three photos of DTU scan 40.
        These 25 saved comparisons open without rendering. Each view shows its measured RGB scores.
        """),
        view_choice,
    ])
    return (view_choice,)


@app.cell
def _(data, mo, panel, result, view_choice):
    _notes = {
        "view24": "The added term reduces the vertical mark inside the opening. Both depth maps still have uneven surfaces.",
        "view10": "Both models distort the top block. RegNeRF + edge L1 has a lower colour score in this view.",
        "view43": "The vertical mark is reduced, but the depth boundary around the opening remains blurred.",
    }
    _score = next(row for row in data["per_view"] if f"view{row['view']}" == view_choice.value)
    mo.vstack([
        mo.ui.tabs({"Depth": panel(view_choice.value, "depth"),
                    "Colour": panel(view_choice.value, "rgb")}),
        result(f"Foreground RGB PSNR: RegNeRF **{_score['regnerf_db']:.2f} dB**; "
               f"RegNeRF + edge L1 **{_score['partner_db']:.2f} dB**; "
               f"difference **{_score['partner_minus_regnerf_db']:+.2f} dB**. "
               + _notes.get(view_choice.value, "Compare the opening, pillar edges, and top block.")
               + " The left panel is a real photo. We have no reference depth for this scan, "
               "so a smoother depth map alone cannot tell us which shape is more accurate."),
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
    _timing_note = mo.md("")
    if _timing_path.is_file():
        _timing = json.loads(_timing_path.read_text(encoding="utf-8"))
        _rows = []
        for _device, _label in (("cpu", "CPU"), ("cuda", "GPU")):
            if _device in _timing["devices"]:
                _times = _timing["devices"][_device]["median_seconds"]
                _rows.append(f"| {_label} | {_times['first_request']:.1f} s | "
                             f"{_times['new_angle']:.1f} s | {_times['repeat_angle']:.2f} s |")
        _timing_note = mo.callout(mo.md(
            "For a guide, notebook 2 measured these median times over three trials "
            "at 100 × 75 pixels per image. Notebook 3 uses different models and also "
            "builds depth images, so its times can differ.\n\n"
            "| Device | First render | New angle | Repeated angle |\n"
            "| --- | ---: | ---: | ---: |\n" + "\n".join(_rows) + "\n\n"
            "**First render:** start and moved positions for both models. "
            "**New angle:** the start images are reused. "
            "**Repeated angle:** all images are reused. At zero degrees, the start "
            "and moved positions are the same, so fewer images need rendering.\n\n"
            f"Notebook 2 was measured on {_timing['cpu']} with {_timing['cpu_threads']} CPU threads, "
            f"and {_timing['gpu']}. Your computer and other running tasks can change the time."
        ), kind="neutral", title="How long does it take?")
    mo.vstack([
        mo.md("""
        ## 5. Move around the arch
        Turn the camera left or right, or look from above or below. Select **Render view** to draw
        both models from that position. The top row stays at camera 25; the bottom row follows your settings.

        Start with both angles at zero. Then try a 20° horizontal turn and a 10° vertical turn.
        Watch the opening and pillars. You can turn through a full circle to inspect the back as well.
        Set both angles to zero to return to the start.

        This renders new colour and depth images from the saved models at 100 × 75 pixels.
        Both models use the same camera and keep their learned weights. The first render loads the models;
        repeated angles reuse the images. A GPU is faster than a CPU. The elapsed time appears below each render.
        """),
        _timing_note,
        orbit_settings,
    ])
    return (orbit_settings,)


@app.cell
def _(mo, orbit_settings, result, show):
    if orbit_settings.value is None:
        _body = mo.md("Choose the two angles, then select **Render view**.")
    else:
        try:
            import time as _time
            from notebooks.extension_camera import render_comparison as _render_comparison
            _start = _time.perf_counter()
            with mo.status.spinner(title="Drawing both models from your camera position..."):
                _boards, _device, _step = _render_comparison(**orbit_settings.value)
            _elapsed = _time.perf_counter() - _start
            _body = mo.vstack([
                mo.ui.tabs({"Colour": show(_boards["rgb"]), "Depth": show(_boards["depth"])}),
                mo.md(f"Rendered in **{_elapsed:.2f} s** on {'GPU' if _device == 'cuda' else 'CPU'}. "
                      f"Both models use their saved weights at step {_step:,}."),
                result("At zero degrees, each lower image matches the image above it. "
                       "As you move, look for surfaces that block the opening or break a pillar. "
                       "Views far from the training cameras can expose large errors.\n\n"
                       "We have no reference photos at these new positions. Use the 25 test views "
                       "above for measured RGB comparisons; moving this camera does not change their scores."),
            ])
        except (ImportError, FileNotFoundError, ValueError, KeyError) as _error:
            _body = result(f"The camera view could not load: {_error}\n\n"
                           "Install the notebook requirements and restore the model files in `notebooks/models/`. "
                           "You can still inspect all 25 saved views above.")
    _body
    return


@app.cell
def _(data, means, mo):
    mo.vstack([
        mo.md("## 6. What changed across the test views?"),
        mo.hstack([
            mo.stat(f"{means['regnerf']:.2f} dB", label="Standard RegNeRF"),
            mo.stat(f"{means['partner']:.2f} dB", label="RegNeRF + edge L1"),
            mo.stat(f"{data['difference_db']:+.2f} dB", label="Edge L1 minus RegNeRF"),
        ], widths="equal"),
        mo.md(f"**RegNeRF + edge L1 scores higher in {data['wins']} of 25 test views.** "
              "Each score above is the arithmetic mean of 25 per-view foreground RGB PSNR values. "
              "These are test scores, not training-batch scores. One scene and one seed."),
        mo.accordion({
            "All 25 test scores": mo.vstack([
                mo.ui.table([{"View": row["view"], "RegNeRF (dB)": round(row["regnerf_db"], 3),
                              "+ edge L1 (dB)": round(row["partner_db"], 3),
                              "Difference (dB)": round(row["partner_minus_regnerf_db"], 3)}
                             for row in data["per_view"]], selection=None, show_column_summaries=False),
                mo.md("Positive differences favour RegNeRF + edge L1. Each per-view score uses RGB error "
                      "inside the test photo's foreground mask. Both models are evaluated at scale 2."),
            ]),
            "Depth display and limits": mo.md("""
                The depth panels use one shared scale across both models and all 25 cameras.
                Dark means near; light means far. The range is the pooled foreground 2nd to 98th
                percentile. Values outside it are clipped for display. No foreground mask hides
                the predicted depth. The reference panel is a colour photo, not a depth map.

                We cannot measure boundary depth error or flat-surface depth error without reference depth.
                The small simulations above have known depth. Their RMSE results do not apply to this scan.
                This is a single-scan investigation of RegNeRF Table 3. The paper reports a benchmark
                aggregate and uses a different training setup.
                """),
            "Training settings and source": mo.md("""
                Both models: 20,000 steps, seed 42, training views 25/22/28, scale 4 (400 × 300),
                Adam, coarse jitter, 1,024 RGB rays and 1,024 depth rays per step, 8 × 8 depth patches.
                The fine L2 weight falls from 400 to 0.1 over 512 steps. Coarse L2 stays at 0.1.
                The extension adds fine L1 with coefficient 0.1, fixed from the repository example.
                Test photos and masks do not select this coefficient or enter training.

                The saved standard model was reused after checking settings, source, and training inputs.
                RegNeRF + edge L1 used the RTX 5090 and committed source 92d882b.
                The one-time edge-scale calculation ran on CUDA. The loss formula was unchanged.

                This lesson reads 50 saved panels and all 25 scores from
                notebooks/results/partner_stone. It needs no downloaded DTU data or local checkpoints.
                Full checkpoint paths, source hashes, and settings are in results.json in that folder.
                Opening the notebook does not train a model.
                """),
            "Next test": mo.md("Repeat with more seeds. Compare added uniform L1 with added weighted L1, "
                               "keeping the coefficient and loss levels fixed. Use reference depth to measure "
                               "errors at true boundaries and on flat surfaces."),
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
