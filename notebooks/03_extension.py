import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Edge regularizer on DTU scan 40")


@app.cell
def _():
    import json

    import marimo as mo
    import numpy as np
    from PIL import Image

    import dtu
    from notebooks.lab import ROOT, extension_case, plot, rmse, solve_depth, solve_depth_l1
    from notebooks.scan40_figures import per_view_change, small, view_grid
    from notebooks.toy_figures import SCALE

    results = ROOT / "notebooks" / "results"
    images = results / "scan40_edge"
    scene = ROOT / "data" / "dtu" / "scan40"
    downloaded = (scene / "cameras.npz").is_file() and (scene / "image").is_dir()
    record_file = results / "scan40_edge.json"
    data = json.loads(record_file.read_text()) if record_file.is_file() else None
    score_command = "uv run python compare_edge.py --output-dir checkpoint/scan40_edge --score"

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
            mo.md("**The DTU photos are not downloaded yet.** They are not in Git. Run this once "
                  "(about 120 MB, under a minute):\n\n```sh\nuv run python prepare_dtu.py --download\n```"),
            kind="warn",
        )
    return (
        Image,
        data,
        downloaded,
        dtu,
        extension_case,
        images,
        json,
        missing,
        mo,
        no_photos,
        np,
        per_view_change,
        plot,
        result,
        rmse,
        scene,
        score_command,
        show,
        small,
        solve_depth,
        solve_depth_l1,
        stat,
        view_grid,
    )


@app.cell
def _(downloaded, images, mo, scene, small):
    _edge_map = images / "edge_map_view25.png"
    _picture = (mo.image(str(_edge_map), width=380, caption="Edge strength on training view 25 (dark = strong edge)")
                if _edge_map.is_file() else
                mo.image(small(scene / "image" / "000025.png", 400), width=380, caption="Scan 40, training view 25")
                if downloaded else mo.md(""))
    _intro = mo.md(
        """
        **The question.** Notebook 2 showed that RegNeRF's depth smoothing helps on the stone arch.
        But its penalty squares each depth difference, and a squared penalty prefers a ramp to a step:
        on an 8-pixel patch, a ramp costs 7 times less. Can we keep the smoothing and still get sharp
        depth edges?

        **Our change.** Add a second depth term that uses absolute differences and is cheap at colour
        edges. It finds edges with the first two steps of the Canny detector (blur, then Sobel
        gradient), with no thresholds and no learned parameters.

        **Prediction.** If the term works, depth edges move to the outline of the arch and get
        sharper, and the colour score near the outline improves. This can fail: stone texture is also
        a colour edge.

        **How to use this page.** Everything uses saved results, so nothing trains. Read each
        section, then the box beside it. Allow about 35 minutes.

        1. Why does a squared penalty blur edges?
        2. Does the edge weight help when the colour guide is wrong?
        3. What does the term see on the arch?
        4. How was the weight chosen?
        5. New cameras
        6. The score over 25 new cameras
        7. Near the edge and away from it
        """
    )
    mo.vstack([
        mo.md("# Sharpening depth at colour edges"),
        mo.hstack([_picture, _intro], widths=[1, 1.4], align="start", gap=2, wrap=True),
    ])
    return


@app.cell
def _(mo):
    case = mo.ui.dropdown(["Aligned edge", "Texture edge", "Missing colour edge"],
                          value="Aligned edge", label="Colour guide")
    mo.vstack([
        mo.md(
            """
            ---
            ## 1. Why does a squared penalty blur edges?
            One line of pixels: two flat surfaces, one depth step at pixel 63.5, plus noise. Four solvers
            get the same data. Only the penalty changes: squared (L2) or absolute (L1) differences, uniform
            or weighted by `w = exp(−10·|colour difference|)`. **Watch** the blue RegNeRF curve and the
            orange curve (ours) at the step. Reference depth is used only for the score.
            """
        ),
        case,
    ])
    return (case,)


@app.cell
def _(case, extension_case, mo, np, plot, result, rmse, solve_depth, solve_depth_l1, stat):
    truth, observations, guide, boundary, flat = extension_case(case.value)
    weights = np.exp(-10 * np.abs(np.diff(guide)))
    solvers = {
        "L2 uniform (RegNeRF)": solve_depth(observations, 8),
        "L2 edge-weighted": solve_depth(observations, 8, weights),
        "L1 uniform": solve_depth_l1(observations, 1),
        "L1 edge-weighted (ours)": solve_depth_l1(observations, 1, weights),
    }
    _base, _ours = solvers["L2 uniform (RegNeRF)"], solvers["L1 edge-weighted (ours)"]
    _rows = [{"Penalty": "None (noisy data)", "Boundary RMSE": rmse(observations, truth, boundary),
              "Flat RMSE": rmse(observations, truth, flat), "All RMSE": rmse(observations, truth)}]
    _rows += [{"Penalty": name, "Boundary RMSE": rmse(p, truth, boundary), "Flat RMSE": rmse(p, truth, flat),
               "All RMSE": rmse(p, truth)} for name, p in solvers.items()]
    _rows = [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in _rows]
    mo.vstack([
        mo.hstack(
            [
                mo.vstack([
                    mo.Html(plot({"Reference": truth, "Noisy depth": observations,
                                  "L2 uniform (RegNeRF)": _base, "L1 edge-weighted (ours)": _ours},
                                 "Two flat surfaces with one depth step")),
                    mo.Html(plot({"Colour guide": guide}, "Colour guide", "Colour (0 to 1)")),
                ]),
                mo.vstack([
                    mo.hstack([
                        stat(f"{rmse(_base, truth, boundary):.3f}", "Boundary error, RegNeRF", "RMSE"),
                        stat(f"{rmse(_ours, truth, boundary):.3f}", "Boundary error, ours", "RMSE"),
                    ], widths="equal"),
                    result(
                        "The squared penalty turns the step into a ramp, so its boundary error is the largest. "
                        "Absolute differences keep most of the step, even with a uniform weight (see the table). "
                        "The edge weight then helps where the guide has an edge at the true step.\n\n"
                        "**A squared penalty blurs edges. An absolute one keeps them.**"
                    ),
                ]),
            ],
            widths=[2, 1],
            align="start",
            gap=1.5,
            wrap=True,
        ),
        mo.accordion({
            "All four penalties (table)": mo.ui.table(_rows, selection=None, show_column_summaries=False),
            "How the regions are defined": mo.md(
                "Boundary pixels are within 3.5 pixels of the step. Flat pixels are at least 12.5 pixels from "
                "it. Both masks come from the reference, not from the estimate. Depths are in scene units."),
        }),
    ])
    return


@app.cell
def _(extension_case, mo, np, result, rmse, solve_depth, solve_depth_l1):
    _summary = []
    for _case in ["Aligned edge", "Texture edge", "Missing colour edge"]:
        _truth, _obs, _guide, _boundary, _flat = extension_case(_case)
        _w = np.exp(-10 * np.abs(np.diff(_guide)))
        _base = solve_depth(_obs, 8)
        _ours = solve_depth_l1(_obs, 1, _w)
        _summary.append({"Case": _case,
                         "Boundary, RegNeRF": round(rmse(_base, _truth, _boundary), 4),
                         "Boundary, ours": round(rmse(_ours, _truth, _boundary), 4),
                         "Flat, RegNeRF": round(rmse(_base, _truth, _flat), 4),
                         "Flat, ours": round(rmse(_ours, _truth, _flat), 4)})
    mo.vstack([
        mo.md(
            """
            ---
            ## 2. Does the edge weight help when the colour guide is wrong?
            The same data with three guides: an edge at the step (aligned), an extra colour band on a flat
            surface (texture), and no colour edge at all (missing). **Watch** the flat columns in the
            texture case and the boundary columns in the missing case.
            """
        ),
        mo.hstack(
            [
                mo.ui.table(_summary, selection=None, show_column_summaries=False),
                result(
                    "Our term has a lower boundary error than RegNeRF's in all three cases. Most of that gain "
                    "comes from the absolute differences, so it remains when the colour edge is missing. The "
                    "texture band does not create a false step: the term stops resisting a step, but it does "
                    "not push one into being.\n\n"
                    "**A wrong guide costs little here. The real test is a blurry guide in a trained NeRF.**"
                ),
            ],
            widths=[1.6, 1],
            align="start",
            gap=1.5,
            wrap=True,
        ),
    ])
    return


@app.cell
def _(data, downloaded, images, missing, mo, no_photos, result, scene, score_command, small):
    _edge_map = images / "edge_map_view25.png"
    if not downloaded:
        _body = no_photos()
    elif data is None or not _edge_map.is_file():
        _body = missing("The scan 40 edge comparison is", score_command)
    else:
        _scale = data["edge_map"]["edge_scale"]
        _body = mo.hstack(
            [
                mo.image(small(scene / "image" / "000025.png", 400), width=300, caption="Training view 25"),
                mo.image(str(_edge_map), width=300, caption="Edge strength 1 − w (dark = strong edge)"),
                result(
                    "The outline of the arch and the opening are dark, so a depth step there is cheap. These are "
                    "real depth edges. The stone texture is grey: there the term relaxes a little, though the "
                    "surface is flat.\n\n"
                    f"The scale `s` = {_scale:.3f} is the 90th percentile of the gradient over the 3 training "
                    "photos. In training, the term uses the rendered colour of each unseen-camera patch, so early "
                    "edges are weak.\n\n"
                    "**The term sees the outline, and some texture.**",
                ),
            ],
            widths=[1, 1, 1.2],
            align="start",
            gap=1,
            wrap=True,
        )
    mo.vstack([
        mo.md(
            """
            ---
            ## 3. What does the term see on the arch?
            Two models train on the same 3 photos with the same seed, steps, sampling, and RegNeRF depth
            penalty. **The only difference:** the second adds λ × our edge term:
            `mean(opacity · w · |depth difference|)` with `w = exp(−‖∇colour‖ / s)`. The first model is
            the "smoothing on" model of notebook 2. **Watch** which lines in the edge map are dark.
            """
        ),
        _body,
    ])
    return


@app.cell
def _(data, missing, mo, result, score_command):
    if data is None or not data.get("pilot"):
        _body = missing("The weight pilot is",
                        "uv run python compare_edge.py --output-dir checkpoint/scan40_edge --pilot\n" + score_command)
    else:
        _pilot = data["pilot"]
        _rows = [{"λ": ("0 (RegNeRF)" if k == "0" else k), "Validation mean (dB)": round(v["mean_db"], 3),
                  **{f"View {view}": round(db, 2) for view, db in v["per_view_db"].items()}}
                 for k, v in _pilot["scores"].items()]
        _body = mo.hstack(
            [
                mo.ui.table(_rows, selection=None, show_column_summaries=False),
                result(
                    f"Each candidate trained for {_pilot['steps']} steps and was scored on views "
                    f"{', '.join(str(v) for v in _pilot['validation_views'])}. The 3-view models never train on "
                    "these views, and they are not test views. Rule: the highest mean validation PSNR; a tie "
                    "within 0.01 dB goes to the smaller λ.\n\n"
                    f"**λ = {_pilot['chosen_lambda']:g} was fixed before any test view was scored.**",
                ),
            ],
            widths=[1.6, 1],
            align="start",
            gap=1.5,
            wrap=True,
        )
    mo.vstack([
        mo.md(
            """
            ---
            ## 4. How was the weight chosen?
            The weight λ sets how strong the edge term is next to the RegNeRF term. **Watch** whether one λ
            stands out, or whether the candidates are close.
            """
        ),
        _body,
    ])
    return


@app.cell
def _(dtu, mo):
    view_choice = mo.ui.dropdown({f"View {v}": v for v in dtu.TEST_IDS}, value="View 45", label="Test view")
    mo.vstack([
        mo.md(
            """
            ---
            ## 5. New cameras
            Both models render every test camera, which neither saw in training. Start with view 45.
            **Watch** the outline of the pillars and the opening under the top stone, first in colour, then
            in depth. Then **change** the view.
            """
        ),
        view_choice,
    ])
    return (view_choice,)


@app.cell
def _(Image, downloaded, images, json, missing, mo, no_photos, result, scene, show, small, view_choice, view_grid):
    _record_file = images / "render_record.json"
    _view = view_choice.value
    if not _record_file.is_file():
        _body = missing("The test-view renders are",
                        "uv run python -m notebooks.render_scan40_views --output notebooks/results/scan40_edge "
                        "--conditions regnerf=checkpoint/scan40_pair_new/on/checkpoint_20000.pt "
                        "edge=checkpoint/scan40_edge/edge/checkpoint_20000.pt")
    elif not downloaded:
        _body = no_photos()
    else:
        _record = json.loads(_record_file.read_text())

        def _open(name):
            return Image.open(images / f"view{_view}_{name}.jpg")

        _body = mo.vstack([
            show(view_grid(small(scene / "image" / f"{_view:06}.png", 600), _open("rgb_regnerf"), _open("rgb_edge"),
                           _open("depth_regnerf"), _open("depth_edge"), _record["depth_ranges"][str(_view)],
                           labels=("RegNeRF", "RegNeRF + edge term")),
                 caption=f"Test view {_view}, step {_record['step']}, rendered at scale {_record['scale']}. "
                         "Top: colour. Bottom: depth (light is near)."),
            result("**Describe only what you can see.** Is the depth outline of the arch sharper? Does the stone "
                   "surface stay flat, or does its texture show up in depth? Each view is an illustration; "
                   "sections 6 and 7 use all 25 test views.", kind="neutral", title="Look and compare"),
        ])
    _body
    return


@app.cell
def _(data, dtu, json, missing, mo, np, per_view_change, result, score_command, show, stat, view_choice):
    if data is None:
        _body = missing("The scan 40 edge comparison is", score_command)
    else:
        _rows = data["per_view"]
        if tuple(r["view"] for r in _rows) != dtu.TEST_IDS:
            raise ValueError("The record must contain the 25 test views in protocol order.")
        _base = np.array([r["regnerf_db"] for r in _rows])
        _ours = np.array([r["edge_db"] for r in _rows])
        _delta = _ours - _base
        _better = int((_delta > 0).sum())
        _mean = float(_delta.mean())
        _effect = (f"**raised** the mean colour score by **{_mean:.2f} dB**" if _mean >= 0.005 else
                   f"**lowered** the mean colour score by **{-_mean:.2f} dB**" if _mean <= -0.005 else
                   "**did not change** the mean colour score")
        _lam = data["record"]["edge_loss_weight"]
        _body = mo.vstack([
            mo.hstack([
                stat(f"{_base.mean():.2f} dB", "RegNeRF", "mean of 25 test views"),
                stat(f"{_ours.mean():.2f} dB", f"+ edge term (λ = {_lam:g})", "mean of 25 test views"),
                stat(f"{_mean:+.2f} dB", "Difference", f"better in {_better} of 25 views"),
            ], widths="equal", gap=1),
            mo.hstack(
                [
                    show(per_view_change(dtu.TEST_IDS, _delta, highlight=view_choice.value, change="edge term"),
                         caption=f"The view chosen in section 5 (view {view_choice.value}) has a dark label."),
                    result(
                        f"On this scan, the edge term {_effect}. It was better in **{_better} of 25** test views; "
                        f"the change per view ranges from {_delta.min():+.2f} to {_delta.max():+.2f} dB.\n\n"
                        "**Limits.** One scan and one seed. PSNR measures colour, not shape.",
                        kind="success" if _mean >= 0.005 else "warn" if _mean <= -0.005 else "neutral",
                    ),
                ],
                widths=[2, 1],
                align="start",
                gap=1.5,
                wrap=True,
            ),
            mo.accordion({
                "Every test view (table)": mo.ui.table(
                    [{"View": r["view"], "RegNeRF (dB)": round(r["regnerf_db"], 2),
                      "+ edge (dB)": round(r["edge_db"], 2),
                      "Difference (dB)": round(r["edge_minus_regnerf_db"], 2)} for r in _rows],
                    selection=None, show_column_summaries=False),
                "Training and evaluation record": mo.md("```json\n" + json.dumps(
                    {k: v for k, v in data["record"].items() if k != "local_diff"}, indent=2) + "\n```"),
            }),
        ])
    mo.vstack([
        mo.md(
            """
            ---
            ## 6. The score over 25 new cameras
            The same score as notebook 2: for each test photo, only the object's pixels, turned into PSNR.
            The result is the mean of the 25 values; higher is better. It is not the PSNR printed during
            training. **Watch** whether the bars agree in sign, or whether a few views decide the mean.
            """
        ),
        _body,
    ])
    return


@app.cell
def _(data, mo, result, stat):
    if data is None:
        _body = mo.md("")
    else:
        _e = {name: value["means"] for name, value in data["edges"].items()}
        _labels = [
            ("boundary_band_psnr", "Boundary band PSNR (dB)", "higher is better", 2),
            ("interior_psnr", "Interior PSNR (dB)", "higher is better", 2),
            ("strong_depth_edges_near_outline", "Strong depth edges near the outline", "higher is better", 3),
            ("strong_depth_edge_distance_px", "Mean distance of strong depth edges (px)", "lower is better", 2),
        ]
        _rows = [{"Metric": label, "RegNeRF": round(_e["regnerf"][key], digits),
                  "+ edge term": round(_e["edge"][key], digits),
                  "Difference": round(_e["edge"][key] - _e["regnerf"][key], digits), "Direction": direction}
                 for key, label, direction, digits in _labels]
        _band = _e["edge"]["boundary_band_psnr"] - _e["regnerf"]["boundary_band_psnr"]
        _inner = _e["edge"]["interior_psnr"] - _e["regnerf"]["interior_psnr"]
        _near = _e["edge"]["strong_depth_edges_near_outline"] - _e["regnerf"]["strong_depth_edges_near_outline"]
        _body = mo.vstack([
            mo.hstack([
                stat(f"{_band:+.2f} dB", "Boundary band", "edge term minus RegNeRF"),
                stat(f"{_inner:+.2f} dB", "Interior", "edge term minus RegNeRF"),
                stat(f"{_near:+.3f}", "Strong depth edges near outline", "change in share"),
            ], widths="equal", gap=1),
            mo.hstack(
                [
                    mo.ui.table(_rows, selection=None, show_column_summaries=False),
                    result(
                        "The mask outline is a true occlusion boundary. A sharper, better placed depth edge "
                        "should raise the boundary-band score and put more strong depth gradients next to the "
                        "outline. The arch's own corners and the opening also make depth edges, so the last two "
                        "rows are a proxy.\n\n"
                        "**DTU in this repo has no reference depth, so these numbers cannot show a depth error "
                        "directly.**",
                        title="How to read it",
                    ),
                ],
                widths=[1.6, 1],
                align="start",
                gap=1.5,
                wrap=True,
            ),
        ])
    mo.vstack([
        mo.md(
            """
            ---
            ## 7. Near the edge and away from it
            Each test view is split by its object mask. The **boundary band** is the mask pixels within 3 px
            of the outline (scale 2). The **interior** is the mask pixels at least 8 px from it. **Strong depth
            edges** are the 5% of pixels with the largest depth gradient, inside the mask plus a 3 px margin.
            **Watch** whether the boundary band and the interior move in the same direction.
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
            - In section 1, which change sharpened the step: absolute differences, the edge weight, or both?
            - In sections 6 and 7, did the edge term change the colour score, the boundary band, or the
              depth-edge proxy? Do they agree?
            - Compare the size of the change with the seed-to-seed variation in notebook 1. Is one seed enough?
            - Find the test views that changed most in section 6. Open them in section 5: what changed at the
              outline?

            **Next.** Train 2 more seeds of both models. Add reference depth for scan 40 and measure depth error
            at true boundaries and on flat stone. Train with uniform L1 alone, to separate the two parts of the
            change, as in section 1.
            """
        ),
        mo.accordion({
            "How the two models were trained": mo.md(
                """
                - **Same for both:** training views 25, 22, and 28; seed 42; 20,000 steps; Adam with the same
                  learning-rate schedule; training scale 4; the same ray sampling, coarse jitter, sample-space
                  annealing, and RegNeRF depth penalty (fine weight 400 → 0.1 by step 512, coarse weight 0.1).
                - **The only difference:** the second model adds λ × the edge term on the fine level of each
                  unseen-camera patch. It renders each patch with a 2-pixel border in eval mode (no noise, no
                  random numbers) to compute the colour gradient. The colour is detached, so the term moves only
                  depth.
                - **Baseline reuse:** the RegNeRF model is the saved notebook 2 model. The training code has not
                  changed since then, apart from the new term. With λ = 0, the new code gives bit-identical
                  weights after 5 steps.
                - **Origin of the term:** edge-aware L1 smoothness from self-supervised monocular depth (Godard
                  et al., 2017). The new part is its use on RegNeRF's unseen-camera patches, with the rendered
                  colour as the guide.
                - **Scoring:** both models at step 20000, evaluation scale 2, the 25 test views from `dtu.py`.
                - **Record:** `compare_edge.py` writes the code revision, local changes, settings, pilot scores,
                  and checkpoint paths. `eval_edges.py` computes the edge metrics.
                """
            ),
        }),
    ])
    return


if __name__ == "__main__":
    app.run()
