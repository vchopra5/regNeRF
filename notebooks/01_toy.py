import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", app_title="Three views, many shapes")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from notebooks.lab import ROOT, rmse, solve_depth, sphere_profile
    return ROOT, mo, np, rmse, solve_depth, sphere_profile


@app.cell
def _(mo):
    mo.Html("""
    <header style="padding:28px 0 20px;border-top:4px solid #237369;border-bottom:1px solid #8b9997">
      <div style="font:600 12px monospace;letter-spacing:.12em;color:#297769">GEOMETRY LAB &nbsp; / &nbsp; 01 OF 03</div>
      <h1 style="font:normal clamp(36px,6vw,60px)/1.05 Georgia,serif;margin:18px 0 12px">Three views, many shapes</h1>
      <p style="font-size:19px;max-width:60ch">How can a model match the colours in a photo and still put the surface at the wrong depth?</p>
      <small>RegNeRF &nbsp; / &nbsp; 15 minutes &nbsp; / &nbsp; CPU experiment</small>
    </header>""")
    return


@app.cell
def _(ROOT, mo):
    mo.vstack([
        mo.md("## The observation\nLook at the face of the statue. The top row shows the three input photos. Compare the middle and bottom rows **in the same column**. Each pair shows the same new camera view."),
        mo.image(src=str(ROOT / "notebooks/assets/regnerf_figure_1.png"), width=540,
                 alt="Three rows show input photographs of a statue, novel views from NeRF, and the same views from RegNeRF."),
        mo.md("*Figure 1, Niemeyer et al., CVPR 2022.* The middle row loses the face. The bottom row preserves its outline. The authors used depth and appearance regularization and a sampling schedule. This figure compares their full method. Here we isolate the depth term. [Read the paper](https://arxiv.org/pdf/2112.00724#page=1)."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 01 &nbsp; Move the camera
    Imagine a gold mark on the central ray of a camera. Two candidate surfaces place that mark at depths 2 and 4. Both project to pixel 0 in the input view.

    **Predict** which mark will move farther across the image when the camera moves to the right. Then move the camera. It keeps the same viewing direction.
    """)
    return


@app.cell
def _(mo):
    baseline = mo.ui.slider(0, 0.8, step=0.05, value=0, show_value=True, label="Camera shift (scene units)")
    baseline
    return (baseline,)


@app.cell
def _(baseline, mo):
    # For a point on the initial centre ray, u = -f * camera_shift / depth.
    focal = 160
    near_pixel = -focal * baseline.value / 2
    far_pixel = -focal * baseline.value / 4
    _rows = []
    for _y, _label, _u in [(64, "Depth 2", near_pixel), (119, "Depth 4", far_pixel)]:
        _rows.append(f'<text x="18" y="{_y+5}">{_label}</text><path d="M145 {_y}H640" stroke="#b7c8c5"/><circle cx="{595+6*_u}" cy="{_y}" r="8" fill="#ab601c"/><text x="663" y="{_y+5}">{_u:.0f} px</text>')
    mo.vstack([
        mo.Html(f"""<svg viewBox="0 0 750 163" role="img" aria-label="Image position of the same mark at two possible depths" style="width:100%;background:#f3f6f4;color:#163c38;fill:currentColor;font:16px sans-serif">
        <text x="18" y="25" font-size="13">PROJECTED IMAGE POSITION</text>
        <text x="595" y="26" text-anchor="middle" font-size="13">Input pixel 0</text>
        <path d="M595 35V140" stroke="#658b85" stroke-dasharray="3 4"/>{''.join(_rows)}</svg>"""),
        mo.md(f"**Predictions differ by {abs(near_pixel-far_pixel):.0f} pixels.** The focal length is 160 pixels. A sideways move shifts the mark by focal length × camera shift ÷ depth. A second view can distinguish these candidates if the mark is visible and identifiable. With few photos, some regions have little such evidence."),
    ])
    return (focal,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 02 &nbsp; Test a preference for smooth surfaces
    Cut through the centre of a sphere and follow one row of foreground pixels. A smooth surface has nearby depths at adjacent pixels. We add fixed noise to its known depth to mimic a jagged estimate. The black curve gives the answer for evaluation.

    **Try 0, 8, then 1000.** Watch the steep sides of the sphere as well as its centre. Which weight removes noise while keeping the curve?
    """)
    return


@app.cell
def _(mo):
    weight = mo.ui.slider(steps=[0, 1, 2, 4, 8, 16, 40, 100, 300, 1000], value=8,
                          show_value=True, label="Depth smoothness weight")
    weight
    return (weight,)


@app.cell
def _(np, sphere_profile):
    truth = sphere_profile()
    observed = truth + np.random.default_rng(42).normal(0, 0.08, len(truth))
    return observed, truth


@app.cell
def _(baseline, focal, mo, np, observed, rmse, solve_depth, truth, weight):
    from notebooks.lab import plot
    _fit = solve_depth(observed, weight.value)
    _roughness = np.mean(np.diff(_fit)**2)
    _off = rmse(observed, truth)
    _error = rmse(_fit, truth)
    _pixel_error = focal * baseline.value * rmse(1/_fit, 1/truth)
    _raw_pixels = focal * baseline.value * rmse(1/observed, 1/truth)
    mo.vstack([
        mo.Html(plot({"Reference":truth, "Fixed noisy estimate":observed, "Smoothed estimate":_fit},
                     "Depth across the centre of the sphere")),
        mo.md(f"""| Measurement | Weight 0 | Weight {weight.value} |
        |:--|--:|--:|
        | Depth error, RMS (scene units) | {_off:.4f} | {_error:.4f} |
        | Mean squared neighbour difference (scene units²) | {np.mean(np.diff(observed)**2):.5f} | {_roughness:.5f} |
        | Position error after a camera shift of {baseline.value:.2f}, RMS (pixels) | {_raw_pixels:.2f} | {_pixel_error:.2f} |
        """),
        mo.md("Set the camera shift above to **0.50**. Compare the position error at weights 0 and 8. Then set the weight to 1000. A flatter estimate can have a smaller neighbour difference and a larger depth error. The position metric tracks the same points across views and excludes occlusion."),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## From one row to a rendered patch
    The toy solver minimizes
    $$\sum_i(d_i-y_i)^2 + \lambda\sum_i(d_{i+1}-d_i)^2.$$
    The first term keeps the estimate $d$ close to the noisy values $y$. The second penalizes jumps between neighbours. The clean reference enters only the measurements.

    RegNeRF gets its supervision from **RGB photos**. During training it also renders a small depth patch from an unseen camera. For each interior pixel it penalizes squared depth differences to the right and below. Those differences provide a training signal even without a photo at that pose. Our direct fit to one row lets you inspect that preference without training a network.

    **Carry one prediction into Notebook 2.** If smoother geometry helps on a real object, what change would you expect in a held-out RGB view? Where could smoothing erase a real depth jump? Open `02_paper.py` to compare depth smoothing on and off on the stone arch.
    """)
    return


if __name__ == "__main__":
    app.run()
