import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from notebooks.lab import ROOT, plot, rmse, solve_depth, sphere_profile
    return ROOT, mo, np, plot, rmse, solve_depth, sphere_profile


@app.cell
def _(mo):
    mo.md("""
    # 1. Why use depth smoothness?
    **Time: 15 minutes.** Start here. Then open `02_paper.py`.

    **Question:** Can a penalty on adjacent depth differences remove noise?

    A NeRF predicts colour and depth from camera rays. With three photos,
    several shapes can explain the same colours. A smoothness penalty adds
    a preference for small depth differences between adjacent pixels.

    We first isolate that penalty. We use one row of known sphere depth and
    add fixed noise. We solve for depth directly. This is a small depth
    problem, not NeRF training or a simulation of three camera views.

    **Before you move the control:** Predict what a very large penalty will do
    to noise and to the curved surface.
    """)
    return


@app.cell
def _(mo):
    strength = mo.ui.slider(0, 40, step=1, value=8, label="Smoothness weight")
    strength
    return (strength,)


@app.cell
def _(np, sphere_profile):
    reference = sphere_profile()
    noisy = reference + np.random.default_rng(42).normal(0, 0.08, len(reference))
    return noisy, reference


@app.cell
def _(mo, noisy, plot, reference, rmse, solve_depth, strength):
    estimate = solve_depth(noisy, strength.value)
    mo.vstack([
        mo.Html(plot({"Reference": reference, "Noisy depth": noisy,
                      "Smooth depth": estimate}, "One row through the sphere")),
        mo.ui.table([
            {"Condition": "No smoothness", "Depth RMSE": rmse(noisy, reference)},
            {"Condition": "Smoothness", "Depth RMSE": rmse(estimate, reference)},
        ], selection=None),
        mo.md(f"**RMSE change (smooth minus off): {rmse(estimate, reference)-rmse(noisy, reference):+.4f} scene units.** A negative change is better."),
    ])
    return


@app.cell
def _(ROOT, mo):
    mo.vstack([
        mo.md("""
        ## Read the evidence
        Try weights 0, 8, and 40. The noise sample stays fixed.

        We minimize `sum((depth - noisy_depth)^2) + weight * sum(diff(depth)^2)`.
        The first term keeps the observations. The second term favours smooth depth.
        All valid pixels have opacity 1 in this toy case. Background pixels are excluded.
        No reference depth enters the solution. It enters only the error measure.

        **Discuss:** Which weight removes noise with the least shape error?
        Why can a lower smoothness loss still give a larger depth error?

        **Connection:** Notebook 2 applies the same preference to depth patches
        from unobserved camera positions. Notebook 3 tests a real depth step.
        """),
        mo.image(str(ROOT / "toy_sphere/train/r_0.png"), width=220),
    ])
    return


if __name__ == "__main__":
    app.run()
