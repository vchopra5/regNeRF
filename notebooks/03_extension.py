import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from notebooks.lab import extension_case, plot, rmse, solve_depth
    return extension_case, mo, np, plot, rmse, solve_depth


@app.cell
def _(mo):
    mo.md("""
    # 3. Reduce the penalty at colour edges
    **Time: 15 minutes, then 5 minutes for team discussion.** Read `02_paper.py` first.

    **Question:** Can colour edges protect true depth boundaries?

    Change one term: multiply each squared depth difference by
    `exp(-10 * abs(colour difference))`. Keep the data, noise, depth weight,
    and solver fixed. The guide is a fixed synthetic colour signal.
    Reference depth is used only for error measurement.

    This is a controlled test of the penalty. It does not establish a gain
    for NeRF or DTU. It extends the mechanism from Notebook 1.

    **Predict:** What happens when a colour edge has no depth edge?
    What happens when a depth edge has no colour edge?
    """)
    return


@app.cell
def _(mo):
    case = mo.ui.dropdown(["Aligned edge", "Texture edge", "Missing colour edge"],
                          value="Aligned edge", label="Colour guide")
    case
    return (case,)


@app.cell
def _(case, extension_case, mo, np, plot, rmse, solve_depth):
    truth, observations, guide, boundary, flat = extension_case(case.value)
    baseline = solve_depth(observations, 8)
    weights = np.exp(-10 * np.abs(np.diff(guide)))
    changed = solve_depth(observations, 8, weights)
    rows = []
    for _label, _prediction in [("Off", observations), ("Uniform", baseline), ("Edge weighted", changed)]:
        rows.append({"Condition": _label, "All RMSE": rmse(_prediction, truth),
                     "Boundary RMSE": rmse(_prediction, truth, boundary),
                     "Flat RMSE": rmse(_prediction, truth, flat)})
    mo.vstack([
        mo.Html(plot({"Reference": truth, "Noisy depth": observations,
                      "Uniform": baseline, "Edge weighted": changed}, "Two flat surfaces with one depth step")),
        mo.Html(plot({"Colour guide": guide}, "Fixed colour guide", "Colour (0 to 1)")),
        mo.ui.table(rows, selection=None),
        mo.md(f"""
        **Change from uniform to edge weighted:**
        Boundary RMSE: {rmse(changed, truth, boundary)-rmse(baseline, truth, boundary):+.4f}.
        Flat RMSE: {rmse(changed, truth, flat)-rmse(baseline, truth, flat):+.4f}.
        Negative is better. All depths use scene units.

        Boundary pixels are within 3.5 pixels of the true step at 63.5.
        Flat pixels are at least 12.5 pixels from that step. These fixed masks
        do not depend on the estimate. The texture case adds a colour band
        on a flat surface. The missing-edge case has a constant colour guide.
        """),
    ])
    return


@app.cell
def _(extension_case, mo, np, rmse, solve_depth):
    summary = []
    for _case in ["Aligned edge", "Texture edge", "Missing colour edge"]:
        _truth, _obs, _guide, _boundary, _flat = extension_case(_case)
        _base = solve_depth(_obs, 8)
        _edge = solve_depth(_obs, 8, np.exp(-10 * np.abs(np.diff(_guide))))
        summary.append({"Case": _case,
                        "Boundary RMSE change": rmse(_edge, _truth, _boundary)-rmse(_base, _truth, _boundary),
                        "Flat RMSE change": rmse(_edge, _truth, _flat)-rmse(_base, _truth, _flat)})
    mo.vstack([mo.md("## All three cases: same seed and weight"),
               mo.ui.table(summary, selection=None)])
    return


@app.cell
def _(mo):
    mo.md("""
    ## Team conclusion: 5 minutes
    State one result from each notebook. Give the metric and its limit.

    1. Did uniform smoothness improve the sphere depth error?
    2. What does the available scan 40 evidence establish?
    3. When did edge weighting help, fail, or leave the result unchanged?

    **Next experiment:** Apply the edge factor to both horizontal and vertical
    patch differences in the NeRF loss. Keep all other settings fixed.
    Specify the source of colour edges at virtual views before training.
    If the guide is predicted RGB, detach it from the gradient and test its errors.
    Use training or validation evidence to fix parameters. Keep test images
    and masks out of that decision. Compare at the same step and scale.
    Measure reference-depth error at true boundaries and on flat surfaces,
    as well as foreground RGB PSNR. A better RGB score alone does not prove
    better geometry. A toy gain alone does not prove a scan 40 gain.
    """)
    return


if __name__ == "__main__":
    app.run()
