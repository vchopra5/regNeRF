import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import marimo as mo
    from notebooks.lab import PAPER, ROOT
    import dtu
    return PAPER, ROOT, dtu, json, mo


@app.cell
def _(PAPER, mo):
    mo.md(f"""
    # 2. Does the effect transfer to sparse views?
    **Time: 25 minutes.** Read `01_toy.py` first. Continue with `03_extension.py`.

    **Target:** A single-scan investigation of RegNeRF Table 3.
    The paper reports an aggregate benchmark result. This project uses
    DTU scan 40, the stone arch from Figure 3, with a different training setup.
    Earlier scan 114 files show Buddha. They are excluded here.

    [Read the paper: Table 3 and Figure 3]({PAPER}).

    **Prediction:** If the mechanism from Notebook 1 transfers, depth
    regularization can improve colour prediction from new camera positions.
    This prediction can fail. Colour PSNR does not measure depth accuracy.
    """)
    return


@app.cell
def _(mo):
    mo.vstack([
        mo.md("## Published reference: Table 3\nThese are paper values, not local measurements."),
        mo.ui.table([
            {"Paper condition": "Without scene space annealing", "PSNR (dB)": 10.17},
            {"Paper condition": "Without geometry regularization", "PSNR (dB)": 14.34},
            {"Paper condition": "Without appearance regularization", "PSNR (dB)": 18.34},
            {"Paper condition": "Full method", "PSNR (dB)": 18.89},
        ], selection=None),
        mo.md("""
        The full method exceeds the row without geometry regularization by
        4.55 dB. Our local pair does not reproduce all Table 3 conditions.
        It tests the effect of the depth weight in the local implementation.
        Do not compare one scan directly with the benchmark mean.
        """),
    ])
    return


@app.cell
def _(ROOT, mo):
    _photos = [ROOT / "data/dtu/scan40/image" / f"{_i:06}.png" for _i in (25, 22, 28)]
    mo.vstack([
        mo.md("## Three training views\nUse these photos for training. Keep all test photos and masks out of parameter selection."),
        mo.hstack([mo.vstack([mo.md(f"View {_i}"), mo.image(str(_p), width=210)])
                   for _i, _p in zip((25, 22, 28), _photos) if _p.exists()]),
    ])
    return


@app.cell
def _(ROOT, dtu, mo):
    _off = ROOT / "renders/scan40_pair/view45_depth_off.png"
    _on = ROOT / "renders/scan40_pair/view45_depth_on.png"
    _target = ROOT / "data/dtu/scan40/image/000045.png"
    mo.vstack([
        mo.md("""
        ## Saved qualitative example: held-out view 45
        These existing render files are for inspection. Their filenames do not
        prove the training revision or settings. Do not use them to select a
        model or a depth weight. Verify the pair with the evaluation record below.
        """),
        mo.hstack([mo.vstack([mo.md(_label), mo.image(str(_path), width=210)])
                   for _label, _path in [("Depth off", _off), ("Depth on", _on), ("Reference photo", _target)]
                   if _path.exists()]),
        mo.md(f"""
        **Evaluation IDs from `dtu.py`:** {', '.join(map(str, dtu.TEST_IDS))}.

        View 45 is a fixed illustration. The score must include all 25 test views.
        Inspect missing structure and colour errors. Describe only what is visible.
        Use reference depth for a geometry conclusion.
        """),
    ])
    return


@app.cell
def _(ROOT, dtu, json, mo):
    _record = ROOT / "notebooks/results/scan40.json"
    if _record.exists():
        result = json.loads(_record.read_text())
        _rows = result["per_view"]
        if len(_rows) != 25 or tuple(_r["view"] for _r in _rows) != dtu.TEST_IDS:
            raise ValueError("The result must contain the 25 test IDs in protocol order.")
        _off_mean = sum(_r["off_db"] for _r in _rows) / 25
        _on_mean = sum(_r["on_db"] for _r in _rows) / 25
        _delta = _on_mean - _off_mean
        _outcome = "increase" if _delta > 0 else "decrease" if _delta < 0 else "tie"
        mo.output.replace(mo.vstack([
            mo.md(f"""
            ## Local measurement
            Off: **{_off_mean:.3f} dB**. On: **{_on_mean:.3f} dB**.
            On minus off: **{_delta:+.3f} dB** ({_outcome}).
            This is the arithmetic mean of 25 foreground RGB PSNR values.
            It is not training batch PSNR.
            """),
            mo.ui.table(_rows, selection=None),
            mo.accordion({"Evaluation record and limits": mo.md("```json\n" + json.dumps(result["record"], indent=2) + "\n```")}),
        ]))
    else:
        mo.output.replace(mo.md("""
        ## Local measurement: not yet verified
        No evaluation record is present. The saved images above do not establish
        a numerical result or a completed Table 3 reproduction.

        Before the review, run this command from the project root:

        ```text
        python -m notebooks.evaluate_pair
        ```

        It evaluates both saved step-40000 models at scale 2. It uses
        `eval_dtu.py` for each model. It does not train a model. It saves all
        25 scores per condition, checkpoint hashes, and the evaluation code state.
        It cannot recover missing training records. Reload this notebook after it finishes.
        """))
    return


@app.cell
def _(mo):
    mo.md("""
    ## Control the comparison
    For a new verified pair, use views 25, 22, and 28; seed 42; Adam;
    44000 steps; and identical ray samples, bounds schedule, and resolution.
    Change only the depth weight. Use a separate random generator for depth
    samples. Skip depth patches when the depth weight is zero.
    In the on condition, the scheduled weight decreases from 400 to 0.1
    by step 512. In the off condition, the weight stays zero.

    Compare saved models at the same step. Use scale 2 for both evaluations.
    For each view, compute RGB MSE on foreground pixels, then compute
    `PSNR = -10 * log10(MSE)`. Take the arithmetic mean of the 25 PSNR values.
    Report `on mean - off mean`, including a tie or a decrease.

    **Preparation limit:** `compare_depth.py` currently uses scale 4 for its
    automatic evaluation. Use `eval_dtu.py --scale 2` for both saved models
    to follow the review protocol. Do not mix scores from different scales.

    **Discuss:** Does the evidence support better colour prediction on this scan?
    What reference-depth evidence is still needed? Which controls lack a
    training record for the existing pair?

    **Connection:** A smooth surface is useful inside an object. A true depth
    boundary needs a different treatment. Test that limit in Notebook 3.
    """)
    return


if __name__ == "__main__":
    app.run()
