import marimo

__generated_with = "0.19.11"
app = marimo.App(
    width="medium",
    layout_file="layouts/wavefit-statistics.slides.json",
)


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, plt


@app.cell
def _(mo):
    mo.md("""
    # Wavefit Statistics

    This notebook provides an overview of wavefit fitting results.
    """)
    return


@app.cell
def _(mo):
    filename = "work/ma05-tbn80-run001//wavetool-field_burst1/wavefit-result.h5"
    fitfile = mo.ui.text(
        label="Wavefit results file",
        placeholder="work/ma05-tbn80-run001//wavetool-field_burst1/wavefit-result.h5",
        full_width=True,
    )
    return (filename,)


@app.cell
def _(filename):
    from shock.wavefit import read_wavefit_results
    from shock.wavefit.analysis import (
        filter_valid,
        add_phase_speed,
        add_k_magnitude,
        overview_stats,
        fitting_statistics,
        wave_statistics,
        background_statistics,
        helicity_counts,
        mode_counts,
    )

    df = read_wavefit_results(filename)
    df = add_k_magnitude(df)
    df = add_phase_speed(df)
    df_valid = filter_valid(df)
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df_well_fitted = df[df["nrmse_balanced"]<0.5]
    df_well_fitted
    return (df_well_fitted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dispersion Relation
    """)
    return


@app.cell
def _(df_well_fitted, plt):
    omega = df_well_fitted["omega"]
    k = df_well_fitted["k"]
    wc = df_well_fitted["wc"]
    wp = df_well_fitted["wp"]

    fig = plt.figure(1)
    plt.scatter(k/wp, omega/wc, marker=".")
    plt.xlim(-1.0, +1.0)
    plt.ylim(+0.0, +0.5)
    plt.ylabel(r"$\omega / \Omega_{ce}$")
    plt.xlabel(r"$k c / \omega_{pe}$")
    plt.grid()
    fig
    return


if __name__ == "__main__":
    app.run()
