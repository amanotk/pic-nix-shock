import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return mo, np, plt


@app.cell
def _(mo):
    mo.md("""
    # Wavefit Statistics

    This notebook provides an overview of wavefit fitting results.
    """)
    return


@app.cell
def _(mo):
    default_filename = "work/ma05-tbn80-run001/wavetool-field_burst1/wavefit-result.h5"
    fitfile = mo.ui.text(
        label="Wavefit results file",
        placeholder=default_filename,
        full_width=True,
    )
    fitfile
    return default_filename, fitfile


@app.cell
def _(default_filename, fitfile):
    filename = fitfile.value.strip() or default_filename
    return (filename,)


@app.cell
def _(filename):
    from shock.wavefit import read_wavefit_results
    from shock.wavefit.analysis import (
        add_k_magnitude,
        add_phase_speed,
        filter_valid,
    )

    df = read_wavefit_results(filename)
    df = add_k_magnitude(df)
    df = add_phase_speed(df)
    df_valid = filter_valid(df)
    return df, df_valid


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    df_well_fitted = df[df["nrmse_balanced"] < 0.4]
    df_well_fitted
    return (df_well_fitted,)


@app.cell
def _(mo):
    mo.md(r"""
    # Fitting Quality
    """)
    return


@app.cell
def _(df_valid, plt):
    nrmse = df_valid["nrmse_balanced"]

    fig_nrmse = plt.figure(2)
    plt.hist(nrmse, bins=50, edgecolor="black", alpha=0.7)
    plt.xlabel("nrmse_balanced")
    plt.ylabel("Count")
    plt.title(f"Fitting Quality (N={len(nrmse)})")
    plt.grid()
    fig_nrmse
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dispersion Relation
    """)
    return


@app.cell
def _(df_well_fitted, kk, plt, ww):
    omega = df_well_fitted["omega"]
    k = df_well_fitted["k"]
    wc = df_well_fitted["wc"]
    wp = df_well_fitted["wp"]

    fig_disp = plt.figure(1)
    plt.plot(+kk, ww[:, 1], "k--")
    plt.plot(+kk, ww[:, 1] + 0.10 * kk, "r--")
    plt.plot(-kk, ww[:, 1], "k--")
    plt.plot(-kk, ww[:, 1] - 0.05 * kk, "r--")
    plt.scatter(k / wp, omega / wc, s=0.2, marker=".")
    plt.xlim(-1.0, +1.0)
    plt.ylim(+0.0, +0.5)
    plt.ylabel(r"$\omega / \Omega_{ce}$")
    plt.xlabel(r"$k c / \omega_{pe}$")
    plt.grid()
    fig_disp
    return


@app.cell
def _(np, plt):
    def lowfreq_em_para_wave_dispersion(k, *, c, wpe, wpi, wce, wci):
        w = np.zeros((len(k), 2))
        for i in range(len(k)):
            ki = (k[i] * c / wpi) ** 2
            ke = (k[i] * c / wpe) ** 2
            a2 = 1 + 1 / ki + 1 / ke
            a1 = wci + wce
            a0 = wci * wce
            ww = np.sort(np.roots([a2, a1, a0]))
            w[i, :] = ww
        return w

    c = 1.0e2
    mie = 400
    vai = 1.0
    wci = 1.0
    wpi = c
    wce = -wci * mie
    wpe = wpi * np.sqrt(mie)

    N = 100
    kk = np.geomspace(1.0e-1, 5.0e2, N)
    ww = lowfreq_em_para_wave_dispersion(kk, wpe=wpe, wce=wce, wpi=wpi, wci=wci, c=c)

    kk = kk / np.sqrt(mie)
    ww = ww / mie

    fig_linear_disp = plt.figure()
    plt.plot(kk, ww)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 0.5)
    plt.grid()
    fig_linear_disp
    return kk, ww


if __name__ == "__main__":
    app.run()
