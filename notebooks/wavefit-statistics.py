import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go

    return go, mo, np


@app.cell
def _(mo):
    mo.md("""
    # Wavefit Statistics
    """)
    return


@app.cell
def _(mo):
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    work_dir = repo_root / "work"

    h5_files = sorted(work_dir.rglob("wavefit-result.h5"))
    h5_options = {str(f.relative_to(work_dir)): str(f) for f in h5_files}

    default_selection = list(h5_options.keys())[0] if h5_options else None
    file_selector = mo.ui.dropdown(
        options=h5_options,
        label="Wavefit results file",
        value=default_selection,
        full_width=True,
    )

    threshold_slider = mo.ui.slider(
        start=0.1,
        stop=1.0,
        step=0.05,
        value=0.4,
        label="NRMSE threshold",
        full_width=True,
    )

    mo.vstack([file_selector, threshold_slider])
    return file_selector, threshold_slider


@app.cell
def _(file_selector):
    from shock.wavefit import read_wavefit_results
    from shock.wavefit.analysis import (
        add_k_magnitude,
        add_phase_speed,
        filter_valid,
    )

    df = read_wavefit_results(file_selector.value)
    df = add_k_magnitude(df)
    df = add_phase_speed(df)
    df_valid = filter_valid(df)
    return df, df_valid


@app.cell
def _(df_valid, go, mo, threshold_slider):
    nrmse = df_valid["nrmse_balanced"]
    threshold = threshold_slider.value
    fig_nrmse = go.Figure()
    fig_nrmse.add_trace(
        go.Histogram(
            x=nrmse,
            nbinsx=50,
            marker_line_width=1,
            marker_line_color="black",
            opacity=0.7,
        )
    )
    fig_nrmse.add_shape(
        type="line",
        x0=threshold,
        x1=threshold,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", dash="dash", width=2),
    )
    fig_nrmse.update_layout(
        xaxis_title="nrmse_balanced",
        yaxis_title="Count",
        title=f"Fitting Quality (N={len(nrmse)}, threshold={threshold:.2f})",
        template="plotly_white",
    )
    mo.ui.plotly(fig_nrmse)
    return


@app.cell
def _(df, threshold_slider):
    df_well_fitted = df[df["nrmse_balanced"] < threshold_slider.value]
    df_well_fitted
    return (df_well_fitted,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dispersion Relation
    """)
    return


@app.cell
def _(df_well_fitted, go, mo, np):
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
    wci = 1.0
    wpi = c
    wce = -wci * mie
    wpe = wpi * np.sqrt(mie)

    N = 100
    kk = np.geomspace(1.0e-1, 5.0e2, N)
    ww = lowfreq_em_para_wave_dispersion(kk, wpe=wpe, wce=wce, wpi=wpi, wci=wci, c=c)
    kk = kk / np.sqrt(mie)
    ww = ww / mie

    omega = df_well_fitted["omega"]
    k = df_well_fitted["k"]
    wc = df_well_fitted["wc"]
    wp = df_well_fitted["wp"]

    kk_theory = np.concatenate([-kk[::-1], kk])
    ww_theory = np.concatenate([ww[::-1, 1], ww[:, 1]])

    fig_disp = go.Figure()
    fig_disp.add_trace(
        go.Scatter(
            x=kk_theory,
            y=ww_theory,
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Cold Plasma Theory",
        )
    )
    fig_disp.add_trace(
        go.Scatter(
            x=k / wp,
            y=omega / wc,
            mode="markers",
            marker=dict(size=2),
            name="Simulation",
        )
    )
    fig_disp.update_layout(
        xaxis_title=r"$k c / \omega_{pe}$",
        yaxis_title=r"$\omega / \Omega_{ce}$",
        xaxis_range=[-1.0, 1.0],
        yaxis_range=[0.0, 0.5],
        template="plotly_white",
    )
    mo.ui.plotly(fig_disp)
    return


if __name__ == "__main__":
    app.run()
