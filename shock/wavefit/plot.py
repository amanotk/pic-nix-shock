#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib as mpl
import numpy as np

mpl.use("Agg") if __name__ == "__main__" else None
import matplotlib.pyplot as plt

from .model import rms_floor


def save_diagnostic_plot(filename, envelope, xx, yy, ix, iy, fit_result):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=120, constrained_layout=True)
    ax = axs[0, 0]
    img = ax.imshow(
        envelope,
        origin="lower",
        extent=[xx.min(), xx.max(), yy.min(), yy.max()],
        aspect="equal",
        cmap="viridis",
    )
    ax.scatter([xx[ix]], [yy[iy]], color="r", s=25)
    ax.set_title("Envelope and candidate")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(img, ax=ax, shrink=0.8)

    bx_data = fit_result["windowed_data_B"][..., 0]
    bx_model = fit_result["windowed_model_B"][..., 0]
    bx_res = bx_data - bx_model

    ez_data = fit_result["windowed_data_E"][..., 2]
    ez_model = fit_result["windowed_model_E"][..., 2]
    px = fit_result["patch_x"]
    py = fit_result["patch_y"]
    extent_patch = [px.min(), px.max(), py.min(), py.max()]

    panels = [
        (axs[0, 1], bx_data, "Bx data"),
        (axs[0, 2], bx_model, "Bx model"),
        (axs[1, 0], bx_res, "Bx residual"),
        (axs[1, 1], ez_data, "Ez data"),
        (axs[1, 2], ez_model, "Ez model"),
    ]
    for axi, arr, title in panels:
        vmax = np.max(np.abs(arr))
        img = axi.imshow(
            arr,
            origin="lower",
            extent=extent_patch,
            aspect="equal",
            cmap="bwr",
            vmin=-vmax,
            vmax=vmax,
        )
        axi.set_title(title)
        axi.set_xlabel("x")
        axi.set_ylabel("y")
        fig.colorbar(img, ax=axi, shrink=0.8)

    txt = (
        "success={:s}  reason={:s}\n"
        "kx={:+.4f} ky={:+.4f}\n"
        "Ew={:.4e} Bw={:.4e}\n"
        "phiE={:+.3f} phiB={:+.3f}\n"
        "nrmse_bal={:.4f} (E={:.4f}, B={:.4f})\n"
        "k={:.4f} lambda/sigma={:.3f}\n"
        "redchi={:.4e} support={:.3f}"
    ).format(
        str(fit_result["success"]),
        fit_result["reason"],
        fit_result["kx"],
        fit_result["ky"],
        fit_result["Ew"],
        fit_result["Bw"],
        fit_result["phiE"],
        fit_result["phiB"],
        fit_result["nrmse"],
        fit_result.get("nrmseE", np.nan),
        fit_result.get("nrmseB", np.nan),
        fit_result.get("k", np.nan),
        fit_result.get("wavelength_over_sigma", np.nan),
        fit_result["redchi"],
        fit_result["support_fraction"],
    )
    fig.text(0.015, 0.01, txt, fontsize=9, family="monospace")
    fig.savefig(filename)
    plt.close(fig)


def save_quickcheck_plot_12panel(filename, fit_result, title=None, rms_normalize=True):
    dE = np.array(fit_result["windowed_data_E"], copy=True)
    dB = np.array(fit_result["windowed_data_B"], copy=True)
    mE = np.array(fit_result["windowed_model_E"], copy=True)
    mB = np.array(fit_result["windowed_model_B"], copy=True)

    if rms_normalize:
        rms_e = rms_floor(dE)
        rms_b = rms_floor(dB)
        dE = dE / rms_e
        dB = dB / rms_b
        mE = mE / rms_e
        mB = mB / rms_b
        labels = ["Ex/rmsE", "Ey/rmsE", "Ez/rmsE", "Bx/rmsB", "By/rmsB", "Bz/rmsB"]
    else:
        labels = ["Ex", "Ey", "Ez", "Bx", "By", "Bz"]

    comps_data = [dE[..., 0], dE[..., 1], dE[..., 2], dB[..., 0], dB[..., 1], dB[..., 2]]
    comps_model = [mE[..., 0], mE[..., 1], mE[..., 2], mB[..., 0], mB[..., 1], mB[..., 2]]

    px = np.asarray(fit_result["patch_x"])
    py = np.asarray(fit_result["patch_y"])
    extent = [float(px.min()), float(px.max()), float(py.min()), float(py.max())]

    e_max = max(np.max(np.abs(dE)), np.max(np.abs(mE)), 1.0e-12)
    b_max = max(np.max(np.abs(dB)), np.max(np.abs(mB)), 1.0e-12)

    fig, axs = plt.subplots(2, 6, figsize=(18, 6), dpi=120, constrained_layout=True)
    for j in range(6):
        vmax = e_max if j < 3 else b_max
        im0 = axs[0, j].imshow(
            comps_data[j],
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="bwr",
            vmin=-vmax,
            vmax=vmax,
        )
        im1 = axs[1, j].imshow(
            comps_model[j],
            origin="lower",
            extent=extent,
            aspect="equal",
            cmap="bwr",
            vmin=-vmax,
            vmax=vmax,
        )
        axs[0, j].set_title("data " + labels[j])
        axs[1, j].set_title("model " + labels[j])
        axs[0, j].set_xlabel("x")
        axs[1, j].set_xlabel("x")
        axs[0, j].set_ylabel("y")
        axs[1, j].set_ylabel("y")
        fig.colorbar(im0, ax=axs[0, j], shrink=0.65)
        fig.colorbar(im1, ax=axs[1, j], shrink=0.65)

    if title is None:
        title = (
            "nrmse_bal={:.3f} (E={:.3f}, B={:.3f})  "
            "lambda/sigma={:.3f}  good=({:d},{:d})  "
            "k=({:+.3f},{:+.3f}) h={:+.0f}"
        ).format(
            float(fit_result.get("nrmse_balanced", fit_result.get("nrmse", np.nan))),
            float(fit_result.get("nrmseE", np.nan)),
            float(fit_result.get("nrmseB", np.nan)),
            float(fit_result.get("wavelength_over_sigma", np.nan)),
            int(bool(fit_result.get("is_good_nrmse", False))),
            int(bool(fit_result.get("is_good_scale", False))),
            float(fit_result.get("kx", np.nan)),
            float(fit_result.get("ky", np.nan)),
            float(fit_result.get("helicity", np.nan)),
        )

    fig.suptitle(title)
    fig.savefig(filename)
    plt.close(fig)
