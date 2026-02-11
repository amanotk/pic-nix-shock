#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pathlib
import sys

import h5py
import matplotlib as mpl
import numpy as np
import scipy.ndimage as ndimage
import tqdm

mpl.use("Agg") if __name__ == "__main__" else None
import matplotlib.pyplot as plt

if "PICNIX_DIR" in os.environ:
    sys.path.append(str(pathlib.Path(os.environ["PICNIX_DIR"]) / "script"))
try:
    from . import base
except ImportError:
    import base

try:
    import lmfit
except ImportError as exc:
    raise ImportError(
        "wavefit.py requires lmfit. Install project dependencies first (e.g., uv sync)."
    ) from exc


def wrap_to_pi(phi):
    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi


def periodic_delta(value, center, period):
    return ((value - center + 0.5 * period) % period) - 0.5 * period


def build_xy(xx, yy):
    X, Y = np.meshgrid(xx, yy)
    return X, Y


def build_window(X, Y, x0, y0, sigma, Ly):
    dx = X - x0
    dy = periodic_delta(Y, y0, Ly)
    return np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))


def circular_model_cartesian(X, Y, x0, y0, sigma, Ly, kx, ky, Ew, Bw, phiE, phiB, helicity=1.0):
    theta = np.arctan2(ky, kx)
    phase = kx * X + ky * Y
    window = build_window(X, Y, x0, y0, sigma, Ly)

    E1 = Ew * np.cos(phase + phiE) * window
    E2 = helicity * Ew * np.sin(phase + phiE) * window
    B1 = Bw * np.cos(phase + phiB) * window
    B2 = helicity * Bw * np.sin(phase + phiB) * window

    Ex = -np.sin(theta) * E1
    Ey = +np.cos(theta) * E1
    Ez = E2

    Bx = -np.sin(theta) * B1
    By = +np.cos(theta) * B1
    Bz = B2

    E = np.stack([Ex, Ey, Ez], axis=-1)
    B = np.stack([Bx, By, Bz], axis=-1)
    return E, B, window


def rms_floor(x, eps=1.0e-12):
    return max(np.sqrt(np.mean(x**2)), eps)


def calc_r2(y, yhat):
    y = y.reshape(-1)
    yhat = yhat.reshape(-1)
    ymean = np.mean(y)
    sst = np.sum((y - ymean) ** 2)
    if sst <= 1.0e-30:
        return 0.0
    sse = np.sum((y - yhat) ** 2)
    return 1.0 - sse / sst


def evaluate_fit_quality(nrmse_balanced, kx, ky, sigma, options):
    nrmse_limit = float(options.get("good_nrmse_bal_max", options.get("good_nrmse_max", 0.4)))
    lambda_factor = float(options.get("good_lambda_factor_max", 4.0))

    k_mag = float(np.sqrt(kx**2 + ky**2))
    if k_mag > 0.0:
        wavelength = 2.0 * np.pi / k_mag
    else:
        wavelength = np.inf

    is_good_nrmse = bool(nrmse_balanced <= nrmse_limit)
    is_good_scale = bool(wavelength <= lambda_factor * float(sigma))

    return {
        "k": k_mag,
        "wavelength": wavelength,
        "wavelength_over_sigma": wavelength / float(sigma) if sigma > 0.0 else np.inf,
        "is_good_nrmse": is_good_nrmse,
        "is_good_scale": is_good_scale,
        "is_good_overall": bool(is_good_nrmse and is_good_scale),
        "nrmse_limit": nrmse_limit,
        "lambda_factor": lambda_factor,
    }


def pick_candidate_points(xx, yy, envelope, sigma, options):
    smooth_sigma = float(options.get("envelope_smooth_sigma", 0.5))
    max_candidates = int(options.get("max_candidates", 32))
    threshold_fraction = float(options.get("envelope_threshold_fraction", 0.5))
    threshold_quantile = options.get("envelope_threshold_quantile", None)
    min_distance_sigma = float(options.get("candidate_min_distance_sigma", 3.0))

    env = np.array(envelope, copy=True)
    if smooth_sigma > 0.0:
        env = ndimage.gaussian_filter1d(env, sigma=smooth_sigma, axis=0, mode="wrap")
        env = ndimage.gaussian_filter1d(env, sigma=smooth_sigma, axis=1, mode="nearest")

    if threshold_quantile is None:
        threshold = threshold_fraction * float(env.max())
    else:
        qvalue = float(np.quantile(env, float(threshold_quantile)))
        threshold = max(threshold_fraction * float(env.max()), qvalue)

    xdx = np.median(np.diff(xx))
    ydy = np.median(np.diff(yy))
    sx = max(1, int(np.ceil(min_distance_sigma * sigma / max(abs(xdx), 1.0e-12))))
    sy = max(1, int(np.ceil(min_distance_sigma * sigma / max(abs(ydy), 1.0e-12))))
    size = (2 * sy + 1, 2 * sx + 1)
    localmax = env == ndimage.maximum_filter(env, size=size, mode=("wrap", "nearest"))
    mask = localmax & (env >= threshold)

    cand_iy, cand_ix = np.where(mask)
    if cand_ix.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), env

    amp = env[cand_iy, cand_ix]
    order = np.argsort(amp)[::-1]

    Ly = (yy[-1] - yy[0]) + ydy
    min_distance = min_distance_sigma * sigma
    selected_ix = []
    selected_iy = []

    for idx in order:
        ix = int(cand_ix[idx])
        iy = int(cand_iy[idx])
        x0 = xx[ix]
        y0 = yy[iy]

        keep = True
        for jx, jy in zip(selected_ix, selected_iy):
            dx = xx[jx] - x0
            dy = periodic_delta(yy[jy], y0, Ly)
            if np.sqrt(dx**2 + dy**2) < min_distance:
                keep = False
                break

        if keep:
            selected_ix.append(ix)
            selected_iy.append(iy)
        if len(selected_ix) >= max_candidates:
            break

    return np.array(selected_ix, dtype=np.int64), np.array(selected_iy, dtype=np.int64), env


def build_patch_masks(xx, yy, x0, y0, sigma, options):
    patch_radius_sigma = float(options.get("patch_radius_sigma", 3.0))
    radius = patch_radius_sigma * sigma
    dx = xx - x0
    Ly = (yy[-1] - yy[0]) + np.median(np.diff(yy))
    dy = periodic_delta(yy, y0, Ly)
    xmask = np.abs(dx) <= radius
    ymask = np.abs(dy) <= radius
    return xmask, ymask, Ly


def fit_one_candidate(E, B, xx, yy, x0, y0, sigma, options):
    xmask, ymask, Ly = build_patch_masks(xx, yy, x0, y0, sigma, options)
    min_points = int(options.get("fit_min_points", 64))

    if xmask.sum() * ymask.sum() < min_points:
        return {
            "success": False,
            "reason": "insufficient_points",
            "message": "patch has too few points",
            "x0": x0,
            "y0": y0,
        }

    xxp = xx[xmask]
    yyp = yy[ymask]
    Ep = E[np.ix_(ymask, xmask, np.arange(3))]
    Bp = B[np.ix_(ymask, xmask, np.arange(3))]
    Xp, Yp = build_xy(xxp, yyp)

    fullX, fullY = build_xy(xx, yy)
    Wfull = build_window(fullX, fullY, x0, y0, sigma, Ly)
    Wpatch = Wfull[np.ix_(ymask, xmask)]

    Ew_data = Ep * Wpatch[..., np.newaxis]
    Bw_data = Bp * Wpatch[..., np.newaxis]

    rms_e = rms_floor(Ew_data)
    rms_b = rms_floor(Bw_data)

    kx_min = float(options.get("kx_min", 0.1))
    kx_max = float(options.get("kx_max", 1.0))
    ky_abs_max = float(options.get("ky_abs_max", 1.0))
    kx_init = float(options.get("kx_init", 0.3))
    ky_init = float(options.get("ky_init", 0.0))

    def clip_value(value, vmin, vmax):
        return min(max(float(value), vmin), vmax)

    def unique_values(values, tol=1.0e-12):
        out = []
        for value in values:
            if all(abs(value - prev) > tol for prev in out):
                out.append(value)
        return out

    ky_scan_user = options.get("ky_init_scan", None)
    if ky_scan_user is None:
        ky_scan = [
            ky_init,
            -ky_init,
            0.0,
            0.2,
            -0.2,
            0.5,
            -0.5,
            0.8,
            -0.8,
            1.0,
            -1.0,
        ]
    else:
        ky_scan = [float(v) for v in ky_scan_user]
    ky_scan = unique_values([clip_value(v, -ky_abs_max, ky_abs_max) for v in ky_scan])

    kx_scan_user = options.get("kx_init_scan", None)
    if kx_scan_user is None:
        kx_scan = [kx_init, kx_min, 0.0, 0.2]
    else:
        kx_scan = [float(v) for v in kx_scan_user]
    kx_scan = unique_values([clip_value(v, kx_min, kx_max) for v in kx_scan])

    helicity_scan_user = options.get("helicity_scan", None)
    if helicity_scan_user is None:
        helicity_scan = [1.0, -1.0]
    else:
        helicity_scan = [1.0 if float(v) >= 0.0 else -1.0 for v in helicity_scan_user]
    helicity_scan = unique_values(helicity_scan)

    fit_multi_start = bool(options.get("fit_multi_start", True))
    if not fit_multi_start:
        ky_scan = [clip_value(ky_init, -ky_abs_max, ky_abs_max)]
        kx_scan = [clip_value(kx_init, kx_min, kx_max)]
        helicity_scan = [helicity_scan[0]]

    best = None
    best_helicity = 1.0

    for helicity in helicity_scan:
        for kx0 in kx_scan:
            for ky0 in ky_scan:
                params = lmfit.Parameters()
                params.add("kx", value=kx0, min=kx_min, max=kx_max)
                params.add("ky", value=ky0, min=-ky_abs_max, max=ky_abs_max)
                params.add("Ew", value=max(np.std(Ew_data), 1.0e-6), min=0.0)
                params.add("Bw", value=max(np.std(Bw_data), 1.0e-6), min=0.0)
                params.add("phiE", value=0.0, min=-np.pi, max=np.pi)
                params.add("phiB", value=0.0, min=-np.pi, max=np.pi)

                def objective(pars):
                    Em, Bm, _ = circular_model_cartesian(
                        Xp,
                        Yp,
                        x0,
                        y0,
                        sigma,
                        Ly,
                        pars["kx"].value,
                        pars["ky"].value,
                        pars["Ew"].value,
                        pars["Bw"].value,
                        pars["phiE"].value,
                        pars["phiB"].value,
                        helicity=helicity,
                    )
                    Ew_model = Em
                    Bw_model = Bm
                    rE = (Ew_data - Ew_model) / rms_e
                    rB = (Bw_data - Bw_model) / rms_b
                    return np.concatenate([rE.reshape(-1), rB.reshape(-1)])

                result = lmfit.minimize(objective, params, method="least_squares")

                if best is None:
                    best = result
                    best_helicity = helicity
                else:
                    best_chisqr = float(best.chisqr) if np.isfinite(best.chisqr) else np.inf
                    test_chisqr = float(result.chisqr) if np.isfinite(result.chisqr) else np.inf
                    if test_chisqr < best_chisqr:
                        best = result
                        best_helicity = helicity

    if best is None:
        return {
            "success": False,
            "reason": "fit_not_run",
            "message": "no fit trial executed",
            "x0": float(x0),
            "y0": float(y0),
        }

    result = best
    p = result.params
    Em, Bm, _ = circular_model_cartesian(
        Xp,
        Yp,
        x0,
        y0,
        sigma,
        Ly,
        p["kx"].value,
        p["ky"].value,
        p["Ew"].value,
        p["Bw"].value,
        p["phiE"].value,
        p["phiB"].value,
        helicity=best_helicity,
    )
    Ew_model = Em
    Bw_model = Bm

    diff_e = Ew_data - Ew_model
    diff_b = Bw_data - Bw_model
    data_all = np.concatenate([Ew_data.reshape(-1), Bw_data.reshape(-1)])
    diff_all = np.concatenate([diff_e.reshape(-1), diff_b.reshape(-1)])

    nrmse_raw = np.sqrt(np.mean(diff_all**2)) / rms_floor(data_all)
    nrmse_e = np.sqrt(np.mean(diff_e**2)) / rms_floor(Ew_data)
    nrmse_b = np.sqrt(np.mean(diff_b**2)) / rms_floor(Bw_data)
    nrmse_balanced = np.sqrt(0.5 * (nrmse_e**2 + nrmse_b**2))
    nrmse = nrmse_balanced

    r2e = calc_r2(Ew_data, Ew_model)
    r2b = calc_r2(Bw_data, Bw_model)

    support_fraction = float(np.sum(Wpatch) / max(np.sum(Wfull), 1.0e-30))
    min_support = float(options.get("min_support_fraction", 0.5))
    quality = evaluate_fit_quality(
        nrmse_balanced, float(p["kx"].value), float(p["ky"].value), sigma, options
    )

    reason = "ok"
    if not result.success:
        reason = "no_converge"
    elif support_fraction < min_support:
        reason = "low_support"
    elif not quality["is_good_nrmse"]:
        reason = "low_quality_nrmse"
    elif not quality["is_good_scale"]:
        reason = "low_quality_scale"

    return {
        "success": bool(result.success),
        "reason": reason,
        "message": str(result.message),
        "x0": float(x0),
        "y0": float(y0),
        "kx": float(p["kx"].value),
        "ky": float(p["ky"].value),
        "Ew": float(p["Ew"].value),
        "Bw": float(p["Bw"].value),
        "phiE": float(wrap_to_pi(p["phiE"].value)),
        "phiB": float(wrap_to_pi(p["phiB"].value)),
        "helicity": float(best_helicity),
        "nfev": int(result.nfev),
        "redchi": float(result.redchi) if np.isfinite(result.redchi) else np.nan,
        "nrmse": float(nrmse),
        "nrmse_balanced": float(nrmse_balanced),
        "nrmseE": float(nrmse_e),
        "nrmseB": float(nrmse_b),
        "nrmse_raw": float(nrmse_raw),
        "k": float(quality["k"]),
        "wavelength": float(quality["wavelength"]),
        "wavelength_over_sigma": float(quality["wavelength_over_sigma"]),
        "is_good_nrmse": bool(quality["is_good_nrmse"]),
        "is_good_scale": bool(quality["is_good_scale"]),
        "r2E": float(r2e),
        "r2B": float(r2b),
        "support_fraction": float(support_fraction),
        "is_good": bool(reason == "ok" and quality["is_good_overall"]),
        "patch_xmin": float(xxp.min()),
        "patch_xmax": float(xxp.max()),
        "patch_ymin": float(yyp.min()),
        "patch_ymax": float(yyp.max()),
        "windowed_data_E": Ew_data,
        "windowed_data_B": Bw_data,
        "windowed_model_E": Ew_model,
        "windowed_model_B": Bw_model,
        "patch_x": xxp,
        "patch_y": yyp,
    }


def save_diagnostic_plot(filename, envelope, xx, yy, ix, iy, fit_result):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), dpi=120, constrained_layout=True)
    X, Y = np.meshgrid(xx, yy)
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
    """Save 2x6 quickcheck plot for E/B data-model comparison.

    Top row: data, bottom row: model.
    Columns: Ex, Ey, Ez, Bx, By, Bz.

    Parameters
    ----------
    filename : str
        Output PNG file path.
    fit_result : dict
        Result dict returned by ``fit_one_candidate``.
    title : str or None
        Optional title; if None, a default title with fit metrics is used.
    rms_normalize : bool
        If True, normalize E by rms(E_data) and B by rms(B_data).
    """

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


class WaveFitAnalyzer(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "analyze" in self.options:
            for key in self.options["analyze"]:
                self.options[key] = self.options["analyze"][key]

    def read_parameter(self):
        return None

    def fit_single_snapshot(self, E, B, xx, yy):
        sigma = float(self.options.get("sigma", 3.0))
        envelope = np.linalg.norm(B, axis=-1)
        cand_ix, cand_iy, env_used = pick_candidate_points(xx, yy, envelope, sigma, self.options)

        fit_results = []
        for ix, iy in zip(cand_ix, cand_iy):
            x0 = float(xx[ix])
            y0 = float(yy[iy])
            fit_result = fit_one_candidate(E, B, xx, yy, x0, y0, sigma, self.options)
            fit_result["ix"] = int(ix)
            fit_result["iy"] = int(iy)
            fit_results.append(fit_result)

        return {
            "candidate_ix": cand_ix,
            "candidate_iy": cand_iy,
            "envelope": env_used,
            "fits": fit_results,
        }

    def write_snapshot_result(self, output_file, step, time, result):
        overwrite = bool(self.options.get("overwrite", False))
        if os.path.exists(output_file) and overwrite:
            os.remove(output_file)
        if os.path.exists(output_file) and not overwrite:
            print(
                "Output file {} already exists. Set overwrite=true to replace.".format(output_file)
            )
            return

        with h5py.File(output_file, "w") as fp:
            grp = fp.create_group("snapshots/{:08d}".format(int(step)))
            grp.attrs["step"] = int(step)
            grp.attrs["t"] = float(time)

            fits = result["fits"]
            nfit = len(fits)
            grp.create_dataset("candidate_ix", data=result["candidate_ix"])
            grp.create_dataset("candidate_iy", data=result["candidate_iy"])
            grp.create_dataset("envelope", data=result["envelope"])

            keys_float = [
                "x0",
                "y0",
                "kx",
                "ky",
                "Ew",
                "Bw",
                "phiE",
                "phiB",
                "redchi",
                "nrmse",
                "nrmse_balanced",
                "nrmseE",
                "nrmseB",
                "nrmse_raw",
                "k",
                "wavelength",
                "wavelength_over_sigma",
                "r2E",
                "r2B",
                "support_fraction",
                "patch_xmin",
                "patch_xmax",
                "patch_ymin",
                "patch_ymax",
            ]
            keys_int = ["ix", "iy", "nfev"]
            keys_bool = ["success", "is_good", "is_good_nrmse", "is_good_scale"]
            keys_str = ["reason", "message"]

            for key in keys_float:
                arr = np.full((nfit,), np.nan, dtype=np.float64)
                for i, item in enumerate(fits):
                    if key in item:
                        arr[i] = item[key]
                grp.create_dataset(key, data=arr)

            for key in keys_int:
                arr = np.full((nfit,), -1, dtype=np.int64)
                for i, item in enumerate(fits):
                    if key in item:
                        arr[i] = item[key]
                grp.create_dataset(key, data=arr)

            for key in keys_bool:
                arr = np.zeros((nfit,), dtype=np.int8)
                for i, item in enumerate(fits):
                    if key in item:
                        arr[i] = int(bool(item[key]))
                grp.create_dataset(key, data=arr)

            dt = h5py.string_dtype(encoding="utf-8")
            for key in keys_str:
                arr = np.array([str(item.get(key, "")) for item in fits], dtype=dt)
                grp.create_dataset(key, data=arr)

            for key in self.options:
                value = self.options[key]
                if isinstance(value, (int, float, str, bool)):
                    grp.attrs["option_{}".format(key)] = value

    def generate_diagnostics(self, result, xx, yy, step):
        if not bool(self.options.get("debug_plot", True)):
            return
        fit_results = result["fits"]
        if len(fit_results) == 0:
            return

        debug_plot_count = int(self.options.get("debug_plot_count", 8))
        output_prefix = str(self.options.get("debug_plot_prefix", "wavefit-debug"))
        outdir = pathlib.Path(self.get_filename(output_prefix, "")).parent
        outdir.mkdir(parents=True, exist_ok=True)

        rank = np.argsort(
            [item.get("nrmse_balanced", item.get("nrmse", np.inf)) for item in fit_results]
        )
        for i in rank[:debug_plot_count]:
            item = fit_results[int(i)]
            if "windowed_data_E" not in item:
                continue
            filename = outdir / ("{:s}-{:08d}-{:03d}.png".format(output_prefix, int(step), int(i)))
            title = (
                "step={:08d} cand={:03d} ix={} iy={} nrmse_bal={:.3f} "
                "(E={:.3f}, B={:.3f}) lambda/sigma={:.3f} good=({:d},{:d})"
            ).format(
                int(step),
                int(i),
                int(item.get("ix", -1)),
                int(item.get("iy", -1)),
                float(item.get("nrmse_balanced", item.get("nrmse", np.nan))),
                float(item.get("nrmseE", np.nan)),
                float(item.get("nrmseB", np.nan)),
                float(item.get("wavelength_over_sigma", np.nan)),
                int(bool(item.get("is_good_nrmse", False))),
                int(bool(item.get("is_good_scale", False))),
            )
            save_quickcheck_plot_12panel(str(filename), item, title=title, rms_normalize=True)

    def cleanup_large_arrays(self, result):
        for item in result["fits"]:
            for key in [
                "windowed_data_E",
                "windowed_data_B",
                "windowed_model_E",
                "windowed_model_B",
                "patch_x",
                "patch_y",
            ]:
                if key in item:
                    del item[key]

    def main(self):
        wavefile = self.get_filename(self.options.get("wavefile", "wavefilter"), ".h5")
        fitfile = self.get_filename(self.options.get("fitfile", "wavefit"), ".h5")
        snapshot_index = int(self.options.get("snapshot_index", 0))

        with h5py.File(wavefile, "r") as fp:
            nstep = fp["step"].shape[0]
            if snapshot_index < 0 or snapshot_index >= nstep:
                raise ValueError(
                    "snapshot_index out of range: {} (nstep={})".format(snapshot_index, nstep)
                )

            step = int(fp["step"][snapshot_index])
            time = float(fp["t"][snapshot_index])

            x = fp["x"][snapshot_index, ...] if fp["x"].ndim == 2 else fp["x"][()]
            y = fp["y"][snapshot_index, ...] if fp["y"].ndim == 2 else fp["y"][()]
            E = fp["E"][snapshot_index, ...]
            B = fp["B"][snapshot_index, ...]

        result = self.fit_single_snapshot(E, B, x, y)
        self.generate_diagnostics(result, x, y, step)
        self.cleanup_large_arrays(result)
        self.write_snapshot_result(fitfile, step, time, result)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Wave fitting tool")
    parser.add_argument(
        "-j",
        "--job",
        type=str,
        default="analyze",
        help="Type of job to perform (analyze)",
    )
    parser.add_argument("config", nargs=1, help="configuration file")
    args = parser.parse_args()
    config = args.config[0]

    jobs = args.job.split(",")
    for job in tqdm.tqdm(jobs):
        if job == "analyze":
            obj = WaveFitAnalyzer(config)
            obj.main()
        else:
            raise ValueError("Unknown job: {:s}".format(job))


if __name__ == "__main__":
    main()
