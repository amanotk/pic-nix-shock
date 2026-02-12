#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np

try:
    import lmfit
except ImportError as exc:
    raise ImportError(
        "wavefit requires lmfit. Install project dependencies first (e.g., uv sync)."
    ) from exc

from .model import (
    build_window,
    build_xy,
    calc_r2,
    circular_model_cartesian,
    evaluate_fit_quality,
    periodic_delta,
    rms_floor,
    wrap_to_pi,
)


PATCH_RADIUS_SIGMA = 3.0


def build_patch_masks(xx, yy, x0, y0, sigma, options):
    radius = PATCH_RADIUS_SIGMA * sigma
    dx = xx - x0
    Ly = (yy[-1] - yy[0]) + np.median(np.diff(yy))
    dy = periodic_delta(yy, y0, Ly)
    xmask = np.abs(dx) <= radius
    ymask = np.abs(dy) <= radius
    return xmask, ymask, Ly


def build_patch_coordinates(xx, yy, x0, y0, sigma, options):
    xmask, ymask, Ly = build_patch_masks(xx, yy, x0, y0, sigma, options)
    x_idx = np.where(xmask)[0]
    y_idx = np.where(ymask)[0]

    y_rel = periodic_delta(yy[y_idx], y0, Ly)
    y_order = np.argsort(y_rel)
    y_idx = y_idx[y_order]
    y_rel = y_rel[y_order]

    xxp = xx[x_idx]
    yyp = y0 + y_rel
    return x_idx, y_idx, xxp, yyp, Ly


def fit_one_candidate(E, B, xx, yy, x0, y0, sigma, options):
    x_idx, y_idx, xxp, yyp, Ly = build_patch_coordinates(xx, yy, x0, y0, sigma, options)
    min_points = int(options.get("fit_min_points", 64))

    if x_idx.size * y_idx.size < min_points:
        return {
            "success": False,
            "reason": "insufficient_points",
            "message": "patch has too few points",
            "x0": x0,
            "y0": y0,
        }

    Ep = E[np.ix_(y_idx, x_idx, np.arange(3))]
    Bp = B[np.ix_(y_idx, x_idx, np.arange(3))]
    Xp, Yp = build_xy(xxp, yyp)

    fullX, fullY = build_xy(xx, yy)
    Wfull = build_window(fullX, fullY, x0, y0, sigma, Ly)
    Wpatch = Wfull[np.ix_(y_idx, x_idx)]

    Ew_data = Ep * Wpatch[..., np.newaxis]
    Bw_data = Bp * Wpatch[..., np.newaxis]

    rms_e = rms_floor(Ew_data)
    rms_b = rms_floor(Bw_data)

    kx_min = float(options.get("kx_min", 0.1))
    kx_max = float(options.get("kx_max", 2.0))
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
        ky_scan = [ky_init, -ky_init, 0.0, 0.2, -0.2, 0.5, -0.5, 0.8, -0.8, 1.0, -1.0]
    else:
        ky_scan = [float(v) for v in ky_scan_user]
    ky_scan = unique_values([clip_value(v, -ky_abs_max, ky_abs_max) for v in ky_scan])

    kx_scan_user = options.get("kx_init_scan", None)
    if kx_scan_user is None:
        kx_scan = [kx_init, kx_min, 0.0, 0.2]
    else:
        kx_scan = [float(v) for v in kx_scan_user]
    kx_scan = unique_values([clip_value(v, kx_min, kx_max) for v in kx_scan])

    helicity_scan = [1.0, -1.0]

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
                    rE = (Ew_data - Em) / rms_e
                    rB = (Bw_data - Bm) / rms_b
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

    def get_stderr(name):
        value = p[name].stderr
        if value is None:
            return np.nan
        value = float(value)
        if not np.isfinite(value):
            return np.nan
        return value

    kx_err = get_stderr("kx")
    ky_err = get_stderr("ky")
    Ew_err = get_stderr("Ew")
    Bw_err = get_stderr("Bw")
    phiE_err = get_stderr("phiE")
    phiB_err = get_stderr("phiB")
    has_errorbars = bool(getattr(result, "errorbars", False))
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

    diff_e = Ew_data - Em
    diff_b = Bw_data - Bm
    data_all = np.concatenate([Ew_data.reshape(-1), Bw_data.reshape(-1)])
    diff_all = np.concatenate([diff_e.reshape(-1), diff_b.reshape(-1)])

    nrmse_raw = np.sqrt(np.mean(diff_all**2)) / rms_floor(data_all)
    nrmse_e = np.sqrt(np.mean(diff_e**2)) / rms_floor(Ew_data)
    nrmse_b = np.sqrt(np.mean(diff_b**2)) / rms_floor(Bw_data)
    nrmse_balanced = np.sqrt(0.5 * (nrmse_e**2 + nrmse_b**2))

    r2e = calc_r2(Ew_data, Em)
    r2b = calc_r2(Bw_data, Bm)

    quality = evaluate_fit_quality(
        nrmse_balanced, float(p["kx"].value), float(p["ky"].value), sigma, options
    )

    reason = "ok"
    if not result.success:
        reason = "no_converge"
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
        "kx_err": float(kx_err),
        "ky": float(p["ky"].value),
        "ky_err": float(ky_err),
        "Ew": float(p["Ew"].value),
        "Ew_err": float(Ew_err),
        "Bw": float(p["Bw"].value),
        "Bw_err": float(Bw_err),
        "phiE": float(wrap_to_pi(p["phiE"].value)),
        "phiE_err": float(phiE_err),
        "phiB": float(wrap_to_pi(p["phiB"].value)),
        "phiB_err": float(phiB_err),
        "helicity": float(best_helicity),
        "nfev": int(result.nfev),
        "redchi": float(result.redchi) if np.isfinite(result.redchi) else np.nan,
        "nrmse": float(nrmse_balanced),
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
        "is_good": bool(reason == "ok" and quality["is_good_overall"]),
        "has_errorbars": has_errorbars,
        "patch_xmin": float(xxp.min()),
        "patch_xmax": float(xxp.max()),
        "patch_ymin": float(yyp.min()),
        "patch_ymax": float(yyp.max()),
        "windowed_data_E": Ew_data,
        "windowed_data_B": Bw_data,
        "windowed_model_E": Em,
        "windowed_model_B": Bm,
        "patch_x": xxp,
        "patch_y": yyp,
    }
