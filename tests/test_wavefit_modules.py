"""Unit tests for refactored wavefit helper modules."""

from pathlib import Path

import numpy as np


def test_wavefit_model_periodic_delta_wraps_minimum_distance():
    from shock.wavefit.model import periodic_delta

    period = 10.0
    center = 9.0
    value = 1.0
    delta = periodic_delta(value, center, period)
    assert np.isclose(delta, 2.0)


def test_wavefit_model_evaluate_quality_threshold_logic():
    from shock.wavefit.model import evaluate_fit_quality

    options = {"good_nrmse_bal_max": 0.4, "good_lambda_factor_max": 4.0}

    good = evaluate_fit_quality(0.35, kx=0.0, ky=1.0, sigma=4.0, options=options)
    assert good["is_good_overall"]

    bad_nrmse = evaluate_fit_quality(0.45, kx=0.0, ky=1.0, sigma=4.0, options=options)
    assert not bad_nrmse["is_good_nrmse"]
    assert not bad_nrmse["is_good_overall"]

    bad_scale = evaluate_fit_quality(0.3, kx=0.0, ky=0.1, sigma=4.0, options=options)
    assert not bad_scale["is_good_scale"]
    assert not bad_scale["is_good_overall"]


def test_wavefit_candidates_pick_points_respects_spacing():
    from shock.wavefit.candidates import pick_candidate_points

    xx = np.linspace(0.0, 20.0, 41)
    yy = np.linspace(0.0, 10.0, 21)
    env = np.zeros((yy.size, xx.size), dtype=np.float64)
    env[10, 20] = 10.0
    env[10, 21] = 9.0

    options = {
        "max_candidates": 8,
        "envelope_threshold_fraction": 0.1,
        "candidate_min_distance_sigma": 1.5,
        "envelope_smooth_sigma": 0.0,
    }
    cand_ix, cand_iy, _ = pick_candidate_points(xx, yy, env, sigma=1.0, options=options)
    assert cand_ix.size == 1
    assert cand_iy.size == 1
    assert int(cand_ix[0]) == 20
    assert int(cand_iy[0]) == 10


def test_wavefit_plot_quickcheck_smoke_writes_png(temp_dir):
    from shock.wavefit.plot import save_quickcheck_plot_12panel

    ny, nx = 16, 18
    patch_x = np.linspace(0.0, 1.0, nx)
    patch_y = np.linspace(0.0, 1.0, ny)
    rng = np.random.default_rng(42)

    fit_result = {
        "windowed_data_E": rng.normal(size=(ny, nx, 3)),
        "windowed_data_B": rng.normal(size=(ny, nx, 3)),
        "windowed_model_E": rng.normal(size=(ny, nx, 3)),
        "windowed_model_B": rng.normal(size=(ny, nx, 3)),
        "patch_x": patch_x,
        "patch_y": patch_y,
        "nrmse_balanced": 0.3,
        "nrmseE": 0.31,
        "nrmseB": 0.29,
        "wavelength_over_sigma": 2.0,
        "is_good_nrmse": True,
        "is_good_scale": True,
        "kx": 0.1,
        "ky": 0.8,
        "helicity": -1.0,
    }

    png_path = Path(temp_dir) / "wavefit-quickcheck-smoke.png"
    save_quickcheck_plot_12panel(str(png_path), fit_result, title="smoke", rms_normalize=True)
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_wavefit_fit_one_candidate_recovers_synthetic_wave():
    from shock.wavefit.fit import fit_one_candidate
    from shock.wavefit.model import build_xy, circular_model_cartesian

    xx = np.linspace(-12.0, 12.0, 29)
    yy = np.linspace(-8.0, 8.0, 23)
    X, Y = build_xy(xx, yy)
    Ly = (yy[-1] - yy[0]) + np.median(np.diff(yy))

    true = {
        "x0": 1.2,
        "y0": -1.8,
        "sigma": 2.8,
        "kx": 0.14,
        "ky": 0.78,
        "Ew": 0.25,
        "Bw": 0.45,
        "phiE": 0.4,
        "phiB": -0.9,
        "helicity": -1.0,
    }

    E, B, _ = circular_model_cartesian(
        X,
        Y,
        true["x0"],
        true["y0"],
        true["sigma"],
        Ly,
        true["kx"],
        true["ky"],
        true["Ew"],
        true["Bw"],
        true["phiE"],
        true["phiB"],
        helicity=true["helicity"],
    )

    rng = np.random.default_rng(7)
    E = E + 0.02 * rng.normal(size=E.shape)
    B = B + 0.02 * rng.normal(size=B.shape)

    options = {
        "patch_radius_sigma": 3.0,
        "fit_min_points": 64,
        "kx_init": 0.5,
        "kx_min": 0.0,
        "kx_max": 1.5,
        "ky_init": 0.5,
        "ky_abs_max": 1.5,
        "fit_multi_start": True,
        "kx_init_scan": [0.0, 0.15, 0.5],
        "ky_init_scan": [0.5, 0.8, -0.5],
        "helicity_scan": [1.0, -1.0],
        "good_nrmse_bal_max": 0.4,
        "good_lambda_factor_max": 4.0,
        "min_support_fraction": 0.2,
    }

    result = fit_one_candidate(E, B, xx, yy, true["x0"], true["y0"], true["sigma"], options)
    assert result["success"]
    assert result["nrmse_balanced"] < 0.4
    assert result["is_good_nrmse"]
    assert result["is_good_scale"]
