"""Unit tests for refactored wavefit helper modules."""

import pickle
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
    from shock.wavefit.cli import pick_candidate_points

    xx = np.linspace(0.0, 20.0, 41)
    yy = np.linspace(0.0, 10.0, 21)
    env = np.zeros((yy.size, xx.size), dtype=np.float64)
    env[10, 20] = 10.0
    env[10, 21] = 9.0

    options = {
        "max_candidates": 8,
        "envelope_threshold": 0.1,
        "candidate_distance": 1.5,
        "envelope_smooth_sigma": 0.0,
    }
    cand_ix, cand_iy, _ = pick_candidate_points(xx, yy, env, sigma=1.0, options=options)
    assert cand_ix.size == 1
    assert cand_iy.size == 1
    assert int(cand_ix[0]) == 20
    assert int(cand_iy[0]) == 10


def test_wavefit_candidates_no_limit_when_max_candidates_not_set():
    from shock.wavefit.cli import pick_candidate_points

    xx = np.linspace(0.0, 30.0, 61)
    yy = np.linspace(0.0, 15.0, 31)
    env = np.zeros((yy.size, xx.size), dtype=np.float64)
    env[5, 10] = 9.0
    env[10, 20] = 8.0
    env[15, 30] = 7.0

    options = {
        "envelope_threshold": 0.1,
        "candidate_distance": 1.0,
        "envelope_smooth_sigma": 0.0,
    }
    cand_ix, cand_iy, _ = pick_candidate_points(xx, yy, env, sigma=1.0, options=options)
    assert cand_ix.size == 3
    assert cand_iy.size == 3


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


def test_wavefit_plot_diagnostic_tolerates_missing_support_fraction(temp_dir):
    from shock.wavefit.plot import save_diagnostic_plot

    ny, nx = 16, 18
    xx = np.linspace(0.0, 1.0, nx)
    yy = np.linspace(0.0, 1.0, ny)
    envelope = np.zeros((ny, nx), dtype=np.float64)
    rng = np.random.default_rng(42)
    fit_result = {
        "windowed_data_E": rng.normal(size=(ny, nx, 3)),
        "windowed_data_B": rng.normal(size=(ny, nx, 3)),
        "windowed_model_E": rng.normal(size=(ny, nx, 3)),
        "windowed_model_B": rng.normal(size=(ny, nx, 3)),
        "patch_x": xx,
        "patch_y": yy,
        "success": True,
        "reason": "ok",
        "kx": 0.1,
        "ky": 0.8,
        "Ew": 0.2,
        "Bw": 0.3,
        "phiE": 0.2,
        "phiB": -0.1,
        "nrmse": 0.3,
        "nrmseE": 0.31,
        "nrmseB": 0.29,
        "k": 0.81,
        "wavelength_over_sigma": 2.0,
        "redchi": 1.2,
    }

    png_path = Path(temp_dir) / "wavefit-diagnostic-smoke.png"
    save_diagnostic_plot(str(png_path), envelope, xx, yy, 0, 0, fit_result)
    assert png_path.exists()
    assert png_path.stat().st_size > 0


def test_wavefit_fit_one_candidate_recovers_synthetic_wave():
    from shock.wavefit.cli import get_charges_from_parameter
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

    B_background = np.zeros_like(B)
    B_background[..., 0] = 2.0
    B_background[..., 1] = -1.0
    B_background[..., 2] = 0.5

    ve_true = np.array([0.2, -0.1, 0.05])
    vi_true = np.array([-0.3, 0.15, 0.02])
    Je0 = -2.0
    Ji0 = +3.0
    J_background = np.zeros(B.shape[:-1] + (8,), dtype=np.float64)
    J_background[..., 0] = Je0
    J_background[..., 1:4] = Je0 * ve_true[np.newaxis, np.newaxis, :]
    J_background[..., 4] = Ji0
    J_background[..., 5:8] = Ji0 * vi_true[np.newaxis, np.newaxis, :]

    options = {
        "fit_min_points": 64,
        "kx_init": 0.5,
        "kx_min": 0.0,
        "kx_max": 1.5,
        "ky_init": 0.5,
        "ky_abs_max": 1.5,
        "kx_init_scan": [0.0, 0.15, 0.5],
        "ky_init_scan": [0.5, 0.8, -0.5],
        "good_nrmse_bal_max": 0.4,
        "good_lambda_factor_max": 4.0,
    }

    qe, qi = get_charges_from_parameter({"nppc": 32, "wp": 1.0, "u0": 0.0125})

    result = fit_one_candidate(
        E,
        B,
        xx,
        yy,
        true["x0"],
        true["y0"],
        true["sigma"],
        options,
        B_background=B_background,
        J_background=J_background,
        qe=qe,
        qi=qi,
    )
    assert result["success"]
    assert result["nrmse_balanced"] < 0.4
    assert result["is_good_nrmse"]
    assert result["is_good_scale"]
    for key in ["kx_err", "ky_err", "Ew_err", "Bw_err", "phiE_err", "phiB_err"]:
        assert key in result
    assert "has_errorbars" in result
    for key in ["Bx", "By", "Bz", "Vex", "Vey", "Vez", "Vix", "Viy", "Viz"]:
        assert key in result
    assert np.isclose(result["Bx"], 2.0)
    assert np.isclose(result["By"], -1.0)
    assert np.isclose(result["Bz"], 0.5)
    assert np.isclose(result["Vex"], ve_true[0])
    assert np.isclose(result["Vey"], ve_true[1])
    assert np.isclose(result["Vez"], ve_true[2])
    assert np.isclose(result["Vix"], vi_true[0])
    assert np.isclose(result["Viy"], vi_true[1])
    assert np.isclose(result["Viz"], vi_true[2])
    assert np.isclose(result["Ne"], Je0 / qe)
    assert np.isclose(result["Ni"], Ji0 / qi)


def test_wavefit_patch_coordinates_are_contiguous_across_y_boundary():
    from shock.wavefit.fit import build_patch_coordinates

    xx = np.linspace(0.0, 50.0, 101)
    yy = np.linspace(0.0, 100.8, 85)
    y0 = float(yy[-2])
    sigma = 4.0
    options = {}

    _, y_idx, _, yyp, _ = build_patch_coordinates(
        xx, yy, x0=25.0, y0=y0, sigma=sigma, options=options
    )

    assert y_idx.size > 0
    dy = float(np.median(np.diff(yy)))
    ydiff = np.diff(yyp)
    assert np.all(ydiff > 0.0)
    assert float(np.max(ydiff)) <= 1.5 * dy


def test_wavefit_extract_parameter_from_embedded_wave_config(temp_dir):
    import h5py

    from shock.wavefit.cli import extract_parameter_from_wavefile

    h5_path = Path(temp_dir) / "wavefile-with-config.h5"
    config_obj = {"parameter": {"sigma": 0.0125, "u0": 0.25}}
    encoded = np.frombuffer(pickle.dumps(config_obj), dtype=np.int8)

    with h5py.File(h5_path, "w") as fp:
        fp.create_dataset("config", data=encoded, dtype=np.int8)

    with h5py.File(h5_path, "r") as fp:
        param = extract_parameter_from_wavefile(fp)

    assert isinstance(param, dict)
    assert np.isclose(float(param["sigma"]), 0.0125)
    assert np.isclose(float(param["u0"]), 0.25)


def test_wavefit_cli_select_debug_indices_modes():
    from shock.wavefit.cli import select_debug_indices

    assert np.array_equal(select_debug_indices(5, debug=False), np.array([0, 1, 2, 3, 4]))
    assert np.array_equal(
        select_debug_indices(5, debug=True, debug_count=2, debug_mode="head"), np.array([0, 1])
    )

    uniform = select_debug_indices(10, debug=True, debug_count=4, debug_mode="uniform")
    assert uniform.size == 4
    assert int(uniform[0]) == 0
    assert int(uniform[-1]) == 9


def test_wavefit_plot_job_writes_envelope_maps(temp_dir, monkeypatch):
    import h5py
    import msgpack
    import toml

    from shock.wavefit.cli import WaveFitAnalyzer

    work_root = temp_dir / "work"
    data_root = temp_dir / "data"
    run = "run1"
    dirname = "wf"
    monkeypatch.setenv("SHOCK_WORK_ROOT", str(work_root))
    monkeypatch.setenv("SHOCK_DATA_ROOT", str(data_root))

    profile_dir = data_root / run / "data"
    profile_dir.mkdir(parents=True, exist_ok=True)
    profile_path = profile_dir / "profile.msgpack"
    with open(profile_path, "wb") as fp:
        msgpack.dump({"configuration": {"parameter": {"sigma": 1.0, "u0": 0.2, "mime": 25.0}}}, fp)

    cfg_path = temp_dir / "wavefit-plot.toml"
    cfg = {
        "run": run,
        "dirname": dirname,
        "profile": "data/profile.msgpack",
        "plot": {
            "wavefile": "wavefilter",
            "fitfile": "wavefit",
            "plot_prefix": "wavefit-envelope",
        },
    }
    with open(cfg_path, "w", encoding="utf-8") as fp:
        toml.dump(cfg, fp)

    work_dir = work_root / run / dirname
    work_dir.mkdir(parents=True, exist_ok=True)

    x = np.linspace(0.0, 4.0, 5)
    y = np.linspace(0.0, 3.0, 4)
    B = np.zeros((1, y.size, x.size, 3), dtype=np.float64)
    B[0, ..., 0] = 0.2
    with h5py.File(work_dir / "wavefilter.h5", "w") as fp:
        fp.create_dataset("step", data=np.array([10], dtype=np.int64))
        fp.create_dataset("t", data=np.array([2.5], dtype=np.float64))
        fp.create_dataset("x", data=x)
        fp.create_dataset("y", data=y)
        fp.create_dataset("B", data=B)

    with h5py.File(work_dir / "wavefit.h5", "w") as fp:
        grp = fp.create_group("snapshots/00000010")
        grp.create_dataset("envelope", data=np.ones((y.size, x.size), dtype=np.float64) * 0.25)
        grp.create_dataset("ix", data=np.array([1, 3], dtype=np.int64))
        grp.create_dataset("iy", data=np.array([2, 1], dtype=np.int64))
        grp.create_dataset("is_good", data=np.array([1, 0], dtype=np.int8))
        grp.attrs["option_envelope_threshold"] = 0.1

    obj = WaveFitAnalyzer(str(cfg_path), option_section="plot")
    obj.options["debug"] = False
    obj.main_plot()

    out_png = work_dir / "wavefit-envelope-0000.png"
    assert out_png.exists()
    assert out_png.stat().st_size > 0


def test_wavefit_get_charges_from_parameter():
    from shock.wavefit.cli import get_charges_from_parameter

    parameter = {"nppc": 32, "wp": 1.0, "u0": 0.0125}
    qe, qi = get_charges_from_parameter(parameter)

    assert qe < 0
    assert qi > 0
    assert np.isclose(qi, -qe)

    gamma = np.sqrt(1.0 + 0.0125**2)
    expected_qe = -1.0 / 32 * np.sqrt(gamma)
    assert np.isclose(qe, expected_qe)


def test_wavefit_density_conversion():
    from shock.wavefit.fit import weighted_local_mean_velocity

    Je_patch = np.array([[[-5.0, 0.01, 0.02, 0.03]]])
    W = np.array([[1.0]])
    qe = -0.03125

    vx, vy, vz, ne = weighted_local_mean_velocity(Je_patch, W, qe)

    assert ne > 0
    assert np.isclose(ne, -5.0 / qe)


def test_wavefit_io_helicity_and_omega(temp_dir):
    import h5py

    from shock.wavefit.io import read_wavefit_results

    h5_path = temp_dir / "wavefit-result.h5"
    with h5py.File(h5_path, "w") as fp:
        grp = fp.create_group("snapshots/00000010")
        grp.attrs["step"] = 10
        grp.attrs["t"] = 2.5
        grp.create_dataset("kx", data=np.array([0.1]))
        grp.create_dataset("ky", data=np.array([0.8]))
        grp.create_dataset("Ew", data=np.array([0.2]))
        grp.create_dataset("Bw", data=np.array([0.3]))
        grp.create_dataset("phiE", data=np.array([np.pi / 2]))
        grp.create_dataset("phiB", data=np.array([0.0]))
        grp.create_dataset("helicity", data=np.array([1.0]))
        grp.create_dataset("Ne", data=np.array([100.0]))
        grp.create_dataset("Ni", data=np.array([100.0]))
        grp.create_dataset("Bx", data=np.array([1.0]))
        grp.create_dataset("By", data=np.array([-0.5]))
        grp.create_dataset("Bz", data=np.array([0.0]))

    results = read_wavefit_results(str(h5_path))

    assert results["omega"][0] > 0
    assert np.isclose(results["wp"][0], 10.0)
    assert np.isclose(results["wc"][0], np.sqrt(1.0**2 + 0.5**2))


def test_wavefit_omega_sign_with_helicity(temp_dir):
    import h5py

    from shock.wavefit.io import read_wavefit_results

    h5_path1 = temp_dir / "wavefit1.h5"
    with h5py.File(h5_path1, "w") as fp:
        grp = fp.create_group("snapshots/00000010")
        grp.attrs["step"] = 10
        grp.attrs["t"] = 2.5
        grp.create_dataset("kx", data=np.array([0.1]))
        grp.create_dataset("ky", data=np.array([0.8]))
        grp.create_dataset("Ew", data=np.array([0.2]))
        grp.create_dataset("Bw", data=np.array([0.3]))
        grp.create_dataset("phiE", data=np.array([np.pi / 2]))
        grp.create_dataset("phiB", data=np.array([0.0]))
        grp.create_dataset("helicity", data=np.array([1.0]))
        grp.create_dataset("Bx", data=np.array([1.0]))
        grp.create_dataset("By", data=np.array([-0.5]))

    results1 = read_wavefit_results(str(h5_path1))
    assert results1["omega"][0] > 0

    h5_path2 = temp_dir / "wavefit2.h5"
    with h5py.File(h5_path2, "w") as fp:
        grp = fp.create_group("snapshots/00000010")
        grp.attrs["step"] = 10
        grp.attrs["t"] = 2.5
        grp.create_dataset("kx", data=np.array([0.1]))
        grp.create_dataset("ky", data=np.array([0.8]))
        grp.create_dataset("Ew", data=np.array([0.2]))
        grp.create_dataset("Bw", data=np.array([0.3]))
        grp.create_dataset("phiE", data=np.array([np.pi / 2]))
        grp.create_dataset("phiB", data=np.array([0.0]))
        grp.create_dataset("helicity", data=np.array([-1.0]))
        grp.create_dataset("Bx", data=np.array([1.0]))
        grp.create_dataset("By", data=np.array([-0.5]))

    results2 = read_wavefit_results(str(h5_path2))
    assert results2["omega"][0] < 0


def test_wavefit_io_valid_field(temp_dir):
    import h5py

    from shock.wavefit.io import read_wavefit_results

    h5_path_valid = temp_dir / "wavefit-valid.h5"
    with h5py.File(h5_path_valid, "w") as fp:
        grp = fp.create_group("snapshots/00000010")
        grp.attrs["step"] = 10
        grp.attrs["t"] = 2.5
        grp.create_dataset("kx", data=np.array([0.1]))
        grp.create_dataset("ky", data=np.array([0.8]))
        grp.create_dataset("Ew", data=np.array([0.2]))
        grp.create_dataset("Bw", data=np.array([0.3]))
        grp.create_dataset("phiE", data=np.array([np.pi / 2]))
        grp.create_dataset("phiB", data=np.array([0.0]))
        grp.create_dataset("helicity", data=np.array([1.0]))
        grp.create_dataset("Bx", data=np.array([1.0]))
        grp.create_dataset("By", data=np.array([-0.5]))

    results_valid = read_wavefit_results(str(h5_path_valid))
    assert results_valid["valid"][0] == True

    h5_path_invalid = temp_dir / "wavefit-invalid.h5"
    with h5py.File(h5_path_invalid, "w") as fp:
        grp = fp.create_group("snapshots/00000010")
        grp.attrs["step"] = 10
        grp.attrs["t"] = 2.5
        grp.create_dataset("kx", data=np.array([0.1]))
        grp.create_dataset("ky", data=np.array([0.8]))
        grp.create_dataset("Ew", data=np.array([0.2]))
        grp.create_dataset("Bw", data=np.array([0.3]))
        grp.create_dataset("phiE", data=np.array([np.pi / 2]))
        grp.create_dataset("phiB", data=np.array([0.0]))
        grp.create_dataset("helicity", data=np.array([1.0]))
        grp.create_dataset("Bx", data=np.array([1.0]))
        grp.create_dataset("By", data=np.array([-0.5]))
        grp.create_dataset("Ne", data=np.array([100.0]))
        grp.create_dataset("Ni", data=np.array([100.0]))

    results_invalid = read_wavefit_results(str(h5_path_invalid))
    assert "valid" in results_invalid
