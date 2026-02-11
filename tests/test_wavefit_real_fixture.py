"""Regression tests for wavefit using compact real-data fixtures."""

import json
from pathlib import Path

import numpy as np


def _load_wavefit_fixture(root_dir):
    fixture_dir = root_dir / "tests" / "fixtures" / "wavefit"
    data = np.load(fixture_dir / "real_subset.npz")
    metadata = json.loads((fixture_dir / "real_subset_metadata.json").read_text(encoding="utf-8"))
    return data, metadata


def _prepare_fields(example, data, metadata):
    from shock import wavefilter

    local = int(example["local_snapshot_index"])
    E = np.array(data["E"][local], dtype=np.float64, copy=True)
    B = np.array(data["B"][local], dtype=np.float64, copy=True)
    B_raw = np.asarray(data["B_raw"][local], dtype=np.float64)

    Bmag = np.linalg.norm(B_raw, axis=-1)
    bhat = B_raw / (Bmag[..., np.newaxis] + 1.0e-32)
    E_para = np.sum(E * bhat, axis=-1)
    E = E - E_para[..., np.newaxis] * bhat

    smooth_sigma = float(metadata["preprocess"].get("smooth_sigma", 1.0))
    E = wavefilter.spatial_smooth(E, smooth_sigma)
    B = wavefilter.spatial_smooth(B, smooth_sigma)
    return E, B


def test_wavefit_real_fixture_examples_match_quality_labels(root_dir):
    from shock import wavefit

    data, metadata = _load_wavefit_fixture(root_dir)
    fit_options = dict(metadata["fit_options"])
    fit_options.update(
        {
            "kx_init_scan": [0.0, 0.15, 0.5],
            "ky_init_scan": [-0.5, 0.2, 0.5, 0.8],
            "helicity_scan": [1.0, -1.0],
        }
    )

    for example in metadata["examples"]:
        E, B = _prepare_fields(example, data, metadata)
        local = int(example["local_snapshot_index"])
        x = np.asarray(data["x"][local], dtype=np.float64)
        y = np.asarray(data["y"][local], dtype=np.float64)

        result = wavefit.fit_one_candidate(
            E,
            B,
            x,
            y,
            float(x[int(example["ix"])]),
            float(y[int(example["iy"])]),
            float(fit_options["sigma"]),
            fit_options,
        )

        assert bool(result["is_good"]) is bool(example["expected_good"])
        assert bool(result["is_good_nrmse"]) is bool(example["expected_good"])
        assert bool(result["is_good_scale"]) is bool(example["expected_good"])


def test_wavefit_goodness_rule_thresholds():
    from shock import wavefit

    options = {
        "good_nrmse_bal_max": 0.4,
        "good_lambda_factor_max": 4.0,
    }

    good = wavefit.evaluate_fit_quality(0.39, kx=0.0, ky=np.pi / 8.0, sigma=4.0, options=options)
    assert good["is_good_nrmse"]
    assert good["is_good_scale"]
    assert good["is_good_overall"]

    bad_nrmse = wavefit.evaluate_fit_quality(
        0.41, kx=0.0, ky=np.pi / 8.0, sigma=4.0, options=options
    )
    assert not bad_nrmse["is_good_nrmse"]
    assert bad_nrmse["is_good_scale"]
    assert not bad_nrmse["is_good_overall"]

    bad_scale = wavefit.evaluate_fit_quality(
        0.3, kx=0.0, ky=np.pi / 20.0, sigma=4.0, options=options
    )
    assert bad_scale["is_good_nrmse"]
    assert not bad_scale["is_good_scale"]
    assert not bad_scale["is_good_overall"]
