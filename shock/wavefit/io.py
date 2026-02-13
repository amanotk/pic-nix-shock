#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
I/O functions for wavefit results.

This module provides functions to read and parse HDF5 files produced by the
wavefit analysis.
"""

import h5py
import numpy as np


def read_wavefit_results(fitfile, good_only=False):
    """
    Read wavefit HDF5 results and return concatenated data arrays.

    Parameters
    ----------
    fitfile : str
        Path to the wavefit HDF5 output file.
    good_only : bool, optional
        If True, only return results where is_good=True. Default is False.

    Returns
    -------
    dict
        Dictionary containing:
        - 'step': snapshot step indices (array)
        - 't': snapshot times (array)
        - 'kx', 'ky': wave vectors (arrays)
        - 'kx_err', 'ky_err': errors (arrays)
        - 'Ew', 'Bw': amplitudes (arrays)
        - 'Ew_err', 'Bw_err': amplitude errors (arrays)
        - 'phiE', 'phiB': phases (arrays)
        - 'phiE_err', 'phiB_err': phase errors (arrays)
        - 'nrmse', 'nrmse_balanced': fitting quality metrics (arrays)
        - 'is_good': good fit flag (array)
        - 'is_good_nrmse', 'is_good_scale': individual quality criteria (arrays)
        - 'k': total wavenumber (array)
        - 'wavelength': wavelength (array)
        - 'x0', 'y0': fit centers (arrays)
        - 'helicity': helicity (+1 or -1), derived from phiE and phiB
    """
    float_keys = [
        "x0",
        "y0",
        "kx",
        "kx_err",
        "ky",
        "ky_err",
        "Ew",
        "Ew_err",
        "Bw",
        "Bw_err",
        "phiE",
        "phiE_err",
        "phiB",
        "phiB_err",
        "redchi",
        "nrmse",
        "nrmse_balanced",
        "nrmseE",
        "nrmseB",
        "k",
        "wavelength",
        "wavelength_over_sigma",
        "r2E",
        "r2B",
    ]

    int_keys = ["ix", "iy"]
    bool_keys = ["success", "is_good", "is_good_nrmse", "is_good_scale", "has_errorbars"]

    result = {}

    with h5py.File(fitfile, "r") as fp:
        if "snapshots" not in fp:
            raise KeyError(f"fitfile '{fitfile}' does not contain 'snapshots' group")

        snapshots = fp["snapshots"]

        # Collect all steps
        all_steps = sorted(int(step) for step in snapshots.keys())

        # Initialize lists for each key
        data = {key: [] for key in float_keys + int_keys + bool_keys}
        data["step"] = []
        data["t"] = []

        for step_str in all_steps:
            grp = snapshots[step_str]
            step = int(grp.attrs["step"])
            t = float(grp.attrs["t"])

            nfit = 0
            for key in float_keys + int_keys:
                if key in grp:
                    nfit = grp[key].shape[0]
                    break

            if nfit == 0:
                continue

            data["step"].append(np.full(nfit, step, dtype=np.int64))
            data["t"].append(np.full(nfit, t, dtype=np.float64))

            for key in float_keys:
                if key in grp:
                    data[key].append(grp[key][...])
                else:
                    data[key].append(np.full(nfit, np.nan))

            for key in int_keys:
                if key in grp:
                    data[key].append(grp[key][...])
                else:
                    data[key].append(np.full(nfit, -1, dtype=np.int64))

            for key in bool_keys:
                if key in grp:
                    data[key].append(grp[key][...].astype(bool))
                else:
                    data[key].append(np.full(nfit, False, dtype=bool))

        # Concatenate all snapshots
        for key in data:
            if len(data[key]) > 0:
                result[key] = np.concatenate(data[key])
            else:
                result[key] = np.array([])

    # Compute helicity from phase difference
    # helicity = sign(phiB - phiE) since model uses sin for both
    # but with E2 = h * Ew * sin, B2 = h * Bw * sin
    # The phase relationship determines handedness
    if "phiE" in result and "phiB" in result:
        phi_diff = result["phiB"] - result["phiE"]
        # Normalize to [-pi, pi]
        phi_diff = np.mod(phi_diff + np.pi, 2.0 * np.pi) - np.pi
        # helicity = +1 if phase diff is -pi/2 (model: sin leads cos)
        # helicity = -1 if phase diff is +pi/2 (model: sin lags cos)
        result["helicity"] = np.where(phi_diff < 0, 1, -1)
        # Handle NaN
        mask = np.isnan(result["phiE"]) | np.isnan(result["phiB"])
        result["helicity"][mask] = np.nan

    # Filter good fits if requested
    if good_only and "is_good" in result:
        mask = result["is_good"]
        for key in result:
            if isinstance(result[key], np.ndarray) and result[key].size > 0:
                result[key] = result[key][mask]

    return result


def get_helicity_from_fit(Ew, phiE, Bw, phiB):
    """
    Compute helicity from fit parameters.

    The model uses:
        E1 = Ew * cos(phase + phiE)
        E2 = helicity * Ew * sin(phase + phiE)

    So helicity = +1 when E2 leads E1 by -pi/2 (sin = cos - pi/2)
    and helicity = -1 when E2 lags E1 by +pi/2 (sin = cos + pi/2)

    Parameters
    ----------
    Ew : array-like
        Electric field amplitude
    phiE : array-like
        Electric field phase
    Bw : array-like
        Magnetic field amplitude
    phiB : array-like
        Magnetic field phase

    Returns
    -------
    array
        Helicity values (+1 or -1)
    """
    phi_diff = np.asarray(phiB) - np.asarray(phiE)
    # Normalize to [-pi, pi]
    phi_diff = np.mod(phi_diff + np.pi, 2.0 * np.pi) - np.pi
    # helicity = +1 when phi_diff < 0 (sin leads cos by -pi/2)
    helicity = np.where(phi_diff < 0, 1, -1)
    # Handle NaN
    mask = np.isnan(phi_diff)
    helicity = np.where(mask, np.nan, helicity)
    return helicity
