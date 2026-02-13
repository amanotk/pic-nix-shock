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
        - 'x0', 'y0': fit centers (arrays)
        - 'helicity': helicity (+1 or -1), stored from fit results
        - 'Bx', 'By', 'Bz': background magnetic field (arrays)
        - 'Vex', 'Vey', 'Vez': background electron velocity (arrays)
        - 'Vix', 'Viy', 'Viz': background ion velocity (arrays)
        - 'Ne', 'Ni': electron and ion number density (arrays, converted from charge density)
        - 'wc': absolute electron cyclotron frequency (|B| in normalized units)
        - 'wp': electron plasma frequency (sqrt(Ne) from fit data, where Ne is number density)
        - 'omega': wave frequency (signed, |k|*c*Ew/Bw with sign from phiE-phiB)
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
        "helicity",
        # Background fields and velocities
        "Bx",
        "By",
        "Bz",
        "Vex",
        "Vey",
        "Vez",
        "Vix",
        "Viy",
        "Viz",
        "Ne",
        "Ni",
    ]

    int_keys = ["ix", "iy"]
    bool_keys = ["is_good"]

    result = {}

    with h5py.File(fitfile, "r") as fp:
        if "snapshots" not in fp:
            raise KeyError(f"fitfile '{fitfile}' does not contain 'snapshots' group")

        snapshots = fp["snapshots"]

        # Collect all steps (keep as strings for h5py access)
        all_steps = sorted(step for step in snapshots.keys())

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

    # Compute omega (wave frequency)
    # |omega| = |k| * c * Ew/Bw (in normalized units c=1)
    # Sign: positive if phiE - phiB ~ +pi/2, negative if ~ -pi/2
    if all(k in result for k in ["kx", "ky", "Ew", "Bw"]):
        k_mag = np.sqrt(result["kx"] ** 2 + result["ky"] ** 2)
        omega_abs = k_mag * result["Ew"] / (result["Bw"] + 1e-32)
        # Wrap phiE - phiB to [-pi, pi]
        if all(k in result for k in ["phiE", "phiB"]):
            phi_diff = result["phiE"] - result["phiB"]
            phi_diff = np.mod(phi_diff + np.pi, 2.0 * np.pi) - np.pi
            # Positive if phi_diff > 0 (phiE leads phiB by ~+pi/2)
            # Negative if phi_diff < 0 (phiE lags phiB by ~-pi/2)
            result["omega"] = np.where(phi_diff >= 0, omega_abs, -omega_abs)
            # Handle NaN
            mask = (
                np.isnan(result["phiE"])
                | np.isnan(result["phiB"])
                | np.isnan(result["Ew"])
                | np.isnan(result["Bw"])
            )
            result["omega"][mask] = np.nan
        else:
            result["omega"] = np.abs(omega_abs)

    # Compute wc (electron cyclotron frequency) and wp (electron plasma frequency)
    # In normalized simulation units: wc = |B|, wp = sqrt(Ne)
    # wc = |qe| * B / (me * cc) = |B| (since qe=-1, me=1, cc=1)
    if all(k in result for k in ["Bx", "By", "Bz"]):
        B_mag = np.sqrt(result["Bx"] ** 2 + result["By"] ** 2 + result["Bz"] ** 2)
        result["wc"] = np.abs(B_mag)
        # wp = sqrt(Ne) - compute from Ne in fit results if available
        if "Ne" in result:
            ne_data = result["Ne"]
            result["wp"] = np.sqrt(ne_data)
        else:
            result["wp"] = np.full_like(B_mag, np.nan)

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
