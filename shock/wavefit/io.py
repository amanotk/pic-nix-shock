#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
I/O functions for wavefit results.

This module provides functions to read and parse HDF5 files produced by the
wavefit analysis.

Units and Conventions
---------------------
All quantities are in normalized simulation units where:
- B, E: normalized by b0 (reference magnetic field)
- n_e, n_i: number densities (NOT charge densities)
- wp = sqrt(Ne / nppc), normalized so wp = 1 when Ne = nppc
- wc = |B| (electron cyclotron frequency)
- helicity: +1 or -1 from fit (handedness of wave polarization)

Wave frequency and wavenumber conventions:
- sign_k_dot_b = sign(kx*Bx + ky*By)
- sign_phi_diff = sign(phiE - phiB), normalized to [-pi, pi]
- omega = |k| * c * Ew/Bw * sign_k_dot_b * sign_phi_diff * helicity
- k = -sign_k_dot_b * |k| * helicity

With these definitions:
- omega > 0 corresponds to R-mode polarization (omega < 0 => L-mode)
- sign(omega/k) corresponds to the sign of the Poynting flux

Physical relationships:
- wp^2 = ne * qe^2 / me
- wc = |qe| * B / me
- In normalized units: me = 1/npc, qe = -wp_config/nppc*sqrt(gamma)
- For simplicity: wp = sqrt(Ne / nppc) when Ne = nppc → wp = 1
"""

import os
import pickle

import h5py
import numpy as np
import pandas as pd

FLOAT_KEYS = [
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
    "nrmse_balanced",
    "nrmseE",
    "nrmseB",
    "helicity",
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

INT_KEYS = ["ix", "iy"]

REQUIRED_KEYS = {"kx", "ky", "Ew", "Bw", "phiE", "phiB", "helicity", "Bx", "By"}


def extract_parameter_from_fitfile(fitfile):
    """
    Extract parameter dict from embedded config in wavefile referenced by fitfile.

    Parameters
    ----------
    fitfile : str
        Path to the wavefit HDF5 output file.

    Returns
    -------
    dict or None
        Parameter dict if found, None otherwise.
    """
    with h5py.File(fitfile, "r") as fp:
        if "snapshots" not in fp:
            return None

        snapshots = fp["snapshots"]
        if len(snapshots) == 0:
            return None

        # Get wavefile path from first snapshot's options
        first_step = sorted(snapshots.keys())[0]
        grp = snapshots[first_step]
        wavefile = grp.attrs.get("option_wavefile", None)
        dirname = grp.attrs.get("option_dirname", None)

        if wavefile is None or dirname is None:
            return None

        # Construct wavefile path (same directory as fitfile)
        fitfile_dir = os.path.dirname(fitfile)
        wavefile_path = os.path.join(fitfile_dir, f"{wavefile}.h5")

        if not os.path.exists(wavefile_path):
            return None

        # Extract parameter from wavefile
        return extract_parameter_from_wavefile_path(wavefile_path)


def extract_parameter_from_wavefile_path(wavefile_path):
    """
    Extract parameter dict from embedded config in wavefile.

    Parameters
    ----------
    wavefile_path : str
        Path to the wavefile HDF5 file.

    Returns
    -------
    dict or None
        Parameter dict if found, None otherwise.
    """
    try:
        with h5py.File(wavefile_path, "r") as fp:
            return extract_parameter_from_wavefile(fp)
    except Exception:
        return None


def extract_parameter_from_wavefile(fileobj):
    """
    Extract parameter dict from embedded config in wavefile.

    Parameters
    ----------
    fileobj : h5py.File or h5py.Group
        Open HDF5 file or group containing 'config' dataset.

    Returns
    -------
    dict or None
        Parameter dict if found, None otherwise.
    """
    if "config" not in fileobj:
        return None

    try:
        config_obj = pickle.loads(fileobj["config"][()].tobytes())
    except Exception:
        return None

    if isinstance(config_obj, dict):
        if isinstance(config_obj.get("parameter", None), dict):
            return config_obj["parameter"]
        configuration = config_obj.get("configuration", None)
        if isinstance(configuration, dict) and isinstance(
            configuration.get("parameter", None), dict
        ):
            return configuration["parameter"]
    return None


def _load_raw_data(fitfile):
    """
    Load raw data arrays from wavefit HDF5 file.

    Parameters
    ----------
    fitfile : str
        Path to the wavefit HDF5 output file.

    Returns
    -------
    dict
        Dictionary containing raw data arrays and keys_array for validity check.
    """
    data = {key: [] for key in FLOAT_KEYS + INT_KEYS}
    data["step"] = []
    data["t"] = []
    keys_array = []

    with h5py.File(fitfile, "r") as fp:
        if "snapshots" not in fp:
            raise KeyError(f"fitfile '{fitfile}' does not contain 'snapshots' group")

        snapshots = fp["snapshots"]
        all_steps = sorted(snapshots.keys())

        for step_str in all_steps:
            grp = snapshots[step_str]
            step = int(grp.attrs["step"])
            t = float(grp.attrs["t"])

            nfit = 0
            for key in FLOAT_KEYS + INT_KEYS:
                if key in grp:
                    nfit = grp[key].shape[0]
                    break

            if nfit == 0:
                continue

            available_keys = set(grp.keys())
            data["step"].append(np.full(nfit, step, dtype=np.int64))
            data["t"].append(np.full(nfit, t, dtype=np.float64))
            keys_array.extend([available_keys] * nfit)

            for key in FLOAT_KEYS:
                if key in grp:
                    data[key].append(grp[key][...])
                else:
                    data[key].append(np.full(nfit, np.nan))

            for key in INT_KEYS:
                if key in grp:
                    data[key].append(grp[key][...])
                else:
                    data[key].append(np.full(nfit, -1, dtype=np.int64))

    for key in data:
        if len(data[key]) > 0:
            data[key] = np.concatenate(data[key])
        else:
            data[key] = np.array([])

    data["_keys_array"] = keys_array
    return data


def _compute_validity(keys_array):
    """
    Compute validity flag for each fit result.

    Parameters
    ----------
    keys_array : list of set
        List where each element is a set of available keys for that fit.

    Returns
    -------
    numpy.ndarray
        Boolean array where True indicates all required keys are present.
    """
    if len(keys_array) == 0:
        return np.array([], dtype=bool)

    validity = np.array(
        [all(k in available_keys for k in REQUIRED_KEYS) for available_keys in keys_array],
        dtype=bool,
    )
    return validity


def _compute_omega_and_k(data, valid):
    """
    Compute omega and k for valid fit results.

    Assumes all required keys are available for valid entries.
    Invalid entries will have NaN for omega and k.

    Formulas:
        sign_k_dot_b = sign(kx*Bx + ky*By)
        sign_phi_diff = sign(phiE - phiB) [normalized to [-pi, pi]]
        omega = |k| * c * Ew/Bw * sign_k_dot_b * sign_phi_diff * helicity
        k = -sign_k_dot_b * |k| * helicity

    Parameters
    ----------
    data : dict
        Dictionary containing raw data arrays.
    valid : numpy.ndarray
        Boolean array indicating which entries are valid.
    """
    n = len(valid)
    data["omega"] = np.full(n, np.nan)
    data["k"] = np.full(n, np.nan)

    if not np.any(valid):
        return

    kx = data["kx"][valid]
    ky = data["ky"][valid]
    Ew = data["Ew"][valid]
    Bw = data["Bw"][valid]
    phiE = data["phiE"][valid]
    phiB = data["phiB"][valid]
    helicity = data["helicity"][valid]
    Bx = data["Bx"][valid]
    By = data["By"][valid]

    k_abs = np.sqrt(kx**2 + ky**2)
    omega_abs = k_abs * Ew / (Bw + 1e-32)

    sign_k_dot_b = np.sign(kx * Bx + ky * By)
    sign_k_dot_b = np.where(sign_k_dot_b == 0, 1, sign_k_dot_b)

    phi_diff = phiE - phiB
    phi_diff = np.mod(phi_diff + np.pi, 2.0 * np.pi) - np.pi
    sign_phi_diff = np.sign(phi_diff)
    sign_phi_diff = np.where(sign_phi_diff == 0, 1, sign_phi_diff)

    # negative sign ensures that positive omega corresponds to R-mode polarization
    omega = -omega_abs * sign_k_dot_b * sign_phi_diff * helicity
    k = -sign_k_dot_b * k_abs * helicity

    data["omega"][valid] = omega
    data["k"][valid] = k


def _compute_wc_and_wp(data, valid, fitfile):
    """
    Compute wc and wp for valid fit results.

    Assumes required keys are available for valid entries.

    Parameters
    ----------
    data : dict
        Dictionary containing raw data arrays.
    valid : numpy.ndarray
        Boolean array indicating which entries are valid.
    fitfile : str
        Path to the wavefit HDF5 file (for extracting nppc).
    """
    n = len(valid)
    data["wc"] = np.full(n, np.nan)
    data["wp"] = np.full(n, np.nan)

    if not np.any(valid):
        return

    Bx = data["Bx"][valid]
    By = data["By"][valid]
    Bz = data["Bz"][valid]
    B_mag = np.sqrt(Bx**2 + By**2 + Bz**2)
    wc = np.abs(B_mag)
    data["wc"][valid] = wc

    nppc = 1
    try:
        parameter = extract_parameter_from_fitfile(fitfile)
        if parameter is not None:
            nppc = parameter.get("nppc", 1)
    except Exception:
        pass

    if "Ne" in data and nppc > 0:
        ne_data = data["Ne"][valid]
        wp = np.sqrt(ne_data / nppc)
        data["wp"][valid] = wp


def read_wavefit_results(fitfile):
    """
    Read wavefit HDF5 results and return as a pandas DataFrame.

    Parameters
    ----------
    fitfile : str
        Path to the wavefit HDF5 output file.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing:
        - 'step': snapshot step indices (array)
        - 't': snapshot times (array)
        - 'kx', 'ky': wave vectors (arrays)
        - 'kx_err', 'ky_err': errors (arrays)
        - 'Ew', 'Bw': amplitudes (arrays)
        - 'Ew_err', 'Bw_err': amplitude errors (arrays)
        - 'phiE', 'phiB': phases (arrays)
        - 'phiE_err', 'phiB_err': phase errors (arrays)
        - 'nrmse_balanced', 'nrmseE', 'nrmseB': fitting quality metrics (arrays)
        - 'valid': validity flag (True if all required keys present)
        - 'x0', 'y0': fit centers (arrays)
        - 'helicity': helicity (+1 or -1), stored from fit results
        - 'Bx', 'By', 'Bz': background magnetic field (arrays)
        - 'Vex', 'Vey', 'Vez': background electron velocity (arrays)
        - 'Vix', 'Viy', 'Viz': background ion velocity (arrays)
        - 'Ne', 'Ni': electron and ion number density (arrays, converted from charge density)
        - 'wc': absolute electron cyclotron frequency (|B| in normalized units)
        - 'wp': electron plasma frequency (sqrt(Ne) from fit data, where Ne is number density)
        - 'omega': wave frequency (signed, defined as |k|*c*Ew/Bw * sign_k_dot_b * sign_phi_diff * helicity)
        - 'k': signed wavenumber (defined as -sign_k_dot_b * |k| * helicity)
    """
    data = _load_raw_data(fitfile)
    keys_array = data.pop("_keys_array")

    valid = _compute_validity(keys_array)
    data["valid"] = valid

    _compute_omega_and_k(data, valid)
    _compute_wc_and_wp(data, valid, fitfile)

    return pd.DataFrame(data)
