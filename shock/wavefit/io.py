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
- omega = |k| * c * Ew/Bw with sign from (phiE - phiB) * helicity
  - positive when (phiE - phiB) * helicity > 0
  - negative when (phiE - phiB) * helicity < 0
- helicity: +1 or -1 from fit (handedness of wave polarization)

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
        - 'omega': wave frequency (signed, |k|*c*Ew/Bw with sign from (phiE-phiB)*helicity)
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

    # Fallback: compute helicity from phiE/phiB if not stored in file
    if "helicity" in result and np.all(np.isnan(result["helicity"])):
        if all(k in result for k in ["phiE", "phiB"]):
            result["helicity"] = get_helicity_from_fit(
                result.get("Ew", np.nan),
                result["phiE"],
                result.get("Bw", np.nan),
                result["phiB"],
            )

    # Compute omega (wave frequency)
    # |omega| = |k| * c * Ew/Bw (in normalized units c=1)
    # Sign: (phiE - phiB) * helicity = +pi/2 -> omega positive
    #       (phiE - phiB) * helicity = -pi/2 -> omega negative
    if all(k in result for k in ["kx", "ky", "Ew", "Bw"]):
        k_mag = np.sqrt(result["kx"] ** 2 + result["ky"] ** 2)
        omega_abs = k_mag * result["Ew"] / (result["Bw"] + 1e-32)
        # Wrap phiE - phiB to [-pi, pi]
        if all(k in result for k in ["phiE", "phiB", "helicity"]):
            phi_diff = result["phiE"] - result["phiB"]
            phi_diff = np.mod(phi_diff + np.pi, 2.0 * np.pi) - np.pi
            # Positive if (phiE - phiB) * helicity > 0
            result["omega"] = np.where(phi_diff * result["helicity"] >= 0, omega_abs, -omega_abs)
            # Handle NaN
            mask = (
                np.isnan(result["phiE"])
                | np.isnan(result["phiB"])
                | np.isnan(result["helicity"])
                | np.isnan(result["Ew"])
                | np.isnan(result["Bw"])
            )
            result["omega"][mask] = np.nan
        else:
            result["omega"] = np.abs(omega_abs)

    # Compute wc (electron cyclotron frequency) and wp (electron plasma frequency)
    # In normalized simulation units: wc = |B|
    # wp = sqrt(Ne / nppc), normalized so wp = 1 when Ne = nppc
    # (This comes from wp^2 = ne * qe^2 / me, with me = 1/npc and qe = -wp_config/npc*sqrt(gamma))
    if all(k in result for k in ["Bx", "By", "Bz"]):
        B_mag = np.sqrt(result["Bx"] ** 2 + result["By"] ** 2 + result["Bz"] ** 2)
        result["wc"] = np.abs(B_mag)

    # Get nppc from parameter for wp calculation
    nppc = 1  # default
    try:
        parameter = extract_parameter_from_fitfile(fitfile)
        if parameter is not None:
            nppc = parameter.get("nppc", 1)
    except Exception:
        pass

    if "Ne" in result and nppc > 0:
        ne_data = result["Ne"]
        result["wp"] = np.sqrt(ne_data / nppc)
    elif all(k in result for k in ["Bx", "By", "Bz"]):
        result["wp"] = np.full_like(result["Bx"], np.nan)

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
