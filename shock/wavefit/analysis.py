#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Analysis functions for wavefit results.

This module provides functions to analyze and compute statistics
from wavefit fitting results (pandas DataFrames).
"""

import numpy as np
import pandas as pd


def filter_valid(df):
    """
    Filter to only valid fit results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame returned by read_wavefit_results.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing only valid fits.
    """
    return df[df["valid"] == True].copy()


def add_phase_speed(df):
    """
    Add phase speed (omega/k) column to DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'omega' and 'k' columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with new 'phase_speed' column.
    """
    df = df.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        df["phase_speed"] = df["omega"] / df["k"]
    return df


def add_k_magnitude(df):
    """
    Add k magnitude (sqrt(kx^2 + ky^2)) column to DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'kx' and 'ky' columns.

    Returns
    -------
    pandas.DataFrame
        DataFrame with new 'k_magnitude' column.
    """
    df = df.copy()
    df["k_magnitude"] = np.sqrt(df["kx"] ** 2 + df["ky"] ** 2)
    return df


def overview_stats(df):
    """
    Get overview statistics for the fitting results.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame returned by read_wavefit_results.

    Returns
    -------
    dict
        Dictionary containing:
        - 'total_fits': total number of fit results
        - 'valid_fits': number of valid fits
        - 'invalid_fits': number of invalid fits
        - 't_min': minimum time
        - 't_max': maximum time
    """
    total = len(df)
    valid = df["valid"].sum() if "valid" in df.columns else 0
    invalid = total - valid
    t_min = df["t"].min() if "t" in df.columns and total > 0 else np.nan
    t_max = df["t"].max() if "t" in df.columns and total > 0 else np.nan

    return {
        "total_fits": total,
        "valid_fits": int(valid),
        "invalid_fits": int(invalid),
        "t_min": t_min,
        "t_max": t_max,
    }


def fitting_statistics(df):
    """
    Get fitting quality statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame returned by read_wavefit_results.

    Returns
    -------
    pandas.DataFrame
        Statistics for nrmse_balanced, nrmseE, nrmseB columns.
    """
    cols = ["nrmse_balanced", "nrmseE", "nrmseB"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    return df[available].describe()


def wave_statistics(df):
    """
    Get wave property statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame returned by read_wavefit_results.

    Returns
    -------
    pandas.DataFrame
        Statistics for kx, ky, omega, k, Ew, Bw columns.
    """
    cols = ["kx", "ky", "omega", "k", "k_magnitude", "Ew", "Bw", "phase_speed"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    return df[available].describe()


def background_statistics(df):
    """
    Get background field and velocity statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame returned by read_wavefit_results.

    Returns
    -------
    pandas.DataFrame
        Statistics for Bx, By, Bz, Vex, Vey, Vez columns.
    """
    cols = ["Bx", "By", "Bz", "Vex", "Vey", "Vez", "Vix", "Viy", "Viz", "Ne", "Ni", "wc", "wp"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.DataFrame()
    return df[available].describe()


def helicity_counts(df):
    """
    Get helicity distribution statistics.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'helicity' column.

    Returns
    -------
    dict
        Dictionary with counts and percentages for helicity +1 and -1.
    """
    if "helicity" not in df.columns:
        return {}

    helicity = df["helicity"].dropna()
    total = len(helicity)
    if total == 0:
        return {}

    pos = (helicity == 1).sum()
    neg = (helicity == -1).sum()

    return {
        "total": int(total),
        "helicity_+1": int(pos),
        "helicity_-1": int(neg),
        "helicity_+1_pct": float(pos / total * 100) if total > 0 else 0,
        "helicity_-1_pct": float(neg / total * 100) if total > 0 else 0,
    }


def mode_counts(df):
    """
    Get R-mode vs L-mode distribution.

    R-mode: omega > 0
    L-mode: omega < 0

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with 'omega' column.

    Returns
    -------
    dict
        Dictionary with counts and percentages for R-mode and L-mode.
    """
    if "omega" not in df.columns:
        return {}

    omega = df["omega"].dropna()
    total = len(omega)
    if total == 0:
        return {}

    r_mode = (omega > 0).sum()
    l_mode = (omega < 0).sum()

    return {
        "total": int(total),
        "R_mode": int(r_mode),
        "L_mode": int(l_mode),
        "R_mode_pct": float(r_mode / total * 100) if total > 0 else 0,
        "L_mode_pct": float(l_mode / total * 100) if total > 0 else 0,
    }


def describe_all_columns(df):
    """
    Get describe() for all numeric columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        Statistics for all numeric columns.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    return df[numeric_cols].describe()
