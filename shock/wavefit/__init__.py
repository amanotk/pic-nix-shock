#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .cli import WaveFitAnalyzer, main, pick_candidate_points
from .fit import build_patch_masks, fit_one_candidate
from .io import get_helicity_from_fit, read_wavefit_results
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
from .plot import save_diagnostic_plot, save_quickcheck_plot_12panel

__all__ = [
    "WaveFitAnalyzer",
    "main",
    "fit_one_candidate",
    "pick_candidate_points",
    "build_patch_masks",
    "read_wavefit_results",
    "get_helicity_from_fit",
    "wrap_to_pi",
    "periodic_delta",
    "build_xy",
    "build_window",
    "circular_model_cartesian",
    "rms_floor",
    "calc_r2",
    "evaluate_fit_quality",
    "save_diagnostic_plot",
    "save_quickcheck_plot_12panel",
]
