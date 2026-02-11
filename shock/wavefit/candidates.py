#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.ndimage as ndimage

from .model import periodic_delta


def pick_candidate_points(xx, yy, envelope, sigma, options):
    smooth_sigma = float(options.get("envelope_smooth_sigma", 0.5))
    max_candidates = int(options.get("max_candidates", 32))
    threshold_fraction = float(options.get("envelope_threshold_fraction", 0.5))
    threshold_quantile = options.get("envelope_threshold_quantile", None)
    min_distance_sigma = float(options.get("candidate_min_distance_sigma", 3.0))

    env = np.array(envelope, copy=True)
    if smooth_sigma > 0.0:
        env = ndimage.gaussian_filter1d(env, sigma=smooth_sigma, axis=0, mode="wrap")
        env = ndimage.gaussian_filter1d(env, sigma=smooth_sigma, axis=1, mode="nearest")

    if threshold_quantile is None:
        threshold = threshold_fraction * float(env.max())
    else:
        qvalue = float(np.quantile(env, float(threshold_quantile)))
        threshold = max(threshold_fraction * float(env.max()), qvalue)

    xdx = np.median(np.diff(xx))
    ydy = np.median(np.diff(yy))
    sx = max(1, int(np.ceil(min_distance_sigma * sigma / max(abs(xdx), 1.0e-12))))
    sy = max(1, int(np.ceil(min_distance_sigma * sigma / max(abs(ydy), 1.0e-12))))
    size = (2 * sy + 1, 2 * sx + 1)
    localmax = env == ndimage.maximum_filter(env, size=size, mode=("wrap", "nearest"))
    mask = localmax & (env >= threshold)

    cand_iy, cand_ix = np.where(mask)
    if cand_ix.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), env

    amp = env[cand_iy, cand_ix]
    order = np.argsort(amp)[::-1]

    Ly = (yy[-1] - yy[0]) + ydy
    min_distance = min_distance_sigma * sigma
    selected_ix = []
    selected_iy = []

    for idx in order:
        ix = int(cand_ix[idx])
        iy = int(cand_iy[idx])
        x0 = xx[ix]
        y0 = yy[iy]

        keep = True
        for jx, jy in zip(selected_ix, selected_iy):
            dx = xx[jx] - x0
            dy = periodic_delta(yy[jy], y0, Ly)
            if np.sqrt(dx**2 + dy**2) < min_distance:
                keep = False
                break

        if keep:
            selected_ix.append(ix)
            selected_iy.append(iy)
        if len(selected_ix) >= max_candidates:
            break

    return np.array(selected_ix, dtype=np.int64), np.array(selected_iy, dtype=np.int64), env


def build_patch_masks(xx, yy, x0, y0, sigma, options):
    patch_radius_sigma = float(options.get("patch_radius_sigma", 3.0))
    radius = patch_radius_sigma * sigma
    dx = xx - x0
    Ly = (yy[-1] - yy[0]) + np.median(np.diff(yy))
    dy = periodic_delta(yy, y0, Ly)
    xmask = np.abs(dx) <= radius
    ymask = np.abs(dy) <= radius
    return xmask, ymask, Ly
