#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np


def wrap_to_pi(phi):
    return ((phi + np.pi) % (2.0 * np.pi)) - np.pi


def periodic_delta(value, center, period):
    return ((value - center + 0.5 * period) % period) - 0.5 * period


def build_xy(xx, yy):
    X, Y = np.meshgrid(xx, yy)
    return X, Y


def build_window(X, Y, x0, y0, sigma, Ly):
    dx = X - x0
    dy = periodic_delta(Y, y0, Ly)
    return np.exp(-(dx**2 + dy**2) / (2.0 * sigma**2))


def circular_model_cartesian(X, Y, x0, y0, sigma, Ly, kx, ky, Ew, Bw, phiE, phiB, helicity=1.0):
    theta = np.arctan2(ky, kx)
    phase = kx * X + ky * Y
    window = build_window(X, Y, x0, y0, sigma, Ly)

    E1 = Ew * np.cos(phase + phiE) * window
    E2 = helicity * Ew * np.sin(phase + phiE) * window
    B1 = Bw * np.cos(phase + phiB) * window
    B2 = helicity * Bw * np.sin(phase + phiB) * window

    Ex = -np.sin(theta) * E1
    Ey = +np.cos(theta) * E1
    Ez = E2

    Bx = -np.sin(theta) * B1
    By = +np.cos(theta) * B1
    Bz = B2

    E = np.stack([Ex, Ey, Ez], axis=-1)
    B = np.stack([Bx, By, Bz], axis=-1)
    return E, B, window


def rms_floor(x, eps=1.0e-12):
    return max(np.sqrt(np.mean(x**2)), eps)


def calc_r2(y, yhat):
    y = y.reshape(-1)
    yhat = yhat.reshape(-1)
    ymean = np.mean(y)
    sst = np.sum((y - ymean) ** 2)
    if sst <= 1.0e-30:
        return 0.0
    sse = np.sum((y - yhat) ** 2)
    return 1.0 - sse / sst


def evaluate_fit_quality(nrmse_balanced, kx, ky, sigma, options):
    nrmse_limit = float(options.get("good_nrmse_bal_max", options.get("good_nrmse_max", 0.4)))
    lambda_factor = float(options.get("good_lambda_factor_max", 4.0))

    k_mag = float(np.sqrt(kx**2 + ky**2))
    if k_mag > 0.0:
        wavelength = 2.0 * np.pi / k_mag
    else:
        wavelength = np.inf

    is_good_nrmse = bool(nrmse_balanced <= nrmse_limit)
    is_good_scale = bool(wavelength <= lambda_factor * float(sigma))

    return {
        "k": k_mag,
        "wavelength": wavelength,
        "wavelength_over_sigma": wavelength / float(sigma) if sigma > 0.0 else np.inf,
        "is_good_nrmse": is_good_nrmse,
        "is_good_scale": is_good_scale,
        "is_good_overall": bool(is_good_nrmse and is_good_scale),
        "nrmse_limit": nrmse_limit,
        "lambda_factor": lambda_factor,
    }
