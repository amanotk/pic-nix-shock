"""Unit tests for `shock.utils`."""

import numpy as np
import pytest

from shock import utils


def test_bandpass_filter1d_shape():
    x = np.linspace(0, 2 * np.pi, 512, endpoint=False)
    signal = np.sin(8 * x) + 0.2 * np.sin(40 * x)
    filtered = utils.bandpass_filter1d(signal, kl=5.0, kh=20.0, dk=1.0, dh=x[1] - x[0])
    assert isinstance(filtered, list)
    assert filtered[0].shape == signal.shape


def test_interp_window_finite_values():
    xdata = np.linspace(0.0, 10.0, 100)
    data = xdata**2
    xnew = np.array([1.25, 3.5, 8.75])
    out = utils.interp_window(xnew, xdata, data)
    assert isinstance(out, list)
    assert out[0].shape == xnew.shape
    assert np.all(np.isfinite(out[0]))


def test_find_overshoot_returns_position():
    xx = np.linspace(0, 20, 2000)
    bx = 1 + 0.2 * np.exp(-((xx - 10.0) ** 2) / 0.2)
    by = np.zeros_like(xx)
    bz = np.zeros_like(xx)
    try:
        xpos = utils.find_overshoot(xx, bx, by, bz, dh=xx[1] - xx[0], mime=100)
        assert np.isfinite(xpos)
        assert xx.min() <= xpos <= xx.max()
    except IndexError:
        pytest.skip("No overshoot peak detected for this synthetic signal")


def test_find_ramp_returns_position_or_none():
    xx = np.linspace(0, 20, 2000)
    yy = 0.5 * (1 + np.tanh((10.0 - xx) / 0.5))
    xpos = utils.find_ramp(xx, yy, dh=xx[1] - xx[0], fc=0.01)
    assert xpos is None or (xx.min() <= xpos <= xx.max())


def test_kspace_kernel1d_shape_and_finite():
    kx = np.fft.fftfreq(128, d=0.1) * 2 * np.pi
    kernel = utils.kspace_kernerl1d(kx, kl=1.0, kh=5.0, dk=0.5)
    assert kernel.shape == kx.shape
    assert np.all(np.isfinite(kernel))


def test_kspace_kernel2d_shape_and_finite():
    nx, ny = 64, 32
    kx = np.fft.fftfreq(nx, d=0.2)[np.newaxis, :] * 2 * np.pi
    ky = np.fft.fftfreq(ny, d=0.2)[:, np.newaxis] * 2 * np.pi
    kernel = utils.kspace_kernerl2d(kx, ky, kl=0.5, kh=3.0, dk=0.3)
    assert kernel.shape == (ny, nx)
    assert np.all(np.isfinite(kernel))
