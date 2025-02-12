#!/usr/bin/env python

import numpy as np
import scipy as sp
from scipy import signal
import matplotlib as mpl
from matplotlib import pyplot as plt


def kspace_kernerl1d(kx, kl, kh, dk):
    # filter function in k space
    kk = np.abs(kx)
    ww = 0.5 * (1 + np.tanh((kk - kl) / dk)) - 0.5 * (1 + np.tanh((kk - kh) / dk))
    return ww


def bandpass_filter1d(data, kl, kh, dk, dh):
    # check input
    if not isinstance(data, (list, tuple)):
        data = (data,)
    result = [None] * len(data)
    # apply filter
    for i, f in enumerate(data):
        if f.ndim != 1:
            f = f.mean(axis=(0, 1))  # take mean along y and z
        Nx = f.shape[0]
        fk = np.fft.fft(f)
        kx = 2 * np.pi * np.fft.fftfreq(Nx, dh)
        ww = kspace_kernerl1d(kx, kl, kh, dk)
        result[i] = np.fft.ifft(fk * ww).real
    return result


def kspace_kernerl2d(kx, ky, kl, kh, dk):
    # filter function in k space
    kk = np.sqrt(kx**2 + ky**2)
    ww = 0.5 * (1 + np.tanh((kk - kl) / dk)) - 0.5 * (1 + np.tanh((kk - kh) / dk))
    return ww


def bandpass_filter2d(data, kl, kh, dk, dh):
    # check input
    if not isinstance(data, (list, tuple)):
        data = (data,)
    result = [None] * len(data)
    # apply filter
    for i, f in enumerate(data):
        if f.ndim != 2:
            f = f.mean(axis=(0,))  # take mean along z
        Nx = f.shape[1]
        Ny = f.shape[0]
        fk = np.fft.fft2(f)
        kx = 2 * np.pi * np.fft.fftfreq(Nx, dh)[np.newaxis, :]
        ky = 2 * np.pi * np.fft.fftfreq(Ny, dh)[:, np.newaxis]
        ww = kspace_kernerl2d(kx, ky, kl, kh, dk)
        result[i] = np.fft.ifft2(fk * ww).real
    return result


def interp_window(xnew, xdata, data):
    # check input
    if not isinstance(data, (list, tuple)):
        data = (data,)

    # linear interpolation in x
    index = xdata.searchsorted(xnew)
    delta = (xdata[index] - xnew) / (xdata[index] - xdata[index - 1])
    result = [None] * len(data)
    for i, f in enumerate(data):
        result[i] = f[..., index - 1] * delta + f[..., index] * (1 - delta)

    return result


def find_overshoot(xx, bx, by, bz, dh, mime):
    """Find the overshoot position of the magnetic field strength"""
    # apply smoothing to magnetic field strength
    fs = 1 / dh
    filtb, filta = signal.butter(5, 0.01 * fs, "low", fs=fs)
    bb = signal.filtfilt(filtb, filta, np.sqrt(bx**2 + by**2 + bz**2))
    # find the overshoot position as the right-most peak
    ww = int(0.5 * np.sqrt(mime) / dh)  # half ion inertial length
    index, prop = signal.find_peaks(bb, prominence=0.1, width=ww, distance=ww)
    return xx[index[-1]]


def find_ramp(xx, yy, dh, fc):
    """Find the ramp position of the magnetic field strength"""
    # apply smoothing to magnetic field strength
    fs = 1 / dh
    filtb, filta = signal.butter(5, fc * fs, "low", fs=fs)
    yy = signal.filtfilt(filtb, filta, yy)
    xc = 0.5 * (xx[1:] + xx[:-1])
    dy = -np.diff(yy, n=1) / np.diff(xx)
    # find the ramp position as the peak
    return xc[np.argmax(dy)]


def calc_shock_speed(params, steps, times, xc, var, fc=0.1):
    """Calculate the shock propagation speed in the simulation frame"""
    delh = params["delh"]

    if var.ndim == 2:
        # scalar (such as density)
        yy = var
    elif var.ndim == 3:
        # vector (such as magnetic field)
        yy = np.sqrt(var[..., 0] ** 2 + var[..., 1] ** 2 + var[..., 2] ** 2)

    x_sh = np.zeros((len(steps),))
    t_sh = np.zeros((len(steps),))
    for index, step in enumerate(steps):
        x_sh[index] = find_ramp(xc, yy[step], delh, fc)
        t_sh[index] = times[step]

    # linear fit to the shock position
    poly = np.polyfit(t_sh, x_sh, 1)
    v_sh = poly[0]

    return t_sh, x_sh, v_sh, poly


def calc_shock_potential(params, v_sh, vars, nsmooth=16):
    """Calculate the cross-shock potential in HTF and NIF

    The HTF potential is calculated from the equation of motion
    for the electron fluid ignoring the temporal derivative.
    See, Goodrich and Scudder (1984), for reference.
    """
    nppc = params["nppc"]
    mime = params["mime"]
    wp = params["wp"]
    u0 = params["u0"]
    qe = wp / nppc * np.sqrt(1 + u0**2)
    me = 1 / nppc

    # normalization
    phi_norm = 0.5 * mime * (me / qe) * v_sh**2

    # smoothing
    ww = np.ones(nsmooth)
    ww /= ww.sum()
    smooth = lambda var: signal.convolve(var, ww, mode="same")

    x = vars["x"]
    Ex = smooth(vars["Ex"])
    Bx = smooth(vars["Bx"])
    By = smooth(vars["By"])
    Bz = smooth(vars["Bz"])
    R = smooth(vars["R"])
    Vx = smooth(vars["Vx"])
    Vy = smooth(vars["Vy"])
    Vz = smooth(vars["Vz"])
    Pxx = smooth(vars["Pxx"]) - R * Vx * Vx
    Pxy = smooth(vars["Pxy"]) - R * Vx * Vy
    Pxz = smooth(vars["Pxz"]) - R * Vx * Vz

    dx = x[+1:] - x[:-1]
    xc = 0.5 * (x[+1:] + x[:-1])
    Rc = 0.5 * (R[+1:] + R[:-1])
    Vxc = 0.5 * (Vx[+1:] + Vx[:-1])
    byx = (By[+1:] + By[:-1]) / (Bx[+1:] + Bx[:-1])
    bzx = (Bz[+1:] + Bz[:-1]) / (Bx[+1:] + Bx[:-1])
    epara_pxx = -(Pxx[+1:] - Pxx[:-1]) / dx / Rc * me / qe
    epara_pxy = -(Pxy[+1:] - Pxy[:-1]) / dx / Rc * me / qe
    epara_pxz = -(Pxz[+1:] - Pxz[:-1]) / dx / Rc * me / qe
    epara_vxx = -(Vx[+1:] - Vx[:-1]) / dx * Vxc * me / qe
    epara_vxy = -(Vy[+1:] - Vy[:-1]) / dx * Vxc * me / qe
    epara_vxz = -(Vz[+1:] - Vz[:-1]) / dx * Vxc * me / qe

    phi_gradp = -np.cumsum(epara_pxx + byx * epara_pxy + bzx * epara_pxz) * dx
    phi_gradv = -np.cumsum(epara_vxx + byx * epara_vxy + bzx * epara_vxz) * dx
    phi_htf = (phi_gradp + phi_gradv) / phi_norm
    phi_nif = -np.cumsum(Ex)[1:] * dx / phi_norm

    return xc, phi_htf, phi_nif
