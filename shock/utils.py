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


def calc_shock_speed(run, step):
    """Calculate the shock propagation speed in the simulation frame"""
    mime = run.config["parameter"]["mime"]
    sigma = run.config["parameter"]["sigma"]
    u0 = run.config["parameter"]["u0"]
    b0 = np.sqrt(sigma) / np.sqrt(1 + u0**2)
    xc = run.xc
    dh = run.delh

    t_sh = np.zeros((len(step),))
    x_sh = np.zeros((len(step),))
    for index, i in enumerate(step):
        uf = run.read_at("field", i, "uf")["uf"]
        bx = uf[..., 3].mean(axis=(0, 1)) / b0
        by = uf[..., 4].mean(axis=(0, 1)) / b0
        bz = uf[..., 5].mean(axis=(0, 1)) / b0
        t_sh[index] = run.get_time_at("field", i)
        x_sh[index] = find_overshoot(xc, bx, by, bz, dh, mime)

    # linear fit to the shock position
    poly = np.polyfit(t_sh, x_sh, 1)
    v_sh = poly[0]

    return t_sh, x_sh, poly


def calc_shock_potential(run, step, v_sh, nsmooth=16):
    """Calculate the cross-shock potential in HTF and NIF"""
    nppc = run.config["parameter"]["nppc"]
    mime = run.config["parameter"]["mime"]
    wp = run.config["parameter"]["wp"]
    u0 = run.config["parameter"]["u0"]
    qe = wp / nppc * np.sqrt(1 + u0**2)
    me = 1 / nppc
    dh = run.delh
    xc = run.xc

    # normalization
    phi_norm = 0.5 * mime * (me / qe) * v_sh**2

    # smoothing
    ww = np.ones(nsmooth)
    ww /= ww.sum()
    smooth = lambda var: signal.convolve(var.mean(axis=(0, 1)), ww, mode="same")

    data = run.read_at("field", step)
    um = data["um"]
    uf = data["uf"]
    Ex = smooth(uf[..., 0])
    Bx = smooth(uf[..., 3])
    By = smooth(uf[..., 4])
    Bz = smooth(uf[..., 5])
    Ne = smooth(um[..., 0, 0]) / me
    Vx = smooth(um[..., 0, 1]) / (me * Ne)
    Vy = smooth(um[..., 0, 2]) / (me * Ne)
    Vz = smooth(um[..., 0, 3]) / (me * Ne)
    Pxx = smooth(um[..., 0, 5]) - me * Ne * Vx**2
    Pxy = smooth(um[..., 0, 11]) - me * Ne * Vx * Vy
    Pxz = smooth(um[..., 0, 13]) - me * Ne * Vz * Vx

    delx = xc[+1:] - xc[:-1]
    xcc = 0.5 * (xc[+1:] + xc[:-1])
    Nec = 0.5 * (Ne[+1:] + Ne[:-1])
    Vxc = 0.5 * (Vx[+1:] + Vx[:-1])
    byx = (By[+1:] + By[:-1]) / (Bx[+1:] + Bx[:-1])
    bzx = (Bz[+1:] + Bz[:-1]) / (Bx[+1:] + Bx[:-1])
    epara_pxx = -(Pxx[+1:] - Pxx[:-1]) / delx / (qe * Nec)
    epara_pxy = -(Pxy[+1:] - Pxy[:-1]) / delx / (qe * Nec)
    epara_pxz = -(Pxz[+1:] - Pxz[:-1]) / delx / (qe * Nec)
    epara_vxx = -(Vx[+1:] - Vx[:-1]) / delx * Vxc * me / qe
    epara_vxy = -(Vy[+1:] - Vy[:-1]) / delx * Vxc * me / qe
    epara_vxz = -(Vz[+1:] - Vz[:-1]) / delx * Vxc * me / qe

    phi_gradp = -np.cumsum(epara_pxx + byx * epara_pxy + bzx * epara_pxz) * dh
    phi_gradv = -np.cumsum(epara_vxx + byx * epara_vxy + bzx * epara_vxz) * dh
    phi_htf = (phi_gradp + phi_gradv) / phi_norm
    phi_nif = -np.cumsum(Ex)[1:] * dh / phi_norm

    return xcc, phi_htf, phi_nif
