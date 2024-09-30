#!/usr/bin/env python

import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt

import picnix


def calc_particle_histogram(run, step, species, xbin, ubin, ebin):
    name = "up{:02d}".format(species)
    particle = run.read_at("particle", step, name)[name]
    xx = particle[:, 0]
    ux = particle[:, 3]
    uy = particle[:, 4]
    uz = particle[:, 5]
    xux = picnix.Histogram2D(xx, ux, xbin, ubin)
    xuy = picnix.Histogram2D(xx, uy, xbin, ubin)
    xuz = picnix.Histogram2D(xx, uz, xbin, ubin)
    # energy in log scale
    e = np.sqrt(1 + ux**2 + uy**2 + uz**2) - 1
    xe = picnix.Histogram2D(xx, e, xbin, ebin, logy=True)
    return {"xux": xux, "xuy": xuy, "xuz": xuz, "xe": xe}


def calc_velocity_dist(run, step, species, xmin, xmax, **kwargs):
    # avearage magnetic field and ExB drift velocity
    index = slice(*tuple(np.searchsorted(run.xc, [xmin, xmax])))
    uf = run.read_at("field", step, "uf")["uf"][..., index, :].mean(axis=(0, 1, 2))
    ex = uf[0]
    ey = uf[1]
    ez = uf[2]
    bx = uf[3]
    by = uf[4]
    bz = uf[5]
    bb = np.sqrt(bx**2 + by**2 + bz**2)
    vex = (ey * bz - ez * by) / bb**2
    vey = (ez * bx - ex * bz) / bb**2
    vez = (ex * by - ey * bx) / bb**2

    # calculate velocity distribution
    name = "up{:02d}".format(species)
    particle = run.read_at("particle", step, name)[name]
    mask = (particle[:, 0] >= xmin) & (particle[:, 0] <= xmax)
    particle = particle[mask]
    ux = particle[:, 3] - vex
    uy = particle[:, 4] - vey
    uz = particle[:, 5] - vez
    gamma = np.sqrt(1 + particle[:, 3] ** 2 + particle[:, 4] ** 2 + particle[:, 5] ** 2)
    upara = (ux * bx + uy * by + uz * bz) / bb
    uperp = np.sqrt(
        (ux - upara * bx / bb) ** 2
        + (uy - upara * by / bb) ** 2
        + (uz - upara * bz / bb) ** 2
    )

    # energy distribution
    enebin = kwargs.get("enebin", np.geomspace(1.0e-3, 1.0e1, 61))
    weights = np.ones_like(gamma) / gamma.size
    dist1d, _ = np.histogram(gamma - 1, bins=enebin, weights=weights)
    energy_dist = {"energy": 0.5 * (enebin[+1:] + enebin[:-1]), "dist": dist1d}

    # 2D velocity distribution
    upabin = kwargs.get("upabin", np.linspace(-1.0, +1.0, 81))
    upebin = kwargs.get("upebin", np.linspace(+0.0, +1.0, 41))
    hist2d = picnix.Histogram2D(upara, uperp, upabin, upebin)
    upara, uperp, dist2d = hist2d.pcolormesh_args()
    dist2d = dist2d / (2 * np.pi * uperp)

    velocity_dist = {"upara": upara, "uperp": uperp, "dist": dist2d}

    return energy_dist, velocity_dist


def summary_plot_velocity_dist(velocity_dist):
    fig = plt.figure(figsize=(8, 4), dpi=120)
    fig.subplots_adjust(
        top=0.92,
        bottom=0.10,
        left=0.08,
        right=0.92,
        hspace=0.25,
        wspace=0.02,
    )
    gridspec = fig.add_gridspec(1, 2, height_ratios=[1], width_ratios=[50, 2])
    axs = fig.add_subplot(gridspec[0, 0])
    cxs = fig.add_subplot(gridspec[0, 1])

    upara = velocity_dist["upara"]
    uperp = velocity_dist["uperp"]
    dist = velocity_dist["dist"]
    cntr_num = 4
    cntr_max = np.floor(np.log10(dist.max()))
    cntr_min = cntr_max - cntr_num + 1
    cntr = 10 ** np.linspace(cntr_min, cntr_max, cntr_num)

    plt.sca(axs)
    plt.pcolormesh(upara, uperp, dist, shading="nearest", norm=mpl.colors.LogNorm())
    plt.colorbar(cax=cxs)
    plt.contour(upara, uperp, dist, levels=cntr, colors="k", linewidths=0.5)
    plt.xlabel(r"$v_{\parallel}$")
    plt.ylabel(r"$v_{\perp}$")
    plt.suptitle(r"$f(v_{\parallel}, v_{\perp})$")

    axs.set_aspect("equal")
    pos = axs.get_position()
    cxs.set_position([pos.x0 + pos.width * 1.02, pos.y0, 0.02, pos.height])
    axs.xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    axs.yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.2))
    axs.xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))
    axs.yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.05))


def plot_particle_histogram(ax, cx, hist):
    args = hist.pcolormesh_args()
    plt.sca(ax)
    plt.pcolormesh(*args, shading="nearest", norm=mpl.colors.LogNorm())
    plt.colorbar(cax=cx)


def summary_get_index(coord, lim=None):
    if lim is None:
        return slice(None)
    else:
        return slice(*np.searchsorted(coord, lim))


def summary_plot_1d(run, step, **kwargs):
    param = run.config["parameter"]
    u0 = param["u0"]
    b0 = np.sqrt(param["sigma"]) / np.sqrt(1 + u0**2)
    tt = np.sqrt(param["sigma"]) / param["mime"] * run.get_time_at("particle", step)

    # calculate histogram
    xbine = kwargs.get("xbine", (0, run.Nx * run.delh, run.Nx + 1))
    ubine = kwargs.get("ubine", (-1.5, +1.5, 81))
    ebine = kwargs.get("ebine", np.geomspace(1.0e-3, 1.0e1, 61))
    xbini = kwargs.get("xbini", xbine)
    ubini = kwargs.get("ubini", (-3.0 * u0, +3.0 * u0, 81))
    ebini = kwargs.get("ebini", np.geomspace(1.0e-4, 1.0e-1, 61))
    ele_psd = calc_particle_histogram(run, step, 0, xbine, ubine, ebine)
    ion_psd = calc_particle_histogram(run, step, 1, xbini, ubini, ebini)
    psd = (
        None,
        ion_psd["xux"],
        ion_psd["xuy"],
        ele_psd["xux"],
        ele_psd["xuy"],
        ele_psd["xe"],
    )
    label = (r"B", r"$u_{i,x}$", r"$u_{i,y}$", r"$u_{e,x}$", r"$u_{e,y}$", r"$K_{e}$")
    logy = (False, False, False, False, False, True)

    # plot
    fig = plt.figure(figsize=(10, 10), dpi=120)
    fig.subplots_adjust(
        top=0.95,
        bottom=0.08,
        left=0.08,
        right=0.92,
        hspace=0.25,
        wspace=0.02,
    )
    gridspec = fig.add_gridspec(
        6, 2, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[50, 1]
    )
    axs = [0] * 6
    cxs = [0] * 6
    for i in range(6):
        axs[i] = fig.add_subplot(gridspec[i, 0])
        if psd[i] is not None:
            cxs[i] = fig.add_subplot(gridspec[i, 1])
        axs[i].set_ylabel(label[i])
        axs[i].set_xlim(xbine[0], xbine[1])
        axs[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        if logy[i] == True:
            axs[i].set_yscale("log")
    axs[-1].set_xlabel(r"$x / c/\omega_{pe}$")
    fig.align_ylabels(axs)
    plt.suptitle(r"$\Omega_{{ci}} t$ = {:5.2f}".format(tt))

    # magnetic field
    xindex = summary_get_index(run.xc, (xbine[0], xbine[1]))
    xc = run.xc[xindex]
    uf = run.read_at("field", step, "uf")["uf"].mean(axis=(0, 1))[xindex]
    plt.sca(axs[0])
    plt.plot(xc, uf[:, 3] / b0, "k-", label=r"$B_x$")
    plt.plot(xc, uf[:, 4] / b0, "r-", label=r"$B_y$")
    plt.plot(xc, uf[:, 5] / b0, "b-", label=r"$B_z$")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

    # phase space
    for i in range(6):
        if psd[i] is not None:
            plot_particle_histogram(axs[i], cxs[i], psd[i])

    return ele_psd, ion_psd


def summary_plot_vector_2d(X, Y, time, vars, labels, vlim=None):
    xmin = X.min()
    xmax = X.max()
    ymin = Y.min()
    ymax = Y.max()

    # colorbar limits
    if vlim is None or (not isinstance(vlim, (list, tuple))):
        vlim = [None] * 3
        for i in range(3):
            vlim[i] = [vars[i].min(), vars[i].max()]
    elif len(vlim) == 2:
        vlim = [vlim] * 3
    elif len(vlim) != 3:
        for i in range(3):
            if len(vlim[i]) != 2:
                vlim[i] = [vars[i].min(), vars[i].max()]

    # plot
    fig = plt.figure(figsize=(10, 8), dpi=120)
    fig.subplots_adjust(
        top=0.95,
        bottom=0.08,
        left=0.08,
        right=0.92,
        hspace=0.25,
        wspace=0.02,
    )
    gridspec = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], width_ratios=[50, 1])
    axs = [0] * 3
    cxs = [0] * 3
    for i in range(3):
        axs[i] = fig.add_subplot(gridspec[i, 0])
        plt.sca(axs[i])
        # plot and colorbar
        plt.pcolormesh(X, Y, vars[i], vmin=vlim[i][0], vmax=vlim[i][1])
        cxs[i] = fig.add_subplot(gridspec[i, 1])
        plt.colorbar(cax=cxs[i], label=labels[i])
        # appearance
        axs[i].set_ylabel(r"$y / c/\omega_{pe}$")
        axs[i].set_xlim(xmin, xmax)
        axs[i].set_ylim(ymin, ymax)
        axs[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
        axs[i].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
    axs[-1].set_xlabel(r"$x / c/\omega_{pe}$")
    fig.align_ylabels(axs)
    fig.align_ylabels(cxs)
    plt.suptitle(r"$\Omega_{{ci}} t$ = {:5.2f}".format(time))


def summary_efield_2d(run, step, **kwargs):
    vlim = kwargs.get("vlim", None)
    xindex = summary_get_index(run.xc, kwargs.get("xlim", None))
    yindex = summary_get_index(run.yc, kwargs.get("ylim", None))

    param = run.config["parameter"]
    wci = np.sqrt(param["sigma"]) / param["mime"]
    u0 = param["u0"]
    b0 = np.sqrt(param["sigma"]) / np.sqrt(1 + u0**2)
    e0 = u0 * b0 / np.sqrt(1 + u0**2)

    X, Y = np.meshgrid(run.xc[xindex], run.yc[yindex])
    time = wci * run.get_time_at("field", step)
    data = run.read_at("field", step, "uf")
    uf = data["uf"]

    Ex = uf[..., 0].mean(axis=(0))[yindex, xindex] / e0
    Ey = uf[..., 1].mean(axis=(0))[yindex, xindex] / e0
    Ez = uf[..., 2].mean(axis=(0))[yindex, xindex] / e0
    vars = (Ex, Ey, Ez)
    labels = (r"$E_x$", r"$E_y$", r"$E_z$")

    summary_plot_vector_2d(X, Y, time, vars, labels, vlim)


def summary_bfield_2d(run, step, **kwargs):
    vlim = kwargs.get("vlim", None)
    xindex = summary_get_index(run.xc, kwargs.get("xlim", None))
    yindex = summary_get_index(run.yc, kwargs.get("ylim", None))

    param = run.config["parameter"]
    wci = np.sqrt(param["sigma"]) / param["mime"]
    u0 = param["u0"]
    b0 = np.sqrt(param["sigma"]) / np.sqrt(1 + u0**2)

    X, Y = np.meshgrid(run.xc[xindex], run.yc[yindex])
    time = wci * run.get_time_at("field", step)
    data = run.read_at("field", step, "uf")
    uf = data["uf"]

    Bx = uf[..., 3].mean(axis=(0))[yindex, xindex] / b0
    By = uf[..., 4].mean(axis=(0))[yindex, xindex] / b0
    Bz = uf[..., 5].mean(axis=(0))[yindex, xindex] / b0
    vars = (Bx, By, Bz)
    labels = (r"$B_x$", r"$B_y$", r"$B_z$")

    summary_plot_vector_2d(X, Y, time, vars, labels, vlim)
