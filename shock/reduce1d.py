#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pathlib
import pickle
import sys

import h5py
import matplotlib as mpl
import numpy as np
import tqdm

mpl.use("Agg") if __name__ == "__main__" else None
import matplotlib.pyplot as plt

# global configuration
plt.rcParams.update({"font.size": 12})

if "PICNIX_DIR" in os.environ:
    sys.path.append(str(pathlib.Path(os.environ["PICNIX_DIR"]) / "script"))
import base
import utils

import picnix


class DataReducer(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "analyze" in self.options:
            for key in self.options["analyze"]:
                self.options[key] = self.options["analyze"][key]

    def main(self, basename):
        self.save(self.get_filename(basename, ".h5"))

    def encode(self, data):
        return np.frombuffer(pickle.dumps(data), np.int8)

    def save(self, filename):
        profile = self.options.get("profile", None)
        method = self.options.get("method", "thread")
        overwrite = self.options.get("overwrite", False)
        run = picnix.Run(profile, method=method)
        config = self.encode(run.config)

        # step and index
        particle_step = run.get_step("particle")
        field_step = run.get_step("field")
        common_step = np.intersect1d(particle_step, field_step)
        step_min = self.options.get("step_min", 0)
        step_max = self.options.get("step_max", common_step[-1])
        index_min = np.searchsorted(common_step, step_min)
        index_end = np.searchsorted(common_step, step_max)
        index_range = np.arange(index_min, index_end + 1)

        data = run.read_at("field", field_step[0], "uf")

        xc = data["xc"]
        Nt = index_end - index_min + 1
        Nx = xc.size
        dh = xc[1] - xc[0]

        # binning
        Nbinx = self.options.get("Nbinx")
        Nbinu = self.options.get("Nbinu")
        xbine_lim = (0, Nx * dh)
        xbini_lim = (0, Nx * dh)
        ubine_lim = self.options.get("ubine")
        ebine_lim = self.options.get("ebine")
        ubini_lim = self.options.get("ubini")
        ebini_lim = self.options.get("ebini")
        xbine = np.linspace(xbine_lim[0], xbine_lim[1], Nbinx + 1)
        ubine = np.linspace(ubine_lim[0], ubine_lim[1], Nbinu + 1)
        ebine = np.geomspace(ebine_lim[0], ebine_lim[1], Nbinu + 1)
        xbini = np.linspace(xbini_lim[0], xbini_lim[1], Nbinx + 1)
        ubini = np.linspace(ubini_lim[0], ubini_lim[1], Nbinu + 1)
        ebini = np.geomspace(ebini_lim[0], ebini_lim[1], Nbinu + 1)

        if not os.path.exists(filename) or overwrite == True:
            # create HDF5 file and datasets first
            with h5py.File(filename, "w") as fp:
                ### write data independent of step
                fp.create_dataset("config", data=config, dtype=np.int8)
                fp.create_dataset("x", data=xc, dtype=np.float64)
                fp.create_dataset("xbine", data=xbine, dtype=np.float64)
                fp.create_dataset("ubine", data=ubine, dtype=np.float64)
                fp.create_dataset("ebine", data=ebine, dtype=np.float64)
                fp.create_dataset("xbini", data=xbini, dtype=np.float64)
                fp.create_dataset("ubini", data=ubini, dtype=np.float64)
                fp.create_dataset("ebini", data=ebini, dtype=np.float64)

                ### create datasets for each step
                dummpy_step = (-1) * np.ones((Nt,), np.int32)
                fp.create_dataset("step", (Nt,), data=dummpy_step, dtype=np.int32)
                fp.create_dataset("t", (Nt,), dtype=np.float64)
                fp.create_dataset("E", (Nt, Nx, 3), dtype=np.float64, chunks=(1, Nx, 3))
                fp.create_dataset("B", (Nt, Nx, 3), dtype=np.float64, chunks=(1, Nx, 3))
                fp.create_dataset("Phi", (Nt, Nx), dtype=np.float64, chunks=(1, Nx))
                # electron moments
                fp.create_dataset("Re", (Nt, Nx), dtype=np.float64, chunks=(1, Nx))
                fp.create_dataset("Ve", (Nt, Nx, 3), dtype=np.float64, chunks=(1, Nx, 3))
                fp.create_dataset("Pe", (Nt, Nx, 6), dtype=np.float64, chunks=(1, Nx, 6))
                # ion moments
                fp.create_dataset("Ri", (Nt, Nx), dtype=np.float64, chunks=(1, Nx))
                fp.create_dataset("Vi", (Nt, Nx, 3), dtype=np.float64, chunks=(1, Nx, 3))
                fp.create_dataset("Pi", (Nt, Nx, 6), dtype=np.float64, chunks=(1, Nx, 6))
                # electron phase space
                fp.create_dataset(
                    "Feu",
                    (Nt, Nbinx, Nbinu, 5),
                    dtype=np.float64,
                    chunks=(1, Nbinx, Nbinu, 5),
                )
                # ion phase space
                fp.create_dataset(
                    "Fiu",
                    (Nt, Nbinx, Nbinu, 5),
                    dtype=np.float64,
                    chunks=(1, Nbinx, Nbinu, 5),
                )

        # read step
        with h5py.File(filename, "r") as fp:
            step_in_file = fp["step"][()]

        # read and process data for each step
        for i, index in enumerate(tqdm.tqdm(index_range)):
            # skip if the step is already stored
            if step_in_file[i] == common_step[index]:
                continue

            # particle
            with h5py.File(filename, "a") as fp:
                data = run.read_at("particle", common_step[index])
                for name, dataset, xbin, ubin, ebin in zip(
                    ["up00", "up01"],
                    ["Feu", "Fiu"],
                    [xbine, xbini],
                    [ubine, ubini],
                    [ebine, ebini],
                ):
                    xu = data[name]
                    ke = np.sqrt(1 + xu[:, 3] ** 2 + xu[:, 4] ** 2 + xu[:, 5] ** 2) - 1
                    fux = picnix.Histogram2D(xu[:, 0], xu[:, 3], xbin, ubin)
                    fuy = picnix.Histogram2D(xu[:, 0], xu[:, 4], xbin, ubin)
                    fuz = picnix.Histogram2D(xu[:, 0], xu[:, 5], xbin, ubin)
                    fke = picnix.Histogram2D(xu[:, 0], ke, xbin, ebin, logy=True)
                    fp[dataset][i, ..., 0] = fux.density
                    fp[dataset][i, ..., 1] = fuy.density
                    fp[dataset][i, ..., 2] = fuz.density
                    fp[dataset][i, ..., 3] = fke.density

            # field
            with h5py.File(filename, "a") as fp:
                data = run.read_at("field", common_step[index], "uf")
                uf = data["uf"].mean(axis=(0, 1))
                fp["E"][i, ...] = uf[:, 0:3]
                fp["B"][i, ...] = uf[:, 3:6]

            # moments
            with h5py.File(filename, "a") as fp:
                data = run.read_at("field", common_step[index], "um")
                um = data["um"].mean(axis=(0, 1))
                for species, name in zip(range(2), ["e", "i"]):
                    # R : mass density
                    # V : bulk velocity
                    # P : pressure tensor + R Vi Vj
                    R = um[:, species, 0]
                    Vx = um[:, species, 1] / R
                    Vy = um[:, species, 2] / R
                    Vz = um[:, species, 3] / R
                    Pxx = um[:, species, 5]
                    Pyy = um[:, species, 6]
                    Pzz = um[:, species, 7]
                    Pxy = um[:, species, 11]
                    Pyz = um[:, species, 12]
                    Pzx = um[:, species, 13]
                    fp["R" + name][i, ...] = R
                    fp["V" + name][i, ..., 0] = Vx
                    fp["V" + name][i, ..., 1] = Vy
                    fp["V" + name][i, ..., 2] = Vz
                    fp["P" + name][i, ..., 0] = Pxx
                    fp["P" + name][i, ..., 1] = Pyy
                    fp["P" + name][i, ..., 2] = Pzz
                    fp["P" + name][i, ..., 3] = Pxy
                    fp["P" + name][i, ..., 4] = Pyz
                    fp["P" + name][i, ..., 5] = Pzx

            # time and step; do this last to ensure all data is written
            with h5py.File(filename, "a") as fp:
                fp["t"][i] = run.get_time_at("field", common_step[index])
                fp["step"][i] = common_step[index]


class ShockPositionModel(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "position" in self.options:
            for key in self.options["position"]:
                self.options[key] = self.options["position"][key]

    def main(self, basename):
        filename = self.get_filename(basename, ".h5")
        with h5py.File(filename, "r") as fp:
            params = pickle.loads(fp["config"][()])["parameter"]
            t = fp["t"][()]
            x = fp["x"][()]
            B = fp["B"][()]

        mime = params["mime"]
        wce = np.sqrt(params["sigma"])
        wci = wce / mime
        fit_range = self.options.get("fit_range", None)
        fit_steps = self.handle_range(t, (fit_range[0] / wci, fit_range[1] / wci))
        bx = B[..., 0]
        by = B[..., 1]
        bz = B[..., 2]
        bb = np.sqrt(bx**2 + by**2 + bz**2)
        t_sh, x_sh, v_sh, poly = utils.calc_shock_speed(params, fit_steps, t, x, bb, 0.01)

        self.poly = poly

    def get_position(self):
        return self.poly

    def handle_range(self, t, fit_range):
        if fit_range is None:
            return np.arange(0, t.size)  # whole range

        # assume list or tuple of length 2
        range1 = fit_range[0]
        range2 = fit_range[1]

        # if range in step
        if type(range1) is int and type(range2) is int:
            return np.arange(range1, range2)

        # if range in time (real-valued)
        if np.isreal(range1) and np.isreal(range2):
            tmin, tmax = t[0], t[-1]
            imin, imax = 0, t.size - 1
            index1 = np.argmin(np.abs(t - range1)) if tmin <= range1 else imin
            index2 = np.argmin(np.abs(t - range2)) if tmax >= range2 else imax
            return np.arange(index1, index2 + 1)

        raise ValueError("Invalid fit_range")


class DataPlotter(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "plot" in self.options:
            for key in self.options["plot"]:
                self.options[key] = self.options["plot"][key]
        self.plot_dict = None
        self.panels = self.options.get("panels", ["fi_ux", "fe_uy", "fe_p4"])

    def main(self, basename):
        filename = self.get_filename(basename, ".h5")
        output = self.options.get("output", "plot")
        outpath = os.sep.join([self.get_dirname(), output])

        # calculate shock position polynomial
        model = ShockPositionModel(self.config_file)
        model.main(basename)
        shock_position = model.get_position()
        print("Shock speed polynomial fit:", shock_position)

        with h5py.File(filename, "r") as fp:
            # read common parameters once
            config = pickle.loads(fp["config"][()])
            xbine = fp["xbine"][()]
            ubine = fp["ubine"][()]
            ebine = fp["ebine"][()]
            xbini = fp["xbini"][()]
            ubini = fp["ubini"][()]
            step = fp["step"][()]
            t = fp["t"][()]
            x = fp["x"][()]
            xbinc = 0.5 * (xbine[1:] + xbine[:-1])
            ebinc = 0.5 * (ebine[1:] + ebine[:-1])
            mime = config["parameter"]["mime"]
            sigma = config["parameter"]["sigma"]
            u0 = config["parameter"]["u0"]

            # normalization factors
            b0 = np.sqrt(sigma) / np.sqrt(1 + u0**2)
            vae = np.sqrt(sigma)
            vai = vae / np.sqrt(mime)

            # make plots
            wci = np.sqrt(sigma) / mime
            for i in tqdm.tqdm(range(step.shape[0])):
                fe_ke = fp["Feu"][i, ..., 3]
                pbinc, f_mom = self.convert_to_momentum_spectrum(ebinc, fe_ke)
                pbine = np.sqrt((ebine + 1) ** 2 - 1)
                fe_p4 = f_mom * pbinc[np.newaxis, :] ** 4

                params = {
                    "xbine": xbine,
                    "ubine": ubine,
                    "ebine": ebine,
                    "ebinc": ebinc,
                    "xbini": xbini,
                    "ubini": ubini,
                    "x": x,
                    "B": fp["B"][i] / b0,
                    "Vi": fp["Vi"][i] / vai,
                    "fi_ux": fp["Fiu"][i, ..., 0],
                    "fe_uy": fp["Feu"][i, ..., 1],
                    "fe_ke": fe_ke,
                    "xbinc": xbinc,
                    "pbine": pbine,
                    "fe_p4": fe_p4,
                }
                pngfile = "{:s}-{:08d}".format(outpath, step[i])
                wci_time = wci * t[i]
                x_shock = np.polyval(shock_position, t[i])
                self.plot(x_shock, wci_time, params)
                self.save(pngfile)

        # convert to mp4
        fps = self.options.get("fps", 10)
        picnix.convert_to_mp4("{:s}".format(outpath), fps, False)

    def convert_to_momentum_spectrum(self, energy, dist):
        p = np.sqrt((energy + 1) ** 2 - 1)
        e2p = 1 / (4 * np.pi * p * np.sqrt(p**2 + 1))
        if dist.ndim == 1 and dist.size == p.size:
            return p, dist * e2p
        elif dist.ndim == 2 and dist.shape[1] == p.size:
            return p, dist * e2p[np.newaxis, :]
        else:
            raise ValueError("Invalid shape of distribution function.")

    def pcolormesh_args(self, xedges, yedges):
        x = 0.5 * (xedges[+1:] + xedges[:-1])
        y = 0.5 * (yedges[+1:] + yedges[:-1])
        X, Y = np.broadcast_arrays(x[:, None], y[None, :])
        return X, Y

    def plot_fi_ux(self, ax, params, fig=None):
        fi_ux = params["fi_ux"]
        xbini = params["xbini"]
        ubini = params["ubini"]
        X, Y = self.pcolormesh_args(xbini, ubini)
        img = ax.pcolormesh(X, Y, fi_ux, shading="nearest", norm=mpl.colors.LogNorm())
        ax.set_ylabel(r"$u_{x} / c$")

        cax = None
        if fig is not None:
            cax = fig.add_axes(base.get_colorbar_position_next(ax, pad=0.025))
            plt.colorbar(img, cax=cax)
            cax.set_ylabel(r"$f_i(u_x)$  [arb. unit]")

        return img, cax

    def plot_fe_uy(self, ax, params, fig=None):
        fe_uy = params["fe_uy"]
        xbine = params["xbine"]
        ubine = params["ubine"]
        X, Y = self.pcolormesh_args(xbine, ubine)
        img = ax.pcolormesh(X, Y, fe_uy, shading="nearest", norm=mpl.colors.LogNorm())
        ax.set_ylabel(r"$u_{y} / c$")

        cax = None
        if fig is not None:
            cax = fig.add_axes(base.get_colorbar_position_next(ax, pad=0.025))
            plt.colorbar(img, cax=cax)
            cax.set_ylabel(r"$f_e(u_y)$  [arb. unit]")

        return img, cax

    def plot_fe_p4(self, ax, params, fig=None):
        fe_p4 = params["fe_p4"]
        xbine = params["xbine"]
        pbine = params["pbine"]
        X, Y = self.pcolormesh_args(xbine, pbine)
        img = ax.pcolormesh(X, Y, fe_p4, shading="nearest", norm=mpl.colors.LogNorm())
        ax.set_ylabel(r"$p / m_e c$")
        ax.semilogy()

        cax = None
        if fig is not None:
            cax = fig.add_axes(base.get_colorbar_position_next(ax, pad=0.025))
            plt.colorbar(img, cax=cax)
            cax.set_ylabel(r"$p^4 f_e(p)$  [arb. unit]")

        return img, cax

    def plot_spectrum(self, ax, params, fig=None):
        fe_p4 = params["fe_p4"]
        pbine = params["pbine"]
        xbinc = params["xbinc"]
        psample = [0.6, 0.7, 0.8, 0.9, 1.0]
        pindex = np.searchsorted(pbine, psample)
        for i in range(len(psample)):
            f_norm = fe_p4[:, pindex[i]] / (np.max(fe_p4[:, pindex[i]]) + 1.0e-32)
            ax.plot(
                xbinc,
                f_norm * 10 ** (-i),
                label=r"$p/m_e c = {:.1f}$".format(psample[i]),
            )
        ax.set_ylabel(r"$f(p)$ [arb. unit]")
        ax.semilogy()
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_ylim(1.0e-6, 1.0e1)
        ax.set_xlabel(r"$x / c / \omega_{pe}$")

        return None, None

    def get_plot_function(self, name):
        mapping = {
            "fi_ux": self.plot_fi_ux,
            "fe_uy": self.plot_fe_uy,
            "fe_p4": self.plot_fe_p4,
            "spectrum": self.plot_spectrum,
        }
        return mapping.get(name)

    def plot(self, x_shock, wci_t, params):
        if self.plot_dict is None:
            # create new figure and axes
            self.plot_dict = self.plot_new(params)
        else:
            # update existing figure and axes
            self.plot_dict = self.plot_update(params)

        # update xrange and title
        self.plot_dict["axs"][-1].set_xlim(x_shock - 200, x_shock + 200)
        plt.suptitle(r"$\Omega_{{ci}} t$ = {:.2f}".format(wci_t))

    def plot_new(self, params):
        x = params["x"]
        B = params["B"]
        Vi = params["Vi"]

        num_panels = 2 + len(self.panels)

        # create figure and axes for the first frame
        fig, axs = plt.subplots(num_panels, 1, figsize=(10, 2 * num_panels), sharex=True)
        fig.subplots_adjust(hspace=0.2, right=0.8, left=0.1, top=0.95, bottom=0.05)
        # plot B field
        axs[0].plot(x, B[:, 0], "k-", label="Bx")
        axs[0].plot(x, B[:, 1], "r-", label="By")
        axs[0].plot(x, B[:, 2], "b-", label="Bz")
        axs[0].set_ylabel(r"$B / B_0$")
        axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
        # plot Vi
        axs[1].plot(x, Vi[:, 0], "k-", label="Vx")
        axs[1].plot(x, Vi[:, 1], "r-", label="Vy")
        axs[1].plot(x, Vi[:, 2], "b-", label="Vz")
        axs[1].set_ylabel(r"$V_{i} / V_{A,i}$")
        axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))

        img = [None] * num_panels
        cax = [None] * num_panels

        for i, name in enumerate(self.panels):
            idx = i + 2
            func = self.get_plot_function(name)
            if func:
                img[idx], cax[idx] = func(axs[idx], params, fig=fig)

        for ax in axs:
            ax.grid(True)
        fig.align_ylabels(axs)

        return {
            "fig": fig,
            "axs": axs,
            "img": img,
            "cax": cax,
        }

    def plot_update(self, params):
        x = params["x"]
        B = params["B"]
        Vi = params["Vi"]

        # update figure and axes for subsequent frames
        fig = self.plot_dict["fig"]
        axs = self.plot_dict["axs"]
        # clear each axes before re-plotting
        for ax in axs:
            ax.cla()
        # plot B field
        axs[0].plot(x, B[:, 0], "k-", label="Bx")
        axs[0].plot(x, B[:, 1], "r-", label="By")
        axs[0].plot(x, B[:, 2], "b-", label="Bz")
        axs[0].set_ylabel(r"$B / B_0$")
        axs[0].legend(loc="upper left", bbox_to_anchor=(1, 1))
        # plot Vi
        axs[1].plot(x, Vi[:, 0], "k-", label="Vx")
        axs[1].plot(x, Vi[:, 1], "r-", label="Vy")
        axs[1].plot(x, Vi[:, 2], "b-", label="Vz")
        axs[1].set_ylabel(r"$V_{i} / V_{A,i}$")
        axs[1].legend(loc="upper left", bbox_to_anchor=(1, 1))

        img = self.plot_dict["img"]
        cax = self.plot_dict["cax"]

        for i, name in enumerate(self.panels):
            idx = i + 2
            func = self.get_plot_function(name)
            if func:
                img[idx], _ = func(axs[idx], params)

        for ax in axs:
            ax.grid(True)
        fig.align_ylabels(axs)

        return {
            "fig": fig,
            "axs": axs,
            "img": img,
            "cax": cax,
        }

    def save(self, filename):
        # save the current figure to a PNG file
        if self.plot_dict is not None and "fig" in self.plot_dict:
            self.plot_dict["fig"].savefig(filename)
        else:
            print("no plot to save")


def main():
    import argparse

    script_name = "reduce1d"

    parser = argparse.ArgumentParser(description="Data Reduction Script")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=script_name,
        help="basename used for output files",
    )
    parser.add_argument(
        "-j",
        "--job",
        type=str,
        default="analyze",
        help="Type of job to perform (analyze, position, plot). Can be combined with comma.",
    )
    parser.add_argument("config", nargs=1, help="configuration file for the job")
    args = parser.parse_args()
    config = args.config[0]
    output = args.output

    jobs = args.job.split(",")

    # perform analyze job first if requested
    if "analyze" in jobs:
        obj = DataReducer(config)
        obj.main(output)
        jobs = [j for j in jobs if j != "analyze"]

    # check prerequisite for remaining jobs
    if len(jobs) > 0:
        obj = DataReducer(config)
        filename = obj.get_filename(output, ".h5")
        if not os.path.exists(filename):
            sys.exit("Error: File '{}' not found. Please run 'analyze' job first.".format(filename))

    # perform remaining jobs
    for job in jobs:
        if job == "position":
            obj = ShockPositionModel(config)
            obj.main(output)
            print("shock_position = [{:20.12e}, {:20.12e}]".format(*obj.get_position()))

        elif job == "plot":
            obj = DataPlotter(config)
            obj.main(output)

        else:
            raise ValueError("Unknown job: {:s}".format(job))


if __name__ == "__main__":
    main()
