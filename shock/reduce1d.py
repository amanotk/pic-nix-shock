#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import pathlib
import pickle
import h5py
import msgpack
import toml
import json
import tqdm
import numpy as np
import scipy.ndimage as ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.use("Agg") if __name__ == "__main__" else None

# global configuration
plt.rcParams.update({"font.size": 12})

if "PICNIX_DIR" in os.environ:
    sys.path.append(str(pathlib.Path(os.environ["PICNIX_DIR"]) / "script"))
import picnix
import utils


def get_colorbar_position_next(ax, pad=0.05):
    axpos = ax.get_position()
    caxpos = [
        axpos.x0 + axpos.width * (1 + pad),
        axpos.y0,
        axpos.width * pad,
        axpos.height,
    ]
    return caxpos


def get_vlim(vars, vmag=100):
    vlims = []
    for v in vars:
        vmin = np.sign(v.min()) * np.ceil(np.abs(v.min()) * vmag) / vmag
        vmax = np.sign(v.max()) * np.ceil(np.abs(v.max()) * vmag) / vmag
        vlims.append([vmin, vmax])
    return vlims


class JobExecutor:
    def __init__(self, **kwargs):
        self.options = dict()
        for key in kwargs:
            if not isinstance(kwargs[key], dict):
                self.options[key] = kwargs[key]
        # read parameter from profile
        self.parameter = self.read_parameter()

    def read_parameter(self):
        # read parameter from profile
        with open(self.options["profile"], "rb") as fp:
            obj = msgpack.load(fp)
            parameter = obj["configuration"]["parameter"]
        return parameter

    def get_dirname(self):
        dirname = self.options.get("dirname", None)
        if dirname is None:
            raise ValueError("dirname is not specified")
        elif not os.path.exists(dirname):
            os.makedirs(dirname)
        return dirname

    def get_filename(self, prefix, ext):
        return os.sep.join([self.get_dirname(), prefix + ext])

    def main(self, prefix):
        raise NotImplementedError


class DataReducer(JobExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "reduce" in kwargs:
            for key in kwargs["reduce"]:
                self.options[key] = kwargs["reduce"][key]

    def main(self, prefix):
        self.save(self.get_filename(prefix, ".h5"))

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

        Nt = index_end - index_min + 1
        Nx = run.Nx
        dh = run.delh
        xx = run.xc

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
                fp.create_dataset("x", data=xx, dtype=np.float64)
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
                fp.create_dataset(
                    "Ve", (Nt, Nx, 3), dtype=np.float64, chunks=(1, Nx, 3)
                )
                fp.create_dataset(
                    "Pe", (Nt, Nx, 6), dtype=np.float64, chunks=(1, Nx, 6)
                )
                # ion moments
                fp.create_dataset("Ri", (Nt, Nx), dtype=np.float64, chunks=(1, Nx))
                fp.create_dataset(
                    "Vi", (Nt, Nx, 3), dtype=np.float64, chunks=(1, Nx, 3)
                )
                fp.create_dataset(
                    "Pi", (Nt, Nx, 6), dtype=np.float64, chunks=(1, Nx, 6)
                )
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

            # time and step
            with h5py.File(filename, "a") as fp:
                fp["t"][i] = run.get_time_at("field", common_step[index])
                fp["step"][i] = common_step[index]

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


class DataPlotter(JobExecutor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if "plot" in kwargs:
            for key in kwargs["plot"]:
                self.options[key] = kwargs["plot"][key]

    def get_shock_poly_fit(self, filename, fit_steps=None):
        with h5py.File(filename, "r") as fp:
            config = pickle.loads(fp["config"][()])
            x = fp["x"][()]
            B = fp["B"][()]
            t_arr = fp["t"][()]
            if fit_steps is None:
                fit_steps = np.arange(0, t_arr.size)
            params = config["parameter"]
            bx = B[..., 0]
            by = B[..., 1]
            bz = B[..., 2]
            bb = np.sqrt(bx**2 + by**2 + bz**2)
            t_sh, x_sh, v_sh, poly_sh = utils.calc_shock_speed(
                params, fit_steps, t_arr, x, bb, 0.01
            )
        return poly_sh

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

    def plot(self, idx, poly_sh, png, params):
        # get only step-dependent data
        x = params["x"]
        B = params["B"]
        Vi = params["Vi"]
        fuy = params["fuy"]
        xbinc = params["xbinc"]
        xbine = params["xbine"]
        ubine = params["ubine"]
        ebine = params["ebine"]
        step_arr = params["step_arr"]
        t_arr = params["t_arr"]

        mime = params["mime"]
        sigma = params["sigma"]
        vae = np.sqrt(sigma)
        vai = vae / np.sqrt(mime)
        wci = np.sqrt(sigma) / mime
        u0 = params["u0"]
        b0 = np.sqrt(sigma) / np.sqrt(1 + u0**2)
        x_shock = np.polyval(poly_sh, t_arr[idx])

        # get per-step energy distribution and momentum spectrum
        f_ene_step = params["f_ene"][idx]
        pbinc, f_mom = self.convert_to_momentum_spectrum(params["ebinc"], f_ene_step)
        pbine = np.sqrt((ebine + 1) ** 2 - 1)

        fig, axs = plt.subplots(5, 1, figsize=(8, 12), sharex=True)
        plt.sca(axs[0])
        plt.plot(x, B[idx, :, 0] / b0, "k-", label="Bx")
        plt.plot(x, B[idx, :, 1] / b0, "r-", label="By")
        plt.plot(x, B[idx, :, 2] / b0, "b-", label="Bz")
        plt.ylabel(r"$B / B_0$")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.sca(axs[1])
        plt.plot(x, Vi[idx, :, 0] / vai, "k-", label="Vx")
        plt.plot(x, Vi[idx, :, 1] / vai, "r-", label="Vy")
        plt.plot(x, Vi[idx, :, 2] / vai, "b-", label="Vz")
        plt.ylabel(r"$V / V_{A,i}$")
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

        plt.sca(axs[2])
        X, Y = self.pcolormesh_args(xbine, ubine)
        plt.pcolormesh(X, Y, fuy[idx], shading="nearest", norm=mpl.colors.LogNorm())
        plt.ylabel(r"$u_y$")

        plt.sca(axs[3])
        X, Y = self.pcolormesh_args(xbine, pbine)
        plt.pcolormesh(
            X,
            Y,
            f_mom * Y**4,
            shading="nearest",
            norm=mpl.colors.LogNorm(),
        )
        plt.ylabel(r"$p^4 f(p)$")
        plt.semilogy()

        plt.sca(axs[4])
        psample = [0.6, 0.7, 0.8, 0.9, 1.0]
        pindex = np.searchsorted(pbine, psample)
        for i in range(len(psample)):
            f_norm = f_mom[:, pindex[i]] / np.max(f_mom[:, pindex[i]])
            plt.plot(
                xbinc,
                f_norm * 10 ** (-i),
                label=r"$p/m_e c = {:.1f}$".format(psample[i]),
            )
        plt.ylabel(r"$f(p)$ [arb. unit]")
        plt.semilogy()
        plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
        plt.ylim(1.0e-6, 1.0e1)
        plt.xlim(max(x_shock - 200, 0), x_shock + 200)
        for ax in axs:
            ax.grid(True)
        fig.align_ylabels(axs)
        step = step_arr[idx]
        time = t_arr[idx]
        plt.suptitle(r"$\Omega_{{ci}} t$ = {:.2f}".format(wci * time))
        plt.tight_layout()
        plt.savefig(png)
        plt.close(fig)

    def main(self, prefix):
        filename = self.get_filename(prefix, ".h5")
        output = self.options.get("output", "plot")
        png = os.sep.join([self.get_dirname(), output])

        poly_sh = self.get_shock_poly_fit(filename)
        print("Shock speed polynomial fit:", poly_sh)

        with h5py.File(filename, "r") as fp:
            # read common parameters once
            config = pickle.loads(fp["config"][()])
            xbine = fp["xbine"][()]
            ubine = fp["ubine"][()]
            ebine = fp["ebine"][()]
            x = fp["x"][()]
            B = fp["B"][()]
            Vi = fp["Vi"][()]
            fuy = fp["Feu"][..., 1]
            f_ene = fp["Feu"][..., 3]
            xbinc = 0.5 * (xbine[1:] + xbine[:-1])
            ebinc = 0.5 * (ebine[1:] + ebine[:-1])
            step_arr = fp["step"][()]
            t_arr = fp["t"][()]
            params = {
                "config": config,
                "xbine": xbine,
                "ubine": ubine,
                "ebine": ebine,
                "ebinc": ebinc,
                "x": x,
                "B": B,
                "Vi": Vi,
                "fuy": fuy,
                "f_ene": f_ene,
                "xbinc": xbinc,
                "step_arr": step_arr,
                "t_arr": t_arr,
                "mime": config["parameter"]["mime"],
                "sigma": config["parameter"]["sigma"],
                "u0": config["parameter"]["u0"],
            }
            # make plots
            for i in range(B.shape[0]):
                self.plot(i, poly_sh, png + "-{:08d}.png".format(i), params)
                print("Plot saved to:", png + "-{:08d}.png".format(i))

        # convert to mp4
        fps = self.options.get("fps", 10)
        picnix.convert_to_mp4("{:s}".format(png), fps, False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quicklook Script")
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="reduce1d",
        help="Prefix used for output file",
    )
    parser.add_argument(
        "-j",
        "--job",
        type=str,
        default="reduce",
        help="Type of job to perform (reduce, plot)",
    )
    parser.add_argument("config", nargs=1, help="configuration file for the job")
    args = parser.parse_args()

    # read configuration file in TOML or JSON format
    filename = args.config[0]

    if not os.path.exists(filename):
        print("Configuration file not found")
        sys.exit(1)
    else:
        if filename.endswith(".toml"):
            with open(filename, "r") as fileobj:
                config = toml.load(fileobj)
        elif filename.endswith(".json"):
            with open(filename, "r") as fileobj:
                config = json.load(fileobj)
        else:
            print("Unsupported configuration file")
            sys.exit(1)

    # perform the job
    if args.job == "reduce":
        obj = DataReducer(**config)
        obj.main(args.prefix)
    elif args.job == "plot":
        obj = DataPlotter(**config)
        obj.main(args.prefix)
