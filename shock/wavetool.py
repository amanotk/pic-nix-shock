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

mpl.use("Agg") if __name__ == "__main__" else None
import matplotlib.pyplot as plt

# global configuration
plt.rcParams.update({"font.size": 12})

if "PICNIX_DIR" in os.environ:
    sys.path.append(str(pathlib.Path(os.environ["PICNIX_DIR"]) / "script"))
import picnix

try:
    from . import utils
except ImportError:
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
    def __init__(self, config_file):
        self.config_file = config_file
        self.options = self.read_config()
        # self.parameter is initialized in subclasses if needed

    def read_config(self):
        filename = self.config_file
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Configuration file not found: {filename}")

        if filename.endswith(".toml"):
            with open(filename, "r") as fileobj:
                config = toml.load(fileobj)
        elif filename.endswith(".json"):
            with open(filename, "r") as fileobj:
                config = json.load(fileobj)
        else:
            raise ValueError("Unsupported configuration file format")

        # Resolve profile path relative to config file
        if "profile" in config:
            config_dir = os.path.dirname(os.path.abspath(filename))
            profile_path = os.path.join(config_dir, config["profile"])
            config["profile"] = os.path.normpath(profile_path)

        return config

    def read_parameter(self):
        # read parameter from profile
        with open(self.options["profile"], "rb") as fp:
            obj = msgpack.load(fp)
            parameter = obj["configuration"]["parameter"]
        return parameter

    def get_filename(self, basename, ext):
        dirname = self.options.get("dirname", None)
        if dirname is None:
            raise ValueError("dirname is not specified")
        elif not os.path.exists(dirname):
            os.makedirs(dirname)

        return os.sep.join([dirname, basename + ext])

    def main(self, basename):
        raise NotImplementedError


class DataReducer(JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "reduce" in self.options:
            for key in self.options["reduce"]:
                self.options[key] = self.options["reduce"][key]
        self.parameter = self.read_parameter()

    def main(self, basename):
        self.save(self.get_filename(basename, ".h5"))

    def average1d(self, x, size):
        x = ndimage.uniform_filter(x, size=size, axes=(0,), mode="wrap")
        return x[size // 2 :: size]

    def average2d(self, x, size):
        x = ndimage.uniform_filter(x, size=size, axes=(0, 1), mode="wrap")
        return x[size // 2 :: size, size // 2 :: size]

    def encode(self, data):
        return np.frombuffer(pickle.dumps(data), np.int8)

    def save(self, filename):
        profile = self.options.get("profile", None)
        config = self.options.get("config", None)
        prefix = self.options.get("prefix", "field")
        overwrite = self.options.get("overwrite", False)
        num_average = self.options.get("num_average", 4)
        num_xwindow = self.options.get("num_xwindow", 2048)
        step_min = self.options.get("step_min", 380000)
        step_max = self.options.get("step_max", 380000)
        x_offset = self.options.get("x_offset", -80)
        shock_position = self.options.get(
            "shock_position", [1.66365906e-02, -1.39911575e02]
        )

        method = self.options.get("method", "thread")
        run = picnix.Run(profile, config=config, method=method)
        param = run.config["parameter"]
        gamma = np.sqrt(1 + param["u0"] ** 2)
        qme = -np.sqrt(gamma)
        qmi = +np.sqrt(gamma) / param["mime"]
        config = self.encode(run.config)

        field_step = run.get_step(prefix)
        index_min = np.searchsorted(field_step, step_min)
        index_end = np.searchsorted(field_step, step_max)
        index_range = np.arange(index_min, index_end + 1)

        Nt = index_end - index_min + 1
        Nx = num_xwindow
        Ny = run.Ny
        Mx = Nx // num_average
        My = Ny // num_average
        dh = run.delh
        xc = self.average1d(run.xc, num_average)
        yc = self.average1d(run.yc, num_average)

        # adjust the boundary
        xc[0] = xc[1] - dh * num_average
        xc[-1] = xc[-2] + dh * num_average
        yc[0] = yc[1] - dh * num_average
        yc[-1] = yc[-2] + dh * num_average

        if not os.path.exists(filename) or overwrite == True:
            # create HDF5 file and datasets first
            with h5py.File(filename, "w") as fp:
                dummpy_step = (-1) * np.ones((Nt,), np.int32)
                fp.create_dataset("config", data=config, dtype=np.int8)
                fp.create_dataset("step", (Nt,), data=dummpy_step, dtype=np.int32)
                fp.create_dataset("t", (Nt,), dtype=np.float64)
                fp.create_dataset("x", (Nt, Mx), dtype=np.float64, chunks=(1, Mx))
                fp.create_dataset("y", (Nt, My), dtype=np.float64, chunks=(1, My))
                fp.create_dataset(
                    "E", (Nt, My, Mx, 3), dtype=np.float64, chunks=(1, My, Mx, 3)
                )
                fp.create_dataset(
                    "B", (Nt, My, Mx, 3), dtype=np.float64, chunks=(1, My, Mx, 3)
                )
                fp.create_dataset(
                    "Je", (Nt, My, Mx, 4), dtype=np.float64, chunks=(1, My, Mx, 4)
                )
                fp.create_dataset(
                    "Ji", (Nt, My, Mx, 4), dtype=np.float64, chunks=(1, My, Mx, 4)
                )

        # read step
        with h5py.File(filename, "r") as fp:
            step_in_file = fp["step"][()]

        # read and process data for each step
        for i, index in enumerate(tqdm.tqdm(index_range)):
            # skip if the step is already stored
            if step_in_file[i] == field_step[index]:
                continue

            # read data
            time = run.get_time_at(prefix, field_step[index])
            data = run.read_at(prefix, field_step[index])
            uf = data["uf"].mean(axis=0)
            je = data["um"].mean(axis=0)[..., 0, 0:4] * qme
            ji = data["um"].mean(axis=0)[..., 1, 0:4] * qmi

            # average
            uf = self.average2d(uf, num_average)
            je = self.average2d(je, num_average)
            ji = self.average2d(ji, num_average)

            # linear interpolation in x
            xmin = np.polyval(shock_position, time) + x_offset
            xnew = np.arange(Mx) * num_average * dh + xmin
            xind = xc.searchsorted(xnew)
            delta = ((xc[xind] - xnew) / (xc[xind] - xc[xind - 1]))[
                np.newaxis, :, np.newaxis
            ]
            uf = delta * uf[..., xind - 1, :] + (1 - delta) * uf[..., xind, :]
            je = delta * je[..., xind - 1, :] + (1 - delta) * je[..., xind, :]
            ji = delta * ji[..., xind - 1, :] + (1 - delta) * ji[..., xind, :]

            # store data
            with h5py.File(filename, "a") as fp:
                fp["step"][i] = field_step[index]
                fp["t"][i] = time
                fp["x"][i, ...] = xnew
                fp["y"][i, ...] = yc
                fp["E"][i, ...] = uf[..., 0:3]
                fp["B"][i, ...] = uf[..., 3:6]
                fp["Je"][i, ...] = je
                fp["Ji"][i, ...] = ji


class SummaryPlotter(JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "plot" in self.options:
            for key in self.options["plot"]:
                self.options[key] = self.options["plot"][key]
        self.plot_dict = None
        self.parameter = None  # initialized later

    def read_parameter(self):
        # ignore
        return None

    def main(self, basename):
        filename = self.get_filename(basename, ".h5")
        output = self.options.get("output", "plot")

        # read parameters here
        with h5py.File(filename, "r") as fp:
            self.parameter = pickle.loads(fp["config"][()])["parameter"]

        wci = np.sqrt(self.parameter["sigma"]) / self.parameter["mime"]
        png = self.get_filename(output, "")

        with h5py.File(filename, "r") as fp:
            t = fp["t"]

            for i in tqdm.tqdm(range(t.shape[0])):
                # read data
                X, Y, vars, labels = self.get_plot_variables(fp, i)

                # apply bandpass filter if requested
                if "bandpass" in self.options:
                    kl, kh, dk = self.options["bandpass"]
                    xx = fp["x"][i, :]
                    dh = xx[1] - xx[0]
                    vars = utils.bandpass_filter2d(vars, kl, kh, dk, dh)

                self.plot(X, Y, t[i] * wci, vars, labels)
                self.save(png + "-{:08d}.png".format(i))

        # convert to mp4
        fps = self.options.get("fps", 10)
        picnix.convert_to_mp4("{:s}".format(png), fps, False)

    def save(self, filename):
        if self.plot_dict is not None and "fig" in self.plot_dict:
            self.plot_dict["fig"].savefig(filename)
        else:
            print("No plot to save")

    def get_plot_variables(self, fileobj, index):
        sigma = self.parameter["sigma"]
        u0 = self.parameter["u0"]
        B0 = np.sqrt(sigma) / np.sqrt(1 + u0**2)
        E0 = B0 * u0 / np.sqrt(1 + u0**2)
        J0 = B0

        quantity = self.options.get("quantity", "field")
        xx = fileobj["x"][index, ...]
        yy = fileobj["y"][index, ...]
        X, Y = np.meshgrid(xx, yy)

        if quantity == "field":
            E = fileobj["E"][index, ...]
            B = fileobj["B"][index, ...]

            vars = [
                E[..., 0] / E0,
                E[..., 1] / E0,
                E[..., 2] / E0,
                B[..., 0] / B0,
                B[..., 1] / B0,
                B[..., 2] / B0,
            ]
            labels = [
                r"$E_x / E_0$",
                r"$E_y / E_0$",
                r"$E_z / E_0$",
                r"$B_x / B_0$",
                r"$B_y / B_0$",
                r"$B_z / B_0$",
            ]
        elif quantity == "current":
            Je = fileobj["Je"][index, ...]
            Ji = fileobj["Ji"][index, ...]

            vars = [
                Je[..., 1] / J0,
                Je[..., 2] / J0,
                Je[..., 3] / J0,
                Ji[..., 1] / J0,
                Ji[..., 2] / J0,
                Ji[..., 3] / J0,
            ]
            labels = [
                r"$J_{e,x} / J_0$",
                r"$J_{e,y} / J_0$",
                r"$J_{e,z} / J_0$",
                r"$J_{i,x} / J_0$",
                r"$J_{i,y} / J_0$",
                r"$J_{i,z} / J_0$",
            ]
        else:
            raise ValueError("Invalid quantity")

        return X, Y, vars, labels

    def plot(self, X, Y, time, vars, labels):
        if self.plot_dict is None:
            self.plot_dict = self.plot_new(X, Y, vars, labels)
        else:
            self.plot_dict = self.plot_update(X, Y, vars, labels)

        plt.suptitle(r"$\Omega_{{ci}} t$ = {:5.2f}".format(time))

    def plot_new(self, X, Y, vars, labels):
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()
        vlim = get_vlim(vars, 10)

        fig = plt.figure(figsize=(10, 8), dpi=120)
        fig.subplots_adjust(
            top=0.92,
            bottom=0.05,
            left=0.08,
            right=0.90,
            hspace=0.20,
            wspace=0.22,
        )
        gridspec = fig.add_gridspec(
            3, 3, height_ratios=[1, 1, 1], width_ratios=[50, 1, 50]
        )
        axs = [0] * 6
        axs[0] = fig.add_subplot(gridspec[0, 0])
        axs[1] = fig.add_subplot(gridspec[1, 0])
        axs[2] = fig.add_subplot(gridspec[2, 0])
        axs[3] = fig.add_subplot(gridspec[0, 2])
        axs[4] = fig.add_subplot(gridspec[1, 2])
        axs[5] = fig.add_subplot(gridspec[2, 2])

        cxs = [0] * 6
        img = [0] * 6
        for i in range(6):
            plt.sca(axs[i])
            # plot and colorbar
            img[i] = plt.imshow(
                vars[i],
                extent=[xmin, xmax, ymin, ymax],
                vmin=vlim[i][0],
                vmax=vlim[i][1],
                origin="lower",
            )
            # appearance
            axs[i].set_xlim(xmin, xmax)
            axs[i].set_ylim(ymin, ymax)
            axs[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
            axs[i].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
            axs[i].set_aspect("equal")
            # colorbar
            cxs[i] = plt.axes(get_colorbar_position_next(axs[i], 0.025))
            plt.colorbar(cax=cxs[i])
            axs[i].set_title(labels[i])
        [axs[i].set_ylabel(r"$y / c/\omega_{pe}$") for i in (0, 1, 2)]
        [axs[i].set_xlabel(r"$x / c/\omega_{pe}$") for i in (2, 5)]

        return {"fig": fig, "axs": axs, "img": img, "cxs": cxs}

    def plot_update(self, X, Y, vars, labels):
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()
        vlim = get_vlim(vars, 10)

        fig = self.plot_dict["fig"]
        axs = self.plot_dict["axs"]
        img = self.plot_dict["img"]
        cxs = self.plot_dict["cxs"]
        for i in range(6):
            plt.sca(axs[i])
            # image
            img[i].set_array(vars[i])
            img[i].set_extent([xmin, xmax, ymin, ymax])
            axs[i].set_xlim(xmin, xmax)
            axs[i].set_ylim(ymin, ymax)
            axs[i].set_title(labels[i])
            # colorbar
            img[i].set_clim(vlim[i])
            cxs[i].cla()
            plt.colorbar(img[i], cax=cxs[i])

        return {"fig": fig, "axs": axs, "img": img, "cxs": cxs}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quicklook Script")
    parser.add_argument(
        "-p",
        "--basename",
        type=str,
        default="wavetool",
        help="basename used for output image and movie files",
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

    # perform the job
    if args.job == "reduce":
        obj = DataReducer(args.config[0])
        obj.main(args.basename)

    if args.job == "plot":
        obj = SummaryPlotter(args.config[0])
        obj.main(args.basename)
