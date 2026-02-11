#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pathlib
import pickle
import sys

import h5py
import matplotlib as mpl
import numpy as np
import scipy.ndimage as ndimage
import toml
import tqdm

mpl.use("Agg") if __name__ == "__main__" else None
import matplotlib.pyplot as plt

# global configuration
plt.rcParams.update({"font.size": 12})

if "PICNIX_DIR" in os.environ:
    sys.path.append(str(pathlib.Path(os.environ["PICNIX_DIR"]) / "script"))
try:
    from . import base, utils
except ImportError:
    import base
    import utils

import picnix


def select_debug_indices(size, debug=False, debug_count=8, debug_mode="uniform"):
    size = int(size)
    if size <= 0:
        return np.array([], dtype=np.int64)
    if not debug:
        return np.arange(size, dtype=np.int64)

    debug_count = int(debug_count)
    if debug_count <= 0:
        raise ValueError("debug_count must be a positive integer")

    count = min(debug_count, size)
    if debug_mode == "head":
        return np.arange(count, dtype=np.int64)
    if debug_mode == "uniform":
        return np.unique(np.linspace(0, size - 1, num=count, dtype=np.int64))
    raise ValueError("Unknown debug_mode: {:s}. Use 'head' or 'uniform'.".format(str(debug_mode)))


class DataAnalyzer(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "analyze" in self.options:
            for key in self.options["analyze"]:
                self.options[key] = self.options["analyze"][key]
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

    def diff_central_periodic(self, x, dx, axis):
        return (np.roll(x, -1, axis=axis) - np.roll(x, 1, axis=axis)) / (2.0 * dx)

    def diff_central_zero_boundary(self, x, dx, axis):
        grad = np.zeros_like(x)
        dst = [slice(None)] * x.ndim
        src_m = [slice(None)] * x.ndim
        src_p = [slice(None)] * x.ndim
        dst[axis] = slice(1, -1)
        src_m[axis] = slice(0, -2)
        src_p[axis] = slice(2, None)
        grad[tuple(dst)] = (x[tuple(src_p)] - x[tuple(src_m)]) / (2.0 * dx)
        return grad

    def calc_e_ohm(self, B, M, dx, dy):
        Lambda = M[..., 0]
        Gamma = M[..., 1:4]

        Pxx = M[..., 4]
        Pyy = M[..., 5]
        Pxy = M[..., 7]
        Pyz = M[..., 8]
        Pzx = M[..., 9]

        dPxx_dx = self.diff_central_zero_boundary(Pxx, dx, axis=1)
        dPxy_dx = self.diff_central_zero_boundary(Pxy, dx, axis=1)
        dPzx_dx = self.diff_central_zero_boundary(Pzx, dx, axis=1)
        dPxy_dy = self.diff_central_periodic(Pxy, dy, axis=0)
        dPyy_dy = self.diff_central_periodic(Pyy, dy, axis=0)
        dPyz_dy = self.diff_central_periodic(Pyz, dy, axis=0)

        divPi = np.stack(
            [
                dPxx_dx + dPxy_dy,
                dPxy_dx + dPyy_dy,
                dPzx_dx + dPyz_dy,
            ],
            axis=-1,
        )

        rhs = -np.cross(Gamma, B, axis=-1) + divPi
        denom = Lambda
        E_ohm = rhs / (denom[..., np.newaxis] + 1.0e-32)
        return E_ohm

    def save(self, filename):
        profile = self.options.get("profile", None)
        config = self.options.get("config", None)
        prefix = self.options.get("prefix", "field")
        overwrite = self.options.get("overwrite", False)
        num_average = self.options.get("num_average", 4)
        num_xwindow = self.options.get("num_xwindow", 2048)
        step_min = self.options.get("step_min", 0)
        step_max = self.options.get("step_max", sys.maxsize)
        x_offset = self.options.get("x_offset", -80)
        debug = self.options.get("debug", False)
        debug_count = self.options.get("debug_count", 8)
        debug_mode = self.options.get("debug_mode", "uniform")

        shock_position = self.options.get("shock_position", None)
        if shock_position is None:
            reduce1d_result = (
                pathlib.Path(self.get_dirname()).parent / "reduce1d" / "reduce1d_result.toml"
            )
            if not reduce1d_result.exists():
                sys.exit(
                    "Error: 'shock_position' is not provided and default reduce1d result file "
                    "'{}' was not found.".format(reduce1d_result)
                )
            shock_position = toml.load(str(reduce1d_result))["shock_position"]

        method = self.options.get("method", "thread")
        run = picnix.Run(profile, config=config, method=method)
        param = run.config["parameter"]
        gamma = np.sqrt(1 + param["u0"] ** 2)
        qme = -np.sqrt(gamma)
        qmi = +np.sqrt(gamma) / param["mime"]
        config = self.encode(run.config)

        field_step = run.get_step(prefix)
        step_min = max(field_step.min(), step_min)
        step_max = min(field_step.max(), step_max)
        index_min = np.searchsorted(field_step, step_min)
        index_end = np.searchsorted(field_step, step_max)
        index_range = np.arange(index_min, index_end + 1)
        debug_indices = select_debug_indices(index_range.size, debug, debug_count, debug_mode)
        index_range = index_range[debug_indices]
        Nt = index_range.size
        if Nt == 0:
            raise ValueError("No snapshots selected. Adjust step range or debug settings.")

        data = run.read_at(prefix, field_step[index_min], "uf")
        xc = data["xc"]
        yc = data["yc"]

        Nx = num_xwindow
        Ny = yc.size
        Mx = Nx // num_average
        My = Ny // num_average
        dh = xc[1] - xc[0]
        xc = self.average1d(xc, num_average)
        yc = self.average1d(yc, num_average)

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
                fp.create_dataset("E", (Nt, My, Mx, 3), dtype=np.float64, chunks=(1, My, Mx, 3))
                fp.create_dataset("E_ohm", (Nt, My, Mx, 3), dtype=np.float64, chunks=(1, My, Mx, 3))
                fp.create_dataset("B", (Nt, My, Mx, 3), dtype=np.float64, chunks=(1, My, Mx, 3))
                fp.create_dataset("J", (Nt, My, Mx, 8), dtype=np.float64, chunks=(1, My, Mx, 8))
                fp.create_dataset("M", (Nt, My, Mx, 10), dtype=np.float64, chunks=(1, My, Mx, 10))

        # support old output files without transformed moments/current dataset
        rewrite_all = False
        with h5py.File(filename, "a") as fp:
            if "J" not in fp:
                fp.create_dataset("J", (Nt, My, Mx, 8), dtype=np.float64, chunks=(1, My, Mx, 8))
                rewrite_all = True
            if "M" not in fp:
                fp.create_dataset("M", (Nt, My, Mx, 10), dtype=np.float64, chunks=(1, My, Mx, 10))
                rewrite_all = True
            if "E_ohm" not in fp:
                fp.create_dataset("E_ohm", (Nt, My, Mx, 3), dtype=np.float64, chunks=(1, My, Mx, 3))
                rewrite_all = True

        # read step
        with h5py.File(filename, "r") as fp:
            step_in_file = fp["step"][()]

        # read and process data for each step
        for i, index in enumerate(tqdm.tqdm(index_range)):
            # skip if the step is already stored
            if not rewrite_all and step_in_file[i] == field_step[index]:
                continue

            # read data
            time = run.get_time_at(prefix, field_step[index])
            data = run.read_at(prefix, field_step[index])
            uf = data["uf"].mean(axis=0)
            um = data["um"].mean(axis=0)
            je = um[..., 0, 0:4] * qme
            ji = um[..., 1, 0:4] * qmi
            m = np.empty(um.shape[:-2] + (10,), dtype=um.dtype)
            m[..., 0:4] = um[..., 0, 0:4] * qme**2 + um[..., 1, 0:4] * qmi**2
            m[..., 4:10] = um[..., 0, 8:14] * qme + um[..., 1, 8:14] * qmi

            # average
            uf = self.average2d(uf, num_average)
            je = self.average2d(je, num_average)
            ji = self.average2d(ji, num_average)
            m = self.average2d(m, num_average)
            j = np.concatenate([je, ji], axis=-1)

            # linear interpolation in x
            xmin = np.polyval(shock_position, time) + x_offset
            xnew = np.arange(Mx) * num_average * dh + xmin
            xind = xc.searchsorted(xnew)
            delta = ((xc[xind] - xnew) / (xc[xind] - xc[xind - 1]))[np.newaxis, :, np.newaxis]
            uf = delta * uf[..., xind - 1, :] + (1 - delta) * uf[..., xind, :]
            j = delta * j[..., xind - 1, :] + (1 - delta) * j[..., xind, :]
            m = delta * m[..., xind - 1, :] + (1 - delta) * m[..., xind, :]
            e_ohm = self.calc_e_ohm(uf[..., 3:6], m, num_average * dh, num_average * dh)

            # store data
            with h5py.File(filename, "a") as fp:
                fp["step"][i] = field_step[index]
                fp["t"][i] = time
                fp["x"][i, ...] = xnew
                fp["y"][i, ...] = yc
                fp["E"][i, ...] = uf[..., 0:3]
                fp["E_ohm"][i, ...] = e_ohm
                fp["B"][i, ...] = uf[..., 3:6]
                fp["J"][i, ...] = j
                fp["M"][i, ...] = m


class SummaryPlotter(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "plot" in self.options:
            for key in self.options["plot"]:
                self.options[key] = self.options["plot"][key]
        self.plot_dict = None
        self.parameter = None  # initialized later
        self.shock_position = None

    def read_parameter(self):
        # ignore
        return None

    def main(self, basename):
        filename = self.get_filename(basename, ".h5")
        output = self.options.get("output", "wavetool")

        step_min = self.options.get("step_min", 0)
        step_max = self.options.get("step_max", sys.maxsize)
        debug = self.options.get("debug", False)
        debug_count = self.options.get("debug_count", 8)
        debug_mode = self.options.get("debug_mode", "uniform")

        # read parameters here
        with h5py.File(filename, "r") as fp:
            self.parameter = pickle.loads(fp["config"][()])["parameter"]

        self.shock_position = self.options.get("shock_position", None)
        if self.shock_position is None:
            reduce1d_result = (
                pathlib.Path(self.get_dirname()).parent / "reduce1d" / "reduce1d_result.toml"
            )
            if reduce1d_result.exists():
                self.shock_position = toml.load(str(reduce1d_result)).get("shock_position", None)

        wci = np.sqrt(self.parameter["sigma"]) / self.parameter["mime"]
        png = self.get_filename(output, "")

        with h5py.File(filename, "r") as fp:
            t = fp["t"]
            step = fp["step"]
            index_min = np.searchsorted(step, step_min)
            index_max = np.searchsorted(step, step_max)
            frame_indices = np.arange(index_min, index_max, dtype=np.int64)
            debug_indices = select_debug_indices(frame_indices.size, debug, debug_count, debug_mode)
            frame_indices = frame_indices[debug_indices]

            for i in tqdm.tqdm(frame_indices):
                # read data
                X, Y, Az, vars, labels, xlim = self.get_plot_variables(fp, i)

                # apply bandpass filter if requested
                if "bandpass" in self.options:
                    kl, kh, dk = self.options["bandpass"]
                    xx = fp["x"][i, :]
                    dh = xx[1] - xx[0]
                    vars = utils.bandpass_filter2d(vars, kl, kh, dk, dh)

                self.plot(X, Y, t[i] * wci, Az, vars, labels, xlim=xlim)
                self.save(png + "-{:08d}.png".format(step[i]))

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
        dh = self.parameter["delh"]
        B0 = np.sqrt(sigma) / np.sqrt(1 + u0**2)
        E0 = B0 * u0 / np.sqrt(1 + u0**2)
        J0 = B0

        quantity = self.options.get("quantity", "field")
        xx = fileobj["x"][index, ...]
        yy = fileobj["y"][index, ...]
        X, Y = np.meshgrid(xx, yy)
        xlim = self.get_x_window(xx, yy, fileobj["t"][index])

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
            if "J" in fileobj:
                J = fileobj["J"][index, ...]
                Je = J[..., 0:4]
                Ji = J[..., 4:8]
            else:
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

        # vector potential
        B = fileobj["B"][index, ...]
        Az = utils.calc_vector_potential2d(B, dh)

        return X, Y, Az, vars, labels, xlim

    def get_x_window(self, xx, yy, time):
        xx = np.asarray(xx)
        yy = np.asarray(yy)
        xmin = float(xx.min())
        xmax = float(xx.max())
        ymin = float(yy.min())
        ymax = float(yy.max())

        aspect_ratio = float(self.options.get("aspect_ratio", 2.0))
        if aspect_ratio <= 0.0:
            raise ValueError("aspect_ratio must be positive")

        x_center_offset = float(self.options.get("x_center_offset", 0.0))
        if self.shock_position is not None:
            x_center = float(np.polyval(self.shock_position, time)) + x_center_offset
        else:
            x_center = 0.5 * (xmin + xmax) + x_center_offset

        ly = ymax - ymin
        lx = xmax - xmin
        if ly <= 0.0 or lx <= 0.0:
            return xmin, xmax

        target_lx = aspect_ratio * ly
        if target_lx >= lx:
            return xmin, xmax

        left = x_center - 0.5 * target_lx
        right = x_center + 0.5 * target_lx
        if left < xmin:
            right += xmin - left
            left = xmin
        if right > xmax:
            left -= right - xmax
            right = xmax

        left = max(left, xmin)
        right = min(right, xmax)
        return left, right

    def plot(self, X, Y, time, Az, vars, labels, xlim=None):
        if self.plot_dict is None:
            self.plot_dict = self.plot_new(X, Y, Az, vars, labels, xlim=xlim)
        else:
            self.plot_dict = self.plot_update(X, Y, Az, vars, labels, xlim=xlim)

        plt.suptitle(r"$\Omega_{{ci}} t$ = {:5.2f}".format(time))

    def plot_new(self, X, Y, Az, vars, labels, xlim=None):
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()
        if xlim is not None:
            xmin, xmax = xlim

        # automatic colorbar limits
        vlim = base.get_vlim(vars, 10)

        common_args = {
            "extent": [xmin, xmax, ymin, ymax],
            "origin": "lower",
            "cmap": "viridis",
        }
        args = [{**common_args, "vmin": vlim[i][0], "vmax": vlim[i][1]} for i in range(6)]

        fig = plt.figure(figsize=(10, 8), dpi=120)
        fig.subplots_adjust(
            top=0.92,
            bottom=0.05,
            left=0.08,
            right=0.90,
            hspace=0.20,
            wspace=0.22,
        )
        gridspec = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[50, 1, 50])
        axs = [0] * 6
        axs[0] = fig.add_subplot(gridspec[0, 0])
        axs[1] = fig.add_subplot(gridspec[1, 0])
        axs[2] = fig.add_subplot(gridspec[2, 0])
        axs[3] = fig.add_subplot(gridspec[0, 2])
        axs[4] = fig.add_subplot(gridspec[1, 2])
        axs[5] = fig.add_subplot(gridspec[2, 2])

        cxs = [0] * 6
        img = [0] * 6
        cnt = [0] * 6
        clb = [0] * 6

        for i in range(6):
            plt.sca(axs[i])
            # plot image
            img[i] = plt.imshow(vars[i], **args[i])
            # field lines
            cnt[i] = plt.contour(X, Y, Az, levels=25, colors="k", linewidths=0.5)
            # appearance
            axs[i].set_xlim(xmin, xmax)
            axs[i].set_ylim(ymin, ymax)
            axs[i].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
            axs[i].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(5))
            axs[i].set_aspect("equal")
            # colorbar
            cxs[i] = plt.axes(base.get_colorbar_position_next(axs[i], 0.025))
            clb[i] = plt.colorbar(img[i], cax=cxs[i])
            axs[i].set_title(labels[i])

        [axs[i].set_ylabel(r"$y / c/\omega_{pe}$") for i in (0, 1, 2)]
        [axs[i].set_xlabel(r"$x / c/\omega_{pe}$") for i in (2, 5)]

        return {"fig": fig, "axs": axs, "img": img, "cnt": cnt, "cxs": cxs, "clb": clb}

    def plot_update(self, X, Y, Az, vars, labels, xlim=None):
        xmin = X.min()
        xmax = X.max()
        ymin = Y.min()
        ymax = Y.max()
        if xlim is not None:
            xmin, xmax = xlim
        vlim = base.get_vlim(vars, 10)

        fig = self.plot_dict["fig"]
        axs = self.plot_dict["axs"]
        img = self.plot_dict["img"]
        cnt = self.plot_dict["cnt"]
        cxs = self.plot_dict["cxs"]
        clb = self.plot_dict["clb"]

        # remove contours
        for i in range(6):
            cnt[i].remove()

        for i in range(6):
            plt.sca(axs[i])
            # image
            img[i].set_array(vars[i])
            img[i].set_extent([xmin, xmax, ymin, ymax])
            # field lines
            cnt[i] = plt.contour(X, Y, Az, levels=25, colors="k", linewidths=0.5)
            axs[i].set_xlim(xmin, xmax)
            axs[i].set_ylim(ymin, ymax)
            axs[i].set_title(labels[i])
            # colorbar
            img[i].set_clim(vlim[i])
            clb[i].update_normal(img[i])

        return {"fig": fig, "axs": axs, "img": img, "cnt": cnt, "cxs": cxs, "clb": clb}


def main():
    import argparse

    script_name = "wavetool"

    parser = argparse.ArgumentParser(description="Quicklook Script")
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
        help="Type of job to perform (analyze, plot). Can be combined with comma.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        default="field",
        help="diagnostic prefix to read from run data",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="run a reduced subset of snapshots",
    )
    parser.add_argument(
        "--debug-count",
        type=int,
        default=8,
        help="number of snapshots to process in debug mode",
    )
    parser.add_argument(
        "--debug-mode",
        type=str,
        default="uniform",
        choices=["head", "uniform"],
        help="snapshot selection mode for debug subset",
    )
    parser.add_argument("config", nargs=1, help="configuration file for the job")
    args = parser.parse_args()
    config = args.config[0]
    output = args.output
    prefix = args.prefix
    debug = args.debug
    debug_count = args.debug_count
    debug_mode = args.debug_mode

    def apply_runtime_options(obj):
        obj.options["prefix"] = prefix
        obj.options["debug"] = debug
        obj.options["debug_count"] = debug_count
        obj.options["debug_mode"] = debug_mode
        dirname = obj.options.get("dirname", None)
        if dirname is not None:
            suffix = "-{:s}".format(str(prefix))
            if not dirname.endswith(suffix):
                obj.options["dirname"] = dirname + suffix

    jobs = args.job.split(",")

    # perform analyze job first if requested
    if "analyze" in jobs:
        obj = DataAnalyzer(config)
        apply_runtime_options(obj)
        obj.main(output)
        jobs = [j for j in jobs if j != "analyze"]

    # check prerequisite for remaining jobs
    if len(jobs) > 0:
        obj = DataAnalyzer(config)
        apply_runtime_options(obj)
        filename = obj.get_filename(output, ".h5")
        if not os.path.exists(filename):
            sys.exit("Error: File '{}' not found. Please run 'analyze' job first.".format(filename))

    # perform remaining jobs
    for job in jobs:
        if job == "plot":
            obj = SummaryPlotter(config)
            apply_runtime_options(obj)
            obj.main(output)
        else:
            raise ValueError("Unknown job: {:s}".format(job))


if __name__ == "__main__":
    main()
