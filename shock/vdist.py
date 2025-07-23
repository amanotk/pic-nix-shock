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
        prefix = self.options.get("prefix", "particle")
        overwrite = self.options.get("overwrite", False)
        num_average = self.options.get("num_average", 20)
        num_xwindow = self.options.get("num_xwindow", 2048)
        step_min = self.options.get("step_min", 380000)
        step_max = self.options.get("step_max", 380000)
        x_offset = self.options.get("x_offset", -80)
        shock_position = self.options.get(
            "shock_position", [1.66365906e-02, -1.39911575e02]
        )

        # binning parameters
        upara_nbins = self.options.get("upara_nbins", 80)
        upara_range = self.options.get("upara_range", [-1.2, +1.2])
        uperp_nbins = self.options.get("uperp_nbins", 40)
        uperp_range = self.options.get("uperp_range", [0.0, +1.2])
        uabs_nbins = self.options.get("uabs_nbins", 80)
        uabs_range = self.options.get("uabs_range", [1.0e-3, 1.0e1])
        ucos_nbins = self.options.get("ucos_nbins", 17)
        ucos_range = [-1.0, 1.0]

        method = self.options.get("method", "thread")
        run = picnix.Run(profile, config=config, method=method)
        config = self.encode(run.config)

        field_step = run.get_step("field")
        particle_step = run.get_step(prefix)
        index_min = np.searchsorted(particle_step, step_min)
        index_end = np.searchsorted(particle_step, step_max)
        index_range = np.arange(index_min, index_end + 1)

        Nt = index_end - index_min + 1
        Nx = num_xwindow
        Ny = run.Ny
        Mx = Nx // num_average
        My = Ny // num_average
        dh = run.delh
        xc = self.average1d(run.xc, num_average)
        # adjust the boundary
        xc[0] = xc[1] - dh * num_average
        xc[-1] = xc[-2] + dh * num_average

        # bins
        x_bins = np.linspace(0.0, dh * num_average * Mx, Mx + 1)
        y_bins = np.linspace(0.0, dh * num_average * My, My + 1)
        upara_bins = np.linspace(upara_range[0], upara_range[1], upara_nbins + 1)
        uperp_bins = np.linspace(uperp_range[0], uperp_range[1], uperp_nbins + 1)
        uabs_bins = np.geomspace(uabs_range[0], uabs_range[1], uabs_nbins + 1)
        ucos_bins = np.linspace(ucos_range[0], ucos_range[1], ucos_nbins + 1)

        # data to be saved
        bb = np.zeros((My, Mx, 3), dtype=np.float64)
        vb = np.zeros((My, Mx, 3), dtype=np.float64)
        c_dist = np.zeros((uperp_nbins, upara_nbins, My, Mx), dtype=np.float64)
        p_dist = np.zeros((ucos_nbins, uabs_nbins, My, Mx), dtype=np.float64)

        if not os.path.exists(filename) or overwrite == True:
            # create HDF5 file and datasets first
            with h5py.File(filename, "w") as fp:
                dummpy_step = (-1) * np.ones((Nt,), np.int32)
                fp.create_dataset("config", data=config, dtype=np.int8)
                fp.create_dataset("step", (Nt,), data=dummpy_step, dtype=np.int32)
                fp.create_dataset("t", (Nt,), dtype=np.float64)
                fp.create_dataset("x", (Nt, Mx), dtype=np.float64, chunks=(1, Mx))
                fp.create_dataset("y", (Nt, My), dtype=np.float64, chunks=(1, My))
                fp.create_dataset("bb", (Nt, My, Mx, 3), dtype=np.float64)
                fp.create_dataset("vb", (Nt, My, Mx, 3), dtype=np.float64)
                fp.create_dataset(
                    "c_dist", (Nt, uperp_nbins, upara_nbins, My, Mx), dtype=np.float64
                )
                fp.create_dataset(
                    "p_dist", (Nt, ucos_nbins, uabs_nbins, My, Mx), dtype=np.float64
                )
                # bins
                fp.create_dataset("x_bins", (Nt, Mx + 1), dtype=np.float64)
                fp.create_dataset("y_bins", (Nt, My + 1), dtype=np.float64)
                fp.create_dataset("upara_bins", data=upara_bins, dtype=np.float64)
                fp.create_dataset("uperp_bins", data=uperp_bins, dtype=np.float64)
                fp.create_dataset("uabs_bins", data=uabs_bins, dtype=np.float64)
                fp.create_dataset("ucos_bins", data=ucos_bins, dtype=np.float64)

        # read step
        with h5py.File(filename, "r") as fp:
            step_in_file = fp["step"][()]

        # read and process data for each step
        for i, index in enumerate(tqdm.tqdm(index_range)):
            # skip if the step is already stored
            if step_in_file[i] == particle_step[index]:
                continue

            # clear
            c_dist[...] = 0.0
            p_dist[...] = 0.0
            bb[...] = 0.0
            vb[...] = 0.0

            # read data
            step = particle_step[index]
            time = run.get_time_at(prefix, step)

            # linear interpolation in x
            xmin = np.polyval(shock_position, time) + x_offset
            ymin = 0.0
            xnew = 0.5 * (x_bins[:-1] + x_bins[+1:]) + xmin
            xind = xc.searchsorted(xnew)
            delta = (xc[xind] - xnew) / (xc[xind] - xc[xind - 1])[np.newaxis, :]

            # magnetic field direction
            data = run.read_at("field", step, "uf")
            uf = self.average2d(data["uf"].mean(axis=0), num_average)
            Bx = delta * uf[..., xind - 1, 3] + (1 - delta) * uf[..., xind, 3]
            By = delta * uf[..., xind - 1, 4] + (1 - delta) * uf[..., xind, 4]
            Bz = delta * uf[..., xind - 1, 5] + (1 - delta) * uf[..., xind, 5]
            BB = np.sqrt(Bx**2 + By**2 + Bz**2)
            bb[..., 0] = Bx / BB
            bb[..., 1] = By / BB
            bb[..., 2] = Bz / BB

            # bulk velocity
            data = run.read_at("field", step, "um")
            ue = self.average2d(data["um"][..., 0, 0:4].mean(axis=0), num_average)
            ro = delta * ue[..., xind - 1, 0] + (1 - delta) * ue[..., xind, 0]
            jx = delta * ue[..., xind - 1, 1] + (1 - delta) * ue[..., xind, 1]
            jy = delta * ue[..., xind - 1, 2] + (1 - delta) * ue[..., xind, 2]
            jz = delta * ue[..., xind - 1, 3] + (1 - delta) * ue[..., xind, 3]
            vb[..., 0] = jx / ro
            vb[..., 1] = jy / ro
            vb[..., 2] = jz / ro

            # process electons
            options = {
                "bb": bb,
                "vb": vb,
                "c_dist": c_dist,
                "p_dist": p_dist,
                "x_bins": x_bins + xmin,
                "y_bins": y_bins + ymin,
                "upara_bins": upara_bins,
                "uperp_bins": uperp_bins,
                "uabs_bins": uabs_bins,
                "ucos_bins": ucos_bins,
                "blocksize": 2**20,
            }
            jsonfiles = run.diag_handlers[prefix].find_json_at_step(step)
            self.process_particle(jsonfiles, "up00", **options)

            # store data
            with h5py.File(filename, "a") as fp:
                fp["step"][i] = step
                fp["t"][i] = time
                fp["x"][i, ...] = 0.5 * (x_bins[:-1] + x_bins[+1:]) + xmin
                fp["y"][i, ...] = 0.5 * (y_bins[:-1] + y_bins[+1:]) + ymin
                fp["bb"][i, ...] = options["bb"]
                fp["vb"][i, ...] = options["vb"]
                fp["c_dist"][i, ...] = options["c_dist"]
                fp["p_dist"][i, ...] = options["p_dist"]
                fp["x_bins"][i, ...] = options["x_bins"]
                fp["y_bins"][i, ...] = options["y_bins"]

    def process_particle(self, jsonfiles, dsname, **opt):
        blocksize = opt.get("blocksize")
        c_dist = opt.get("c_dist")
        p_dist = opt.get("p_dist")

        for f in jsonfiles:
            dataset, meta = picnix.read_jsonfile(f)
            byteorder, layout, datafile = picnix.process_meta(meta)
            offset, dtype, shape = picnix.get_dataset_info(dataset[dsname], byteorder)
            Np = shape[0]  # number of particles
            Nc = shape[1]  # number of components

            with open(datafile, "r") as fp:
                for ip in range(0, Np, blocksize):
                    fp.seek(offset + ip * Nc * np.dtype(dtype).itemsize)
                    count = min(blocksize, Np - ip)
                    data = np.fromfile(fp, dtype=dtype, count=count * Nc)
                    data = data.reshape((count, Nc))
                    result = utils.calc_velocity_dist4d(data, **opt)
                    c_dist[...] += result[0]
                    p_dist[...] += result[1]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quicklook Script")
    parser.add_argument(
        "-p",
        "--basename",
        type=str,
        default="vdist",
        help="basename used for output image and movie files",
    )
    parser.add_argument(
        "-j",
        "--job",
        type=str,
        default="reduce",
        help="Type of job to perform",
    )
    parser.add_argument("config", nargs=1, help="configuration file for the job")
    args = parser.parse_args()

    # perform the job
    if args.job == "reduce":
        obj = DataReducer(args.config[0])
        obj.main(args.basename)
