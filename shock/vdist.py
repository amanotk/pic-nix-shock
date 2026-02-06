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
import concurrent.futures
from mpi4py import MPI
from mpi4py.futures import MPICommExecutor

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
    from . import base
    from . import utils
except ImportError:
    import base
    import utils


class DataReducer(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "reduce" in self.options:
            for key in self.options["reduce"]:
                self.options[key] = self.options["reduce"][key]
        self.parameter = self.read_parameter()
        # for MPI
        self.is_root = MPI.COMM_WORLD.Get_rank() == 0

    def main(self, basename):
        self.worker_params = {}
        self.save(self.get_filename(basename, ".h5"))

    def average1d(self, x, size):
        x = ndimage.uniform_filter(x, size=size, axes=(0,), mode="wrap")
        return x[size // 2 :: size]

    def average2d(self, x, size):
        x = ndimage.uniform_filter(x, size=size, axes=(0, 1), mode="wrap")
        return x[size // 2 :: size, size // 2 :: size]

    def encode(self, data):
        return np.frombuffer(pickle.dumps(data), np.int8)

    def message(self, msg):
        if self.is_root:
            print(msg)
            sys.stdout.flush()

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
        shock_position = self.options.get("shock_position", [1.66365906e-02, -1.39911575e02])

        # binning parameters
        upara_nbins = self.options.get("upara_nbins", 80)
        upara_range = self.options.get("upara_range", [-1.2, +1.2])
        uperp_nbins = self.options.get("uperp_nbins", 40)
        uperp_range = self.options.get("uperp_range", [0.0, +1.2])
        uabs_nbins = self.options.get("uabs_nbins", 80)
        uabs_range = self.options.get("uabs_range", [1.0e-3, 1.0e1])
        ucos_nbins = self.options.get("ucos_nbins", 25)
        ucos_range = [-1.0, 1.0]

        method = self.options.get("method", "async")
        run = picnix.Run(profile, config=config, method=method)
        config = self.encode(run.config)

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

        self.worker_params["config_encoded"] = config
        self.worker_params["Nt"] = Nt
        self.worker_params["Mx"] = Mx
        self.worker_params["My"] = My
        self.worker_params["uperp_nbins"] = uperp_nbins
        self.worker_params["upara_nbins"] = upara_nbins
        self.worker_params["ucos_nbins"] = ucos_nbins
        self.worker_params["uabs_nbins"] = uabs_nbins
        self.worker_params["upara_bins"] = upara_bins
        self.worker_params["uperp_bins"] = uperp_bins
        self.worker_params["uabs_bins"] = uabs_bins
        self.worker_params["ucos_bins"] = ucos_bins
        self.worker_params["name"] = "up00"
        self.worker_params["bb"] = bb
        self.worker_params["vb"] = vb
        self.worker_params["c_dist"] = c_dist
        self.worker_params["p_dist"] = p_dist
        self.worker_params["x_bins"] = x_bins
        self.worker_params["y_bins"] = y_bins
        self.worker_params["blocksize"] = 2**20

        # create HDF5 file and datasets first
        self.create_hdf5_file(filename, overwrite)
        MPI.COMM_WORLD.Barrier()

        # read step
        with h5py.File(filename, "r") as fp:
            step_in_file = fp["step"][()]

        # process data for each step
        for index, step_index in enumerate(index_range):
            # skip if the step is already stored
            if step_in_file[index] == particle_step[step_index]:
                continue

            self.message(f"Processing step : {step_index:8d}")

            # clear first
            bb[...] = 0.0
            vb[...] = 0.0
            c_dist[...] = 0.0
            p_dist[...] = 0.0

            # process field data
            step = particle_step[step_index]
            time = run.get_time_at(prefix, step)
            xmin = np.polyval(shock_position, time) + x_offset
            ymin = 0.0
            x_bins_new = x_bins.copy() + xmin
            y_bins_new = y_bins.copy() + ymin
            self.process_field(run, step, xc, x_bins_new, num_average, bb, vb)

            # process electrons
            self.worker_params["x_bins"] = x_bins_new
            self.worker_params["y_bins"] = y_bins_new
            jsonfiles = run.diag_handlers[prefix].find_json_at_step(step)
            self.process_particle(jsonfiles)

            # store data
            self.write_hdf5_data(filename, index, step, time)
            MPI.COMM_WORLD.Barrier()

    def create_hdf5_file(self, filename, overwrite):
        if not self.is_root:
            return

        if not os.path.exists(filename) or overwrite is True:
            config = self.worker_params["config_encoded"]
            Nt = self.worker_params["Nt"]
            Mx = self.worker_params["Mx"]
            My = self.worker_params["My"]
            uperp_nbins = self.worker_params["uperp_nbins"]
            upara_nbins = self.worker_params["upara_nbins"]
            ucos_nbins = self.worker_params["ucos_nbins"]
            uabs_nbins = self.worker_params["uabs_nbins"]
            upara_bins = self.worker_params["upara_bins"]
            uperp_bins = self.worker_params["uperp_bins"]
            uabs_bins = self.worker_params["uabs_bins"]
            ucos_bins = self.worker_params["ucos_bins"]

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
                fp.create_dataset("p_dist", (Nt, ucos_nbins, uabs_nbins, My, Mx), dtype=np.float64)
                # bins
                fp.create_dataset("x_bins", (Nt, Mx + 1), dtype=np.float64)
                fp.create_dataset("y_bins", (Nt, My + 1), dtype=np.float64)
                fp.create_dataset("upara_bins", data=upara_bins, dtype=np.float64)
                fp.create_dataset("uperp_bins", data=uperp_bins, dtype=np.float64)
                fp.create_dataset("uabs_bins", data=uabs_bins, dtype=np.float64)
                fp.create_dataset("ucos_bins", data=ucos_bins, dtype=np.float64)

    def write_hdf5_data(self, filename, index, step, time):
        if not self.is_root:
            return

        x_bins = self.worker_params["x_bins"]
        y_bins = self.worker_params["y_bins"]

        with h5py.File(filename, "a") as fp:
            fp["step"][index] = step
            fp["t"][index] = time
            fp["x"][index, ...] = 0.5 * (x_bins[:-1] + x_bins[+1:])
            fp["y"][index, ...] = 0.5 * (y_bins[:-1] + y_bins[+1:])
            fp["bb"][index, ...] = self.worker_params["bb"]
            fp["vb"][index, ...] = self.worker_params["vb"]
            fp["c_dist"][index, ...] = self.worker_params["c_dist"]
            fp["p_dist"][index, ...] = self.worker_params["p_dist"]
            fp["x_bins"][index, ...] = x_bins
            fp["y_bins"][index, ...] = y_bins

    def process_field(self, run, step, xc, x_bins, num_average, bb, vb):
        if self.is_root:
            xnew = 0.5 * (x_bins[:-1] + x_bins[+1:])
            xind = xc.searchsorted(xnew)
            delta = (xc[xind] - xnew) / (xc[xind] - xc[xind - 1])[np.newaxis, :]

            # magnetic field direction
            data = run.read_at("field", step, "uf")
            uf_raw = data["uf"].mean(axis=0)
            uf_avg = self.average2d(uf_raw, num_average)
            Bx = delta * uf_avg[..., xind - 1, 3] + (1 - delta) * uf_avg[..., xind, 3]
            By = delta * uf_avg[..., xind - 1, 4] + (1 - delta) * uf_avg[..., xind, 4]
            Bz = delta * uf_avg[..., xind - 1, 5] + (1 - delta) * uf_avg[..., xind, 5]
            BB = np.sqrt(Bx**2 + By**2 + Bz**2)
            bb[..., 0] = Bx / BB
            bb[..., 1] = By / BB
            bb[..., 2] = Bz / BB
            # try to free memory explicitly
            del uf_raw, uf_avg, data
            run.clear_cache()

            # bulk velocity
            data = run.read_at("field", step, "um")
            ue_raw = data["um"][..., 0, 0:4].mean(axis=0)
            ue_avg = self.average2d(ue_raw, num_average)
            ro = delta * ue_avg[..., xind - 1, 0] + (1 - delta) * ue_avg[..., xind, 0]
            jx = delta * ue_avg[..., xind - 1, 1] + (1 - delta) * ue_avg[..., xind, 1]
            jy = delta * ue_avg[..., xind - 1, 2] + (1 - delta) * ue_avg[..., xind, 2]
            jz = delta * ue_avg[..., xind - 1, 3] + (1 - delta) * ue_avg[..., xind, 3]
            vb[..., 0] = jx / ro
            vb[..., 1] = jy / ro
            vb[..., 2] = jz / ro
            # try to free memory explicitly
            del ue_raw, ue_avg, data
            run.clear_cache()

        # broadcast results
        MPI.COMM_WORLD.Bcast([bb, MPI.DOUBLE], root=0)
        MPI.COMM_WORLD.Bcast([vb, MPI.DOUBLE], root=0)

    def process_particle(self, jsonfiles):
        global mpi_vars

        # global variables for MPI workers
        mpi_vars_keys = [
            "bb",
            "vb",
            "c_dist",
            "p_dist",
            "x_bins",
            "y_bins",
            "upara_bins",
            "uperp_bins",
            "uabs_bins",
            "ucos_bins",
        ]
        mpi_vars = {key: value for key, value in self.worker_params.items() if key in mpi_vars_keys}

        executor_args = dict()
        with MPICommExecutor(**executor_args) as executor:
            if executor is None:
                raise RuntimeError("MPICommExecutor is not available!")

            # submit tasks
            future_list = []
            name = self.worker_params.get("name")
            blocksize = self.worker_params.get("blocksize")
            for f in jsonfiles:
                dataset, meta = picnix.read_jsonfile(f)
                byteorder, layout, datafile = picnix.process_meta(meta)
                offset, dtype, shape = picnix.get_dataset_info(dataset[name], byteorder)
                Np = shape[0]  # number of particle
                Nc = shape[1]  # number of components

                for ip in range(0, Np, blocksize):
                    index_range = (ip, min(ip + blocksize, Np))
                    block = {
                        "offset": offset + ip * Nc * np.dtype(dtype).itemsize,
                        "dtype": dtype,
                        "count": (index_range[1] - index_range[0]) * Nc,
                        "shape": (index_range[1] - index_range[0], Nc),
                    }
                    # submit
                    future = executor.submit(self.mpi_work_dist, datafile, block)
                    future_list.append(future)

            # wait for tasks to complete
            concurrent.futures.wait(future_list)

            # reduce results
            comm_size = MPI.COMM_WORLD.Get_size()
            for rank in range(comm_size - 1):
                executor.submit(self.mpi_reduce_dist)
            MPI.COMM_WORLD.Reduce(None, mpi_vars["c_dist"], op=MPI.SUM, root=0)
            MPI.COMM_WORLD.Reduce(None, mpi_vars["p_dist"], op=MPI.SUM, root=0)

    @staticmethod
    def mpi_work_dist(datafile, block):
        global mpi_vars

        offset = block["offset"]
        dtype = block["dtype"]
        count = block["count"]
        shape = block["shape"]

        with open(datafile, "r") as fp:
            fp.seek(offset)
            data = np.fromfile(fp, dtype=dtype, count=count).reshape(shape)
            result = utils.calc_velocity_dist4d(data, **mpi_vars)

        # accumulate results in global variables
        mpi_vars["c_dist"][...] += result[0]
        mpi_vars["p_dist"][...] += result[1]

    @staticmethod
    def mpi_reduce_dist():
        global mpi_vars

        MPI.COMM_WORLD.Reduce(mpi_vars["c_dist"], None, op=MPI.SUM, root=0)
        MPI.COMM_WORLD.Reduce(mpi_vars["p_dist"], None, op=MPI.SUM, root=0)


def main():
    import argparse

    script_name = "vdist"

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
        default="reduce",
        help="Type of job to perform",
    )
    parser.add_argument("config", nargs=1, help="configuration file for the job")
    args = parser.parse_args()
    config = args.config[0]
    output = args.output

    # perform the job
    if args.job == "reduce":
        obj = DataReducer(config)
        obj.main(output)


if __name__ == "__main__":
    main()
