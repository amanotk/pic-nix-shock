#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pathlib
import pickle
import sys

import h5py
import numpy as np
import pywt
import tqdm

if "PICNIX_DIR" in os.environ:
    sys.path.append(str(pathlib.Path(os.environ["PICNIX_DIR"]) / "script"))
try:
    from . import base, utils, waveplot
except ImportError:
    import base
    import utils
    import waveplot

import picnix


class MraAnalyzer(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)

    def read_parameter(self):
        # ignore
        return None

    def main(self):
        rawfile = self.get_filename(self.options["rawfile"], ".h5")
        mrafile = self.get_filename(self.options["mrafile"], ".h5")
        self.apply_mra_and_save(rawfile, mrafile)

    def apply_mra_and_save(self, rawfile, mrafile):
        vars = ["B", "E", "Je", "Ji"]

        # check if the output file already exists
        if os.path.exists(mrafile):
            print(f"Output file {mrafile} already exists. Please choose a different name.")
            return

        with h5py.File(rawfile, "r") as f:
            # create output file if it does not exist
            with h5py.File(mrafile, "w") as f_out:
                f_out.create_dataset("step", data=f["step"][()])
                f_out.create_dataset("t", data=f["t"][()])
                f_out.create_dataset("x", data=f["x"][()])
                f_out.create_dataset("y", data=f["y"][()])

            # perform Multi-Resolution Analysis (MRA) on each variable
            for v in vars:
                if v not in f:
                    print(f"Variable {v} not found in the input file.")
                    continue

                print(f"Processing variable: {v}")
                data = f[v][()]
                data = data[0 : 2 * (data.shape[0] // 2), ...]
                level = pywt.swt_max_level(data.shape[0])
                result = pywt.mra(data, wavelet="db2", axis=0, level=level)
                with h5py.File(mrafile, "a") as f_out:
                    out_shape = (level + 1,) + data.shape
                    f_out.create_dataset(v, shape=out_shape, dtype=data.dtype)
                    for l in range(level + 1):
                        f_out[v][l] = result[l]


class MraPlotter(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "plot" in self.options:
            for key in self.options["plot"]:
                self.options[key] = self.options["plot"][key]
        self.plotter = waveplot.SixPanelWavePlot(self.options)
        self.parameter = None  # initialized later

    def read_parameter(self):
        # ignore
        return None

    def main(self):
        rawfile = self.get_filename(self.options.get("rawfile"), ".h5")
        mrafile = self.get_filename(self.options.get("mrafile"), ".h5")
        output = self.options.get("output", "mra")

        # read parameters here
        with h5py.File(rawfile, "r") as fp_raw:
            self.parameter = pickle.loads(fp_raw["config"][()])["parameter"]

        wci = np.sqrt(self.parameter["sigma"]) / self.parameter["mime"]
        png = self.get_filename(output, "")

        with h5py.File(mrafile, "r") as fp_mra, h5py.File(rawfile, "r") as fp_raw:
            t = fp_raw["t"]
            step = fp_raw["step"]

            for i in tqdm.tqdm(range(1, t.shape[0] - 1)):
                # read data
                X, Y, Az, vars, labels = self.get_plot_variables(fp_mra, fp_raw, i)

                self.plot(X, Y, t[i] * wci, Az, vars, labels)
                self.save(png + "-{:08d}.png".format(step[i]))

        # convert to mp4
        fps = self.options.get("fps", 10)
        picnix.convert_to_mp4("{:s}".format(png), fps, False)

    def save(self, filename):
        self.plotter.save(filename)

    def get_plot_variables(self, fileobj_mra, fileobj_raw, index):
        sigma = self.parameter["sigma"]
        dh = self.parameter["delh"]
        u0 = self.parameter["u0"]
        B0 = np.sqrt(sigma) / np.sqrt(1 + u0**2)
        E0 = B0 * u0 / np.sqrt(1 + u0**2)

        # raw magnetic field
        B_raw = fileobj_raw["B"][index, ...]
        BB = np.sqrt(B_raw[..., 0] ** 2 + B_raw[..., 1] ** 2 + B_raw[..., 2] ** 2)
        bx = B_raw[..., 0] / BB
        by = B_raw[..., 1] / BB
        bz = B_raw[..., 2] / BB
        Az = utils.calc_vector_potential2d(B_raw, dh)

        # MRA result
        level = 2

        xx = fileobj_mra["x"][index, ...]
        yy = fileobj_mra["y"][index, ...]
        X, Y = np.meshgrid(xx, yy)

        E = fileobj_mra["E"][level, index, ...]
        B = fileobj_mra["B"][level, index, ...]
        Sx = E[..., 1] * B[..., 2] - E[..., 2] * B[..., 1]
        Sy = E[..., 2] * B[..., 0] - E[..., 0] * B[..., 2]
        Sz = E[..., 0] * B[..., 1] - E[..., 1] * B[..., 0]
        S0 = E0 * B0

        B_absolute = BB
        B_envelope = np.sqrt(B[..., 0] ** 2 + B[..., 1] ** 2 + B[..., 2] ** 2)
        S_parallel = Sx * bx + Sy * by + Sz * bz

        vars = [
            B[..., 0] / B0,
            B[..., 1] / B0,
            B[..., 2] / B0,
            B_absolute / B0,
            B_envelope / B0,
            S_parallel / S0,
        ]
        labels = [
            r"$\delta B_x / B_0$",
            r"$\delta B_y / B_0$",
            r"$\delta B_z / B_0$",
            r"$|B| / B_0$",
            r"$\delta B_{envelope} / B_0$",
            r"$\delta S_{parallel} / S_0$",
        ]

        return X, Y, Az, vars, labels

    def plot(self, X, Y, time, Az, vars, labels):
        self.plotter.plot(X, Y, time, Az, vars, labels)


def main():
    import argparse

    script_name = "mra"

    parser = argparse.ArgumentParser(description="Multi-Resolution Analysis Tool")
    parser.add_argument(
        "-j",
        "--job",
        type=str,
        default="plot",
        help="Type of job to perform (analyze, plot)",
    )
    parser.add_argument("config", nargs=1, help="configuration file")
    args = parser.parse_args()
    config = args.config[0]

    if args.job == "analyze":
        obj = MraAnalyzer(config)
        obj.main()

    if args.job == "plot":
        obj = MraPlotter(config)
        obj.main()


if __name__ == "__main__":
    main()
