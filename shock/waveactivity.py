#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pathlib
import pickle
import sys

import h5py
import numpy as np
import scipy.signal as signal
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


def highpass(x, fs, fc, order=4):
    nyq = 0.5 * fs
    wn = fc / nyq
    if not (0.0 < wn < 1.0):
        raise ValueError("highpass cutoff must satisfy 0 < fc < fs/2")
    b, a = signal.butter(order, wn, btype="high", analog=False)
    return signal.filtfilt(b, a, x, axis=0)


class WaveActivityAnalyzer(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "analyze" in self.options:
            for key in self.options["analyze"]:
                self.options[key] = self.options["analyze"][key]

    def read_parameter(self):
        return None

    def main(self):
        rawfile = self.get_filename(self.options.get("rawfile", "wavetool"), ".h5")
        wavefile = self.get_filename(self.options.get("wavefile", "waveactivity"), ".h5")
        self.apply_filter_and_save(rawfile, wavefile)

    def apply_filter_and_save(self, rawfile, wavefile):
        overwrite = self.options.get("overwrite", False)
        fs = self.options.get("fs", 4.0)
        fc = self.options.get("fc", 0.5)
        order = self.options.get("order", 4)

        if os.path.exists(wavefile) and not overwrite:
            print(f"Output file {wavefile} already exists. Please choose a different name.")
            return

        with h5py.File(rawfile, "r") as fp_in:
            B = fp_in["B"][()]
            E = fp_in["E"][()]
            Je = fp_in["Je"][()]
            Ji = fp_in["Ji"][()]

            BB = np.linalg.norm(B, axis=-1)
            BB = np.where(BB > 0.0, BB, 1.0)
            B_hat = B / BB[..., np.newaxis]

            Re = np.where(np.abs(Je[..., 0:1]) > 0.0, Je[..., 0:1], 1.0)
            Ri = np.where(np.abs(Ji[..., 0:1]) > 0.0, Ji[..., 0:1], 1.0)
            Ve = Je[..., 1:4] / Re
            Vi = Ji[..., 1:4] / Ri
            Ve_para = np.sum(Ve * B_hat, axis=-1)
            Vi_para = np.sum(Vi * B_hat, axis=-1)
            Ve = Ve - Ve_para[..., np.newaxis] * B_hat
            Vi = Vi - Vi_para[..., np.newaxis] * B_hat

            B_hp = highpass(B, fs=fs, fc=fc, order=order)
            E_hp = highpass(E, fs=fs, fc=fc, order=order)
            Ve_hp = highpass(Ve, fs=fs, fc=fc, order=order)
            Vi_hp = highpass(Vi, fs=fs, fc=fc, order=order)

            with h5py.File(wavefile, "w") as fp_out:
                fp_out.create_dataset("step", data=fp_in["step"][()])
                fp_out.create_dataset("t", data=fp_in["t"][()])
                fp_out.create_dataset("x", data=fp_in["x"][()])
                fp_out.create_dataset("y", data=fp_in["y"][()])
                fp_out.create_dataset("B", data=B_hp)
                fp_out.create_dataset("E", data=E_hp)
                fp_out.create_dataset("Ve", data=Ve_hp)
                fp_out.create_dataset("Vi", data=Vi_hp)


class WaveActivityPlotter(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "plot" in self.options:
            for key in self.options["plot"]:
                self.options[key] = self.options["plot"][key]
        self.parameter = None
        self.plotter = waveplot.SixPanelWavePlot(self.options)

    def read_parameter(self):
        return None

    def main(self):
        rawfile = self.get_filename(self.options.get("rawfile", "wavetool"), ".h5")
        wavefile = self.get_filename(self.options.get("wavefile", "waveactivity"), ".h5")
        output = self.options.get("output", "waveactivity")

        with h5py.File(rawfile, "r") as fp_raw:
            self.parameter = pickle.loads(fp_raw["config"][()])["parameter"]

        wci = np.sqrt(self.parameter["sigma"]) / self.parameter["mime"]
        png = self.get_filename(output, "")

        with h5py.File(wavefile, "r") as fp_wave, h5py.File(rawfile, "r") as fp_raw:
            t = fp_raw["t"]
            step = fp_raw["step"]

            for i in tqdm.tqdm(range(1, t.shape[0] - 1)):
                X, Y, Az, vars, labels = self.get_plot_variables(fp_wave, fp_raw, i)
                self.plotter.plot(X, Y, t[i] * wci, Az, vars, labels)
                self.plotter.save(png + "-{:08d}.png".format(step[i]))

        fps = self.options.get("fps", 10)
        picnix.convert_to_mp4("{:s}".format(png), fps, False)

    def get_plot_variables(self, fileobj_wave, fileobj_raw, index):
        sigma = self.parameter["sigma"]
        dh = self.parameter["delh"]
        u0 = self.parameter["u0"]
        B0 = np.sqrt(sigma) / np.sqrt(1 + u0**2)

        B_raw = fileobj_raw["B"][index, ...] / B0
        BB = np.linalg.norm(B_raw, axis=-1)
        Az = utils.calc_vector_potential2d(B_raw, dh)

        xx = fileobj_wave["x"][index, ...]
        yy = fileobj_wave["y"][index, ...]
        X, Y = np.meshgrid(xx, yy)

        Bf_hp = fileobj_wave["B"][index, ...] / B0
        Ve_hp = fileobj_wave["Ve"][index, ...]

        B_absolute = BB
        B_envelope = np.linalg.norm(Bf_hp, axis=-1)
        S_parallel = -np.sum(Bf_hp * Ve_hp, axis=-1) * BB

        vars = [
            Bf_hp[..., 0],
            Bf_hp[..., 1],
            Bf_hp[..., 2],
            B_absolute,
            B_envelope,
            S_parallel,
        ]
        labels = [
            r"$\delta B_x / B_0$",
            r"$\delta B_y / B_0$",
            r"$\delta B_z / B_0$",
            r"$|B| / B_0$",
            r"$\delta B_{envelope} / B_0$",
            r"$\delta S_{parallel} / c B_0^2/4\pi$",
        ]

        return X, Y, Az, vars, labels


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Wave Activity Analysis Tool")
    parser.add_argument(
        "-j",
        "--job",
        type=str,
        default="analyze",
        help="Type of job to perform (analyze, plot). Can be combined with comma.",
    )
    parser.add_argument("config", nargs=1, help="configuration file")
    args = parser.parse_args()
    config = args.config[0]

    jobs = args.job.split(",")

    if "analyze" in jobs:
        obj = WaveActivityAnalyzer(config)
        obj.main()
        jobs = [j for j in jobs if j != "analyze"]

    if len(jobs) > 0:
        obj = WaveActivityAnalyzer(config)
        wavefile = obj.get_filename(obj.options.get("wavefile", "waveactivity"), ".h5")
        if not os.path.exists(wavefile):
            sys.exit("Error: File '{}' not found. Please run 'analyze' job first.".format(wavefile))

    for job in jobs:
        if job == "plot":
            obj = WaveActivityPlotter(config)
            obj.main()
        else:
            raise ValueError("Unknown job: {:s}".format(job))


if __name__ == "__main__":
    main()
