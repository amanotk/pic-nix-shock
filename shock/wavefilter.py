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


def get_sampling_frequency(t, rtol=1.0e-6, atol=0.0):
    t = np.asarray(t)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("time array must be one-dimensional with at least 2 samples")

    dt = np.diff(t)
    if not np.all(np.isfinite(dt)):
        raise ValueError("time array contains non-finite values")
    if np.any(dt <= 0.0):
        raise ValueError("time array must be strictly increasing")
    if not np.allclose(dt, dt[0], rtol=rtol, atol=atol):
        raise ValueError("time array must be equally sampled")

    return 1.0 / dt[0]


def temporal_filter(x, fs, fc_low=None, fc_high=None, order=4):
    nyq = 0.5 * fs

    if fc_low is not None:
        fc_low = float(fc_low)
        if not np.isfinite(fc_low):
            raise ValueError("fc_low must be finite")
        if not (0.0 < fc_low < nyq):
            raise ValueError(
                "fc_low must satisfy 0 < fc_low < fs/2 (Nyquist frequency = {:g})".format(nyq)
            )

    if fc_high is not None:
        fc_high = float(fc_high)
        if not np.isfinite(fc_high):
            raise ValueError("fc_high must be finite")
        if not (0.0 < fc_high < nyq):
            raise ValueError(
                "fc_high must satisfy 0 < fc_high < fs/2 (Nyquist frequency = {:g})".format(nyq)
            )

    if fc_low is not None and fc_high is not None:
        if not (fc_low < fc_high):
            raise ValueError("fc_low must satisfy fc_low < fc_high")
        wn = [fc_low, fc_high]
        btype = "bandpass"
    elif fc_low is not None:
        wn = fc_low
        btype = "highpass"
    elif fc_high is not None:
        wn = fc_high
        btype = "lowpass"
    else:
        raise ValueError("At least one of fc_low or fc_high must be provided")

    b, a = signal.butter(order, wn, btype=btype, analog=False, fs=fs)
    return signal.filtfilt(b, a, x, axis=0)


class WaveFilterAnalyzer(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "analyze" in self.options:
            for key in self.options["analyze"]:
                self.options[key] = self.options["analyze"][key]

    def read_parameter(self):
        return None

    def main(self):
        rawfile = self.get_filename(self.options.get("rawfile", "wavetool"), ".h5")
        wavefile = self.get_filename(self.options.get("wavefile", "wavefilter"), ".h5")
        self.apply_filter_and_save(rawfile, wavefile)

    def apply_filter_and_save(self, rawfile, wavefile):
        overwrite = self.options.get("overwrite", False)
        fc_low = self.options.get("fc_low", None)
        fc_high = self.options.get("fc_high", None)
        order = self.options.get("order", 4)

        if os.path.exists(wavefile) and not overwrite:
            print(f"Output file {wavefile} already exists. Please choose a different name.")
            return

        with h5py.File(rawfile, "r") as fp_in:
            B = fp_in["B"][()]
            E = fp_in["E_ohm"][()]
            t = fp_in["t"][()]

            fs = get_sampling_frequency(t)

            B_hp = temporal_filter(B, fs=fs, fc_low=fc_low, fc_high=fc_high, order=order)
            E_hp = temporal_filter(E, fs=fs, fc_low=fc_low, fc_high=fc_high, order=order)

            with h5py.File(wavefile, "w") as fp_out:
                fp_out.create_dataset("step", data=fp_in["step"][()])
                fp_out.create_dataset("t", data=fp_in["t"][()])
                fp_out.create_dataset("x", data=fp_in["x"][()])
                fp_out.create_dataset("y", data=fp_in["y"][()])
                fp_out.create_dataset("B", data=B_hp)
                fp_out.create_dataset("E", data=E_hp)


class WaveFilterPlotter(base.JobExecutor):
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
        wavefile = self.get_filename(self.options.get("wavefile", "wavefilter"), ".h5")
        output = self.options.get("output", "wavefilter")

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
        Bhat_raw = B_raw / (BB[..., np.newaxis] + 1.0e-32)
        Az = utils.calc_vector_potential2d(B_raw, dh)

        xx = fileobj_wave["x"][index, ...]
        yy = fileobj_wave["y"][index, ...]
        X, Y = np.meshgrid(xx, yy)

        Bf_hp = fileobj_wave["B"][index, ...] / B0
        Ew_hp = fileobj_wave["E"][index, ...]
        Bw_hp = fileobj_wave["B"][index, ...]

        B_absolute = BB
        B_envelope = np.linalg.norm(Bf_hp, axis=-1)
        S = np.cross(Ew_hp, Bw_hp, axis=-1) / (B0**2)
        S_parallel = np.sum(S * Bhat_raw, axis=-1)

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
            r"$\delta S_{parallel} / c B_0^2$",
        ]

        return X, Y, Az, vars, labels


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Wave Filter Tool")
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
        obj = WaveFilterAnalyzer(config)
        obj.main()
        jobs = [j for j in jobs if j != "analyze"]

    for job in jobs:
        if job == "plot":
            obj = WaveFilterPlotter(config)
            wavefile = obj.get_filename(obj.options.get("wavefile", "wavefilter"), ".h5")
            if not os.path.exists(wavefile):
                sys.exit(
                    "Error: File '{}' not found. Please run 'analyze' job first.".format(wavefile)
                )
            obj.main()
        else:
            raise ValueError("Unknown job: {:s}".format(job))


if __name__ == "__main__":
    main()
