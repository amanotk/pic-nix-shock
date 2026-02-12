#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pathlib
import pickle
import sys

import h5py
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
import toml
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
        debug = self.options.get("debug", False)
        debug_count = self.options.get("debug_count", 8)
        debug_mode = self.options.get("debug_mode", "uniform")

        if os.path.exists(wavefile) and not overwrite:
            print(f"Output file {wavefile} already exists. Please choose a different name.")
            return

        with h5py.File(rawfile, "r") as fp_in:
            B = fp_in["B"][()]
            E = fp_in["E_ohm"][()]
            t = fp_in["t"][()]
            step = fp_in["step"][()]
            x = fp_in["x"][()]
            y = fp_in["y"][()]
            config = fp_in["config"][()] if "config" in fp_in else None
            nstep = step.shape[0]

            fs = get_sampling_frequency(t)

            B_hp = temporal_filter(B, fs=fs, fc_low=fc_low, fc_high=fc_high, order=order)
            E_hp = temporal_filter(E, fs=fs, fc_low=fc_low, fc_high=fc_high, order=order)

            debug_indices = select_debug_indices(nstep, debug, debug_count, debug_mode)
            step = step[debug_indices]
            t = t[debug_indices]
            B_hp = B_hp[debug_indices, ...]
            E_hp = E_hp[debug_indices, ...]

            if x.ndim >= 2 and x.shape[0] == nstep:
                x = x[debug_indices, ...]

            if y.ndim >= 2 and y.shape[0] == nstep:
                y = y[debug_indices, ...]

            with h5py.File(wavefile, "w") as fp_out:
                if config is not None:
                    fp_out.create_dataset("config", data=config, dtype=np.int8)
                fp_out.create_dataset("step", data=step)
                fp_out.create_dataset("t", data=t)
                fp_out.create_dataset("x", data=x)
                fp_out.create_dataset("y", data=y)
                fp_out.create_dataset("B", data=B_hp)
                fp_out.create_dataset("E", data=E_hp)


class WaveFilterPlotter(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "plot" in self.options:
            for key in self.options["plot"]:
                self.options[key] = self.options["plot"][key]
        self.options.setdefault("wave_layout", "diagnostics_left")
        self.parameter = None
        self.shock_position = None
        self.plotter = waveplot.SixPanelWavePlot(self.options)

    def read_parameter(self):
        return None

    def main(self):
        rawfile = self.get_filename(self.options.get("rawfile", "wavetool"), ".h5")
        wavefile = self.get_filename(self.options.get("wavefile", "wavefilter"), ".h5")
        output = self.options.get("output", "wavefilter")
        debug = self.options.get("debug", False)
        debug_count = self.options.get("debug_count", 8)
        debug_mode = self.options.get("debug_mode", "uniform")

        with h5py.File(rawfile, "r") as fp_raw:
            self.parameter = pickle.loads(fp_raw["config"][()])["parameter"]

        self.shock_position = self.options.get("shock_position", None)
        if self.shock_position is None:
            reduce1d_result = (
                pathlib.Path(self.get_dirname()).parent / "reduce1d" / "reduce1d_result.toml"
            )
            if reduce1d_result.exists():
                self.shock_position = toml.load(str(reduce1d_result)).get("shock_position", None)

        wci = np.sqrt(self.parameter["sigma"]) / self.parameter["mime"]
        png = self.get_filename(output, "")

        with h5py.File(wavefile, "r") as fp_wave, h5py.File(rawfile, "r") as fp_raw:
            t_wave = fp_wave["t"]
            step_wave = fp_wave["step"]
            step_raw = fp_raw["step"][()]
            frame_indices = np.arange(1, t_wave.shape[0] - 1, dtype=np.int64)
            debug_indices = select_debug_indices(frame_indices.size, debug, debug_count, debug_mode)
            frame_indices = frame_indices[debug_indices]

            for i_wave in tqdm.tqdm(frame_indices):
                step_value = step_wave[i_wave]
                i_raw = np.searchsorted(step_raw, step_value)
                if i_raw >= step_raw.size or step_raw[i_raw] != step_value:
                    raise ValueError("Step not found in raw file: {:d}".format(int(step_value)))

                X, Y, Az, vars, labels, xlim = self.get_plot_variables(
                    fp_wave, fp_raw, i_wave, i_raw
                )
                self.plotter.plot(X, Y, t_wave[i_wave] * wci, Az, vars, labels, xlim=xlim)
                self.plotter.save(png + "-{:08d}.png".format(step_value))

        fps = self.options.get("fps", 10)
        picnix.convert_to_mp4("{:s}".format(png), fps, False)

    def get_plot_variables(self, fileobj_wave, fileobj_raw, i_wave, i_raw):
        sigma = self.parameter["sigma"]
        dh = self.parameter["delh"]
        u0 = self.parameter["u0"]
        B0 = np.sqrt(sigma) / np.sqrt(1 + u0**2)
        E0 = B0 * u0 / np.sqrt(1 + u0**2)
        quantity = self.options.get("quantity", "wave")
        smooth_sigma = float(self.options.get("smooth_sigma", 0.5))

        xx = fileobj_wave["x"][i_wave, ...]
        yy = fileobj_wave["y"][i_wave, ...]
        X, Y = np.meshgrid(xx, yy)

        if quantity == "wave":
            B_raw = fileobj_raw["B"][i_raw, ...] / B0
            BB = np.linalg.norm(B_raw, axis=-1)
            Bhat_raw = B_raw / (BB[..., np.newaxis] + 1.0e-32)
            Az = utils.calc_vector_potential2d(B_raw, dh)

            Bf_hp = fileobj_wave["B"][i_wave, ...] / B0
            Ew_hp = fileobj_wave["E"][i_wave, ...]
            Bw_hp = fileobj_wave["B"][i_wave, ...]

            Bf_hp = spatial_smooth(Bf_hp, smooth_sigma)
            Ew_hp = spatial_smooth(Ew_hp, smooth_sigma)
            Bw_hp = spatial_smooth(Bw_hp, smooth_sigma)

            B_absolute = BB
            B_envelope = np.linalg.norm(Bf_hp, axis=-1)
            S = np.cross(Ew_hp, Bw_hp, axis=-1) / (E0 * B0)
            S_parallel = np.sum(S * Bhat_raw, axis=-1)

            vars = [
                B_absolute,
                B_envelope,
                S_parallel,
                Bf_hp[..., 0],
                Bf_hp[..., 1],
                Bf_hp[..., 2],
            ]
            labels = [
                r"$|B| / B_0$",
                r"$\delta B_{envelope} / B_0$",
                r"$\delta S_{parallel} / c E_0 B_0$",
                r"$\delta B_x / B_0$",
                r"$\delta B_y / B_0$",
                r"$\delta B_z / B_0$",
            ]
        elif quantity == "field":
            E = fileobj_wave["E"][i_wave, ...]
            B = fileobj_wave["B"][i_wave, ...]
            E = spatial_smooth(E, smooth_sigma)
            B = spatial_smooth(B, smooth_sigma)
            B_raw = fileobj_raw["B"][i_raw, ...] / B0
            Az = utils.calc_vector_potential2d(B_raw, dh)
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
        else:
            raise ValueError("Unknown quantity: {:s}. Use 'wave' or 'field'.".format(str(quantity)))

        xlim = self.get_x_window(xx, yy, fileobj_wave["t"][i_wave])
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


def spatial_smooth(x, smooth_sigma):
    smooth_sigma = float(smooth_sigma)
    if smooth_sigma < 0.0:
        raise ValueError("smooth_sigma must be non-negative")
    if smooth_sigma == 0.0:
        return x

    x = ndimage.gaussian_filter1d(x, sigma=smooth_sigma, axis=0, mode="wrap")
    x = ndimage.gaussian_filter1d(x, sigma=smooth_sigma, axis=1, mode="nearest")
    return x


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
    parser.add_argument("config", nargs=1, help="configuration file")
    args = parser.parse_args()
    config = args.config[0]
    debug = args.debug
    debug_count = args.debug_count
    debug_mode = args.debug_mode

    def apply_runtime_options(obj):
        obj.options["debug"] = debug
        obj.options["debug_count"] = debug_count
        obj.options["debug_mode"] = debug_mode

    jobs = args.job.split(",")

    if "analyze" in jobs:
        obj = WaveFilterAnalyzer(config)
        apply_runtime_options(obj)
        obj.main()
        jobs = [j for j in jobs if j != "analyze"]

    for job in jobs:
        if job == "plot":
            obj = WaveFilterPlotter(config)
            apply_runtime_options(obj)
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
