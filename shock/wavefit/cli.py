#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pathlib
import pickle
import sys
from typing import Union

import h5py
import numpy as np
import scipy.ndimage as ndimage
import tqdm

if "PICNIX_DIR" in os.environ:
    sys.path.append(str(pathlib.Path(os.environ["PICNIX_DIR"]) / "script"))
try:
    from .. import base
except ImportError:
    import base

from .fit import fit_one_candidate
from .model import periodic_delta
from .plot import save_envelope_map_plot, save_quickcheck_plot_12panel


RESULT_FLOAT_KEYS = [
    "x0",
    "y0",
    "kx",
    "kx_err",
    "ky",
    "ky_err",
    "Ew",
    "Ew_err",
    "Bw",
    "Bw_err",
    "phiE",
    "phiE_err",
    "phiB",
    "phiB_err",
    "redchi",
    "nrmse",
    "nrmse_balanced",
    "nrmseE",
    "nrmseB",
    "nrmse_raw",
    "k",
    "wavelength",
    "wavelength_over_sigma",
    "r2E",
    "r2B",
    "patch_xmin",
    "patch_xmax",
    "patch_ymin",
    "patch_ymax",
    "Bx",
    "By",
    "Bz",
    "vex",
    "vey",
    "vez",
    "vix",
    "viy",
    "viz",
]
RESULT_INT_KEYS = ["ix", "iy", "nfev"]
RESULT_BOOL_KEYS = ["success", "is_good", "is_good_nrmse", "is_good_scale", "has_errorbars"]
RESULT_STR_KEYS = ["reason", "message"]
LARGE_ARRAY_KEYS = [
    "windowed_data_E",
    "windowed_data_B",
    "windowed_model_E",
    "windowed_model_B",
    "patch_x",
    "patch_y",
]


def decode_embedded_config(config_dataset_value):
    data = np.asarray(config_dataset_value)
    payload = data.astype(np.uint8).tobytes()
    return pickle.loads(payload)


def extract_parameter_from_wavefile(fileobj):
    if "config" not in fileobj:
        return None
    try:
        config_obj = decode_embedded_config(fileobj["config"][()])
    except Exception:
        return None

    if isinstance(config_obj, dict):
        if isinstance(config_obj.get("parameter", None), dict):
            return config_obj["parameter"]
        configuration = config_obj.get("configuration", None)
        if isinstance(configuration, dict) and isinstance(
            configuration.get("parameter", None), dict
        ):
            return configuration["parameter"]
    return None


def get_h5_dataset(fileobj, key):
    # type: (Union[h5py.File, h5py.Group], str) -> h5py.Dataset
    obj = fileobj[key]
    if not isinstance(obj, h5py.Dataset):
        raise TypeError("{} must be a dataset".format(key))
    return obj


def get_h5_group(fileobj, key):
    # type: (Union[h5py.File, h5py.Group], str) -> h5py.Group
    obj = fileobj[key]
    if not isinstance(obj, h5py.Group):
        raise TypeError("{} must be a group".format(key))
    return obj


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


def pick_candidate_points(xx, yy, envelope, sigma, options):
    smooth_sigma = float(options.get("envelope_smooth_sigma", 0.5))
    max_candidates_opt = options.get("max_candidates", None)
    if max_candidates_opt is None:
        max_candidates = None
    else:
        max_candidates = int(max_candidates_opt)
        if max_candidates <= 0:
            max_candidates = None
    threshold = float(options.get("envelope_threshold", 0.10))
    candidate_distance = float(options.get("candidate_distance", sigma))

    env = np.array(envelope, copy=True)
    if smooth_sigma > 0.0:
        env = ndimage.gaussian_filter1d(env, sigma=smooth_sigma, axis=0, mode="wrap")
        env = ndimage.gaussian_filter1d(env, sigma=smooth_sigma, axis=1, mode="nearest")

    ydy = np.median(np.diff(yy))
    sy = 1
    sx = 1
    env_max = ndimage.maximum_filter1d(env, size=2 * sy + 1, axis=0, mode="wrap")
    env_max = ndimage.maximum_filter1d(env_max, size=2 * sx + 1, axis=1, mode="nearest")
    localmax = env == env_max
    mask = localmax & (env >= threshold)

    cand_iy, cand_ix = np.where(mask)
    if cand_ix.size == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64), env

    amp = env[cand_iy, cand_ix]
    order = np.argsort(amp)[::-1]

    Ly = (yy[-1] - yy[0]) + ydy
    min_distance = candidate_distance
    selected_ix = []
    selected_iy = []

    for idx in order:
        ix = int(cand_ix[idx])
        iy = int(cand_iy[idx])
        x0 = xx[ix]
        y0 = yy[iy]

        keep = True
        for jx, jy in zip(selected_ix, selected_iy):
            dx = xx[jx] - x0
            dy = periodic_delta(yy[jy], y0, Ly)
            if np.sqrt(dx**2 + dy**2) < min_distance:
                keep = False
                break

        if keep:
            selected_ix.append(ix)
            selected_iy.append(iy)
        if max_candidates is not None and len(selected_ix) >= max_candidates:
            break

    return np.array(selected_ix, dtype=np.int64), np.array(selected_iy, dtype=np.int64), env


class WaveFitAnalyzer(base.JobExecutor):
    def __init__(self, config_file, option_section="analyze"):
        super().__init__(config_file)
        self._merge_option_sections(option_section)

    def _merge_option_sections(self, option_section):
        sections = []
        if option_section == "plot" and "analyze" in self.options:
            sections.append("analyze")
        if option_section in self.options:
            sections.append(option_section)

        for section in sections:
            for key in self.options[section]:
                self.options[key] = self.options[section][key]

    def read_parameter(self):
        try:
            return super().read_parameter()
        except (FileNotFoundError, KeyError):
            return None

    def get_reference_b0(self, parameter=None):
        source = parameter if isinstance(parameter, dict) else self.parameter
        if not isinstance(source, dict):
            raise ValueError("parameter is required for B0 normalization")
        if "sigma" not in source or "u0" not in source:
            raise ValueError("parameter must contain sigma and u0 for B0 normalization")

        sigma = float(source["sigma"])
        u0 = float(source["u0"])
        b0 = np.sqrt(sigma) / np.sqrt(1.0 + u0**2)
        if b0 <= 0.0 or not np.isfinite(b0):
            raise ValueError("computed B0 is invalid")
        return float(b0)

    def get_wci(self, parameter=None):
        source = parameter if isinstance(parameter, dict) else self.parameter
        if not isinstance(source, dict):
            return np.nan
        if "sigma" not in source or "mime" not in source:
            return np.nan

        sigma0 = float(source["sigma"])
        mime = float(source["mime"])
        if mime == 0.0:
            return np.nan
        return float(np.sqrt(sigma0) / mime)

    def select_snapshot_indices(self, nstep):
        snapshot_indices_opt = self.options.get("snapshot_indices", [])
        if len(snapshot_indices_opt) > 0:
            indices = []
            for index in snapshot_indices_opt:
                snapshot_index = int(index)
                if snapshot_index < 0 or snapshot_index >= nstep:
                    raise ValueError(
                        "snapshot index out of range: {} (nstep={})".format(snapshot_index, nstep)
                    )
                indices.append(snapshot_index)
            snapshot_indices = np.array(sorted(set(indices)), dtype=np.int64)
            if snapshot_indices.size == 0:
                raise ValueError("No snapshots selected. Adjust snapshot selection options.")
            return snapshot_indices

        debug = bool(self.options.get("debug", False))
        debug_count = int(self.options.get("debug_count", 8))
        debug_mode = str(self.options.get("debug_mode", "uniform"))
        debug_indices = self.options.get("debug_indices", [])

        if debug and len(debug_indices) > 0:
            indices = []
            for index in debug_indices:
                snapshot_index = int(index)
                if snapshot_index < 0 or snapshot_index >= nstep:
                    raise ValueError(
                        "debug snapshot index out of range: {} (nstep={})".format(
                            snapshot_index, nstep
                        )
                    )
                indices.append(snapshot_index)
            snapshot_indices = np.array(sorted(set(indices)), dtype=np.int64)
        else:
            snapshot_indices = select_debug_indices(nstep, debug, debug_count, debug_mode)

        if snapshot_indices.size == 0:
            raise ValueError("No snapshots selected. Adjust debug settings.")
        return snapshot_indices

    def fit_single_snapshot(self, E, B, xx, yy, b0, B_background=None, J_background=None):
        fit_options = dict(self.options)
        if bool(self.options.get("debug", False)):
            fit_options.setdefault("max_candidates", 32)
        else:
            fit_options.pop("max_candidates", None)

        sigma = float(self.options.get("sigma", 3.0))
        envelope = np.linalg.norm(B, axis=-1) / b0
        cand_ix, cand_iy, env_used = pick_candidate_points(xx, yy, envelope, sigma, fit_options)

        fit_results = []
        for ix, iy in zip(cand_ix, cand_iy):
            x0 = float(xx[ix])
            y0 = float(yy[iy])
            fit_result = fit_one_candidate(
                E,
                B,
                xx,
                yy,
                x0,
                y0,
                sigma,
                fit_options,
                B_background=B_background,
                J_background=J_background,
            )
            fit_result["ix"] = int(ix)
            fit_result["iy"] = int(iy)
            fit_results.append(fit_result)

        return {
            "candidate_ix": cand_ix,
            "candidate_iy": cand_iy,
            "envelope": env_used,
            "fits": fit_results,
        }

    def write_snapshot_result(self, fp, step, time, result):
        grp_name = "snapshots/{:08d}".format(int(step))
        if grp_name in fp:
            del fp[grp_name]

        grp = fp.create_group(grp_name)
        grp.attrs["step"] = int(step)
        grp.attrs["t"] = float(time)

        fits = result["fits"]
        nfit = len(fits)
        grp.create_dataset("candidate_ix", data=result["candidate_ix"])
        grp.create_dataset("candidate_iy", data=result["candidate_iy"])

        for key in RESULT_FLOAT_KEYS:
            arr = np.full((nfit,), np.nan, dtype=np.float64)
            for i, item in enumerate(fits):
                if key in item:
                    arr[i] = item[key]
            grp.create_dataset(key, data=arr)

        for key in RESULT_INT_KEYS:
            arr = np.full((nfit,), -1, dtype=np.int64)
            for i, item in enumerate(fits):
                if key in item:
                    arr[i] = item[key]
            grp.create_dataset(key, data=arr)

        for key in RESULT_BOOL_KEYS:
            arr = np.zeros((nfit,), dtype=np.int8)
            for i, item in enumerate(fits):
                if key in item:
                    arr[i] = int(bool(item[key]))
            grp.create_dataset(key, data=arr)

        dt = h5py.string_dtype(encoding="utf-8")
        for key in RESULT_STR_KEYS:
            arr = np.array([str(item.get(key, "")) for item in fits], dtype=dt)
            grp.create_dataset(key, data=arr)

        for key in self.options:
            value = self.options[key]
            if isinstance(value, (int, float, str, bool)):
                grp.attrs["option_{}".format(key)] = value

    def generate_diagnostics(self, result, xx, yy, step, time=None, wci=None):
        if not bool(self.options.get("debug_plot", True)):
            return
        fit_results = result["fits"]

        debug_plot_count = int(self.options.get("debug_plot_count", 8))
        if debug_plot_count < 0:
            raise ValueError("debug_plot_count must be a non-negative integer")
        output_prefix = str(self.options.get("debug_plot_prefix", "wavefit-debug"))
        outdir = pathlib.Path(self.get_filename(output_prefix, "")).parent
        outdir.mkdir(parents=True, exist_ok=True)

        good_points = [item for item in fit_results if bool(item.get("is_good", False))]
        good_x = [float(xx[int(item["ix"])]) for item in good_points if "ix" in item]
        good_y = [float(yy[int(item["iy"])]) for item in good_points if "iy" in item]
        envelope_map_file = outdir / ("{:s}-envelope-{:08d}.png".format(output_prefix, int(step)))
        save_envelope_map_plot(
            str(envelope_map_file),
            result["envelope"],
            xx,
            yy,
            good_x,
            good_y,
            step,
            time=time,
            wci=wci,
            threshold=self.options.get("envelope_threshold", np.nan),
        )

        if len(fit_results) == 0:
            return

        rank = np.argsort(
            [item.get("nrmse_balanced", item.get("nrmse", np.inf)) for item in fit_results]
        )
        for i in rank[:debug_plot_count]:
            item = fit_results[int(i)]
            if "windowed_data_E" not in item:
                continue
            filename = outdir / ("{:s}-{:08d}-{:03d}.png".format(output_prefix, int(step), int(i)))
            kx = float(item.get("kx", np.nan))
            ky = float(item.get("ky", np.nan))
            k_mag = np.sqrt(kx**2 + ky**2)
            wavelength = (2.0 * np.pi / k_mag) if np.isfinite(k_mag) and k_mag > 0.0 else np.nan
            theta_deg = (
                np.degrees(np.arctan2(ky, kx)) if np.isfinite(kx) and np.isfinite(ky) else np.nan
            )
            title = (
                "step={:08d} cand={:03d} ix={} iy={} nrmse_bal={:.3f} "
                "(E={:.3f}, B={:.3f}) redchi={:.3e} lambda/sigma={:.3f} "
                "lambda={:.2f} theta={:+.1f}deg good=({:d},{:d})"
            ).format(
                int(step),
                int(i),
                int(item.get("ix", -1)),
                int(item.get("iy", -1)),
                float(item.get("nrmse_balanced", item.get("nrmse", np.nan))),
                float(item.get("nrmseE", np.nan)),
                float(item.get("nrmseB", np.nan)),
                float(item.get("redchi", np.nan)),
                float(item.get("wavelength_over_sigma", np.nan)),
                wavelength,
                theta_deg,
                int(bool(item.get("is_good_nrmse", False))),
                int(bool(item.get("is_good_scale", False))),
            )
            save_quickcheck_plot_12panel(str(filename), item, title=title, rms_normalize=True)

    def cleanup_large_arrays(self, result):
        for item in result["fits"]:
            for key in LARGE_ARRAY_KEYS:
                if key in item:
                    del item[key]

    def main(self, basename=None):
        wavefile = self.get_filename(self.options.get("wavefile", "wavefilter"), ".h5")
        fitfile = self.get_filename(self.options.get("fitfile", "wavefit"), ".h5")
        debug = bool(self.options.get("debug", False))
        overwrite = bool(self.options.get("overwrite", False))
        if os.path.exists(fitfile) and overwrite:
            os.remove(fitfile)
        if os.path.exists(fitfile) and not overwrite:
            print("Output file {} already exists. Set overwrite=true to replace.".format(fitfile))
            return

        fp_raw = None
        try:
            with h5py.File(wavefile, "r") as fp_wave, h5py.File(fitfile, "w") as fp_fit:
                parameter = extract_parameter_from_wavefile(fp_wave)
                if parameter is None:
                    parameter = self.parameter
                b0 = self.get_reference_b0(parameter)
                wci = self.get_wci(parameter)

                rawfile_opt = self.options.get("rawfile", "wavetool")
                rawfile = self.get_filename(rawfile_opt, ".h5") if rawfile_opt else None
                step_to_raw_index = {}
                if rawfile is not None and os.path.exists(rawfile):
                    fp_raw = h5py.File(rawfile, "r")
                    if "step" in fp_raw:
                        raw_steps = get_h5_dataset(fp_raw, "step")[...]
                        step_to_raw_index = {int(s): i for i, s in enumerate(raw_steps)}

                wave_step_ds = get_h5_dataset(fp_wave, "step")
                wave_t_ds = get_h5_dataset(fp_wave, "t")
                wave_x_ds = get_h5_dataset(fp_wave, "x")
                wave_y_ds = get_h5_dataset(fp_wave, "y")
                wave_E_ds = get_h5_dataset(fp_wave, "E")
                wave_B_ds = get_h5_dataset(fp_wave, "B")

                nstep = int(wave_step_ds.shape[0])
                snapshot_indices = self.select_snapshot_indices(nstep)

                for snapshot_index in tqdm.tqdm(snapshot_indices):
                    step = int(wave_step_ds[snapshot_index])
                    time = float(wave_t_ds[snapshot_index])

                    x = wave_x_ds[snapshot_index, ...] if wave_x_ds.ndim == 2 else wave_x_ds[()]
                    y = wave_y_ds[snapshot_index, ...] if wave_y_ds.ndim == 2 else wave_y_ds[()]
                    E = wave_E_ds[snapshot_index, ...]
                    B = wave_B_ds[snapshot_index, ...]

                    B_background = None
                    J_background = None
                    if fp_raw is not None:
                        raw_index = step_to_raw_index.get(step, None)
                        if raw_index is not None:
                            if "B" in fp_raw:
                                B_background = get_h5_dataset(fp_raw, "B")[raw_index, ...]
                            if "J" in fp_raw:
                                J_background = get_h5_dataset(fp_raw, "J")[raw_index, ...]
                            elif "Je" in fp_raw and "Ji" in fp_raw:
                                Je = get_h5_dataset(fp_raw, "Je")[raw_index, ...]
                                Ji = get_h5_dataset(fp_raw, "Ji")[raw_index, ...]
                                J_background = np.concatenate([Je, Ji], axis=-1)

                    result = self.fit_single_snapshot(
                        E,
                        B,
                        x,
                        y,
                        b0,
                        B_background=B_background,
                        J_background=J_background,
                    )
                    if debug:
                        self.generate_diagnostics(result, x, y, step, time=time, wci=wci)
                    self.cleanup_large_arrays(result)
                    self.write_snapshot_result(fp_fit, step, time, result)
        finally:
            if fp_raw is not None:
                fp_raw.close()

    def main_plot(self):
        wavefile = self.get_filename(self.options.get("wavefile", "wavefilter"), ".h5")
        fitfile = self.get_filename(self.options.get("fitfile", "wavefit"), ".h5")
        output_prefix = str(self.options.get("plot_prefix", "wavefit-envelope"))

        outdir = pathlib.Path(self.get_filename(output_prefix, "")).parent
        outdir.mkdir(parents=True, exist_ok=True)

        with h5py.File(wavefile, "r") as fp_wave, h5py.File(fitfile, "r") as fp_fit:
            parameter = extract_parameter_from_wavefile(fp_wave)
            if parameter is None:
                parameter = self.parameter
            b0 = self.get_reference_b0(parameter)
            wci = self.get_wci(parameter)

            if "snapshots" not in fp_fit:
                raise KeyError("fitfile does not contain snapshots group")

            snapshots_group = get_h5_group(fp_fit, "snapshots")

            fit_steps = sorted(int(step) for step in snapshots_group.keys())
            if len(fit_steps) == 0:
                raise ValueError("No fitted snapshots found in fitfile")

            wave_step_ds = get_h5_dataset(fp_wave, "step")
            wave_t_ds = get_h5_dataset(fp_wave, "t")
            wave_x_ds = get_h5_dataset(fp_wave, "x")
            wave_y_ds = get_h5_dataset(fp_wave, "y")
            wave_B_ds = get_h5_dataset(fp_wave, "B")

            snapshot_indices = self.select_snapshot_indices(len(fit_steps))
            wave_steps = wave_step_ds[...]
            step_to_wave_index = {int(step): i for i, step in enumerate(wave_steps)}

            for snapshot_index in tqdm.tqdm(snapshot_indices):
                step = int(fit_steps[int(snapshot_index)])
                grp_name = "snapshots/{:08d}".format(step)
                grp = get_h5_group(fp_fit, grp_name)

                if step not in step_to_wave_index:
                    raise KeyError("step {:08d} not found in wavefile".format(step))
                wave_index = int(step_to_wave_index[step])

                x = wave_x_ds[wave_index, ...] if wave_x_ds.ndim == 2 else wave_x_ds[()]
                y = wave_y_ds[wave_index, ...] if wave_y_ds.ndim == 2 else wave_y_ds[()]
                time = float(wave_t_ds[wave_index])
                B = wave_B_ds[wave_index, ...]
                envelope = np.linalg.norm(B, axis=-1) / b0

                if "ix" in grp and "iy" in grp:
                    cand_ix = np.asarray(get_h5_dataset(grp, "ix")[...], dtype=np.int64)
                    cand_iy = np.asarray(get_h5_dataset(grp, "iy")[...], dtype=np.int64)
                else:
                    cand_ix = np.asarray(get_h5_dataset(grp, "candidate_ix")[...], dtype=np.int64)
                    cand_iy = np.asarray(get_h5_dataset(grp, "candidate_iy")[...], dtype=np.int64)

                if "is_good" in grp:
                    good_mask = np.asarray(get_h5_dataset(grp, "is_good")[...], dtype=np.int8) > 0
                else:
                    good_mask = np.ones((cand_ix.size,), dtype=bool)

                valid = (
                    good_mask
                    & (cand_ix >= 0)
                    & (cand_ix < x.size)
                    & (cand_iy >= 0)
                    & (cand_iy < y.size)
                )

                good_x = x[cand_ix[valid]].tolist()
                good_y = y[cand_iy[valid]].tolist()
                threshold = grp.attrs.get(
                    "option_envelope_threshold", self.options.get("envelope_threshold", np.nan)
                )

                filename = outdir / ("{:s}-{:08d}.png".format(output_prefix, step))
                save_envelope_map_plot(
                    str(filename),
                    envelope,
                    x,
                    y,
                    good_x,
                    good_y,
                    step,
                    time=time,
                    wci=wci,
                    threshold=threshold,
                )


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Wave fitting tool")
    parser.add_argument(
        "-j",
        "--job",
        type=str,
        default="analyze",
        help="Type of job to perform (analyze, plot)",
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
    parser.add_argument(
        "--debug-index",
        dest="debug_indices",
        type=int,
        action="append",
        default=[],
        help="explicit snapshot index to process in debug mode (repeatable)",
    )
    parser.add_argument(
        "--snapshot-index",
        dest="snapshot_indices",
        type=int,
        action="append",
        default=[],
        help="explicit snapshot index to process (works without --debug; repeatable)",
    )
    parser.add_argument(
        "--debug-plot",
        dest="debug_plot",
        action="store_true",
        default=True,
        help="save quicklook plots in debug mode (default: enabled)",
    )
    parser.add_argument(
        "--no-debug-plot",
        dest="debug_plot",
        action="store_false",
        help="disable quicklook plot generation in debug mode",
    )
    parser.add_argument(
        "--debug-plot-count",
        type=int,
        default=8,
        help="number of quicklook candidates to plot in debug mode",
    )
    parser.add_argument(
        "--debug-plot-prefix",
        type=str,
        default="wavefit-debug",
        help="filename prefix for debug quicklook plots",
    )
    parser.add_argument(
        "--plot-prefix",
        type=str,
        default="wavefit-envelope",
        help="filename prefix for envelope map outputs in plot job",
    )
    parser.add_argument("config", nargs=1, help="configuration file")
    args = parser.parse_args()
    config = args.config[0]
    debug = args.debug
    debug_count = args.debug_count
    debug_mode = args.debug_mode
    debug_indices = args.debug_indices
    snapshot_indices = args.snapshot_indices
    debug_plot = args.debug_plot
    debug_plot_count = args.debug_plot_count
    debug_plot_prefix = args.debug_plot_prefix
    plot_prefix = args.plot_prefix

    if debug_plot_count < 0:
        raise ValueError("debug_plot_count must be a non-negative integer")

    def apply_runtime_options(obj):
        obj.options["debug"] = debug
        obj.options["debug_count"] = debug_count
        obj.options["debug_mode"] = debug_mode
        obj.options["debug_indices"] = debug_indices
        obj.options["snapshot_indices"] = snapshot_indices
        obj.options["debug_plot"] = debug_plot
        obj.options["debug_plot_count"] = debug_plot_count
        obj.options["debug_plot_prefix"] = debug_plot_prefix
        obj.options["plot_prefix"] = plot_prefix

    jobs = args.job.split(",")
    for job in tqdm.tqdm(jobs):
        if job == "analyze":
            obj = WaveFitAnalyzer(config)
            apply_runtime_options(obj)
            obj.main()
        elif job == "plot":
            obj = WaveFitAnalyzer(config, option_section="plot")
            apply_runtime_options(obj)
            obj.main_plot()
        else:
            raise ValueError("Unknown job: {:s}".format(job))
