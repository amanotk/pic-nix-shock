#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import pathlib
import sys

import h5py
import numpy as np
import tqdm

if "PICNIX_DIR" in os.environ:
    sys.path.append(str(pathlib.Path(os.environ["PICNIX_DIR"]) / "script"))
try:
    from .. import base
except ImportError:
    import base

from .candidates import pick_candidate_points
from .fit import fit_one_candidate
from .plot import save_quickcheck_plot_12panel


RESULT_FLOAT_KEYS = [
    "x0",
    "y0",
    "kx",
    "ky",
    "Ew",
    "Bw",
    "phiE",
    "phiB",
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
]
RESULT_INT_KEYS = ["ix", "iy", "nfev"]
RESULT_BOOL_KEYS = ["success", "is_good", "is_good_nrmse", "is_good_scale"]
RESULT_STR_KEYS = ["reason", "message"]
LARGE_ARRAY_KEYS = [
    "windowed_data_E",
    "windowed_data_B",
    "windowed_model_E",
    "windowed_model_B",
    "patch_x",
    "patch_y",
]


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


class WaveFitAnalyzer(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "analyze" in self.options:
            for key in self.options["analyze"]:
                self.options[key] = self.options["analyze"][key]

    def read_parameter(self):
        return None

    def fit_single_snapshot(self, E, B, xx, yy):
        fit_options = dict(self.options)
        if bool(self.options.get("debug", False)):
            fit_options.setdefault("max_candidates", 32)
        else:
            fit_options.pop("max_candidates", None)

        sigma = float(self.options.get("sigma", 3.0))
        envelope = np.linalg.norm(B, axis=-1)
        cand_ix, cand_iy, env_used = pick_candidate_points(xx, yy, envelope, sigma, fit_options)

        fit_results = []
        for ix, iy in zip(cand_ix, cand_iy):
            x0 = float(xx[ix])
            y0 = float(yy[iy])
            fit_result = fit_one_candidate(E, B, xx, yy, x0, y0, sigma, fit_options)
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
        grp.create_dataset("envelope", data=result["envelope"])

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

    def generate_diagnostics(self, result, xx, yy, step):
        if not bool(self.options.get("debug_plot", True)):
            return
        fit_results = result["fits"]
        if len(fit_results) == 0:
            return

        debug_plot_count = int(self.options.get("debug_plot_count", 8))
        output_prefix = str(self.options.get("debug_plot_prefix", "wavefit-debug"))
        outdir = pathlib.Path(self.get_filename(output_prefix, "")).parent
        outdir.mkdir(parents=True, exist_ok=True)

        rank = np.argsort(
            [item.get("nrmse_balanced", item.get("nrmse", np.inf)) for item in fit_results]
        )
        for i in rank[:debug_plot_count]:
            item = fit_results[int(i)]
            if "windowed_data_E" not in item:
                continue
            filename = outdir / ("{:s}-{:08d}-{:03d}.png".format(output_prefix, int(step), int(i)))
            title = (
                "step={:08d} cand={:03d} ix={} iy={} nrmse_bal={:.3f} "
                "(E={:.3f}, B={:.3f}) lambda/sigma={:.3f} good=({:d},{:d})"
            ).format(
                int(step),
                int(i),
                int(item.get("ix", -1)),
                int(item.get("iy", -1)),
                float(item.get("nrmse_balanced", item.get("nrmse", np.nan))),
                float(item.get("nrmseE", np.nan)),
                float(item.get("nrmseB", np.nan)),
                float(item.get("wavelength_over_sigma", np.nan)),
                int(bool(item.get("is_good_nrmse", False))),
                int(bool(item.get("is_good_scale", False))),
            )
            save_quickcheck_plot_12panel(str(filename), item, title=title, rms_normalize=True)

    def cleanup_large_arrays(self, result):
        for item in result["fits"]:
            for key in LARGE_ARRAY_KEYS:
                if key in item:
                    del item[key]

    def main(self):
        wavefile = self.get_filename(self.options.get("wavefile", "wavefilter"), ".h5")
        fitfile = self.get_filename(self.options.get("fitfile", "wavefit"), ".h5")
        debug = bool(self.options.get("debug", False))
        debug_count = int(self.options.get("debug_count", 8))
        debug_mode = str(self.options.get("debug_mode", "uniform"))
        debug_indices = self.options.get("debug_indices", [])
        overwrite = bool(self.options.get("overwrite", False))
        if os.path.exists(fitfile) and overwrite:
            os.remove(fitfile)
        if os.path.exists(fitfile) and not overwrite:
            print("Output file {} already exists. Set overwrite=true to replace.".format(fitfile))
            return

        with h5py.File(wavefile, "r") as fp_wave, h5py.File(fitfile, "w") as fp_fit:
            nstep = int(fp_wave["step"].shape[0])
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

            for snapshot_index in tqdm.tqdm(snapshot_indices):
                step = int(fp_wave["step"][snapshot_index])
                time = float(fp_wave["t"][snapshot_index])

                x = (
                    fp_wave["x"][snapshot_index, ...]
                    if fp_wave["x"].ndim == 2
                    else fp_wave["x"][()]
                )
                y = (
                    fp_wave["y"][snapshot_index, ...]
                    if fp_wave["y"].ndim == 2
                    else fp_wave["y"][()]
                )
                E = fp_wave["E"][snapshot_index, ...]
                B = fp_wave["B"][snapshot_index, ...]

                result = self.fit_single_snapshot(E, B, x, y)
                if debug:
                    self.generate_diagnostics(result, x, y, step)
                self.cleanup_large_arrays(result)
                self.write_snapshot_result(fp_fit, step, time, result)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Wave fitting tool")
    parser.add_argument(
        "-j",
        "--job",
        type=str,
        default="analyze",
        help="Type of job to perform (analyze)",
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
    parser.add_argument("config", nargs=1, help="configuration file")
    args = parser.parse_args()
    config = args.config[0]
    debug = args.debug
    debug_count = args.debug_count
    debug_mode = args.debug_mode
    debug_indices = args.debug_indices

    def apply_runtime_options(obj):
        obj.options["debug"] = debug
        obj.options["debug_count"] = debug_count
        obj.options["debug_mode"] = debug_mode
        obj.options["debug_indices"] = debug_indices

    jobs = args.job.split(",")
    for job in tqdm.tqdm(jobs):
        if job == "analyze":
            obj = WaveFitAnalyzer(config)
            apply_runtime_options(obj)
            obj.main()
        else:
            raise ValueError("Unknown job: {:s}".format(job))
