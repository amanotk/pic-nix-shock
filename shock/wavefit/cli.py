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
    "support_fraction",
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


class WaveFitAnalyzer(base.JobExecutor):
    def __init__(self, config_file):
        super().__init__(config_file)
        if "analyze" in self.options:
            for key in self.options["analyze"]:
                self.options[key] = self.options["analyze"][key]

    def read_parameter(self):
        return None

    def fit_single_snapshot(self, E, B, xx, yy):
        sigma = float(self.options.get("sigma", 3.0))
        envelope = np.linalg.norm(B, axis=-1)
        cand_ix, cand_iy, env_used = pick_candidate_points(xx, yy, envelope, sigma, self.options)

        fit_results = []
        for ix, iy in zip(cand_ix, cand_iy):
            x0 = float(xx[ix])
            y0 = float(yy[iy])
            fit_result = fit_one_candidate(E, B, xx, yy, x0, y0, sigma, self.options)
            fit_result["ix"] = int(ix)
            fit_result["iy"] = int(iy)
            fit_results.append(fit_result)

        return {
            "candidate_ix": cand_ix,
            "candidate_iy": cand_iy,
            "envelope": env_used,
            "fits": fit_results,
        }

    def write_snapshot_result(self, output_file, step, time, result):
        overwrite = bool(self.options.get("overwrite", False))
        if os.path.exists(output_file) and overwrite:
            os.remove(output_file)
        if os.path.exists(output_file) and not overwrite:
            print(
                "Output file {} already exists. Set overwrite=true to replace.".format(output_file)
            )
            return

        with h5py.File(output_file, "w") as fp:
            grp = fp.create_group("snapshots/{:08d}".format(int(step)))
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
        snapshot_index = int(self.options.get("snapshot_index", 0))

        with h5py.File(wavefile, "r") as fp:
            nstep = fp["step"].shape[0]
            if snapshot_index < 0 or snapshot_index >= nstep:
                raise ValueError(
                    "snapshot_index out of range: {} (nstep={})".format(snapshot_index, nstep)
                )

            step = int(fp["step"][snapshot_index])
            time = float(fp["t"][snapshot_index])

            x = fp["x"][snapshot_index, ...] if fp["x"].ndim == 2 else fp["x"][()]
            y = fp["y"][snapshot_index, ...] if fp["y"].ndim == 2 else fp["y"][()]
            E = fp["E"][snapshot_index, ...]
            B = fp["B"][snapshot_index, ...]

        result = self.fit_single_snapshot(E, B, x, y)
        self.generate_diagnostics(result, x, y, step)
        self.cleanup_large_arrays(result)
        self.write_snapshot_result(fitfile, step, time, result)


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
    parser.add_argument("config", nargs=1, help="configuration file")
    args = parser.parse_args()
    config = args.config[0]

    jobs = args.job.split(",")
    for job in tqdm.tqdm(jobs):
        if job == "analyze":
            obj = WaveFitAnalyzer(config)
            obj.main()
        else:
            raise ValueError("Unknown job: {:s}".format(job))
