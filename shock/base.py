import json
import os

import msgpack
import numpy as np
import toml


class JobExecutor:
    def __init__(self, config_file):
        self.config_file = config_file
        self.options = self.read_config()
        self.parameter = self.read_parameter()

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
        if "profile" not in self.options:
            return None
        with open(self.options["profile"], "rb") as fp:
            obj = msgpack.load(fp)
            parameter = obj["configuration"]["parameter"]
        return parameter

    def get_dirname(self):
        dirname = self.options.get("dirname", None)
        if dirname is None:
            raise ValueError("dirname is not specified")
        elif not os.path.exists(dirname):
            os.makedirs(dirname)
        return dirname

    def get_filename(self, basename, ext):
        return os.sep.join([self.get_dirname(), basename + ext])

    def main(self, basename):
        raise NotImplementedError


def get_colorbar_position_next(ax, pad=0.05):
    axpos = ax.get_position()
    caxpos = [
        axpos.x0 + axpos.width * (1 + pad),
        axpos.y0,
        axpos.width * pad,
        axpos.height,
    ]
    return caxpos


def get_vlim(vars, vmag=100):
    vlims = []
    for v in vars:
        vmin = np.sign(v.min()) * np.ceil(np.abs(v.min()) * vmag) / vmag
        vmax = np.sign(v.max()) * np.ceil(np.abs(v.max()) * vmag) / vmag
        vlims.append([vmin, vmax])
    return vlims
