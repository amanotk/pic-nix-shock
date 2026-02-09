import json
import os
import pathlib

import msgpack
import numpy as np
import toml


class JobExecutor:
    def __init__(self, config_file):
        self._load_local_env()
        self.config_file = config_file
        self.options = self.read_config()
        self.parameter = self.read_parameter()

    def _load_local_env(self):
        env_path = os.environ.get("SHOCK_ENV_FILE", None)
        if env_path is None:
            repo_root = pathlib.Path(__file__).resolve().parent.parent
            env_file = repo_root / ".shock.env"
        else:
            env_file = pathlib.Path(env_path).expanduser()

        if not env_file.exists() or not env_file.is_file():
            return

        with open(env_file, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if len(line) == 0 or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                    value = value[1:-1]
                if key not in os.environ:
                    os.environ[key] = value

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

        run_path = self._normalize_relative_subpath(config.get("run", None), "run")
        profile_path = self._normalize_relative_subpath(
            config.get("profile", "data/profile.msgpack"),
            "profile",
        )
        data_root = pathlib.Path(os.environ.get("SHOCK_DATA_ROOT", "./data")).expanduser()

        config["run"] = run_path.as_posix()
        config["profile"] = str(data_root / run_path / profile_path)

        return config

    def _normalize_relative_subpath(self, value, key):
        if not isinstance(value, str) or len(value.strip()) == 0:
            raise ValueError(f"{key} must be a non-empty string")

        value_path = pathlib.PurePath(value)
        if value_path.is_absolute():
            raise ValueError(f"{key} must be a relative path")

        parts = [part for part in value_path.parts if part not in ("", ".")]
        if len(parts) == 0:
            raise ValueError(f"{key} must be a non-empty relative path")
        if any(part == ".." for part in parts):
            raise ValueError(f"{key} must not contain '..'")

        return pathlib.Path(*parts)

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
        run = self.options.get("run", None)
        run_path = self._normalize_relative_subpath(run, "run")
        dirname_path = self._normalize_relative_subpath(dirname, "dirname")
        work_root = pathlib.Path(os.environ.get("SHOCK_WORK_ROOT", "./work")).expanduser()
        dirname_path = work_root / run_path / dirname_path
        dirname_path.mkdir(parents=True, exist_ok=True)
        return str(dirname_path)

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
