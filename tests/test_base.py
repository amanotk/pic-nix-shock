"""Unit tests for `shock.base`."""

import json

import matplotlib
import msgpack
import numpy as np

from shock import base

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _write_profile(path):
    payload = {
        "configuration": {
            "parameter": {
                "dummy": 1,
            }
        }
    }
    with open(path, "wb") as fp:
        msgpack.dump(payload, fp)


def test_job_executor_read_config_toml(temp_dir):
    profile = temp_dir / "profile.msgpack"
    _write_profile(profile)

    config_path = temp_dir / "config.toml"
    config_path.write_text(
        'dirname = "out"\nprofile = "profile.msgpack"\noverwrite = true\n',
        encoding="utf-8",
    )

    executor = base.JobExecutor(str(config_path))
    assert executor.options is not None
    assert executor.parameter == {"dummy": 1}


def test_job_executor_read_config_json(temp_dir):
    profile = temp_dir / "profile.msgpack"
    _write_profile(profile)

    config_data = {
        "dirname": "out",
        "profile": "profile.msgpack",
        "overwrite": True,
    }
    config_path = temp_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(config_data, fp)

    executor = base.JobExecutor(str(config_path))
    assert executor.options["dirname"] == "out"
    assert executor.parameter == {"dummy": 1}


def test_get_colorbar_position_next():
    fig, ax = plt.subplots()
    try:
        caxpos = base.get_colorbar_position_next(ax)
    finally:
        plt.close(fig)
    assert len(caxpos) == 4


def test_get_vlim():
    data1 = np.random.randn(100)
    data2 = np.random.randn(100)
    vlims = base.get_vlim([data1, data2])
    assert len(vlims) == 2
    assert len(vlims[0]) == 2
