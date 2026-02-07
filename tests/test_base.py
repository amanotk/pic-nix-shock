"""Unit tests for `shock.base`."""

import json

import matplotlib
import msgpack
import numpy as np
import pytest

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
    data_root = temp_dir / "data"
    run_dir = data_root / "run1" / "data"
    run_dir.mkdir(parents=True)

    profile = run_dir / "profile.msgpack"
    _write_profile(profile)

    config_path = temp_dir / "config.toml"
    config_path.write_text(
        'run = "run1"\ndirname = "out"\noverwrite = true\n',
        encoding="utf-8",
    )

    import os

    old_data_root = os.environ.get("SHOCK_DATA_ROOT")
    old_work_root = os.environ.get("SHOCK_WORK_ROOT")
    os.environ["SHOCK_DATA_ROOT"] = str(data_root)
    os.environ["SHOCK_WORK_ROOT"] = str(temp_dir / "work")
    try:
        executor = base.JobExecutor(str(config_path))
        output_dir = executor.get_dirname()
    finally:
        if old_data_root is None:
            os.environ.pop("SHOCK_DATA_ROOT", None)
        else:
            os.environ["SHOCK_DATA_ROOT"] = old_data_root
        if old_work_root is None:
            os.environ.pop("SHOCK_WORK_ROOT", None)
        else:
            os.environ["SHOCK_WORK_ROOT"] = old_work_root

    assert executor.options is not None
    assert executor.parameter == {"dummy": 1}
    assert executor.options["profile"] == str(data_root / "run1" / "data" / "profile.msgpack")
    assert output_dir == str(temp_dir / "work" / "run1" / "out")


def test_job_executor_read_config_toml_with_nested_run(temp_dir):
    data_root = temp_dir / "data"
    run_dir = data_root / "campaignA" / "run2" / "data"
    run_dir.mkdir(parents=True)

    profile = run_dir / "profile.msgpack"
    _write_profile(profile)

    config_path = temp_dir / "config.toml"
    config_path.write_text(
        'run = "campaignA/run2"\ndirname = "reduce1d"\nprofile = "data/profile.msgpack"\noverwrite = true\n',
        encoding="utf-8",
    )

    import os

    old_data_root = os.environ.get("SHOCK_DATA_ROOT")
    old_work_root = os.environ.get("SHOCK_WORK_ROOT")
    os.environ["SHOCK_DATA_ROOT"] = str(data_root)
    os.environ["SHOCK_WORK_ROOT"] = str(temp_dir / "work")
    try:
        executor = base.JobExecutor(str(config_path))
        output_dir = executor.get_dirname()
    finally:
        if old_data_root is None:
            os.environ.pop("SHOCK_DATA_ROOT", None)
        else:
            os.environ["SHOCK_DATA_ROOT"] = old_data_root
        if old_work_root is None:
            os.environ.pop("SHOCK_WORK_ROOT", None)
        else:
            os.environ["SHOCK_WORK_ROOT"] = old_work_root

    assert executor.options["profile"] == str(
        data_root / "campaignA" / "run2" / "data" / "profile.msgpack"
    )
    assert output_dir == str(temp_dir / "work" / "campaignA" / "run2" / "reduce1d")


def test_job_executor_requires_run(temp_dir):
    profile = temp_dir / "profile.msgpack"
    _write_profile(profile)

    config_path = temp_dir / "config.toml"
    config_path.write_text(
        'dirname = "out"\nprofile = "data/profile.msgpack"\noverwrite = true\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run"):
        base.JobExecutor(str(config_path))


def test_job_executor_read_config_json(temp_dir):
    data_root = temp_dir / "data"
    run_dir = data_root / "run-json"
    run_dir.mkdir(parents=True)
    profile = run_dir / "custom.msgpack"
    _write_profile(profile)

    config_data = {
        "run": "run-json",
        "dirname": "out",
        "profile": "custom.msgpack",
        "overwrite": True,
    }
    config_path = temp_dir / "config.json"
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(config_data, fp)

    import os

    old_data_root = os.environ.get("SHOCK_DATA_ROOT")
    old_work_root = os.environ.get("SHOCK_WORK_ROOT")
    os.environ["SHOCK_DATA_ROOT"] = str(data_root)
    os.environ["SHOCK_WORK_ROOT"] = str(temp_dir / "work")
    try:
        executor = base.JobExecutor(str(config_path))
    finally:
        if old_data_root is None:
            os.environ.pop("SHOCK_DATA_ROOT", None)
        else:
            os.environ["SHOCK_DATA_ROOT"] = old_data_root
        if old_work_root is None:
            os.environ.pop("SHOCK_WORK_ROOT", None)
        else:
            os.environ["SHOCK_WORK_ROOT"] = old_work_root

    assert executor.options["dirname"] == "out"
    assert executor.parameter == {"dummy": 1}


def test_job_executor_rejects_parent_traversal_in_run(temp_dir):
    config_path = temp_dir / "config.toml"
    config_path.write_text(
        'run = "../run1"\ndirname = "out"\nprofile = "data/profile.msgpack"\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not contain"):
        base.JobExecutor(str(config_path))


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
