"""Tests for wavefilter analyze output."""

import sys
import types

import h5py
import numpy as np


def test_wavefilter_analyze_writes_filtered_fields(temp_dir, monkeypatch):
    data_root = temp_dir / "data"
    work_root = temp_dir / "work"
    monkeypatch.setenv("SHOCK_DATA_ROOT", str(data_root))
    monkeypatch.setenv("SHOCK_WORK_ROOT", str(work_root))
    sys.modules.setdefault("picnix", types.ModuleType("picnix"))

    run_dir = work_root / "run-test" / "wavetool"
    run_dir.mkdir(parents=True)
    raw_path = run_dir / "wavetool.h5"

    nt, ny, nx = 16, 4, 6
    t = np.linspace(0.0, 3.75, nt)
    step = np.arange(nt, dtype=np.int32)
    x = np.tile(np.linspace(0.0, 5.0, nx), (nt, 1))
    y = np.tile(np.linspace(-1.0, 1.0, ny), (nt, 1))

    tt = t[:, np.newaxis, np.newaxis]
    yy = np.linspace(-1.0, 1.0, ny)[np.newaxis, :, np.newaxis]
    xx = np.linspace(0.0, 5.0, nx)[np.newaxis, np.newaxis, :]

    B = np.stack(
        [
            1.0 + 0.1 * np.sin(2.0 * np.pi * tt) + 0.02 * xx + 0.0 * yy,
            0.05 * np.cos(2.0 * np.pi * tt) + 0.01 * yy + 0.0 * xx,
            0.03 * np.sin(4.0 * np.pi * tt) + 0.0 * xx + 0.0 * yy,
        ],
        axis=-1,
    )
    E_ohm = np.stack(
        [
            0.2 * np.sin(2.0 * np.pi * tt) + 0.0 * xx + 0.0 * yy,
            0.1 * np.cos(2.0 * np.pi * tt) + 0.0 * xx + 0.0 * yy,
            0.05 * np.sin(4.0 * np.pi * tt) + 0.0 * xx + 0.0 * yy,
        ],
        axis=-1,
    )

    with h5py.File(raw_path, "w") as fp:
        fp.create_dataset("step", data=step)
        fp.create_dataset("t", data=t)
        fp.create_dataset("x", data=x)
        fp.create_dataset("y", data=y)
        fp.create_dataset("B", data=B)
        fp.create_dataset("E_ohm", data=E_ohm)

    config_path = temp_dir / "wavefilter.toml"
    config_path.write_text(
        "\n".join(
            [
                'run = "run-test"',
                'dirname = "wavetool"',
                'profile = "data/profile.msgpack"',
                "overwrite = true",
                "",
                "[analyze]",
                'rawfile = "wavetool"',
                'wavefile = "wavefilter"',
                "fc_low = 0.5",
                "order = 2",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    from shock import wavefilter

    analyzer = wavefilter.WaveFilterAnalyzer(str(config_path))
    analyzer.main()
    out_path = analyzer.get_filename("wavefilter", ".h5")

    with h5py.File(out_path, "r") as fp:
        assert fp["B"].shape == (nt, ny, nx, 3)
        assert fp["E"].shape == (nt, ny, nx, 3)
        assert np.all(np.isfinite(fp["B"][...]))
        assert np.all(np.isfinite(fp["E"][...]))
