"""Fast integration test for wavetool analyze output."""

import sys
import types

import h5py
import msgpack
import numpy as np


class _FakeRun:
    def __init__(self, profile, config=None, method=None):
        self.config = {
            "parameter": {
                "u0": 1.0,
                "mime": 25.0,
                "sigma": 1.0,
                "delh": 1.0,
            }
        }
        self._steps = np.array([10, 20], dtype=np.int32)
        self._xc = np.arange(16, dtype=np.float64)
        self._yc = np.arange(8, dtype=np.float64)
        self._uf = {}
        self._um = {}

        gx, gy = np.meshgrid(np.arange(16, dtype=np.float64), np.arange(8, dtype=np.float64))
        for step in self._steps:
            base = 0.01 * step + gx + 0.1 * gy

            uf = np.zeros((2, 8, 16, 6), dtype=np.float64)
            um = np.zeros((2, 8, 16, 2, 14), dtype=np.float64)

            for k in range(6):
                uf[..., k] = base + k

            for species in (0, 1):
                for moment in range(14):
                    um[..., species, moment] = base + 10 * species + 0.1 * moment

            self._uf[step] = uf
            self._um[step] = um

    def get_step(self, prefix):
        return self._steps

    def get_time_at(self, prefix, step):
        return 0.1 * float(step)

    def read_at(self, prefix, step, component=None):
        data = {
            "xc": self._xc,
            "yc": self._yc,
            "uf": self._uf[int(step)],
            "um": self._um[int(step)],
        }
        if component == "uf":
            return {"xc": data["xc"], "yc": data["yc"], "uf": data["uf"]}
        return data


def test_wavetool_analyze_writes_hdf5_and_transformed_moments(temp_dir, monkeypatch):
    data_root = temp_dir / "data"
    work_root = temp_dir / "work"
    run_data_dir = data_root / "run-test" / "data"
    run_data_dir.mkdir(parents=True)

    profile_path = run_data_dir / "profile.msgpack"
    profile_payload = {
        "configuration": {
            "parameter": {
                "u0": 1.0,
                "mime": 25.0,
                "sigma": 1.0,
                "delh": 1.0,
            }
        }
    }
    with open(profile_path, "wb") as fp:
        msgpack.dump(profile_payload, fp)

    config_path = temp_dir / "wavetool.toml"
    config_path.write_text(
        "\n".join(
            [
                'run = "run-test"',
                'dirname = "wavetool-test"',
                'profile = "data/profile.msgpack"',
                "overwrite = true",
                "",
                "[analyze]",
                "num_average = 2",
                "num_xwindow = 8",
                "step_min = 10",
                "step_max = 20",
                "x_offset = 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    reduce1d_dir = work_root / "run-test" / "reduce1d"
    reduce1d_dir.mkdir(parents=True)
    (reduce1d_dir / "reduce1d_result.toml").write_text(
        "shock_position = [0.0, 0.0]\n",
        encoding="utf-8",
    )

    monkeypatch.setenv("SHOCK_DATA_ROOT", str(data_root))
    monkeypatch.setenv("SHOCK_WORK_ROOT", str(work_root))
    sys.modules.setdefault("picnix", types.ModuleType("picnix"))

    from shock import wavetool

    monkeypatch.setattr(wavetool.picnix, "Run", _FakeRun, raising=False)

    analyzer = wavetool.DataAnalyzer(str(config_path))
    analyzer.main("quick")
    output_path = analyzer.get_filename("quick", ".h5")

    with h5py.File(output_path, "r") as fp:
        assert "E" in fp
        assert "E_ohm" in fp
        assert "B" in fp
        assert "J" in fp
        assert "M" in fp
        assert fp["E"].shape == (2, 4, 4, 3)
        assert fp["E_ohm"].shape == (2, 4, 4, 3)
        assert fp["B"].shape == (2, 4, 4, 3)
        assert fp["J"].shape == (2, 4, 4, 8)
        assert fp["M"].shape == (2, 4, 4, 10)
        assert np.all(np.isfinite(fp["M"][...]))


def test_calc_e_ohm_uses_small_epsilon_divisor(monkeypatch):
    sys.modules.setdefault("picnix", types.ModuleType("picnix"))
    from shock import wavetool

    analyzer = object.__new__(wavetool.DataAnalyzer)

    B = np.array([[[[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]]], dtype=np.float64)
    M = np.zeros((1, 1, 2, 10), dtype=np.float64)
    M[..., 1] = 1.0
    M[..., 0] = np.array([[[0.0, 2.0]]], dtype=np.float64)

    E_ohm = analyzer.calc_e_ohm(B, M, dx=1.0, dy=1.0)

    assert np.isfinite(E_ohm).all()
    assert E_ohm[0, 0, 0, 1] > 1.0e20
    assert np.isclose(E_ohm[0, 0, 1, 1], 0.5)
