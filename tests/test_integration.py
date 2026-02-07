"""Integration tests for basic CLI workflows."""

import os
import subprocess
import sys
from pathlib import Path

import h5py


def _make_stub_picnix(stub_dir: Path):
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "picnix.py").write_text("class Run:\n    pass\n", encoding="utf-8")


def _script_env(root_dir: Path, tmp_path: Path):
    stub_dir = tmp_path / "stub"
    _make_stub_picnix(stub_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub_dir), str(root_dir), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env.setdefault("PICNIX_DIR", str(Path.home() / "raid" / "simulation" / "pic-nix"))
    return env


def test_reduce1d_cli_parsing(root_dir, tmp_path):
    env = _script_env(root_dir, tmp_path)
    result = subprocess.run(
        [sys.executable, "shock/reduce1d.py", "--help"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(root_dir),
    )
    assert result.returncode == 0
    assert "reduce1d" in result.stdout.lower()


def test_hdf5_structure_validation(minimal_hdf5_file):
    with h5py.File(minimal_hdf5_file, "r") as fp:
        required = ["step", "time", "E", "B", "Je", "Ji"]
        for name in required:
            assert name in fp

        step = fp["step"][:]
        time = fp["time"][:]
        assert len(step) == len(time)
        assert fp["E"].shape[0] == len(step)
        assert fp["B"].shape[0] == len(step)


def test_wavetool_cli_parsing(root_dir, tmp_path):
    env = _script_env(root_dir, tmp_path)
    result = subprocess.run(
        [sys.executable, "shock/wavetool.py", "--help"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(root_dir),
    )
    assert result.returncode == 0
    assert "wavetool" in result.stdout.lower()


def test_mra_cli_parsing(root_dir, tmp_path):
    env = _script_env(root_dir, tmp_path)
    result = subprocess.run(
        [sys.executable, "shock/mra.py", "--help"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(root_dir),
    )
    assert result.returncode == 0
    assert "multi-resolution" in result.stdout.lower()


def test_vdist_cli_parsing(root_dir, tmp_path):
    try:
        from mpi4py import MPI  # noqa: F401
    except Exception:
        return

    env = _script_env(root_dir, tmp_path)
    result = subprocess.run(
        [sys.executable, "shock/vdist.py", "--help"],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(root_dir),
    )
    assert result.returncode == 0
    assert "vdist" in result.stdout.lower()
