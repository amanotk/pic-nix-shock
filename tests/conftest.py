"""Pytest fixtures and shared test configuration."""

import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest


@pytest.fixture(scope="session")
def root_dir():
    """Path to repository root."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def sample_config_path(root_dir):
    """Path to sample configuration directory."""
    return root_dir / "sample"


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    dirname = tempfile.mkdtemp(prefix="shock-test-")
    try:
        yield Path(dirname)
    finally:
        shutil.rmtree(dirname, ignore_errors=True)


@pytest.fixture
def minimal_hdf5_file(temp_dir):
    """Create a minimal HDF5 file for structure checks."""
    h5_path = temp_dir / "minimal.h5"
    with h5py.File(h5_path, "w") as fp:
        fp.create_dataset("step", data=np.array([0, 1, 2, 3, 4], dtype=np.int32))
        fp.create_dataset("time", data=np.array([0, 1, 2, 3, 4], dtype=np.float64))
        fp.create_dataset("E", data=np.random.randn(5, 10))
        fp.create_dataset("B", data=np.random.randn(5, 10))
        fp.create_dataset("Je", data=np.random.randn(5, 10))
        fp.create_dataset("Ji", data=np.random.randn(5, 10))
        fp.create_dataset("Feu", data=np.random.randn(5, 10, 8))
        fp.create_dataset("Fiu", data=np.random.randn(5, 10, 8))
    return h5_path


@pytest.fixture
def synthetic_2d_data():
    """Generate simple synthetic 2D field-like data."""
    x = np.linspace(-10, 10, 100)
    y = np.linspace(-10, 10, 100)
    xx, yy = np.meshgrid(x, y)
    data = np.exp(-(xx**2 + yy**2) / 20.0)
    return {"x": x, "y": y, "data": data}
