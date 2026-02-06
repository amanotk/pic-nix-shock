# Analysis Tools for PIC Simulations of Collisionless Shocks

[![CI](https://github.com/amanotk/pic-nix-shock/actions/workflows/ci.yml/badge.svg)](https://github.com/amanotk/pic-nix-shock/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This package provides analysis tools for Particle-In-Cell (PIC) simulations of
collisionless shocks. It includes scripts for data reduction, wave analysis,
velocity distribution computation, and visualization.

## Installation

### From Source

```bash
git clone https://github.com/amanotk/pic-nix-shock.git
cd pic-nix-shock
pip install -e .
```

### With MPI Support

```bash
pip install -e ".[mpi]"
```

### For Development

```bash
pip install -e .
pip install pytest ruff pre-commit
```

## Dependencies

Required:

- Python >= 3.8
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.4
- h5py >= 3.0
- toml >= 0.10
- msgpack >= 1.0
- tqdm >= 4.60
- PyWavelets >= 1.1

Optional:

- mpi4py >= 3.0 (for MPI workflows)

## Quick Start

### Reduce simulation data to 1D profiles

```bash
python shock/reduce1d.py -j analyze sample/reduce1d-config.toml
python shock/reduce1d.py -j plot sample/reduce1d-config.toml
```

### Analyze 2D field data

```bash
python shock/wavetool.py -j analyze sample/wavetool-config.toml
python shock/wavetool.py -j plot sample/wavetool-config.toml
```

### Compute velocity distributions

```bash
python shock/vdist.py -j reduce config.toml
```

## Documentation

- [Configuration Reference](CONFIGURATION.md)

## Development

### Running Tests

```bash
pytest tests/ -q
```

### Code Quality

```bash
ruff check shock/ tests/
ruff format shock/ tests/
pre-commit run --all-files
```

## Notes

- The `picnix` module is required to process real simulation outputs.
- Set `PICNIX_DIR` to your local PIC-NIX installation when needed.
