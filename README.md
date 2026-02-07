# Analysis Tools for PIC Simulations of Collisionless Shocks

[![CI](https://github.com/amanotk/pic-nix-shock/actions/workflows/ci.yml/badge.svg)](https://github.com/amanotk/pic-nix-shock/actions)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

## Overview

This package provides analysis tools for Particle-In-Cell (PIC) simulations of
collisionless shocks. It includes scripts for data reduction, wave analysis,
velocity distribution computation, and visualization.

These tools are intended to analyze simulation outputs produced by PIC-NIX:
https://github.com/amanotk/pic-nix/

## Installation

Use `uv` as the default workflow:

```bash
git clone https://github.com/amanotk/pic-nix-shock.git
cd pic-nix-shock
bash scripts/setup.sh
```

If `uv` is unavailable, `scripts/setup.sh` falls back to a local `.venv` + `pip install -e .` automatically.

Manual fallback (if you prefer not to use the setup script):

```bash
git clone https://github.com/amanotk/pic-nix-shock.git
cd pic-nix-shock
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

With MPI support (fallback pip path):

```bash
pip install -e ".[mpi]"
```

**Setup script** (`scripts/setup.sh`):
- Creates the `work/` directory if it doesn't exist
- Sets up `.shock.env` from example (if not already present)
- Syncs the Python environment with `uv sync` (or pip fallback when `uv` is missing)
- Run once after cloning or pulling

## Environment Variables and Local Runtime Settings

- `SHOCK_WORK_ROOT`: base directory for relative `dirname` output paths.
  - default: `work`
- `SHOCK_DATA_ROOT`: optional user metadata/convention for where simulation run directories live.
  - profile paths are still selected explicitly per run/config.
- `SHOCK_ENV_FILE`: optional path to an env file to auto-load.
  - if unset, runtime auto-loads repo-root `.shock.env` when present.

Runtime precedence is: exported shell/job vars > `.shock.env` values > defaults.

Create local runtime settings (git-ignored):

```bash
cp .shock.env.example .shock.env
```

Example `.shock.env`:

```bash
SHOCK_DATA_ROOT=/path/to/simulation-runs
SHOCK_WORK_ROOT=work
```

## Batch Scheduler Usage

### Slurm example

```bash
#!/bin/bash
#SBATCH -J shock-reduce
#SBATCH -t 01:00:00

cd /path/to/shock
export SHOCK_DATA_ROOT=/path/to/sim-data
export SHOCK_WORK_ROOT=work

uv run python shock/reduce1d.py -j analyze sample/reduce1d-config.toml
```

### PJM / pjsub example

```bash
#!/bin/bash
#PJM -N "shock-reduce"
#PJM -L "elapse=01:00:00"

cd /path/to/shock
export SHOCK_DATA_ROOT=/path/to/sim-data
export SHOCK_WORK_ROOT=work

uv run python shock/reduce1d.py -j analyze sample/reduce1d-config.toml
```

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
