# Installation Guide

## System Requirements

- Python 3.8 or newer
- Typical scientific Python stack
- For MPI support: OpenMPI or MPICH

## Install from Source

```bash
git clone https://github.com/amanotk/pic-nix-shock.git
cd pic-nix-shock
pip install -e .
```

## Optional Extras

### MPI support

```bash
pip install -e ".[mpi]"
```

### Development tooling

```bash
pip install -e ".[dev]"
```

## Verify Installation

```bash
python --version
python -m compileall shock
pytest tests/test_smoke.py -v
```

## PICNIX dependency

The `picnix` package is required for real simulation data workflows and is not installed
from PyPI with this project. Set the location in your environment:

```bash
export PICNIX_DIR=${HOME}/raid/simulation/pic-nix
```
