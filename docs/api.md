# API Overview

This page summarizes the main modules and entry points in `shock`.

## Package Layout

```text
shock/
├── __init__.py
├── base.py
├── utils.py
├── reduce1d.py
├── wavetool.py
├── vdist.py
├── mra.py
├── printparam.py
├── summary.py
├── make_tracer_hdf5.py
```

## Core Modules

### `shock.base`

- `JobExecutor`: Base helper for loading configs and parameters.
- `get_colorbar_position_next(ax, pad=0.05)`
- `get_vlim(vars, vmag=100)`

### `shock.utils`

- K-space filters:
  - `kspace_kernerl1d`, `kspace_kernerl2d`
  - `bandpass_filter1d`, `bandpass_filter2d`
- Shock analysis helpers:
  - `find_overshoot`, `find_ramp`, `calc_shock_speed`, `calc_shock_potential`
- Misc:
  - `interp_window`, `calc_velocity_dist4d`, `calc_particle_flux`

### `shock.reduce1d`

- `DataReducer`
- `ShockPositionModel`
- `DataPlotter`
- CLI entrypoint: `main()`

### `shock.wavetool`

- `DataAnalyzer`
- `DataPlotter`
- CLI entrypoint: `main()`

### `shock.vdist`

- `DataReducer` (MPI-enabled workflow)
- CLI entrypoint: `main()`

### `shock.mra`

- `MraAnalyzer`
- `MraPlotter`
- CLI entrypoint: `main()`

### `shock.printparam`

- `read_and_print_config`
- CLI entrypoint: `cli_main()`

## Console Scripts

Installed through `pyproject.toml`:

- `reduce1d`
- `wavetool`
- `mra`
- `vdist`
- `printparam`

## External Dependencies

- `picnix`: Required for reading PIC simulation datasets.
- `mpi4py`: Required only for MPI-enabled `vdist` workflows.
