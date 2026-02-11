# Configuration Reference

All scripts use TOML configuration files.

## Common Keys

```toml
run = "run1"
dirname = "output_directory"
profile = "data/profile.msgpack"
overwrite = true
```

- `run`: Run directory name (or subpath) under `SHOCK_DATA_ROOT`
- `dirname`: Output directory under `SHOCK_WORK_ROOT/run`
- `profile`: Profile path relative to `run` (default: `data/profile.msgpack`)
- `overwrite`: Whether existing outputs may be overwritten

## `reduce1d.py`

Example: `sample/reduce1d-config.toml`

```toml
run = "run1"
dirname = "reduce1d"
method = "async"
overwrite = true

[analyze]
step_min = 0
step_max = 100000000
Nbinx = 4096
Nbinu = 80
ubine = [-0.8, +0.8]
ebine = [1.0e-3, 1.0e+0]
ubini = [-0.05, +0.05]
ebini = [1.0e-4, 1.0e-1]

[plot]
panels = ["fi_ux", "fi_uy", "fe_ux", "fe_uy", "fe_p4"]

[position]
fit_range = [5.0, 16.0]
```

## `wavetool.py`

Example: `sample/wavetool-config.toml`

`wavetool.py` takes diagnostic prefix from CLI option `--prefix` (default:
`field`), not from TOML. Output directory is automatically suffixed as
`<dirname>-<prefix>`.

```toml
run = "run1"
dirname = "wavetool"
overwrite = true

[analyze]
num_average = 4
num_xwindow = 1024
step_min = 0
x_offset = -80
shock_position = [0.01, -75.0]

[plot]
fps = 10
quantity = "field"
aspect_ratio = 2.0
x_center_offset = 0.0
```

`aspect_ratio` controls the x-window length as `x_length / y_length` while keeping
equal axis scaling. The full y-range is always shown, and x-window center is
`shock_position(t) + x_center_offset` (fallback: x-domain center if shock position is unavailable).

Example commands:

```bash
python shock/wavetool.py -j analyze --prefix field sample/wavetool-config.toml
python shock/wavetool.py -j plot --prefix field sample/wavetool-config.toml
```

## `vdist.py`

```toml
run = "run1"
dirname = "vdist"
overwrite = true

[reduce]
step_min = 0
step_max = 100000000
upara_nbins = 80
```

## `mra.py`

```toml
run = "run1"
dirname = "mra"
overwrite = true

[analyze]
rawfile = "wavetool"
mrafile = "mra"
```

## `wavefilter.py`

```toml
run = "run1"
dirname = "wavetool"
profile = "data/profile.msgpack"
overwrite = true

[analyze]
rawfile = "wavetool"
wavefile = "wavefilter"
fc_low = 0.5
# fc_high = 1.5
order = 4

[plot]
wavefile = "wavefilter"
output = "wavefilter"
fps = 10
quantity = "wave"
aspect_ratio = 2.0
x_center_offset = 0.0
# quantity = "field"
# E_lim = [-0.2, 0.2]
# B_lim = [-0.2, 0.2]
smooth_sigma = 0.5
B_wave_lim = [-0.25, 0.25]
B_env_lim = [0.0, 0.5]
B_abs_lim = [0.5, 5.0]
S_para_lim = [-0.5, 0.5]
```

`wavefilter.py` chooses the temporal filter mode from cutoff keys:

- `fc_low` + `fc_high`: band-pass
- `fc_low` only: high-pass
- `fc_high` only: low-pass

Cutoff frequencies are in inverse time units of the input HDF5 `t` array.
Sampling frequency is inferred from `t` and requires equally sampled time points.

`smooth_sigma` applies optional Gaussian smoothing (in cell units) during plotting.
Smoothing uses periodic boundary in `y` (`wrap`) and non-periodic boundary in `x` (`nearest`).
Set `smooth_sigma = 0` to disable smoothing.

`aspect_ratio` controls the x-window length as `x_length / y_length` while keeping
equal axis scaling. The full y-range is always shown, and x-window center is
`shock_position(t) + x_center_offset` (fallback: x-domain center if shock position is unavailable).

`analyze` stores filtered `E` (from `E_ohm`) and filtered `B`.
`plot` computes Poynting flux from these filtered fields as
`S = E x B` (in the project normalization; no explicit `4\pi` factor).
`S_parallel` is the projection of `S` onto the local, instantaneous
ambient magnetic-field direction `B_raw / |B_raw|`.

`plot.quantity` selects panel content:

- `wave` (default): `\delta Bx, \delta By, \delta Bz, |B|, B_envelope, S_parallel`
- `field`: `Ex/E0, Ey/E0, Ez/E0, Bx/B0, By/B0, Bz/B0`

For `field`, normalization follows `wavetool.py`:

- `B0 = sqrt(sigma) / sqrt(1 + u0^2)`
- `E0 = B0 * u0 / sqrt(1 + u0^2)`

Optional `field` color limits are `E_lim` and `B_lim`.

## Environment Variables

- `SHOCK_DATA_ROOT` default: `./data`
- `SHOCK_WORK_ROOT` default: `./work`

```bash
export PICNIX_DIR=/path/to/picnix
```
