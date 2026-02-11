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
```

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
dirname = "wavetool"
profile = "profile.msgpack"
overwrite = true

[analyze]
rawfile = "wavetool"
wavefile = "wavefilter"
fs = 4.0
fc = 0.5
order = 4

[plot]
wavefile = "wavefilter"
output = "wavefilter"
fps = 10
B_wave_lim = [-0.25, 0.25]
B_env_lim = [0.0, 0.5]
B_abs_lim = [0.5, 5.0]
S_para_lim = [-0.5, 0.5]
```

## Environment Variables

- `SHOCK_DATA_ROOT` default: `./data`
- `SHOCK_WORK_ROOT` default: `./work`

```bash
export PICNIX_DIR=/path/to/picnix
```
