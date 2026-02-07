# Configuration Reference

All scripts use TOML configuration files.

## Common Keys

```toml
dirname = "output_directory"
profile = "path/to/profile.msgpack"
overwrite = true
```

- `dirname`: Output directory
- `profile`: Simulation profile file path
- `overwrite`: Whether existing outputs may be overwritten

## `reduce1d.py`

Example: `sample/reduce1d-config.toml`

```toml
dirname = "reduce1d"
profile = "profile.msgpack"
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

```toml
dirname = "wavetool"
profile = "profile.msgpack"
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

## `vdist.py`

```toml
dirname = "vdist"
profile = "profile.msgpack"
overwrite = true

[reduce]
step_min = 0
step_max = 100000000
upara_nbins = 80
```

## `mra.py`

```toml
dirname = "mra"
profile = "profile.msgpack"
overwrite = true

[analyze]
rawfile = "wavetool"
mrafile = "mra"
```

## `waveactivity.py`

```toml
dirname = "waveactivity"
profile = "profile.msgpack"
overwrite = true

[analyze]
rawfile = "wavetool"
wavefile = "waveactivity"
fs = 4.0
fc = 0.5
order = 4

[plot]
wavefile = "waveactivity"
output = "waveactivity"
fps = 10
B_wave_lim = [-0.25, 0.25]
B_env_lim = [0.0, 0.5]
B_abs_lim = [0.5, 5.0]
S_para_lim = [-0.5, 0.5]
```

## Environment Variables

```bash
export PICNIX_DIR=/path/to/picnix
```
