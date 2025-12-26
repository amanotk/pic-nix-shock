# Analysis Tools for PIC Simulations of Collisionless Shocks

## Tips
To merge the content of this repository with an existing directory, you can do as follows (at your own risk):
```
$ git init
$ git branch -m main
$ git remote add origin git@github.com:amanotk/pic-nix-shock.git
$ git fetch
$ git merge origin/main --allow-unrelated-histories
$ git branch --set-upstream-to=origin/main main
```

## Description
### Data Reduction to 1D `reduce1d.py`
The script `reduce1d.py` reduces simulation data to 1D profiles by averaging over the transverse direction.

#### Usage
```bash
$ python shock/reduce1d.py [options] config_file
```

#### Options
- `-o`, `--output`: Basename for output files (default: `reduce1d`).
- `-j`, `--job`: Type of job to perform. Available jobs are `analyze`, `position`, `plot`. Multiple jobs can be specified separated by commas (e.g., `analyze,plot`).

#### Jobs
- `analyze`: Reads raw simulation data, performs reduction/binning, and saves to an HDF5 file (`<output>.h5`).
- `position`: Calculates the shock position based on the reduced data.
- `plot`: Generates plots (PNG) and a movie (MP4) from the reduced data.

#### Examples
- Perform data reduction only:
  ```bash
  $ python shock/reduce1d.py -j analyze reduce1d-config.toml
  ```
- Perform data reduction and then plotting:
  ```bash
  $ python shock/reduce1d.py -j analyze,plot reduce1d-config.toml
  ```
- Calculate shock position only (requires existing HDF5 file):
  ```bash
  $ python shock/reduce1d.py -j position reduce1d-config.toml
  ```

### Field Data Plot in 2D `wavetool.py`
The script `wavetool.py` extracts 2D field or current data, transforms it into the shock rest frame, and generates animations.

#### Usage
```bash
$ python shock/wavetool.py [options] config_file
```

#### Options
- `-o`, `--output`: Basename for output files (default: `wavetool`).
- `-j`, `--job`: Type of job to perform. Available jobs are `analyze`, `plot`. Multiple jobs can be specified separated by commas (e.g., `analyze,plot`).

#### Jobs
- `analyze`: Reads raw simulation data, performs spatial averaging, transforms coordinates to the shock rest frame, and saves to an HDF5 file (`<output>.h5`). **Note**: This job requires `shock_position` to be defined in the configuration or options.
- `plot`: Generates 2D plots (PNG) and a movie (MP4) from the analyzed data.

#### Examples
- Perform analysis and plotting:
  ```bash
  $ python shock/wavetool.py -j analyze,plot wavetool-config.toml
  ```
