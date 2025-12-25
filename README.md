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
  $ python shock/reduce1d.py -j analyze config.toml
  ```
- Perform data reduction and then plotting:
  ```bash
  $ python shock/reduce1d.py -j analyze,plot config.toml
  ```
- Calculate shock position only (requires existing HDF5 file):
  ```bash
  $ python shock/reduce1d.py -j position config.toml
  ```
