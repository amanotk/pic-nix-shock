# AGENTS.md

Guidance for coding agents working in this repository.
This document is based only on files currently tracked by git.

## Scope and Source of Truth

- Primary code lives in `shock/`.
- Primary docs live in `README.md`.
- Example runtime configs live in `sample/*.toml`.
- Packaging metadata is tracked in `pyproject.toml`.
- For wavefit CLI/runtime semantics, prefer `CONFIGURATION.md` and
  `sample/wavefit-config.toml` as the source of truth for current options.
- For interactive wavefit tuning workflow, prefer
  `docs/wavefit-interactive-tuning.md`.

## Agent Scope

- This repository is primarily intended for OpenCode-style agent workflows.

## Git Operations

- **DO NOT commit or push any changes without explicit instruction from the user.**
- If you create new files that should be committed, ask the user first.
- Always verify the changes with `git diff` before committing.
- Follow the commit message style used in this repository.

## Repository Size and File-Reading Policy

- This workspace may contain a very large number of files outside the git-tracked source tree.
- To keep response latency low, only read/analyze files that are tracked by git.
- Additionally, it is acceptable to read files that the agent has just touched or created during the current task.
- Do not perform broad filesystem scans over untracked directories.
- Prefer targeted inspection: `git ls-files`, then read only the minimum subset needed for the requested change.

## Runtime and Dependency Assumptions

- Codebase is Python-first and script-oriented.
- Preferred environment manager is `uv` (`uv sync`, `uv run ...`).
- `pip` workflows remain acceptable fallback on systems where `uv` is unavailable.
- Typical dependencies inferred from imports:
  - `numpy`, `scipy`, `matplotlib`, `h5py`, `toml`, `msgpack`, `tqdm`, `pywt`, `mpi4py`.
  - Domain dependency: `picnix` (imported as a Python module, and sometimes via `PICNIX_DIR/script` on `sys.path`).
- Some scripts use MPI (`shock/vdist.py`) and may need `mpiexec`.

### Repository setup

- After cloning, run `bash scripts/setup.sh` once to:
  - Create the `work/` directory
  - Set up `.shock.env` from example
  - Sync the Python environment with `uv` (or pip fallback if `uv` is unavailable)
- Run this again if you accidentally delete the `work/` directory or need to reinstall the environment.
- The setup script is local and not pushed to remote.

### Local env file and path behavior

- The runtime may auto-load a repo-root `.shock.env` (or file pointed by `SHOCK_ENV_FILE`).
- `SHOCK_WORK_ROOT` defaults to `./work`; outputs are written under `SHOCK_WORK_ROOT/run/dirname`.
- `SHOCK_DATA_ROOT` defaults to `./data`; profile is resolved as `SHOCK_DATA_ROOT/run/profile` (default profile is `data/profile.msgpack`).

## Build / Lint / Test Commands

The repository does not define a formal build system, linter config, or test suite in tracked files.
Use the following practical commands for validation.

### Environment smoke checks

- Repository setup:
  - `bash scripts/setup.sh` (creates `.shock.env`, `work/`, and syncs env)
- Sync environment (preferred):
  - `uv sync`
- Check Python version:
  - `python --version`
- Verify key imports:
  - `python -c "import numpy, scipy, matplotlib, h5py, toml, msgpack, tqdm"`

### "Build" (syntax / import sanity)

- Bytecode-compile tracked package modules:
  - `python -m compileall shock`
- Script help checks (fast CLI sanity):
  - `python shock/reduce1d.py --help`
  - `python shock/wavetool.py --help`
  - `python shock/mra.py --help`
  - `python shock/wavefilter.py --help`
  - `python shock/vdist.py --help`
  - `python shock/make_tracer_hdf5.py --help`
  - `python shock/printparam.py --help`

### Lint / format

- No tracked formatter/linter configuration found.
- If linting is needed, prefer non-destructive local checks:
  - `python -m py_compile shock/*.py`
- Do not introduce a formatter/linter config unless explicitly requested.

### Tests

- No tracked `tests/` directory or pytest config found.
- If tests are added later and pytest is used:
  - Run all tests: `pytest`
  - Run single file: `pytest path/to/test_file.py`
  - Run single test: `pytest path/to/test_file.py::test_name`
  - Run single parametrized case: `pytest path/to/test_file.py::test_name[param]`
- For now, validate changes with targeted script execution and small-scope dry runs.

## Common Project Commands

- Reduce data to 1D (analyze only):
  - `python shock/reduce1d.py -j analyze sample/reduce1d-config.toml`
- Reduce + plot:
  - `python shock/reduce1d.py -j analyze,plot sample/reduce1d-config.toml`
- Shock position from reduced data:
  - `python shock/reduce1d.py -j position sample/reduce1d-config.toml`
- 2D wave analysis + plot:
  - `python shock/wavetool.py -j analyze,plot sample/wavetool-config.toml`
- MRA analyze:
  - `python shock/mra.py -j analyze <config.toml>`
- MRA plot:
  - `python shock/mra.py -j plot <config.toml>`
- Wave activity analyze + plot:
  - `python shock/wavefilter.py -j analyze,plot <config.toml>`
- Velocity distribution reduction (MPI-capable script):
  - `python shock/vdist.py -j reduce <config.toml>`

## Coding Style Conventions Observed in Tracked Code

Follow existing style in `shock/*.py` unless asked to refactor broadly.

### Imports

- Standard library imports appear first, then third-party, then local imports.
- `matplotlib` backend is set conditionally in script entry modules:
  - `mpl.use("Agg") if __name__ == "__main__" else None`
- Some modules append `PICNIX_DIR/script` to `sys.path` before importing `picnix`.
- Local package imports often use plain module names (e.g., `import base`, `import utils`) in script files.

### Formatting

- Use 4-space indentation.
- Keep lines readable; long expressions are often split with parentheses.
- Prefer explicit intermediate variables for physics quantities over compressed one-liners.
- Existing code uses both f-strings and `.format`; match local file style.

### Types and APIs

- Type hints are mostly absent in tracked code.
- Prefer preserving current untyped style unless adding hints materially improves safety in touched code.
- Functions generally accept and return `numpy.ndarray`, dict-like config objects, and tuples.
- Avoid unnecessary API shape changes; many scripts are used as CLI tools.

### Naming

- Class names use `CapWords` (`DataReducer`, `SummaryPlotter`, `JobExecutor`).
- Functions/variables use `snake_case`.
- Physics/domain variables use concise conventional symbols (`Bx`, `u0`, `wci`, `mime`, `delh`).
- Dataset keys and HDF5 names are short and stable (`E`, `B`, `Je`, `Ji`, `Feu`, `Fiu`).

### Error handling and exits

- Prefer explicit guards for required files/options:
  - Raise `FileNotFoundError` / `ValueError` for invalid inputs.
  - Use `sys.exit(...)` for CLI-fatal precondition failures.
- In long-running loops, skip already-processed steps when possible.
- Preserve existing behavior where scripts fail fast on missing required config values.

### I/O and data integrity

- HDF5 writes are commonly staged:
  - Create datasets first.
  - Fill per-step data.
  - Write time/step markers last.
- Respect `overwrite` flags; do not silently clobber files.
- Keep dataset shapes/chunks consistent with existing schema.

### Numerical and scientific patterns

- Use vectorized `numpy` operations by default.
- Keep normalization constants explicit and close to where they are used.
- For smoothing/filtering, current code uses `scipy.signal` and `scipy.ndimage` helpers.
- For derived quantities, prefer readability and traceability over micro-optimizations.

### Plotting conventions

- Use Matplotlib directly, usually with explicit `fig`, `axs`, colorbars, and minor tick locators.
- Preserve axis labels and units (often LaTeX-style strings).
- Preserve frame-by-frame update patterns (`plot_new` / `plot_update`) for movie generation.
- Avoid changing colormaps/limits semantics unless requested.

### Parallel / MPI behavior

- `shock/vdist.py` uses MPI (`MPICommExecutor`, collective `Reduce`, `Bcast`).
- Keep root/non-root responsibilities explicit (`self.is_root`).
- Preserve barriers and reduction ordering; correctness is more important than cosmetic refactor.

## Change Management for Agents

- Keep edits minimal and localized.
- Avoid broad style-only rewrites.
- When adding new commands/docs, prefer updating `README.md` and this file together.
- If you introduce tests or lint tooling, document exact commands here (including single-test command examples).

## Quick Pre-PR Validation Checklist

- Run `python -m compileall shock`.
- Run `--help` for any CLI script you touched.
- If logic changed, execute the smallest relevant job path with sample config.
- Confirm no accidental output artifacts are committed (large HDF5/PNG/MP4 files).
