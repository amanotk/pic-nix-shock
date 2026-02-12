# Test Fixtures

This directory contains small configuration files and synthetic fixtures used by the
pytest suite.

## Files

- `minimal_config.toml`: Minimal TOML configuration for test jobs.
- `__init__.py`: Package marker for fixture imports.
- `wavefit/real_subset.npz`: Compact real-data subset for wavefit regression tests.
- `wavefit/real_subset_metadata.json`: Labels/options for wavefit fixture examples.

## Notes

Fixtures are intentionally lightweight to avoid large binary test assets.
The wavefit fixture stores only a small subset of snapshots and selected
candidate points to keep runtime and repository size reasonable.
