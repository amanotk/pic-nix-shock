# Wavefit Interactive Tuning Playbook

This document defines a reusable workflow for interactive quality-tuning
sessions of `shock.wavefit` without depending on machine-specific paths.

## Session Contract Template

Copy and fill this at the start of a chat/session.

```text
We are doing an interactive wavefit tuning session.

Context:
- Config file: <path/to/wavefit-config.toml>
- Wave data: <path/to/wavefilter.h5>
- Focus snapshot indices: [<i0>, <i1>, ...]

Workflow rules:
1) Make one parameter/code change at a time.
2) After each run, report:
   - nfit / success / good
   - nrmse stats (min/median/max)
   - output plot paths (envelope+candidates, quicklooks)
3) Do not commit unless explicitly requested.
4) Keep output filenames stable (overwrite same names when possible).

Goal:
- <describe current tuning objective>
```

## Standard Run Loop

For each candidate setting:

1. Run one snapshot in debug mode.
2. Generate/update envelope map with selected candidates.
3. Review top quicklook panels.
4. Record metrics and compare with the previous run.

Prefer this order for tuning candidate density:

1. `candidate_distance` (absolute spacing in x/y units)
2. `envelope_threshold` (`|B_wave| / B0` threshold)
3. `envelope_smooth_sigma`

## Current Wavefit Notes

- Candidate spacing uses `candidate_distance` (absolute units).
- Fit patch radius is fixed to `3 * sigma`.
- `y` boundary handling uses periodic unwrapped patch coordinates during fit.
- Quicklook titles include `nrmse_bal`, `redchi`, `lambda`, and `theta`.

## Suggested Local Notes (Untracked)

Keep machine-specific context in a local file under `work/`, for example:

- `work/<run>/wavefit-tuning-notes.md`

Include concrete paths, selected snapshots, and short findings there.
