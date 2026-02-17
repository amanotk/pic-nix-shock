#!/bin/bash
#SBATCH -p gpu1
#SBATCH -n 1
#SBATCH -t 180
#SBATCH --mem 64G

# Sample Fugaku Slurm helper for running `shock/reduce1d.py` across
# configured runs. Submit it with (for example):
#     sbatch sample/slurm-reduce1d.sh ma05-tbn80-run002 ma05-tbn80-run003
# Each argument should match a `work/<run>/reduce1d-config.toml` entry.
# The script loops over its arguments and runs `python shock/reduce1d.py
# -j analyze,plot` once per run in submission order, so a single job can
# cover multiple datasets without re-submitting.

export PICNIX_DIR=${HOME}/pic-nix

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/.shock.env" ]]; then
    REPO_ROOT="${SLURM_SUBMIT_DIR}"
else
    SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
    CANDIDATE_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
    if [[ -f "${CANDIDATE_ROOT}/.shock.env" ]]; then
        REPO_ROOT="${CANDIDATE_ROOT}"
    else
        echo "ERROR: Could not locate repo root (.shock.env not found)." >&2
        echo "       Submit from repo root or export SHOCK_ENV_FILE explicitly." >&2
        exit 1
    fi
fi

cd "${REPO_ROOT}" || exit 1

. "${REPO_ROOT}/.shock.env"

if command -v uv >/dev/null 2>&1; then
    PYTHON_CMD=(uv run python)
else
    PYTHON_CMD=(python)
fi

if [[ $# -eq 0 ]]; then
    echo "ERROR: at least one run name argument required" >&2
    echo "Usage: $0 <run-name> [<run-name> ...]" >&2
    exit 1
fi

ulimit -n 4096
for RUN in "$@"; do
    TOML="${SHOCK_WORK_ROOT}/${RUN}/reduce1d-config.toml"
    "${PYTHON_CMD[@]}" -m shock.reduce1d -j analyze,plot "${TOML}"
done
