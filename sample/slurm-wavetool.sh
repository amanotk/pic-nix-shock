#!/bin/bash
#SBATCH -p gpu1
#SBATCH -n 1
#SBATCH -t 180
#SBATCH --mem 64G

# Sample Fugaku Slurm helper for running `shock/wavetool.py` across
# configured runs. Submit it with (for example):
#     sbatch sample/slurm-wavetool.sh --prefix field ma05-tbn80-run002 ma05-tbn80-run003
# Each argument should match a `work/<run>/wavetool-config.toml` entry.
# The script loops over its arguments and runs `python shock/wavetool.py
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

PREFIX="field"
RUNS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prefix)
            if [[ -z "${2:-}" ]]; then
                echo "ERROR: --prefix requires a value" >&2
                echo "Usage: $0 [--prefix <prefix>] <run-name> [<run-name> ...]" >&2
                exit 1
            fi
            PREFIX="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [--prefix <prefix>] <run-name> [<run-name> ...]"
            exit 0
            ;;
        --*)
            echo "ERROR: unknown option '$1'" >&2
            echo "Usage: $0 [--prefix <prefix>] <run-name> [<run-name> ...]" >&2
            exit 1
            ;;
        *)
            RUNS+=("$1")
            shift
            ;;
    esac
done

if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "ERROR: at least one run name argument required" >&2
    echo "Usage: $0 [--prefix <prefix>] <run-name> [<run-name> ...]" >&2
    exit 1
fi

ulimit -n 4096
for RUN in "${RUNS[@]}"; do
    TOML="${SHOCK_WORK_ROOT}/${RUN}/wavetool-config.toml"
    "${PYTHON_CMD[@]}" -m shock.wavetool -j analyze,plot --prefix "${PREFIX}" "${TOML}"
done
