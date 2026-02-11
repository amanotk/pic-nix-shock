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

. .shock.env

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
    python ./shock/wavetool.py -j analyze,plot --prefix "${PREFIX}" "${TOML}"
done
