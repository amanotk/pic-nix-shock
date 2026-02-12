#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

if [ -f .shock.env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.shock.env
    set +a
fi

resolve_binary() {
    local requested="$1"
    local fallback_name="$2"
    if [ -n "$requested" ]; then
        if [ -x "$requested" ]; then
            printf "%s" "$requested"
            return 0
        fi
        if command -v "$requested" > /dev/null 2>&1; then
            command -v "$requested"
            return 0
        fi
        return 1
    fi
    if command -v "$fallback_name" > /dev/null 2>&1; then
        command -v "$fallback_name"
        return 0
    fi
    return 1
}

usage() {
    cat <<'EOF'
Usage: scripts/mpi-wavefit.sh [wrapper options] [wavefit args]

Wrapper options:
  -n, --nproc <N>   Number of MPI ranks (default: WAVEFIT_NPROC, SLURM_NTASKS, or 4)
      --dry-run     Print resolved command and exit
  -h, --help        Show this help

Examples:
  scripts/mpi-wavefit.sh -n 4 -j analyze work/ma05-tbn80-run002/wavefit-config.toml
  scripts/mpi-wavefit.sh -n 8 -j analyze,plot --snapshot-index 10 work/ma05-tbn80-run002/wavefit-config.toml
EOF
}

NPROC="${WAVEFIT_NPROC:-${SLURM_NTASKS:-4}}"
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        -n|--nproc|--np)
            if [ "$#" -lt 2 ]; then
                echo "Error: missing value for $1" >&2
                usage
                exit 2
            fi
            NPROC="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [ "$#" -lt 1 ]; then
    echo "Error: missing wavefit arguments" >&2
    usage
    exit 2
fi

if ! [[ "$NPROC" =~ ^[0-9]+$ ]] || [ "$NPROC" -lt 1 ]; then
    echo "Error: nproc must be a positive integer (got '$NPROC')" >&2
    exit 2
fi

if ! command -v uv > /dev/null 2>&1; then
    echo "Error: uv is not available in PATH" >&2
    exit 1
fi

MPIEXEC_BIN=""
if MPIEXEC_BIN="$(resolve_binary "${MPIEXEC:-}" mpiexec)"; then
    :
else
    echo "Error: MPIEXEC is not configured and 'mpiexec' was not found in PATH" >&2
    echo "Set MPIEXEC in .shock.env or export MPIEXEC before running this script." >&2
    exit 1
fi

CMD=("$MPIEXEC_BIN" -n "$NPROC" uv run python -m shock.wavefit "$@")

if [ "$DRY_RUN" -eq 1 ]; then
    printf '[dry-run] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
fi

exec "${CMD[@]}"
