#!/bin/bash
# Test script to verify mpi4py works with the wavefit MPI setup

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$REPO_ROOT"

# Source .shock.env if it exists
if [ -f .shock.env ]; then
    set -a
    . ./.shock.env
    set +a
fi

# Set LD_LIBRARY_PATH for mpi4py (same as mpi-wavefit.sh)
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

MPICC_BIN=""
if MPICC_BIN="$(resolve_binary "${MPICC:-}" mpicc)"; then
    MPI_LIB_PATH="$($MPICC_BIN --showme:link 2>/dev/null | grep -oP -- '-L\K[^ ]+' | head -1)"
    if [ -n "$MPI_LIB_PATH" ] && [ -d "$MPI_LIB_PATH" ]; then
        export LD_LIBRARY_PATH="$MPI_LIB_PATH:$LD_LIBRARY_PATH"
        echo "Set LD_LIBRARY_PATH to: $MPI_LIB_PATH"
    fi
fi

# Resolve mpiexec
MPIEXEC_BIN=""
if MPIEXEC_BIN="$(resolve_binary "${MPIEXEC:-}" mpiexec)"; then
    :
else
    echo "Error: MPIEXEC is not configured and 'mpiexec' was not found" >&2
    exit 1
fi

# Default to 2 ranks if not specified
NPROC="${1:-2}"

echo "Running mpi4py test with $NPROC ranks..."

# Simple MPI test
"$MPIEXEC_BIN" -n "$NPROC" uv run python -c "
from mpi4py import MPI
import sys

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

print(f'Rank {rank}/{size}: Hello from mpi4py!', flush=True)

if rank == 0:
    print(f'Total MPI size: {size}', flush=True)

# Test MPI reduction
data = rank
total = comm.reduce(data, op=MPI.SUM, root=0)
if rank == 0:
    print(f'Sum of ranks: {total} (expected {sum(range(size))})', flush=True)

sys.exit(0)
"

echo "MPI test completed successfully!"
