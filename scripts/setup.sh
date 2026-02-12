#!/bin/bash
set -e

echo "Setting up shock repository..."

# Check if we're in a git repository (works for worktrees too)
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "Error: Not in a git repository"
    exit 1
fi

# Create work directory if it doesn't exist
if [ ! -d work ]; then
    mkdir -p work
    echo "Created work directory"
else
    echo "work directory already exists"
fi

# Copy .shock.env if it doesn't exist (keep user customizations if it does)
if [ ! -f .shock.env ]; then
    cp .shock.env.example .shock.env
    echo "Created .shock.env from example"
else
    echo ".shock.env already exists (keeping your customizations)"
fi

# Load local environment settings if available
if [ -f .shock.env ]; then
    set -a
    # shellcheck disable=SC1091
    . ./.shock.env
    set +a
fi

echo ""
echo "Directory configuration (.shock.env):"
echo "- Set SHOCK_WORK_ROOT (default: ./work)"
echo "- Set SHOCK_DATA_ROOT (default: ./data)"
echo "- Set PICNIX_DIR if you use local PIC-NIX scripts"
echo "- Optional MPI toolchain: MPICC and MPIEXEC"

resolve_binary() {
    requested="$1"
    fallback_name="$2"
    if [ -n "$requested" ]; then
        if [ -x "$requested" ]; then
            printf "%s" "$requested"
            return 0
        fi
        if command -v "$requested" > /dev/null 2>&1; then
            command -v "$requested"
            return 0
        fi
        echo "Warning: configured path not executable: $requested" >&2
        return 1
    fi
    if command -v "$fallback_name" > /dev/null 2>&1; then
        command -v "$fallback_name"
        return 0
    fi
    return 1
}

mpicc_bin=""
mpiexec_bin=""
if mpicc_bin="$(resolve_binary "${MPICC:-}" mpicc)"; then
    :
else
    mpicc_bin=""
fi
if mpiexec_bin="$(resolve_binary "${MPIEXEC:-}" mpiexec)"; then
    :
else
    mpiexec_bin=""
fi

# Sync uv environment
if command -v uv &> /dev/null; then
    echo "Syncing uv environment..."
    uv sync
    echo "uv environment ready"

    if [ -n "$mpicc_bin" ] && [ -n "$mpiexec_bin" ]; then
        echo "Configuring mpi4py with MPICC=$mpicc_bin"
        MPICC="$mpicc_bin" uv run python -m pip install --no-binary=mpi4py --force-reinstall "mpi4py>=3.0"
        echo "MPI ready (MPIEXEC=$mpiexec_bin)"
    else
        echo "MPI toolchain not fully available; setup remains serial-only"
    fi
else
    echo "uv not found - using pip fallback"
    pybin="$(command -v python3 || command -v python || true)"
    if [ -z "$pybin" ]; then
        echo "Error: python is not available for pip fallback"
        exit 1
    fi

    if [ ! -d .venv ]; then
        "$pybin" -m venv .venv
    fi

    if ! .venv/bin/python -m pip --version > /dev/null 2>&1; then
        .venv/bin/python -m ensurepip --upgrade
    fi

    .venv/bin/python -m pip install -e .

    if [ -n "$mpicc_bin" ] && [ -n "$mpiexec_bin" ]; then
        echo "Configuring mpi4py with MPICC=$mpicc_bin"
        MPICC="$mpicc_bin" .venv/bin/python -m pip install --no-binary=mpi4py --force-reinstall "mpi4py>=3.0"
        echo "MPI ready (MPIEXEC=$mpiexec_bin)"
    else
        echo "MPI toolchain not fully available; setup remains serial-only"
    fi

    echo "pip environment ready (.venv)"
fi

echo "Setup complete!"
