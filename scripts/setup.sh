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

echo ""
echo "Directory configuration (.shock.env):"
echo "- Set SHOCK_WORK_ROOT (default: work)"
echo "- Set SHOCK_DATA_ROOT to your simulation directory"
echo "- Set PICNIX_DIR if you use local PIC-NIX scripts"

# Sync uv environment
if command -v uv &> /dev/null; then
    echo "Syncing uv environment..."
    uv sync
    echo "uv environment ready"
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
    echo "pip environment ready (.venv)"
fi

echo "Setup complete!"
