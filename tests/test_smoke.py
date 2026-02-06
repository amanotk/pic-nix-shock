"""Smoke tests to verify basic functionality."""

import os
import subprocess
import sys
import types
from pathlib import Path


def _make_stub_picnix(stub_dir: Path):
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "picnix.py").write_text("class Run:\n    pass\n", encoding="utf-8")


def test_imports():
    """Import core modules without runtime execution."""
    sys.modules.setdefault("picnix", types.ModuleType("picnix"))

    import shock  # noqa: F401
    import shock.base  # noqa: F401
    import shock.utils  # noqa: F401
    import shock.reduce1d  # noqa: F401
    import shock.wavetool  # noqa: F401
    import shock.mra  # noqa: F401
    try:
        import shock.vdist  # noqa: F401
    except RuntimeError as exc:
        if "cannot load MPI library" not in str(exc):
            raise
    import shock.printparam  # noqa: F401
    import shock.summary  # noqa: F401


def test_cli_help(root_dir, tmp_path):
    """CLI scripts should print help without crashing."""
    stub_dir = tmp_path / "stub"
    _make_stub_picnix(stub_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(stub_dir), str(root_dir), env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env.setdefault("PICNIX_DIR", str(Path.home() / "raid" / "simulation" / "pic-nix"))

    scripts = [
        root_dir / "shock" / "reduce1d.py",
        root_dir / "shock" / "wavetool.py",
        root_dir / "shock" / "mra.py",
    ]

    try:
        from mpi4py import MPI  # noqa: F401

        scripts.append(root_dir / "shock" / "vdist.py")
    except Exception:
        pass

    for script in scripts:
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(root_dir),
        )
        assert result.returncode == 0, f"{script} --help failed:\n{result.stderr}"
        assert "usage:" in result.stdout.lower()


def test_bytecode_compilation(root_dir):
    """All shock Python files should compile."""
    import compileall

    shock_dir = root_dir / "shock"
    assert compileall.compile_dir(str(shock_dir), force=True, quiet=True)


def test_package_version():
    """Package exports a version string."""
    import shock

    assert isinstance(shock.__version__, str)
