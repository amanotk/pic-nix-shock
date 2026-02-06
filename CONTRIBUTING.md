# Contributing Guide

Thank you for your interest in contributing to `shock2d`.

## Setup

1. Fork the repository.
2. Clone your fork:

   ```bash
   git clone https://github.com/YOUR_USERNAME/pic-nix-shock.git
   cd pic-nix-shock
   ```

3. Install development dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

4. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Branching Strategy

We use a simplified Git Flow model:

- `main`: production-ready code and tagged releases
- `develop`: integration branch
- `feature/*`: new features from `develop`
- `bugfix/*`: bug fixes from `develop`
- `hotfix/*`: urgent fixes from `main`

## Development Workflow

### Feature work

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Bug fixes

```bash
git checkout develop
git pull origin develop
git checkout -b bugfix/your-bugfix-name
```

### Commit style

Use concise intent-first messages such as:

- `Add: support for ...`
- `Fix: correct ...`
- `Docs: update ...`
- `Test: add ...`
- `Refactor: simplify ...`
- `Chore: maintain ...`

## Quality Checks

Before opening a pull request, run:

```bash
pytest tests/ -v
ruff check shock/ tests/
black --check shock/ tests/
pre-commit run --all-files
```

## Pull Request Checklist

1. Update docs as needed.
2. Ensure tests and checks pass.
3. Keep changes focused and reviewable.
4. Request review and address feedback.
