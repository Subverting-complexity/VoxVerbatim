# Contributing to VoxVerbatim

Thank you for your interest in contributing.

## Reporting bugs and requesting features

Open an issue on [GitHub Issues](https://github.com/Subverting-complexity/VoxVerbatim/issues). Include steps to reproduce for bugs, and describe the use case for feature requests.

## Making changes

1. Fork the repository.
2. Create a branch from `main` for your change.
3. Make your changes and add or update tests where applicable.
4. Open a pull request against `main`.

## Development setup

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements-dev.txt   # Windows
# or: .venv/bin/pip install -r requirements-dev.txt  # Linux / macOS
```

## Running tests

```bash
pytest
```

The tests run without a visible desktop. On Linux or in CI, set
`QT_QPA_PLATFORM=offscreen` so Qt does not look for a display server.
The test configuration already does this in `conftest.py`, but the
environment variable is the safest way to ensure it takes effect early
enough.

## Code style

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting,
configured in `pyproject.toml`:

- Line length: 100
- Target: Python 3.11

Run `ruff check .` before opening a pull request. CI runs the same
check. `bash scripts/quality-gate.sh` runs Ruff and the tests together.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
