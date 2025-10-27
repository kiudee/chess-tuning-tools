# Repository Guidelines

## Project Structure & Module Organization
- `tune/`: core Python package with CLI entry points (`tune/cli.py`), sampling logic, and database workers.
- `tests/`: pytest suite mirroring package modules; add new tests alongside the feature under test.
- `docs/`, `examples/`, `README.rst`: user-facing documentation and usage samples. Update when behavior changes.
- `.github/workflows/`: CI pipelines; edit these when adding sessions or adjusting supported Python versions.

## Build, Test, and Development Commands
- `uv sync --group dev --extra dist`: install the project with development dependencies and optional dist extras.
- `uv run --group dev pytest`: execute unit tests locally.
- `uv run --group dev nox -s tests-3.13`: run the CI-aligned test session for Python 3.13.
- `uv run --group dev nox -s pre-commit`: lint and format the codebase; pass `install` to set up hooks.
- `uv build`: produce wheel and sdist artifacts; `uv publish` pushes to PyPI when credentials are configured.

## Coding Style & Naming Conventions
- Follow the existing code patterns: 4-space indentation, snake_case for variables and functions, PascalCase for classes.
- Run `ruff format` and `ruff check` via `uv run --group dev nox -s pre-commit` or the individual tools before committing.
- Keep log messages short; prefer structured logging (e.g., `logger.info("Testing %s", value)`).

## Testing Guidelines
- Use `pytest` with descriptive test names like `test_feature_behavior`.
- Mirror module names inside `tests/` (e.g., `tests/test_local.py` covers `tune/local.py`).
- Add regression tests for bug fixes; aim to keep warnings clean unless explicitly tracked.

## Commit & Pull Request Guidelines
- Write commits in the imperative mood (e.g., `Add uv build workflow`), bundling related changes together.
- Every pull request should link to relevant issues, describe functional changes, and note how to reproduce validation steps (tests, lint).
- Include screenshots or logs when touching CLI output or user-visible behavior.
