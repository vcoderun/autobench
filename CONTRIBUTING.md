# Contributing

Autobench uses `uv` for dependency management and tool execution.

## Prerequisites

- Python `3.11+`
- `uv`
- `git`

## Clone And Install

```bash
git clone https://github.com/vcoderun/autobench
cd autobench
uv sync --extra dev
```

## Common Commands

- `make format`: run `ruff format`
- `make check`: run `ruff`, `ty`, and `basedpyright`
- `make tests`: run the test suite
- `make docs`: build the Zensical documentation in strict mode
- `make docs-serve`: preview the documentation locally
- `make pre-commit`: run pre-commit hooks across the repository

## Pre-commit

Install hooks with:

```bash
uv run --extra dev pre-commit install
```

Run them manually with:

```bash
uv run --extra dev pre-commit run --all-files
```

## Recommended Workflow

1. Sync dependencies with `uv sync --extra dev`.
2. Make your change.
3. Run `make format`.
4. Run `make prod`.
5. Run `make pre-commit`.

## Notes

- Prefer `uv run ...` over ad hoc environment activation.
- Keep core generic and move domain-specific logic into examples or adapters.
- Preserve compatibility with the `src/` layout and documented public contracts.
