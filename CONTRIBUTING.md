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
- `make check`: run `ruff check` and `basedpyright`
- `make tests`: run the test suite
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
4. Run `make check`.
5. Run `make tests`.

## Notes

- Prefer `uv run ...` over ad hoc environment activation.
- Keep core generic and move domain-specific logic into examples or adapters.
- Preserve compatibility with the `src/` layout and the phase plan in `docs/`.
