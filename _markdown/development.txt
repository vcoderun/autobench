# Development

Autobench uses `uv` for dependency management and exposes stable repository operations through
the Makefile.

## Environment

```bash
uv sync --extra dev
```

The committed lock file is the reproducible dependency contract used by CI.

## Quality Gates

```bash
make format
make prod
make pre-commit
```

`make prod` runs the test suite, enforces `100%` line and branch coverage, checks formatting and
typing, builds the documentation, validates Python 3.11 through 3.13, and executes the offline
examples end to end.

## Documentation

The site uses Zensical's modern theme while retaining `mkdocs.yml` as the supported migration
configuration format.

```bash
make docs
make docs-serve
```

Pushes to `main` build the site in strict mode. The workflow stores generated files in the
`gh-pages` branch and deploys the same artifact through GitHub Pages Actions.

## Release Artifacts

```bash
make build
```

The build produces a wheel and source distribution under `dist/`. Generated documentation,
benchmark runs, internal planning files, references, and agent instructions are excluded from the
published package.
