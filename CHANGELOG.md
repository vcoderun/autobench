# Changelog

## Unreleased

- Added verified OpenAI Python 2.53 compatibility and a dedicated CI matrix target while retaining
  OpenAI 2.52 as the minimum supported release.
- Made behavioral asset definitions full-capture by default while retaining metadata-first runtime
  evidence, and added an independent `asset_default_level` privacy control.
- Externalized versioned asset snapshots and diffs into one transaction-safe, content-addressed
  `artifacts/asset-content.sqlite3` registry with typed manifest references and loaders.
- Reorganized and expanded the documentation around installation, architecture, first-run workflow,
  practical AI and non-AI use cases, current Python/CLI APIs, automatic asset discovery, and
  troubleshooting.

## 0.2.0

- Migrated the documentation build and GitHub Pages deployment to Zensical.
- Expanded the documentation to cover Autobench's complete public feature set.
- Replaced top-level navigation tabs with a persistent vertical documentation menu.
- Added `llms.txt`, `llms-full.txt`, and a Copy as Markdown action to every documentation page.
- Added the Autobench Instrumentation Protocol (ABP), immutable traces, capture policy, semantic
  source mapping, accounting-safe extraction, and partial-trace replay.
- Added native Pydantic AI, OpenAI Python, OpenAI Agents, and HTTPX instrumentors with typed Python
  and YAML configuration, compatibility diagnostics, privacy-first capture, and streaming lifecycle.
- Added benchmark-scoped `instrument_all()` discovery with exclusions, strict mode, explicit
  override precedence, duplicate prevention, and persisted skip diagnostics.
- Added `autobench instrumentation doctor` and `autobench instrumentation trace` Rich diagnostics.
- Added real offline ABP concurrency, OpenAI streaming, OpenAI Agents, and extraction examples.
- Added a live Pydantic AI/OpenRouter example that uses `instrument_all()` to collect layered
  framework, client, and HTTPX transport evidence without manual task telemetry.
- Added automatic behavioral asset discovery, versioning, source/effective lineage, capability
  scopes, capture-aware persistence, and exact replay for Pydantic AI, OpenAI, OpenAI Agents, and
  custom method instrumentors.
- Extended the supported quality matrix through Python 3.14 and added built-wheel/no-extras smoke
  tests plus target-library compatibility CI.

## 0.1.0

- Added the YAML-first benchmark DSL and typed Python runtime.
- Added semantic evidence, spans, artifacts, scoring, derivation, policies, and comparisons.
- Added immutable recording, replay, Rich terminal reporting, and file exports.
- Added asset tracking with persistent version history and diffs.
- Added offline minimal through advanced examples and a live CodeMode integration.
- Added Python 3.11-3.13 CI, strict documentation builds, and 100% line/branch coverage gates.
