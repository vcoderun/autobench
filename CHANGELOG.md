# Changelog

## Unreleased

- Fixed case-paired comparisons of boolean metrics: pass/fail scores such as `result.success` were
  excluded from wins, ties, and losses, so their paired counts were always empty.
- Added exact sign (McNemar) tests to every paired comparison metric, plus the paired 2x2 table and
  Newcombe's interval for the success-rate difference of boolean metrics, in Rich and Markdown
  reports.
- Added named `report.compare` entries with case `tags`, and pre-declared comparison hypotheses that
  are Holm-adjusted within each family and reported as supported, not supported, or untestable.
- Added `autobench compare --tag` and `--metric SEMANTIC:AGGREGATE`.
- Fixed recording of assets that change within one run: every version a run used is persisted with
  its snapshot and diff, where only the latest version was written before.
- Added `AssetCandidate.parent_version` for producers that know a non-chronological parent; lineage
  and diffs follow the declared parent.
- Added asset trajectories: `build_asset_trajectory` and `report.trajectories` compare each recorded
  version of an asset with its parent version on the cases both ran, with the same paired tests as
  variant comparisons. Step p-values are also Holm-adjusted across the trajectory, `variants`
  restricts the runs, and several records of one benchmark can be pooled, for example one record
  per generation of a self-modifying agent. `compare_runs` compares any two caller-selected run
  groups.
- Added asset version comparisons: `compare_asset_versions`, `report.compare` entries with `asset`,
  and `autobench compare --asset --variant` compare two versions or the `root` and `leaf` of a
  lineage, and can carry Holm-adjusted hypotheses. Unresolvable selectors are explained in the
  comparison's diagnostics and make its hypothesis untestable.
- Added `autobench trajectory RUN_DIR...`, which lists the recorded assets or renders trajectories
  across one or more records.
- Added 95% BCa bootstrap intervals for the mean paired difference of numeric metrics, beside the
  Newcombe interval of boolean metrics, in comparisons, trajectories, and reports.
- A case with more than one run on either side of a comparison, as when several variants share an
  asset version, is never paired; it is counted as a case with several runs. Factors that differ
  between runs of one side are reported as mixed and mark the comparison confounded.
- Fixed spec YAML views dropping the name, case tags, and hypothesis of comparison entries.
- Extended Pydantic AI instrumentation support to `>=2.22,<2.67`, with a 2.66.0 CI target.
- Updated the locked development dependencies to patched releases for every open security
  advisory. The development and matrix environments now use Pydantic AI 2.66.0 and OpenAI 3.46.0,
  with an OpenAI 3.46.0 CI target; older SDK versions stay covered by isolated CI targets.
- Split case-paired comparison into `autobench.reports.comparison` and metric value resolution into
  `autobench.reports.values`; `autobench.reports.reporting` keeps its public names.

## 0.4.0 - 2026-10-10

- Preserve skipped GEPA evaluation events without closing an evaluation span that never started
  or has already completed.
- Keep legacy HTTPX transport coverage type-correct with OpenAI 3's HTTPX2 annotations, and
  document which transport layer the HTTPX instrumentor captures.
- Added native pydantic-gepa instrumentation with typed event subscriptions, keyed ABP optimizer
  lifecycles, Optimize Anything engine/composition evidence, semantic budgets and scores,
  candidate/component asset lineage, replay-safe projections, and dedicated Rich reports.
- Added credential-free standard GEPA, Optimize Anything Omni, multi-component, and checkpoint
  resume benchmarks plus a credential-gated Pydantic AI/OpenRouter layering example.
- Added incremental crash-recoverable recording, cooperative cancellation checkpoints, typed
  progress dispatch, file-backed streaming artifacts, and cross-invocation execution correlation.
- Added typed pre-run generated dataset preparation with review states, determinism claims,
  provenance manifests, content hashes, safe complete publication, and explicit incomplete
  sidecars.
- Added an optional ABP-to-OTLP HTTP/protobuf exporter with immutable record replay, hierarchy and
  identity preservation, privacy-first content omission, Rich CLI output, and offline examples.
- Updated the OpenAI integration to SDK 3 with matching dependency and discovery ranges,
  CI targets for 3.0, 3.6, and 3.27, and an OpenAI Agents 0.22 compatibility target.
- Require the PyPI release of pydantic-gepa 0.1.1, including the
  candidate-reselection lifecycle and released-GEPA compatibility fixes.
- Extended native Pydantic AI instrumentation compatibility through 2.36 and added a dedicated
  compatibility matrix target while retaining Pydantic AI 2.22 as the minimum supported release.
- Made behavioral asset definitions full-capture by default while retaining metadata-first runtime
  evidence, and added an independent `asset_default_level` privacy control.
- Externalized versioned asset snapshots and diffs into one transaction-safe, content-addressed
  `artifacts/asset-content.sqlite3` registry with typed manifest references and loaders.
- Reorganized and expanded the documentation around installation, architecture, first-run workflow,
  practical AI and non-AI use cases, current Python/CLI APIs, automatic asset discovery, and
  troubleshooting.
- Made capture diagnostics best-effort after the per-trace diagnostic limit is reached so
  instrumentation saturation cannot interrupt the observed workload.
- Made per-run metric reports prefer aggregate measurements over earlier progress snapshots.
- Marked terminal pydantic-gepa budget snapshots as aggregate measurements so reports select final
  usage and remaining-budget values instead of intermediate progress updates.

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
