## Purpose

Write production-grade Python that is clean, explicit, maintainable, strongly typed, and consistent with a strict, reviewable codebase style.

The codebase should prioritize:

- clarity over cleverness
- composition over inheritance
- explicit contracts over implicit behavior
- type safety over convenience shortcuts
- small, cohesive modules over sprawling abstractions
- stable public APIs over incidental internal exposure

Always generate code that is compatible with:

- `ruff`
- `ty`
- `basedpyright`

Do not silence lint or type errors unless the reason is real, localized, and unavoidable.

---

## Core Principles

- Follow Python best practices and relevant PEPs.
- Prefer readability, predictability, and long-term maintainability.
- Keep public APIs small and intentional.
- Avoid hidden side effects, magic state, and ambiguous behavior.
- Favor explicit data flow and explicit ownership of responsibilities.
- Use composition to assemble behavior instead of creating overly specialized classes.
- Respect single responsibility at the class, function, and module level.
- Do not over-engineer simple logic.
- Treat internal consistency as a feature.
- Write code that looks like it belongs to a high-quality, manually-designed codebase, even when it is AI-assisted.
- Do not claim to follow repo conventions unless the repo context is actually visible.
- If repo context is unavailable, fall back to the standards in this file and prefer conservative, explicit patterns over speculative imitation.

---

## Repo Context Discovery

Before adding new abstractions or copying repo style, inspect the visible repository structure and follow what is actually present.

Look for:

- package layout
- naming conventions
- import style
- public API boundaries
- typing idioms
- dataclass or pydantic usage
- testing patterns
- tool configuration files
- existing extension seams such as protocols, ABCs, wrappers, registries, hooks, or graph nodes
- documentation structure
- source code URL and project metadata in `pyproject.toml`

If repo context is not available:

- do not invent repo-specific rules
- do not say "match existing repo conventions"
- instead apply this AGENTS.md as the default house style

Prefer concrete seams that commonly appear in strict Python repos, for example:

- private implementation modules such as `_utils.py`, `_schema.py`, `_agent_graph.py`
- explicit public re-exports from `__init__.py`
- protocol-based interfaces such as `Model`, `Toolset`, `Serializer`, `Transport`, or `CacheBackend`
- typed settings containers instead of loose `dict[str, Any]`
- discriminated unions for tagged message or event variants

Example private/public split:

```python
# package/_serializer.py
from __future__ import annotations as _annotations

class JsonSerializer:
    ...
````

```python
# package/__init__.py
from ._serializer import JsonSerializer

__all__ = ('JsonSerializer',)
```

Example interface shape:

```python
from __future__ import annotations as _annotations

from typing import Protocol

class ModelBackend(Protocol):
    async def request(self, prompt: str) -> str: ...
```

When extending a repo, prefer adding another implementation behind an existing interface over creating a parallel abstraction stack.

---

## Module and Package Conventions

### Module Preamble

* Every Python module should start with:

  ```python
  from __future__ import annotations as _annotations
  ```

* Use postponed evaluation of annotations consistently across the codebase.

* Prefer a uniform module preamble over mixing annotation strategies.

### Public vs Internal API

* Public modules should define an explicit `__all__` tuple.
* Treat `__all__` as the contract for what the module intentionally exports.
* Internal implementation details should remain private.
* Prefer `_private.py`-style modules and selective re-export from package `__init__.py` files instead of exposing internals directly.
* Use `_` prefixes for private symbols and private modules.

### Module Structure

* Organize code by responsibility, not by arbitrary fragmentation.
* Avoid spreading one concept across many tiny modules without a strong reason.
* If a module grows beyond roughly 1000 lines, split it into helper modules with clearly differentiated responsibilities.
* Group related helpers under cohesive subpackages.
* Keep model-related helpers inside dedicated model namespaces.

Example:

* prefer `module/models/model_registry.py`
* avoid flat structures when a domain already has a clear home

### Circular Import Avoidance

* Use `if TYPE_CHECKING:` imports for type-only dependencies.
* Avoid runtime circular imports by separating typing imports from runtime imports.
* Prefer better module boundaries first; use `TYPE_CHECKING` as a tool, not as a crutch.

---

## Design Patterns

Use design patterns when they make the design clearer, not because they are fashionable.

### Composition First

* Prefer composition over inheritance for assembling behavior.
* Prefer wrapping collaborating components instead of introducing specialized subclasses.

Example:

* prefer `Compiler(Runtime())`
* avoid `CompilerWithFixedRuntime()`

### Decorator Pattern

Use the decorator pattern when responsibilities can be layered cleanly.

* Split distinct responsibilities into composable wrappers.
* Chain behavior through explicit decorators when each layer adds a meaningful concern.
* Use this for cross-cutting or staged behavior, not for trivial wrappers.

Guideline:

* simple ownership like `cls(db)` is acceptable
* when responsibilities become more involved, prefer explicit composition and wrapping

### Factory Pattern

Use factory methods for clear and simple object construction.

* Prefer factory methods when object creation requires validation, normalization, or dependency wiring.
* Keep factories simple and intention-revealing.
* Do not hide complex application logic inside factories.

### Alternative Constructors

* Use `from_*()` constructors for meaningful construction paths.
* Prefer named constructors over overloading `__init__`.
* Keep `__init__` simple, direct, and unsurprising.
* Prefer a fluent or builder-style API only when it truly improves clarity.

### Singleton-Like Behavior

* Do not introduce global singleton patterns casually.
* If controlled instance reuse is needed, prefer explicit `from_*()` or cached construction paths.
* Singleton behavior must be deliberate, testable, and easy to replace.

### Adapter Pattern

Use adapters to bridge incompatible interfaces.

* Prefer protocol-based adapters with `typing.Protocol`.
* Keep adapters thin and focused on interface translation.
* Do not leak implementation-specific details through the adapter boundary.

### Iterator Pattern

Use iterator-based designs for custom data traversal, streaming, and REPL-like data flows.

* Implement iterators when the abstraction is naturally sequential.
* Prefer lazy iteration when it improves memory usage or streaming ergonomics.
* Keep iterator contracts explicit and predictable.

### Strategy and Hook Systems

* Prefer strategy-style extension points for optional behavior.
* Use before, after, or wrap hooks only when they represent real extension seams.
* Keep hooks orthogonal and composable.
* Centralize validation and normalization in one layer instead of duplicating it across subclasses.

### Template Method Pattern

* Use abstract base classes when a shared algorithm skeleton exists and subclasses should override only designated steps.
* Keep the template flow centralized and subclass responsibilities narrow.
* Do not duplicate cross-cutting request preparation, validation, or normalization logic across subclasses.

### Other Patterns

Use other design patterns only when they materially improve clarity, extensibility, or correctness.

Do not introduce patterns that increase indirection without clear payoff.

---

## Type System Rules

### General

* Always write fully typed Python.
* Every public function, method, class attribute, and module-level constant should have explicit types where relevant.
* Prefer precise types over broad ones.
* Keep type annotations readable.
* Treat type annotations as part of the API contract, not as optional metadata.

### Built-in Generics

Use built-in generics whenever supported.

* use `list[str]` instead of `List[str]`
* use `dict[str, int]` instead of `Dict[str, int]`
* use `set[str]` instead of `Set[str]`

Do not import legacy container aliases from `typing` unless compatibility truly requires it.

### `typing_extensions` Preference

* Prefer `typing_extensions` for backported or version-sensitive typing features.
* Use `typing_extensions.TypedDict` instead of `typing.TypedDict`.
* Use `typing_extensions.assert_never` instead of `typing.assert_never`.
* Prefer consistency across supported Python versions over mixing import locations.

### Advanced Typing

Use advanced typing constructs when they improve correctness or API precision.

Use when needed:

* `TypeVar`
* `TypeAlias`
* `TypeAliasType`
* `ParamSpec`
* `TypeVarTuple`
* `Protocol`
* `ABC`
* `Self`
* `Literal`
* `Final`
* `Never`
* `overload`
* `TypeGuard`
* `TypeIs`
* `Concatenate`

Do not use advanced typing as decoration. Use it only when it clarifies behavior or constrains APIs meaningfully.

### Type Variables and Variance

* Use variance intentionally.
* Dependency-like input type variables should usually be contravariant.
* Output and result type variables should usually be covariant.
* Use `default=` on type variables when it materially improves ergonomics and the codebase supports it.
* Do not introduce generic parameters that do not improve correctness.

### Type Aliases

* Use `TypeAlias` for readable names around complex types.
* Use `TypeAliasType` when opaque or recursive aliases improve readability and IDE presentation.
* Prefer a named alias over repeating a large union or callable signature.

### Protocols and Abstract Base Classes

* Prefer `Protocol` for structural contracts.
* Use `ABC` when runtime-enforced inheritance is important.
* Mark abstract members with `@abstractmethod`.
* Raise `NotImplementedError` only in real abstract boundaries, not as a substitute for unfinished design.

### Final APIs

* Use `@final` for classes and methods that are not intended to be extended or overridden.
* Use `Final` for constants and immutable design-level values.

### Overloads

Use `@overload` for APIs with truly different call signatures or return behaviors.

* Make overloads reflect real behavioral differences.
* Keep implementation bodies aligned with overload contracts.
* Do not add overloads for superficial variations.
* Use overloads when return type depends on input shape or argument values.

### Narrowing

* Use `TypeGuard` and `TypeIs` for runtime narrowing when it improves downstream type precision.
* Prefer explicit narrowing helpers over repeated casts.
* Use `assert_never` for exhaustiveness when all variants must be handled.

### Discriminated Unions

* Prefer `Literal` discriminator fields such as `kind`, `type`, or `part_kind`.
* Use discriminated unions when modeling tagged variants.
* Keep discriminator names stable and explicit.
* Prefer explicit tagged unions over loose dict shapes.

---

## Function and API Design

* Prefer small, focused functions.
* Use descriptive names that communicate intent.
* Keep parameter lists compact and explicit.
* Prefer keyword-only arguments for functions with several optional parameters.
* Avoid boolean traps in public APIs.
* Return rich, typed values instead of ambiguous tuples or loosely shaped dictionaries.
* Do not return different unrelated types from the same function unless overloads make that contract explicit.
* Prefer dedicated settings objects or typed dicts over broad `**kwargs` bags.
* If backward compatibility requires deprecated kwargs, capture them intentionally and warn explicitly.

### Higher-Order Functions

* Use `ParamSpec` and `Concatenate` when modeling callables that preserve or prepend parameters.
* Do not erase callable signatures with overly broad `Callable[..., Any]` unless truly unavoidable.

### Settings and Flexible Structures

* Prefer `TypedDict` for flexible dict-shaped settings that are intentionally partial.
* Use `total=False` where partial configuration is expected.
* Prefer typed dict merging over untyped config bags.

---

## Classes and State

* Keep classes small and responsibility-driven.
* Prefer immutable or minimally mutable state where practical.
* Keep state transitions explicit.
* Avoid classes that act as vague service containers.
* Inject dependencies explicitly through constructors or named factories.
* Do not hide global dependencies behind import-time state.

### Dataclass Style

* Prefer `@dataclass(kw_only=True)` for data-heavy structures.
* Prefer keyword-only construction for readability and API safety.
* Prefer `frozen=True` when mutation is not required.
* Prefer `dataclasses.replace()` over in-place mutation when evolving dataclass state.
* Use `@dataclass(init=False)` only when you need dataclass benefits but require tightly controlled construction.
* Keep custom `__init__` methods simple and validation-oriented.

### Representations

* Keep `__repr__` outputs concise and meaningful.
* Prefer reprs that highlight non-default or state-relevant fields instead of dumping everything blindly.
* Optimize debug readability, not exhaustiveness.

### Sentinel and Option-Like Values

* Distinguish carefully between:

  * no value provided
  * explicit `None`
  * real value present

* Use a dedicated sentinel when `None` is a valid value.

* Prefer explicit sentinel-based APIs over ambiguous `None` semantics.

* Use option-like wrappers when the domain needs to distinguish "unset" from "set to None".

### Cached Computation

* Use `cached_property` for expensive, pure, lazily-computed values.
* Do not cache values with surprising side effects or unclear invalidation semantics.

---

## Async, Concurrency, and Runtime Context

### Async Boundaries

* Keep async and sync boundaries explicit.
* When sync code must run in async paths, prefer sending it to a thread pool via a dedicated helper.
* Avoid blocking the event loop with CPU-bound or sync I/O work.

### Lifecycle Management

* Prefer `@asynccontextmanager` for async lifecycle and cleanup flows.
* Use context managers when resources require deterministic setup and teardown.
* Ensure cancellation and cleanup paths are explicit and testable.

### Context-Local Overrides

* Use `contextvars.ContextVar` for task-safe runtime overrides, request-local state, or scoped execution flags.
* Do not use mutable globals for async-sensitive state.

### Streaming and Iteration

* Prefer explicit async iterator wrappers when buffering, peeking, debouncing, or grouping stream items.
* Keep streaming utilities single-purpose and predictable.

---

## Error Handling

* Fail loudly and clearly.
* Raise specific exceptions.
* Do not swallow exceptions without a good reason.
* Preserve useful error context.
* Use custom exception types when they clarify domain failures.

### Exception Design

* Build semantic exception hierarchies when the domain benefits from them.

* Distinguish:

  * user misuse
  * runtime failure
  * transport failure
  * validation failure
  * retry signal
  * deferred or approval-required control flow

* Exception-as-signal patterns are acceptable only when they are a deliberate part of the public control-flow design.

* Do not use generic exceptions where the caller needs structured meaning.

### Deprecation

* Mark deprecated APIs explicitly.
* Prefer `@deprecated` plus targeted warnings for old parameters or old call styles.
* If legacy kwargs are temporarily supported, isolate them and emit a clear `DeprecationWarning`.
* Comment future deprecation intent only when it captures a real pending decision or migration path.

---

## Comments Policy

Comments must be sparse, intentional, and high value.

* Do not add comments that merely restate what the code already says.
* Do not place a comment on every line or every small block just to explain obvious behavior.
* Prefer self-explanatory naming and clear structure over explanatory inline comments.
* Add comments only when they provide information the code itself cannot communicate cleanly.

Good reasons to add comments:

* non-obvious design decisions
* temporary workarounds or intentionally unusual implementations
* behavior constrained by external systems or library quirks
* implementation details that are likely to change in the future
* migration notes, compatibility notes, or deprecation planning
* warnings about tricky edge cases that future edits could easily break

### Multi-line Notes

Use multi-line comments only when a longer note is genuinely useful.

Examples of valid multi-line notes:

* a temporary workaround with conditions for future removal
* a note that a certain path should be deprecated if a future decision is made
* a note documenting an unstable integration boundary
* a note explaining why a less-obvious implementation was intentionally chosen

Comment goal:

* explain **why**
* document **risk**
* capture **future decision points**

Do not use comments to narrate **what** the code is doing unless that behavior is genuinely non-obvious.

---

## Style and Readability

* Prefer explicit control flow over compressed one-liners.
* Avoid unnecessary metaprogramming.
* Avoid dynamic attribute creation unless the design truly requires it.
* Keep docstrings concise and useful.
* Add comments only when intent is not obvious from the code itself.
* Do not restate the code in comments.
* Prefer single quotes unless the codebase or tooling requires otherwise.
* Keep line length and formatting aligned with repo tooling instead of personal taste.

### Docstrings

* Use Google-style docstrings unless the repository explicitly uses another style.
* Public APIs should have concise, useful docstrings where they add value.
* Keep docstrings aligned with actual behavior and current types.
* If docstrings feed downstream tooling, schema generation, or tool descriptions, treat them as contract-bearing text.
* Do not write decorative docstrings.

---

## Imports and Dependencies

* Keep imports clean and minimal.
* Avoid circular dependencies through better module boundaries.
* Import only what is used.
* Prefer stable abstractions over reaching into internal modules.
* Do not introduce dependencies for trivial problems.

### Dependency Boundaries

* Keep the core package as dependency-light as practical.
* Prefer optional integrations and clear extras over forcing all users to install all providers.
* Separate reusable infrastructure from higher-level orchestration concerns when it materially improves reuse.

### Package Manager and Build Backend Policy

* Use `uv` as the primary and preferred package manager for all dependency management, environment management, tool execution, and script invocation.

* Prefer `uv` in every workflow:

  * dependency installation
  * lockfile management
  * virtual environment creation
  * running linters, type checkers, tests, and scripts
  * adding, removing, or updating packages

* If `uv` is unavailable, use `pip` as the only fallback.

* Do not use other package managers or workflow tools such as:

  * `poetry`
  * `pdm`
  * `hatch` as a package manager
  * `pip-tools`
  * `pipenv`
  * `conda`
  * `pixi`

* If a build backend is needed, prefer `hatchling`.

* Do not introduce alternative build backends unless the repo already requires one and changing it is out of scope.

---

## Formatting, Linting, and Type Validation Workflow

### Config Discovery First

Before running any formatter, linter, or type checker, discover and honor the repo’s existing configuration.

Search in this order:

* for Ruff:

  * `.ruff.toml`
  * `ruff.toml`
  * `pyproject.toml` with `[tool.ruff]`

* for ty:

  * `pyproject.toml` with `[tool.ty]`
  * `ty.toml` only if `[tool.ty]` is absent

* for basedpyright:

  * `pyrightconfig.json`
  * `pyproject.toml` with `[tool.basedpyright]`
  * `pyproject.toml` with `[tool.pyright]` only for compatibility with an existing repo

Prefer `pyproject.toml` as the single source of truth whenever possible.

Do not assume tool defaults if project config exists.

### ty Policy

* Prefer configuring `ty` in `pyproject.toml` under `[tool.ty]`.
* Use `ty.toml` only if the repo already uses it.
* Do not invent a strictness level for `ty`.
* Do not assume hidden defaults are part of the project style.
* Read the repo’s existing `ty` configuration first.
* If the repo’s CI, scripts, or docs define a canonical `ty` invocation, use that exact invocation.
* If `ty` configuration is absent, run `uv run ty check` from the repo root and treat the repo root as the project boundary.
* Do not scope `ty` to guessed subdirectories unless the repo already does so.
* Do not add ad hoc CLI overrides to simulate a stricter or looser mode.
* If the project later adds explicit `ty` settings, treat those settings as authoritative.

### basedpyright Policy

* Prefer configuring basedpyright in `pyproject.toml` under `[tool.basedpyright]`.
* Use `pyrightconfig.json` if the repo already uses it or if that is the visible project standard.
* Use `[tool.pyright]` only for compatibility with an existing repo that has not migrated.
* Do not invent a strictness level for basedpyright.
* Do not assume hidden defaults are part of the project style.
* Read the repo’s existing basedpyright or pyright configuration first.
* If the repo’s CI, scripts, or docs define a canonical basedpyright invocation, use that exact invocation.
* If basedpyright configuration is absent, run `uv run basedpyright` from the repo root with no additional flags and treat the tool’s own defaults as the project baseline.
* Do not scope basedpyright to guessed subdirectories unless the repo already does so.
* Do not add ad hoc CLI overrides to simulate a stricter or looser mode.
* If the project later adds explicit basedpyright settings, treat those settings as authoritative.

### Invocation Rules

* Run tools from the repo root whenever possible.
* Prefer `uv run` for executing formatters, linters, type checkers, tests, and project scripts.
* Prefer `uv add`, `uv remove`, and related `uv` commands for dependency changes.
* If `uv` is unavailable, use `pip` as the only fallback.
* Do not use Poetry, PDM, Pipenv, Conda, Pixi, or other package managers.
* If the repo already defines canonical commands, still prefer invoking them through `uv` when possible.
* Do not invent alternative package-manager workflows.

### Formatting During Edits

* After every batch edit, run the repo’s canonical format command through `uv` when possible.
* If no wrapper command exists, run Ruff formatting through `uv` from the repo root.
* If `uv` is unavailable, use the direct tool invocation inside the active environment.
* Do not allow large unformatted diffs to accumulate.

### Validation After Major Changes

After major changes, run the repo’s canonical validation commands through `uv` when possible.

If the repo does not define wrappers, run all of the following from the repo root:

* `uv run ruff check`
* `uv run ty check`
* `uv run basedpyright`

If `uv` is unavailable, use the only allowed fallback:

* `ruff check`
* `ty check`
* `basedpyright`

Additional rules:

* for ty, prefer repo-root execution unless the repo explicitly scopes it differently
* prefer `[tool.ty]` in `pyproject.toml` over `ty.toml` when both are plausible
* for basedpyright, prefer `[tool.basedpyright]` in `pyproject.toml`, then `pyrightconfig.json`, then `[tool.pyright]` only for compatibility
* do not pass ad hoc CLI overrides unless the task specifically requires them
* do not create temporary config files to force success
* do not guess or simulate a strictness mode that is not present in repo config

### Lint and Type Standards

* Keep the code compatible with strict type checking.
* Prefer modernized syntax that satisfies pyupgrade-style rules.
* Keep import ordering stable and tool-friendly.
* Respect repo-level lint rules for quotes, docstrings, complexity, banned imports, and file inclusions.
* Do not add ignore comments, blanket exclusions, or type escapes just to force green checks unless the reason is concrete and justified.

### Quality Gate

A major edit is not complete unless:

* formatting has been applied with repo-aware config
* lint checks pass with repo-aware config
* type checks pass with repo-aware config

---

## Agent / Agentic Task Specifics

When the repository contains agents, model backends, tool systems, message graphs, or other agentic workflows, preserve the agent architecture instead of flattening it into generic helper code.

### Agent Architecture Rules

* Prefer explicit agent boundaries:

  * model backend
  * tool or toolset layer
  * run context
  * message or event types
  * state or graph nodes
  * output validation
  * retry, approval, and deferred execution boundaries

* Keep model-facing logic separate from pure domain logic.

* Keep provider-specific code behind interfaces or adapters.

* Do not hardcode a provider into shared agent abstractions.

* Prefer typed request and response parts over loose dict payloads.

* Preserve discriminators such as `kind`, `part_kind`, `role`, or similar tagged fields when modeling agent messages or events.

* Prefer semantic exception types for retry, skip, approval, deferred execution, validation failure, and model behavior errors.

### Tooling and Tool Definitions

* Treat tools as typed contracts, not arbitrary callables.

* Preserve:

  * explicit tool names
  * descriptions
  * schema generation boundaries
  * argument validation
  * retry semantics
  * approval, timeout, and sequential execution flags when present

* Prefer wrappers, filters, prefixes, or preparers around toolsets instead of duplicating tool registration logic.

* If the repo uses docstrings as tool descriptions or schema input, keep docstrings accurate and contract-bearing.

### Run Context and State

* Keep run context typed and explicit.
* Pass dependencies through typed context objects instead of hidden globals.
* If the repo uses graph nodes, state machines, or step objects, extend that structure instead of replacing it with ad hoc branching.
* Prefer immutable or replace-style state evolution where the existing architecture suggests it.

### Model Abstraction

* Separate:

  * request preparation
  * provider transport
  * response parsing
  * structured output validation
  * profile or capability checks

* Do not duplicate request normalization in every provider implementation if the repo already has a central preparation layer.

* Prefer capability or profile objects over scattered provider conditionals.

### Agent Testing Rules

For agentic repos, tests should validate behavior at the agent boundary rather than private internal choreography.

Prefer testing:

* generated outputs
* tool calls
* retry prompts
* approval or deferred paths
* captured messages
* structured output validation
* context propagation
* graph transitions when graph behavior is public

Prefer:

* project-provided test models and fakes
* deterministic tool or model substitutes
* snapshots for message sequences or structured outputs
* cassette-backed HTTP tests only when provider behavior is genuinely under test

Avoid:

* deep mocks of internal agent plumbing
* brittle assertions on helper call order
* flattening agent tests into generic unit tests that lose the behavioral contract

### Agent Change Checklist

An agent-related edit is not complete unless it preserves or intentionally updates:

* provider abstraction boundaries
* typed tool contracts
* message or event schemas
* structured output guarantees
* retry, approval, and deferred control flow
* repo-native testing style for agents

---

## Markdown and Documentation Policy

Markdown must be clean, restrained, metadata-aware, and useful.

### General Markdown Rules

* Do not add emoji clutter or decorative emoji slop.
* Prefer plain, professional technical writing.
* Use headings, bullets, tables, code blocks, and structured sections only where they improve readability.
* Do not make docs noisy with banners, badges, or filler prose unless the repo already explicitly uses them.
* Prefer high-signal documentation over marketing-style fluff.

### README Scope

* Keep `README.md` focused on the project intro and top-level navigation.

* The README should briefly explain:

  * what the project is
  * who it is for
  * the primary installation paths
  * the shortest getting-started path
  * where to find deeper documentation

* Do not turn the README into the full manual.

* Move detailed sections into dedicated docs pages and reference them from README.

Examples:

* `CLI reference: <full-url-to-docs/CLI.md>`
* `Configuration guide: <full-url-to-docs/CONFIGURATION.md>`
* `API reference: <full-url-to-docs/API.md>`

### Markdown Link Policy

* Prefer full absolute repository links over relative links in generated Markdown when linking to project files and docs.

* Construct documentation and source links from the project's source code URL.

* Do not emit relative links such as:

  * `docs/CLI.md`
  * `../API.md`

* Prefer absolute forms derived from repo metadata, for example:

  * `<source-url>/blob/main/docs/CLI.md`
  * `<source-url>/blob/main/docs/API.md`

* If the default branch is discoverable from repo context, use it.

* If it is not discoverable, prefer the repo's existing convention if visible; otherwise fall back conservatively and avoid inventing fake links.

### Metadata-Driven Documentation
https://deepeval.com/llms.txt
* Do not use placeholder values such as:

  * `yourusername`
  * `your-org`
  * `your-project`
  * `my-package`

* Read project metadata from `pyproject.toml` whenever possible, including:

  * project name
  * package name
  * author or maintainer name when needed
  * repository or source URL
  * optional dependency groups

* Use the source code URL to construct doc and reference links.

* If metadata is unavailable, prefer omitting speculative values rather than inventing placeholders.

### Installation Instructions

* Show installation instructions with a `uv`-first approach.
* Always include a `pip` fallback unless the task explicitly says otherwise.
* Distinguish clearly between production and development installation.

Prefer patterns such as:

Production:

```bash
uv add package-name
```

```bash
pip install package-name
```

Development:

```bash
uv sync --extra dev --extra mcp
```

```bash
pip install ".[dev,mcp]"
```

If the repo uses optional extras, document them explicitly and accurately from project metadata.
Do not invent extras that are not present in `pyproject.toml`.

### Dev vs Prod Documentation

* Distinguish production usage from contributor setup.

* Production docs should focus on installing and using the package.

* Development docs should cover:

  * editable or workspace setup when relevant
  * dev extras
  * linting
  * type checking
  * testing
  * local docs or CLI workflows if applicable

* Keep contributor details out of the top of README unless they are essential.

* Prefer linking to dedicated contributor docs for full setup instructions.

### Markdown Structure for Project Docs

When generating docs pages, prefer separate focused files such as:

* `docs/CLI.md`
* `docs/INSTALLATION.md`
* `docs/DEVELOPMENT.md`
* `docs/CONFIGURATION.md`
* `docs/API.md`
* `docs/TESTING.md`

Each file should own one primary responsibility.
Do not dump all documentation into README.

### Tables and Rich Documentation Elements

* Use tables when they genuinely improve comparison or scanning.

* Use diagrams, charts, or interactive-friendly structures when the medium supports them and they add real value.

* Prefer tables for:

  * install option comparison
  * extras and feature mapping
  * environment variable reference
  * provider capability matrices
  * CLI command summaries

* Do not force tables where plain bullets are clearer.

* Do not add decorative diagrams without informational value.

### Documentation Accuracy Rules

* Do not describe commands, extras, entry points, modules, env vars, or file paths that are not visible in repo context.

* Do not invent CLI examples, config keys, or extras.

* Keep docs aligned with:

  * `pyproject.toml`
  * actual package names
  * actual import paths
  * actual source URLs
  * actual docs file paths

* If repo metadata is incomplete, stay conservative and explicit about uncertainty rather than fabricating details.

---

## Testing Policy

Tests must reflect the same standards as production code: explicit, deterministic, strongly typed, easy to review, and hard to accidentally weaken.

### Testing Philosophy

* Test public behavior, not private implementation details.
* Prefer testing through the public API surface only.
* Do not write tests that directly target `_private` functions, `_private` methods, or hidden module internals unless there is no public seam and the project explicitly allows it.
* Validate observable behavior, returned values, raised exceptions, emitted messages, and state transitions.
* Do not couple tests to incidental implementation details such as internal helper call order.

### Test Scope Hierarchy

Use the lightest test style that still proves the behavior.

1. **Pure unit tests**

   * no network
   * no real model requests
   * no filesystem or clock dependence unless explicitly controlled
   * fastest and most common test layer

2. **Boundary tests**

   * validate integration with serializers, adapters, schemas, tool definitions, retries, hooks, context propagation, or async boundaries
   * may use fakes, stub transports, snapshots, or recorded HTTP

3. **Integration tests**

   * use real external APIs only when this is the point of the test
   * should be explicit, isolated, and usually recorded or gated

### Public API Rule

* Test what consumers use.
* Prefer assertions on exported classes, functions, methods, and documented behavior.
* If internal complexity matters, expose it through a stable public seam instead of testing internals directly.
* Refactoring internal implementation should not force large test rewrites if behavior is unchanged.

### Real API vs Mock Policy

* Prefer real behavior over deep mocking.
* Prefer the project's own testing models, fakes, stubs, and recorded responses over large mock trees.
* For agent and model flows, prefer tools like `TestModel` and `FunctionModel` instead of mocking every internal step.
* Real external API calls should not happen in normal unit tests.
* By default, tests should assume model requests are blocked, for example via a global `ALLOW_MODEL_REQUESTS = False` style safeguard.
* Real API usage belongs in explicit integration or recorded tests, not in ordinary unit coverage.

### Test Layout and Naming

* Mirror the source tree with a `test_{module}.py` convention whenever practical.
* Keep a near 1:1 mapping between production modules and test modules.
* Co-locate highly specific tests with the module they validate.
* Put broad shared fixtures only in `conftest.py`.
* Keep helper utilities out of `conftest.py` unless they are genuinely fixture-related.

Examples:

* `package/foo.py` -> `tests/test_foo.py`
* `package/models/bar.py` -> `tests/models/test_bar.py`

### Test Function Design

* Each test should validate one behavior or one tightly related behavior cluster.
* Prefer descriptive test names that state the condition and expected outcome.
* Avoid giant scenario tests that validate many unrelated concerns at once.
* Keep setup small and local unless it is reused enough to justify a fixture.
* Prefer Arrange / Act / Assert structure, even if not written as comments.

### Fixtures

Fixtures should reduce duplication without hiding meaning.

#### General Fixture Rules

* Prefer small, composable fixtures.
* Use the narrowest scope that makes sense.
* Default to `function` scope unless a broader scope is clearly beneficial and safe.
* Keep fixtures deterministic and side-effect aware.
* Type annotate fixture return values.
* Name fixtures after the object or state they provide, not after vague setup actions.

#### When to Use Fixtures

Use fixtures for:

* reusable object construction
* shared environment setup
* reusable temporary directories or files
* monkeypatch-based environment control
* async resource lifecycle
* repeated agent/model/tool setup
* test data factories reused across multiple tests

Do not use fixtures when:

* inline setup is shorter and clearer
* the fixture hides too much important context
* the fixture produces many optional variants through booleans or branching

#### Yield Fixtures

* Use `yield` fixtures for anything requiring teardown.
* Teardown must be deterministic and idempotent.
* Cleanup belongs in the fixture, not repeated manually in tests.

Examples of valid `yield` fixture use:

* temporary servers
* patched environment variables
* context-local overrides
* temp files requiring explicit cleanup
* captured registries or restored globals

#### Factory Fixtures

* Prefer factory fixtures when many tests need similar objects with slight variations.
* Factory fixtures should create clearly typed objects and accept explicit keyword arguments.
* Do not build giant "do everything" fixtures with many optional flags.

#### Autouse Fixtures

* Use `autouse=True` sparingly.

* Autouse is acceptable for global safety rails such as:

  * blocking real model or network requests by default
  * forcing deterministic environment settings
  * resetting global registries or contextvars

* Do not use autouse fixtures for ordinary object construction.

### Parametrization

* Use `@pytest.mark.parametrize` for behavior matrices, edge cases, and backend variations.
* Prefer parametrization over copy-pasted test bodies.
* Keep parameter sets readable.
* Use `ids=` for non-trivial parameter groups so failures are understandable.
* Prefer combining parametrization with factory fixtures rather than giant conditional tests.

### Async Testing

* Keep async tests first-class.
* Use pytest's async support consistently across the repo.
* If the project standardizes on AnyIO, keep the backend fixed and explicit, for example with an `anyio_backend` fixture returning `asyncio`.
* Do not mix multiple async test styles arbitrarily.
* Do not call `asyncio.run()` inside pytest async tests.
* Use async fixtures for async resources.
* Keep cancellation, timeout, and cleanup behavior testable.

#### Async Rules

* Prefer `pytest.mark.anyio` or the repo's chosen async marker consistently.
* Use `AsyncMock` only when mocking an async boundary is truly necessary.
* Avoid real sleeps when possible; prefer controlled timing, fake clocks, or deterministic event sequencing.
* For streams and async iterators, assert both yielded values and cleanup semantics.

### Mocks, Stubs, Fakes, and Monkeypatching

Prefer the least invasive test double.

#### Priority Order

Prefer, in order:

1. real pure object
2. project-specific fake or test model
3. lightweight stub
4. monkeypatch
5. `Mock` / `MagicMock` / `AsyncMock`

#### Mock Rules

* Do not mock deep internal call graphs.

* Mock only true side-effect boundaries:

  * network
  * clocks
  * environment
  * subprocesses
  * filesystem edges when necessary
  * provider SDK boundaries

* Assert outcomes first, call counts second.

* Do not verify every internal call unless that call choreography is the behavior being specified.

#### Patch Location Rule

* Patch where the symbol is looked up, not where it was originally defined.
* Keep patch scope narrow.
* Prefer context-managed patching or fixture-based patching over global mutation.

#### Monkeypatch Usage

Use `monkeypatch` for:

* environment variables
* module-level constants
* temporary function replacement
* context-local switches
* working directory or path adjustments

Do not use monkeypatch to rewrite large parts of the system under test.

### Recommended Built-in Pytest Features

Use these standard pytest tools intentionally:

* `monkeypatch` for env and symbol overrides
* `tmp_path` for filesystem tests
* `capsys` / `capfd` for stdout-stderr assertions
* `caplog` for logging assertions
* `recwarn` for warning assertions
* `parametrize` for input matrices
* `fixture` + `yield` for reusable setup or teardown
* `mark.anyio` or repo-standard async markers for async tests
* `xfail` only for known, documented, temporary cases
* `skip` only when the environment truly cannot support the test

### Snapshot Testing

* Use `inline-snapshot` for complex, structured, or high-noise assertions.
* Prefer importing snapshots through the project's local helper if one exists, otherwise use `from inline_snapshot import snapshot`.
* Use `snapshot()` directly in assertions instead of maintaining external golden files for ordinary structured expectations.
* Prefer snapshots when asserting:

  * large JSON-like structures
  * generated schemas
  * message sequences
  * structured outputs from agents or tools
  * multiline stdout or stderr
  * nested mappings or lists where direct literals remain readable

#### Snapshot Assertion Style

* Prefer direct patterns such as:

  * `assert value == snapshot()`
  * `assert value == snapshot({...})`
  * `assert item in snapshot([...])`
  * `assert value <= snapshot(limit)`
  * `assert value >= snapshot(limit)`

* Use nested snapshots when the structure is stable and readable.

* Keep snapshots close to the assertion site so intent is obvious.

* Do not move ordinary snapshots into separate files unless the payload is genuinely too large.

#### Snapshot Update Workflow

* Record or review snapshot changes intentionally.

* Prefer review-style updates instead of silently rewriting expectations.

* Use the project's agreed command for snapshot review or update, for example:

  * `pytest --inline-snapshot=review`

* Do not bulk-accept snapshot changes without reading the diff.

* Snapshot changes are code changes and must be reviewed with the same care as production edits.

#### Large Snapshot Payloads

* For very large strings or payloads, prefer external storage helpers such as `outsource(...)` and `external(...)`.
* Use externalized snapshots when inline expectations would make the test unreadable.
* Keep external snapshot content diff-friendly and deterministic.
* Do not externalize snapshots prematurely; inline remains the default.

#### Snapshot Hygiene

* Normalize unstable values before snapshotting:

  * timestamps
  * UUIDs
  * request IDs
  * ordering that is not semantically relevant
  * environment-dependent paths

* Do not snapshot values that are intentionally random or volatile unless they are first normalized.

* Do not use snapshots for tiny scalar assertions that are clearer as direct equality checks.

* Snapshot-heavy tests should still make it clear what behavior is being protected.

### Cassetter / Recorded HTTP Testing

* Use `cassetter` for recorded HTTP integration tests.
* Prefer cassette-backed tests when mocking would erase the behavior under test.
* Keep cassette tests explicit and separate from ordinary pure unit tests.
* Prefer pytest integration with:

  * `@pytest.mark.vcr`
  * `cassette` fixture when direct cassette inspection is needed
  * `vcr_config` fixture for shared test-level recording configuration
  * `vcr_cassette_dir` when the repo needs explicit cassette placement

#### Core Cassetter Usage

* Mark recorded tests with `@pytest.mark.vcr`.
* Use the `cassette` fixture only when the test needs to inspect recorded interactions directly.
* Prefer one cassette per test scenario.
* Keep test names and cassette names aligned and predictable.

#### Record Mode Policy

Use record modes intentionally:

* `none`

  * replay only
  * fail if no matching interaction exists
  * preferred for CI and deterministic local verification

* `once`

  * record only if the cassette does not exist
  * replay thereafter
  * good default for stable cassette-based tests

* `new_episodes`

  * replay existing interactions and append new ones
  * use only when extending an intentionally evolving scenario

* `all`

  * re-record everything and overwrite the cassette
  * use only for deliberate refreshes

* Prefer running CI with replay-only behavior.

* Do not allow accidental network fallback in deterministic test runs.

* Do not leave broad overwrite modes as the default.

#### Cassette Storage and Format

* Store cassettes in a dedicated `tests/cassettes/` hierarchy or another clearly named test-owned directory.
* Keep cassette paths stable and reviewable.
* Prefer YAML cassettes by default for readability.
* TOML cassettes are acceptable when the repo explicitly wants smaller or faster-loading cassette files.
* Keep cassette layout deterministic and scenario-specific.

#### Filtering and Secret Hygiene

* Never allow secrets into cassette files.

* Rely on cassetter's safe-by-default filtering, but still configure project-specific filtering explicitly when needed.

* Add extra protection with:

  * `filter_headers=[...]`
  * `body_scrub_patterns=[...]`
  * `filter_query_parameters`
  * `before_record_request`
  * `before_record_response`

* Strip or scrub:

  * authorization headers
  * cookies
  * API keys
  * bearer tokens
  * request IDs when unstable
  * account-specific identifiers if not required for the assertion

* Treat cassette files as committed artifacts that must already be sanitized when written.

#### Matching Policy

* Keep request matching as strict as practical while remaining stable.

* Default matching on method + URI is usually sufficient.

* Add body-aware matching when response shape depends on payload, for example:

  * `match_on=['method', 'uri', 'json_body']`

* Ignore unstable JSON paths when matching request bodies, for example:

  * timestamps
  * request IDs
  * trace IDs
  * nonce-like fields

* Do not make matching looser than necessary, or unrelated requests may replay incorrectly.

#### Expiry and Refresh Policy

* Use cassette expiry for endpoints that change materially over time.

* Prefer explicit freshness windows for tests that validate recent provider behavior.

* Configure expiry with:

  * `max_age`
  * `on_expiry`

* Allowed expiry behaviors:

  * `warn`
  * `fail`
  * `rerecord`

* Prefer `fail` or `rerecord` for behavior that must stay fresh.

* Prefer `warn` only when stale data is acceptable for that specific test class.

#### Hooks and Project-Level Configuration

* Use `vcr_config` for shared project or module configuration.

* Use `before_record_request` to:

  * remove secrets
  * drop irrelevant requests
  * normalize headers or URLs

* Use `before_record_response` to:

  * strip unstable headers
  * skip recording server failures when appropriate
  * normalize volatile payload fragments

* Keep hook logic minimal, deterministic, and well-commented.

#### Concurrency and Context Safety

* Cassetter supports concurrent cassette contexts through context-local state.
* Prefer isolated cassette scopes for concurrent async tests.
* If work moves into threads, ensure context propagation is explicit when needed.
* Do not assume cassette context automatically survives arbitrary thread boundaries.

#### Orphan and Drift Control

* Detect unused cassettes regularly.
* Remove orphans when tests are deleted or renamed.
* Keep recorded fixtures minimal and current.
* Prefer repo automation or a maintenance command for orphan detection.

#### Migration and Compatibility Notes

* Prefer cassetter over legacy `pytest-recording` / `VCR.py` usage for new work.
* Existing VCR cassettes may remain temporarily, but new recordings should follow the repo's cassetter conventions.
* Do not rely on features that are absent or intentionally unsupported in cassetter.
* Keep the test suite aligned with cassetter's supported fixture and marker model.

### Agent and Model Testing

* For agentic workflows, prefer domain-aware testing tools such as `TestModel` and `FunctionModel` rather than generic mocks.
* Test tool registration, retries, validation, output shaping, structured outputs, approval flows, and message emission through realistic model substitutes.
* Verify behavior at the agent boundary:

  * generated output
  * tool calls
  * retry prompts
  * captured messages
  * final structured values

### Determinism Requirements

Tests must be deterministic by default.

* No hidden dependence on:

  * real network
  * current time
  * random state
  * process-global mutable state
  * ambient environment
  * external credentials

* If a test depends on one of these, control it explicitly with fixtures or marks.

* Normalize unstable values before assertion, snapshotting, or cassette matching.

### Warnings Policy

* Treat warnings as errors by default.
* Do not add broad warning filters casually.
* If a 3rd-party warning must be ignored, make the filter narrow and documented.
* New code should not introduce fresh warnings.
* Use `recwarn` or explicit warning assertions when warnings are part of the API contract.

### Coverage Policy

* Maintain full coverage expectations for changed behavior.
* Treat `fail_under = 100` as a real gate, not a suggestion.
* Cover both happy paths and edge cases.
* Use `# pragma: no cover` or lax variants only when truly justified.
* Coverage exclusions must be intentional, rare, and reviewable.

### Failure Signal Quality

* Test failures should explain what behavior regressed.
* Prefer direct, high-signal assertions.
* Avoid vague assertions like `result is not None` when stronger behavior is known.
* Keep snapshots, parametrized IDs, fixture names, and cassette paths readable so failures are diagnosable.

### What to Avoid in Tests

* testing `_private` methods directly
* huge fixtures that hide business meaning
* pervasive deep mocking
* network access in ordinary unit tests
* arbitrary sleeps
* brittle call-order assertions for internal helpers
* random data without seeding or normalization
* snapshots for trivial values
* cassette files with secrets or unstable junk fields
* broad warning ignores
* global mutable state that leaks across tests

### Test Completion Checklist

A test change is not complete unless:

* the public behavior is covered
* the chosen test level is appropriate
* fixtures are minimal and readable
* snapshots are deterministic
* snapshot diffs were reviewed intentionally
* cassette matching and filtering are stable
* network or model access is explicit
* warnings remain clean
* coverage remains at the required threshold
* the full quality gate still passes

---

## What to Avoid

* unnecessary inheritance hierarchies
* god objects
* overly generic utility modules
* meaningless wrappers
* constructor overloading for unrelated behaviors
* excessive module fragmentation
* untyped public APIs
* `Any` without strong justification
* type ignores used as shortcuts
* banned typing imports when `typing_extensions` should be used
* pattern usage without clear value
* abstractions that make simple logic harder to follow
* mutable global state for runtime overrides
* hidden public APIs without `__all__`

---

## Decision Heuristics

When choosing between two implementations, prefer the one that:

1. has the clearer responsibility boundary
2. has the more precise type contract
3. uses fewer hidden assumptions
4. is easier to test
5. is easier to extend through composition
6. keeps the public API smaller and more explicit
7. preserves backward compatibility more intentionally
8. aligns better with the repo's existing patterns

---

## Output Standard

Generated Python must be:

* clean
* explicit
* composable
* type-safe
* lint-clean
* easy to test
* easy to navigate
* suitable for long-term maintenance
* consistent with strict repo conventions
