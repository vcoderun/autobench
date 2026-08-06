# Behavioral Asset Tracking

## Contents

- [Why assets matter](#why-assets-matter)
- [Asset kinds](#asset-kinds)
- [Explicit tracking](#explicit-tracking)
- [Automatic discovery](#automatic-discovery)
- [Identity and versions](#identity-and-versions)
- [Definition and effective forms](#definition-and-effective-forms)
- [Persistence](#persistence)
- [Capture and privacy](#capture-and-privacy)
- [Rules for optimization](#rules-for-optimization)

## Why Assets Matter

Metrics are not enough to reproduce a successful run. Autobench also records the behavioral
components that influenced it: prompts, tools, schemas, agents, capabilities, and configuration.
Versioned assets make “prompt v7 was better” actionable because the exact content and diff can be
retrieved.

Asset lineage is evidence, not source control. It complements Git by capturing runtime-visible
definitions, dynamically assembled components, and effective SDK values.

## Asset Kinds

Supported or discoverable families include:

- prompt/system prompt/instructions;
- tool function, name, docstring, parameters, and return type;
- Pydantic model, dataclass, or class/output schema;
- output field name, type, description, examples, default, alias, and constraints;
- agent and toolset;
- capability and capability instructions;
- guardrail, handoff, policy, and runtime configuration;
- custom generic asset kinds through extension contracts.

Pydantic models hash normalized JSON Schema plus source identity where applicable. Dataclasses use
their field definitions rather than only `__init__`. Other classes use stable structural and source
information. Type aliases must be resolved structurally so changing a `Literal` behind an alias
changes the asset version.

## Explicit Tracking

```python
from autobench import track

SYSTEM_PROMPT = track.prompt(
    name="support_system",
    source="prompts/support.md",
)


@track.tool
def lookup_order(order_id: str) -> dict[str, str]:
    """Return the current order status."""
    ...
```

Useful surfaces include `track.prompt`, `track.tool`, `track.type`, `track.dataclass`, and
`track.asset`. Decorators preserve the original callable or class type and signature.

`track.decorate_type()` can compose another class decorator, such as `dataclass`, before tracking
the transformed class. Use it when decorator order would otherwise produce poor typing or DX.

Tracked text exposes its raw string and string conversion. File-backed prompts retain source and
content identity.

## Automatic Discovery

Native instrumentors inspect supported SDK call boundaries so users do not need tracking
decorators for already-visible components. Pydantic AI, OpenAI, OpenAI Agents, and custom
`InstrumentAssetSpec` extraction can attach `AssetUse` records to spans and runs.

Configure families and representations through `instrument_all()` or typed/YAML instrumentation
settings. Discovery failure becomes a diagnostic and must not fail the SDK call.

Use explicit tracking for application-owned identity or source files; use automatic discovery for
SDK-visible effective behavior. Both can correlate through locators and aliases.

## Identity And Versions

An asset version includes:

- stable asset ID and kind;
- version/content hash;
- parent version when one exists;
- changed paths and diff reference;
- source locator and aliases;
- scope, representation, and span use;
- typed content reference.

Changing content creates a new version. Changing only capture policy must not create a behavioral
version. Repeated observation of identical content reuses the version. Independent workers merge
histories safely.

Identity preference:

1. explicit application identity;
2. stable Python/source locator;
3. SDK semantic locator plus scope;
4. deterministic structural fallback.

Do not derive unnamed multi-asset identity from collection order when stable mapping keys or
content identities exist.

## Definition And Effective Forms

- **definition**: application-authored source, such as the original prompt or tool declaration;
- **effective**: value visible after SDK composition, capability expansion, defaults, or runtime
  overrides.

An effective use can link to its definition without rewriting the definition version. This allows
later analysis of both what the developer authored and what the model actually received.

Capability scope separates otherwise similar assets. For example, two instruction assets can share
kind and local ID while remaining distinct under `retrieval` and `safety` scopes.

## Persistence

Explicit registry persistence:

```python
from pathlib import Path

from autobench import load_asset_content, load_asset_diff, track

track.write_assets(Path(".autobench/assets"))
version = track.asset_version_of(SYSTEM_PROMPT)
snapshot = load_asset_content(
    Path(".autobench/assets/content.sqlite3"),
    asset_id=version.asset_id,
    version=version.version,
)
```

The directory contains:

```text
.autobench/assets/
  index.yaml
  <safe-asset-id>.yaml
  content.sqlite3
```

YAML manifests hold identity, lineage, changed paths, and references. Exact content and readable
diffs live in one transaction-safe, content-addressed SQLite registry. `load_asset_diff()` resolves
a version-to-parent diff.

Recorded experiments use the same contract at `artifacts/asset-content.sqlite3` with manifests
under `assets/`.

## Capture And Privacy

Behavioral assets default to full capture because hashes alone cannot reconstruct a successful
prompt, tool, or schema. Runtime evidence remains metadata-first.

Set `asset_default_level` to hash, redacted, metadata, or none when the experiment directory must
not retain exact behavioral content. Apply path and semantic rules for mixed sensitivity.

Never place raw asset bodies or diffs directly in run YAML. Records carry typed references to the
local registry. Secret paths and denied fields remain protected even under full capture.

## Rules For Optimization

- A candidate is a proposed asset value; a version is observed lineage. Do not conflate them.
- Record all asset versions active in every run, not only the changed asset.
- Do not mix “best” versions independently without validating interactions.
- Compare baseline, asset-only candidates, and combinations to identify confounding and interaction.
- Keep held-out validation and promotion decisions separate from candidate generation.
- Promotion creates a recommendation or version reference; it should not silently overwrite source
  files.
- Preserve candidate parentage, feedback, scores, and rejection reasons for later analysis.

