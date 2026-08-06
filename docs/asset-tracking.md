# Asset Tracking

Benchmarks need to know which prompt, tool, schema, or configuration produced each result.
Autobench tracking assigns content-derived versions, captures structured metadata, persists history,
and binds exact asset versions to RunRecords.

Supported SDK instrumentors can discover these assets without decorators. Use explicit tracking on
this page when the application owns a better logical identity or when the component never crosses an
instrumented boundary. See [Automatic Asset Discovery](automatic-asset-discovery.md) for unannotated
Pydantic AI, OpenAI, OpenAI Agents, capability scopes, privacy, and custom SDK extraction.

## Prompts And Text Assets

Track inline text:

```python
from autobench import track

SYSTEM_PROMPT = track.prompt(
    name="support_system_prompt",
    text="Route the request to billing, account, or technical support.",
)
```

Or load it from a file:

```python
SYSTEM_PROMPT = track.prompt(
    name="support_system_prompt",
    source="prompts/support.md",
)
```

`TrackedPrompt.raw` returns the text, and `str(SYSTEM_PROMPT)` provides the same value for APIs that
expect a string. File-backed prompts retain their source path and source hash.

## Tools

`@track.tool` preserves the callable's exact signature and return type while collecting tool
metadata:

```python
from typing import Literal

from autobench import track


@track.tool
def route_ticket(
    queue: Literal["billing", "account", "technical"],
    priority: int = 1,
) -> bool:
    """Route a ticket to a support queue."""
    return priority > 0
```

The resulting `ToolAsset` records:

- qualified name and docstring
- parameter names, kinds, annotations, defaults, and requirements
- return annotation
- source path and source hash
- structured parameter schema
- semantic type and version lineage

Annotations are normalized by structure rather than alias spelling. If the contents of a
`Literal`, union, generic, model, or referenced type change, the asset hash changes even when the
alias name stays the same.

## Pydantic Models, Dataclasses, And Classes

```python
from dataclasses import dataclass
from typing import Literal

from autobench import track
from pydantic import BaseModel, Field


@track.type
class Car(BaseModel):
    make: Literal["audi", "bmw", "mercedes"]
    model: str = Field(examples=["a3", "320i"])
    year: int = Field(gt=0)


@track.dataclass(frozen=True, slots=True)
class CarRequest:
    make: Literal["audi", "bmw", "mercedes"]
    model: str
    year: int
```

Pydantic models are hashed from normalized JSON Schema plus source identity. Standard dataclasses
use dataclass field definitions and resolved annotations. Other typed classes use resolved class
annotations, inspectable signatures, and source hashes.

`TypeAsset` and `FieldAsset` preserve field names, resolved annotations, descriptions, examples,
aliases, defaults, required state, and relevant constraints.

## Composing Another Class Decorator

When `@track.type` above a class-transforming decorator gives poor type-checker inference, use
`track.decorate_type`:

```python
from dataclasses import dataclass

from autobench import track


@track.decorate_type(dataclass, frozen=True, slots=True)
class Request:
    value: str
```

The decorator and its normalized arguments are stored as asset metadata. `track.dataclass(...)` is
the typed convenience form for the standard dataclass decorator.

## Arbitrary Assets

Use `track.asset` for configurations, policies, routing tables, or other application components:

```python
@track.asset(kind="routing_policy", name="enterprise_routing")
def route_policy(ticket):
    return "priority" if ticket["enterprise"] else "standard"
```

The decorator returns the original object unchanged. Callables use source and signature metadata;
manual `version`, `hash`, `source_path`, `parent_version`, and metadata values are available when
automatic identity is not enough.

## Versions, Diffs, And Persistence

`TrackingRegistry` keeps current assets and version history in memory during execution. Persist it
with:

```python
from pathlib import Path

from autobench import track

track.write_assets(Path(".autobench/assets"))
```

The directory contains an index, one lightweight manifest per asset, and `content.sqlite3`, a local
content-addressed registry keyed by asset ID and version. Every new manifest version links to its
parent and stores changed paths plus typed content and diff references. Prompt text, tool bodies,
schema definitions, and readable diffs never appear in the manifests. SQLite transactions avoid
rewriting the complete history as it grows, while content hashes deduplicate identical snapshots.

```python
from autobench import load_asset_content

version = track.asset_version_of(SYSTEM_PROMPT)
snapshot = load_asset_content(
    Path(".autobench/assets/content.sqlite3"),
    asset_id=version.asset_id,
    version=version.version,
)
assert snapshot["raw"] == SYSTEM_PROMPT.raw
```

For a version with a parent, resolve its readable diff with
`load_asset_diff(path, asset_id=..., version=..., parent_version=...)`.

## Binding Assets To Runs

```python
def run_case(ctx, case):
    ctx.attach_tracked_asset(SYSTEM_PROMPT)
    ctx.attach_tracked_asset(route_ticket)
    ctx.attach_tracked_asset(Car)
    return execute(case.input)
```

The exact `AssetVersion` values are copied into the RunRecord. Reports and optimization feedback
can then relate metric changes to prompt, tool, or output-schema versions without guessing from
source control state.
