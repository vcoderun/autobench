from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError
from pydantic.json_schema import JsonSchemaMode, JsonSchemaValue

from autobench import Case, RunContext, Variant, load_asset_content
from autobench.instrumentation import (
    InstrumentAssetSpec,
    InstrumentationRuntime,
    InstrumentCall,
    InstrumentorInfo,
)
from autobench.io import load_yaml
from autobench.protocol import (
    AbstractionLayer,
    ActiveContext,
    CaptureMechanism,
    CapturePolicy,
    LocalCollector,
    new_trace_id,
    use_context,
)
from autobench.runtime.instrumentation import (
    instrument_method,
    reset_active_run_context,
    set_active_run_context,
)
from autobench.tracking import (
    AssetCandidate,
    AssetDefinition,
    AssetProvenance,
    AssetRepresentation,
    AssetSensitivity,
    AssetVersion,
    RegisteredAsset,
    TrackingRegistry,
    asset_to_yaml_view,
    canonical_asset_content,
    canonical_asset_hash,
)

NON_CALLABLE_EXTRACTOR = "not callable"


def _context() -> RunContext:
    return RunContext(
        benchmark_id="asset-discovery",
        case=Case(id="case-1"),
        variant=Variant(id="variant-1"),
    )


def _info() -> InstrumentorInfo:
    return InstrumentorInfo(
        id="autobench.test_sdk",
        version="1",
        mechanism=CaptureMechanism.HOOK,
        layer=AbstractionLayer.FRAMEWORK,
    )


def _candidate(
    content: str,
    *,
    locator: str = "test_sdk:prompt:instructions",
    representation: AssetRepresentation = AssetRepresentation.DEFINITION,
    definition_locator: str | None = None,
    aliases: tuple[str, ...] = (),
    sensitivity: AssetSensitivity = AssetSensitivity.INTERNAL,
) -> AssetCandidate:
    return AssetCandidate(
        kind="prompt",
        local_id="instructions",
        name="instructions",
        source_locator=locator,
        canonical_content=content,
        representation=representation,
        scope="test_sdk",
        definition_locator=definition_locator,
        aliases=aliases,
        sensitivity=sensitivity,
        provenance=AssetProvenance(
            system="test_sdk",
            key="instructions",
            instrumentor="autobench.test_sdk",
        ),
    )


def test_registry_versions_targetless_candidates_and_scoped_names(tmp_path: Path) -> None:
    registry = TrackingRegistry()
    first = registry.register_candidate(_candidate("Be concise."))
    repeated = registry.register_candidate(_candidate("Be concise."))
    changed = registry.register_candidate(_candidate("Be precise."))
    other = registry.register_candidate(
        _candidate("Be concise.", locator="other_sdk:prompt:instructions")
    )

    assert isinstance(first.asset, AssetDefinition)
    assert repeated.version == first.version
    assert changed.version.parent_version == first.version.version
    assert other.asset.id != first.asset.id
    assert len(registry.definitions) == 2
    assert len(registry.versions) == 3
    assert registry.resolve_locator("test_sdk:prompt:instructions") == changed.asset

    registry.write_assets(tmp_path)

    assert (tmp_path / "index.yaml").is_file()
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in tmp_path.glob("*.yaml")
        if path.name != "index.yaml"
    )
    assert "representation: definition" in text
    assert "Be precise." not in text
    assert "parent:" in text
    assert (
        load_asset_content(
            tmp_path / "content.sqlite3",
            asset_id=changed.asset.id,
            version=changed.version.version,
        )["content"]
        == "Be precise."
    )


def test_registry_recovers_from_stale_index_and_rejects_unknown_locators(
    tmp_path: Path,
) -> None:
    first = TrackingRegistry()
    first.register_candidate(_candidate("First"))
    first.write_assets(tmp_path)
    (tmp_path / "index.yaml").write_text("record: stale\n", encoding="utf-8")

    second = TrackingRegistry()
    second.register_candidate(_candidate("Second", locator="test_sdk:prompt:second"))
    second.write_assets(tmp_path)

    index = (tmp_path / "index.yaml").read_text(encoding="utf-8")
    assert "test_sdk:prompt:second" in index
    assert "record: stale" not in index
    with pytest.raises(KeyError, match="Unknown Autobench asset locator"):
        second.resolve_locator("test_sdk:prompt:missing")


def test_explicit_target_identity_wins_over_automatic_locator() -> None:
    registry = TrackingRegistry()

    @registry.tool
    def lookup(query: str) -> str:
        return query

    candidate = AssetCandidate(
        kind="tool",
        local_id="lookup",
        name="lookup",
        source_locator="test_sdk:tool:lookup",
        canonical_content={"name": "lookup", "parameters": {"query": "string"}},
        python_target=lookup,
        provenance=AssetProvenance(
            system="test_sdk",
            key="tools.lookup",
            instrumentor="autobench.test_sdk",
        ),
        aliases=("test_sdk:resolved_tool:lookup",),
    )

    registered = registry.register_candidate(candidate)

    assert registered.asset == registry.asset_of(lookup)
    assert registered.version == registry.asset_version_of(lookup)
    assert registry.resolve_locator("test_sdk:resolved_tool:lookup") == registered.asset
    assert len(registry.versions) == 1


def test_registry_reuses_python_locator_and_explicit_asset_identity() -> None:
    registry = TrackingRegistry()

    def build_lookup() -> Callable[[str], str]:
        def lookup(query: str) -> str:
            return query

        return lookup

    first_lookup = build_lookup()
    second_lookup = build_lookup()
    first = registry.register_candidate(
        AssetCandidate(
            kind="tool",
            local_id="lookup",
            name="lookup",
            source_locator="sdk_one:tool:lookup",
            canonical_content={"name": "lookup"},
            python_target=first_lookup,
            provenance=AssetProvenance(
                system="sdk_one",
                key="tools.lookup",
                instrumentor="autobench.sdk_one",
            ),
        )
    )
    same_python_symbol = registry.register_candidate(
        AssetCandidate(
            kind="tool",
            local_id="lookup",
            name="lookup",
            source_locator="sdk_two:tool:lookup",
            canonical_content={"name": "lookup"},
            python_target=second_lookup,
            provenance=AssetProvenance(
                system="sdk_two",
                key="tools.lookup",
                instrumentor="autobench.sdk_two",
            ),
        )
    )

    assert same_python_symbol.asset.id == first.asset.id
    assert same_python_symbol.version == first.version

    explicit = registry.register_candidate(
        AssetCandidate(
            kind="prompt",
            local_id="instructions",
            name="instructions",
            source_locator="sdk_one:prompt:instructions",
            explicit_asset_id="application.prompt.instructions",
            canonical_content="Be concise.",
            provenance=AssetProvenance(
                system="sdk_one",
                key="instructions",
                instrumentor="autobench.sdk_one",
            ),
        )
    )
    explicit_update = registry.register_candidate(
        AssetCandidate(
            kind="prompt",
            local_id="instructions",
            name="instructions",
            source_locator="sdk_two:prompt:instructions",
            explicit_asset_id="application.prompt.instructions",
            canonical_content="Be precise.",
            provenance=AssetProvenance(
                system="sdk_two",
                key="instructions",
                instrumentor="autobench.sdk_two",
            ),
        )
    )

    assert explicit_update.asset.id == explicit.asset.id
    assert explicit_update.version.parent_version == explicit.version.version


def test_registry_tracks_callable_instances_without_a_python_locator() -> None:
    class DynamicPrompt:
        def __call__(self) -> str:
            return "dynamic"

    registry = TrackingRegistry()
    registered = registry.register_candidate(
        AssetCandidate(
            kind="prompt",
            local_id="dynamic",
            name="dynamic",
            source_locator="test_sdk:prompt:dynamic",
            canonical_content="dynamic",
            python_target=DynamicPrompt(),
            provenance=AssetProvenance(
                system="test_sdk",
                key="dynamic",
                instrumentor="autobench.test_sdk",
            ),
        )
    )

    assert registered.asset.id == "test_sdk:prompt:dynamic"
    assert registry.resolve_locator("test_sdk:prompt:dynamic") == registered.asset


def test_effective_asset_use_links_to_definition_without_rewriting_it() -> None:
    registry = TrackingRegistry()
    definition = registry.register_candidate(_candidate("Hello, {name}."))
    effective = registry.register_candidate(
        _candidate(
            "Hello, Mert.",
            locator="test_sdk:prompt:instructions:effective",
            representation=AssetRepresentation.EFFECTIVE,
            definition_locator="test_sdk:prompt:instructions",
        )
    )

    assert effective.asset.id != definition.asset.id
    assert effective.use.definition_asset_id == definition.asset.id
    assert effective.use.definition_version == definition.version.version
    assert registry.version_by_asset_id(definition.asset.id) == definition.version


def test_registry_merges_cross_layer_locators_without_creating_a_version() -> None:
    registry = TrackingRegistry()
    first = registry.register_candidate(
        _candidate(
            "Use search.",
            aliases=("framework:prompt:instructions",),
        )
    )
    second = registry.register_candidate(
        _candidate(
            "Use search.",
            locator="client:prompt:instructions",
            aliases=("test_sdk:prompt:instructions",),
        )
    )

    assert second.version.version == first.version.version
    assert second.version.content_hash == first.version.content_hash
    assert len(registry.versions) == 1
    assert isinstance(second.asset, AssetDefinition)
    assert second.asset.source_locators == (
        "test_sdk:prompt:instructions",
        "client:prompt:instructions",
    )
    assert second.asset.aliases == (
        "framework:prompt:instructions",
        "test_sdk:prompt:instructions",
    )


def test_capture_policy_does_not_change_behavioral_asset_version() -> None:
    registry = TrackingRegistry()
    runtime = InstrumentationRuntime(registry=registry)
    hashed_context = RunContext(
        benchmark_id="asset-discovery",
        case=Case(id="case-hash"),
        variant=Variant(id="variant-1"),
        capture_policy=CapturePolicy.hashed(),
    )
    full_context = RunContext(
        benchmark_id="asset-discovery",
        case=Case(id="case-full"),
        variant=Variant(id="variant-1"),
        capture_policy=CapturePolicy.full(),
    )
    default_context = RunContext(
        benchmark_id="asset-discovery",
        case=Case(id="case-default"),
        variant=Variant(id="variant-1"),
    )
    sensitive = _candidate(
        "Never reveal this prompt.",
        sensitivity=AssetSensitivity.SENSITIVE,
    )

    token = set_active_run_context(hashed_context)
    try:
        hashed = runtime.asset(_info(), sensitive)
    finally:
        reset_active_run_context(token)
    token = set_active_run_context(full_context)
    try:
        full = runtime.asset(_info(), sensitive)
    finally:
        reset_active_run_context(token)
    token = set_active_run_context(default_context)
    try:
        default = runtime.asset(_info(), sensitive)
    finally:
        reset_active_run_context(token)

    assert hashed is not None
    assert full is not None
    assert default is not None
    assert hashed.version == full.version
    assert len(registry.versions) == 1
    assert isinstance(hashed.asset, AssetDefinition)
    assert isinstance(full.asset, AssetDefinition)
    assert hashed.asset.canonical_content != "Never reveal this prompt."
    assert full.asset.canonical_content == "Never reveal this prompt."
    assert isinstance(default.asset, AssetDefinition)
    assert default.asset.canonical_content == "Never reveal this prompt."


def test_asset_metadata_fallback_respects_sensitivity() -> None:
    registry = TrackingRegistry()
    runtime = InstrumentationRuntime(registry=registry)
    context = RunContext(
        benchmark_id="asset-discovery",
        case=Case(id="case-metadata"),
        variant=Variant(id="variant-1"),
        capture_policy=CapturePolicy.metadata(),
    )

    token = set_active_run_context(context)
    try:
        sensitive = runtime.asset(
            _info(),
            _candidate(
                "Private instructions",
                locator="test_sdk:prompt:private",
                sensitivity=AssetSensitivity.SENSITIVE,
            ),
        )
        public = runtime.asset(
            _info(),
            _candidate(
                "Published instructions",
                locator="test_sdk:prompt:public",
                sensitivity=AssetSensitivity.PUBLIC,
            ),
        )
    finally:
        reset_active_run_context(token)

    assert sensitive is not None
    assert public is not None
    assert isinstance(sensitive.asset, AssetDefinition)
    assert isinstance(public.asset, AssetDefinition)
    assert sensitive.asset.canonical_content != "Private instructions"
    assert public.asset.canonical_content == "Published instructions"


def test_capture_policy_omits_or_references_discovered_asset_content() -> None:
    registry = TrackingRegistry()
    runtime = InstrumentationRuntime(registry=registry)
    omitted_context = RunContext(
        benchmark_id="asset-discovery",
        case=Case(id="case-none"),
        variant=Variant(id="variant-1"),
        capture_policy=CapturePolicy.none(),
    )
    referenced_context = RunContext(
        benchmark_id="asset-discovery",
        case=Case(id="case-reference"),
        variant=Variant(id="variant-1"),
        capture_policy=CapturePolicy.full(max_inline_bytes=1),
    )

    token = set_active_run_context(omitted_context)
    try:
        omitted = runtime.asset(
            _info(),
            _candidate("omitted", locator="test_sdk:prompt:omitted"),
        )
    finally:
        reset_active_run_context(token)
    token = set_active_run_context(referenced_context)
    try:
        referenced = runtime.asset(
            _info(),
            _candidate("referenced", locator="test_sdk:prompt:referenced"),
        )
    finally:
        reset_active_run_context(token)

    assert omitted is not None
    assert isinstance(omitted.asset, AssetDefinition)
    omitted_content = omitted.asset.canonical_content
    assert isinstance(omitted_content, dict)
    assert omitted_content == {
        "omitted": True,
        "sha256": omitted_content["sha256"],
    }
    assert referenced is not None
    assert isinstance(referenced.asset, AssetDefinition)
    referenced_content = referenced.asset.canonical_content
    assert isinstance(referenced_content, dict)
    assert referenced_content == {
        "kind": "artifact",
        "id": referenced_content["id"],
        "version": None,
        "media_type": "application/json",
    }
    assert omitted_context.trace.diagnostics[0].code == "capture_omitted"
    assert referenced_context.trace.diagnostics[0].code == "capture_referenced"


def test_runtime_reports_asset_discovery_failures_without_affecting_the_run() -> None:
    class FailingRegistry(TrackingRegistry):
        def register_candidate(
            self,
            candidate: AssetCandidate,
            *,
            span_id: str | None = None,
        ) -> RegisteredAsset:
            del candidate, span_id
            raise RuntimeError("registry unavailable")

    context = _context()
    runtime = InstrumentationRuntime(registry=FailingRegistry())
    token = set_active_run_context(context)
    try:
        registered = runtime.asset(_info(), _candidate("still runs"))
    finally:
        reset_active_run_context(token)

    assert registered is None
    assert context.trace.diagnostics[0].code == "asset_discovery_failed"
    assert "registry unavailable" in context.trace.diagnostics[0].message


def test_instrumentation_runtime_attaches_assets_and_suppresses_failures() -> None:
    registry = TrackingRegistry()
    runtime = InstrumentationRuntime(registry=registry)
    context = _context()
    token = set_active_run_context(context)
    try:
        registered = runtime.asset(_info(), _candidate("Use tools."))
        repeated = runtime.asset(_info(), _candidate("Use tools."))
    finally:
        reset_active_run_context(token)

    assert registered is not None
    assert repeated is not None
    assert context.asset_versions == [registered.version]
    assert context.asset_uses == [registered.use]
    assert runtime.asset(_info(), _candidate("Outside a run.")) is None


def test_asset_discovery_is_a_noop_without_a_benchmark_run_context() -> None:
    runtime = InstrumentationRuntime(registry=TrackingRegistry())
    protocol_context = ActiveContext(
        collector=LocalCollector(),
        trace_id=new_trace_id(),
    )

    with use_context(protocol_context):
        registered = runtime.asset(_info(), _candidate("Outside a benchmark."))

    assert registered is None


def test_asset_yaml_preserves_optional_definition_context_and_empty_collections() -> None:
    minimal = AssetDefinition(
        id="application.prompt.minimal",
        kind="prompt",
        name="minimal",
        canonical_content="Minimal",
    )
    minimal_version = AssetVersion(
        asset_id=minimal.id,
        version="minimal-v1",
        content_hash="a" * 64,
    )
    minimal_view = asset_to_yaml_view(
        minimal,
        minimal_version,
        existing={"asset": {"source_locators": [], "aliases": []}},
    )

    assert "scope" not in minimal_view["asset"]
    assert "owner" not in minimal_view["asset"]
    assert "source_locators" not in minimal_view["asset"]
    assert "aliases" not in minimal_view["asset"]

    rich = AssetDefinition(
        id="application.prompt.rich",
        kind="prompt",
        name="rich",
        semantic_type="prompt.version",
        metadata={"team": "retrieval"},
        canonical_content="Rich",
        scope="retrieval",
        owner_locator="application:agent:retrieval",
        source_locators=("application:prompt:rich",),
        aliases=("provider:prompt:rich",),
    )
    rich_version = AssetVersion(
        asset_id=rich.id,
        version="rich-v1",
        content_hash="b" * 64,
    )
    rich_view = asset_to_yaml_view(
        rich,
        rich_version,
        existing={"asset": "stale"},
    )
    list_existing_view = asset_to_yaml_view(rich, rich_version, existing=[])

    assert rich_view["asset"]["scope"] == "retrieval"
    assert rich_view["asset"]["owner"] == "application:agent:retrieval"
    assert rich_view["asset"]["source_locators"] == ["application:prompt:rich"]
    assert rich_view["asset"]["aliases"] == ["provider:prompt:rich"]
    assert rich_view["versions"][0]["content_ref"] == {
        "asset_id": rich.id,
        "version": rich_version.version,
        "path": "content.sqlite3",
    }
    assert list_existing_view["asset"] == rich_view["asset"]


def test_canonical_asset_content_normalizes_models_types_and_collections() -> None:
    class Output(BaseModel):
        answer: str

    output_schema = canonical_asset_content(Output)
    assert isinstance(output_schema, dict)
    properties = output_schema["properties"]
    assert isinstance(properties, dict)
    answer = properties["answer"]
    assert isinstance(answer, dict)
    assert answer["type"] == "string"
    assert canonical_asset_content(Output(answer="yes")) == {"answer": "yes"}
    assert canonical_asset_content({"z": 1, "a": [True]}) == {"a": [True], "z": 1}

    @dataclass
    class DataclassOutput:
        answer: str
        confidence: float

    assert canonical_asset_content(DataclassOutput("yes", 0.9)) == {
        "answer": "yes",
        "confidence": 0.9,
    }


def test_callable_asset_content_is_stable_and_schema_normalization_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build_callback() -> Callable[[str], str]:
        def callback(value: str) -> str:
            return value

        return callback

    first = canonical_asset_content(build_callback())
    second = canonical_asset_content(build_callback())
    assert first == second
    assert "0x" not in str(first)

    class Output(BaseModel):
        answer: str

    schema_calls: list[str] = []
    original = Output.model_json_schema

    def model_json_schema(
        cls: type[Output],
        *,
        mode: JsonSchemaMode = "validation",
    ) -> JsonSchemaValue:
        del cls
        schema_calls.append(mode)
        return original(mode=mode)

    monkeypatch.setattr(Output, "model_json_schema", classmethod(model_json_schema))

    assert canonical_asset_content(Output) == canonical_asset_content(Output)
    assert schema_calls == ["validation"]


def test_callable_asset_content_supports_methods_builtins_and_opaque_instances() -> None:
    class Handler:
        def transform(self, value: str) -> str:
            return value

    class OpaqueCallback:
        @property
        def __signature__(self) -> str:
            return "not-a-signature"

        def __call__(self, value: str) -> str:
            return value

    method = canonical_asset_content(Handler().transform)
    builtin = canonical_asset_content(len)
    opaque = canonical_asset_content(OpaqueCallback())

    assert isinstance(method, dict)
    method_qualname = method["qualname"]
    assert isinstance(method_qualname, str)
    assert method_qualname.endswith("Handler.transform")
    assert method["source_hash"] is not None
    assert builtin == {
        "module": "builtins",
        "qualname": "len",
        "signature": "(obj, /)",
        "source_hash": None,
    }
    assert isinstance(opaque, dict)
    opaque_qualname = opaque["qualname"]
    assert isinstance(opaque_qualname, str)
    assert opaque_qualname.endswith("OpaqueCallback")
    assert opaque["signature"] is None


def test_asset_history_merges_versions_written_by_independent_workers(tmp_path: Path) -> None:
    first = TrackingRegistry()
    second = TrackingRegistry()
    first.register_candidate(_candidate("Version one", aliases=("worker:first",)))
    second.register_candidate(_candidate("Version two", aliases=("worker:second",)))

    with ThreadPoolExecutor(max_workers=2) as pool:
        writes = [pool.submit(registry.write_assets, tmp_path) for registry in (first, second)]
        for write in writes:
            write.result()

    prompt_record = load_yaml(tmp_path / "test_sdk_prompt_instructions.yaml")
    version_one = first.version_by_asset_id("test_sdk:prompt:instructions").version
    version_two = second.version_by_asset_id("test_sdk:prompt:instructions").version
    first_content = load_asset_content(
        tmp_path / "content.sqlite3",
        asset_id="test_sdk:prompt:instructions",
        version=version_one,
    )
    second_content = load_asset_content(
        tmp_path / "content.sqlite3",
        asset_id="test_sdk:prompt:instructions",
        version=version_two,
    )

    assert first_content["content"] == "Version one"
    assert second_content["content"] == "Version two"
    assert len(prompt_record["versions"]) == 2
    assert prompt_record["versions"][1]["parent"] == prompt_record["versions"][0]["version"]
    assert set(prompt_record["asset"]["aliases"]) == {"worker:first", "worker:second"}


def extract_result(call: InstrumentCall) -> str:
    assert isinstance(call.result, str)
    return call.result


def test_instrument_method_discovers_single_multiple_and_imported_assets() -> None:
    class Client:
        def run(self, prompt: str, *, tools: list[dict[str, str]]) -> str:
            return f"effective:{prompt}:{len(tools)}"

    handle = instrument_method(
        Client,
        "run",
        span="test_sdk.run",
        operation_family="test_sdk.Client.run",
        assets=[
            InstrumentAssetSpec(
                kind="prompt",
                local_id="prompt",
                value_path="kwargs.prompt",
            ),
            InstrumentAssetSpec(
                kind="tool",
                local_id="tools",
                value_path="kwargs.tools",
                many=True,
            ),
            InstrumentAssetSpec(
                kind="prompt",
                local_id="effective",
                representation=AssetRepresentation.EFFECTIVE,
                extractor_target=f"{__name__}:extract_result",
            ),
        ],
    )
    context = _context()
    token = set_active_run_context(context)
    try:
        result = Client().run(
            prompt="Route this",
            tools=[{"name": "search"}, {"name": "lookup"}],
        )
    finally:
        reset_active_run_context(token)
        handle.close()

    assert result == "effective:Route this:2"
    assert {use.source_locator for use in context.asset_uses} == {
        "python:test_sdk.Client.run:prompt:prompt",
        "python:test_sdk.Client.run:tool:tools:search",
        "python:test_sdk.Client.run:tool:tools:lookup",
        "python:test_sdk.Client.run:prompt:effective",
    }
    assert {use.span_id for use in context.asset_uses} == {context.spans[0].id}


def test_instrument_asset_spec_requires_one_extraction_strategy() -> None:
    with pytest.raises(ValidationError, match="exactly one"):
        InstrumentAssetSpec(kind="prompt", local_id="missing")
    with pytest.raises(ValidationError, match="exactly one"):
        InstrumentAssetSpec(
            kind="prompt",
            local_id="duplicate",
            value_path="result",
            value_factory=lambda call: call.result,
        )


def test_instrument_method_normalizes_candidates_and_isolates_extractor_errors() -> None:
    candidate = _candidate(
        "pre-normalized",
        locator="custom_sdk:prompt:normalized",
    )

    def policy() -> str:
        return "policy"

    class Client:
        def run(self) -> str:
            return "result"

    handle = instrument_method(
        Client,
        "run",
        span="custom_sdk.run",
        assets=[
            InstrumentAssetSpec(
                kind="prompt",
                local_id="candidate",
                value_factory=lambda call: candidate,
            ),
            InstrumentAssetSpec(
                kind="policy",
                local_id="policies",
                source_locator="custom_sdk:policies",
                many=True,
                value_factory=lambda call: [
                    candidate,
                    {"name": ""},
                    policy,
                ],
            ),
            InstrumentAssetSpec(
                kind="tool",
                local_id="invalid-many",
                many=True,
                value_path="result",
            ),
            InstrumentAssetSpec(
                kind="prompt",
                local_id="bad-target",
                extractor_target="invalid-target",
            ),
            InstrumentAssetSpec(
                kind="prompt",
                local_id="not-callable",
                extractor_target=f"{__name__}:NON_CALLABLE_EXTRACTOR",
            ),
        ],
    )
    context = _context()
    token = set_active_run_context(context)
    try:
        assert Client().run() == "result"
    finally:
        reset_active_run_context(token)
        handle.close()

    anonymous_id = canonical_asset_hash(canonical_asset_content({"name": ""}))[:12]
    assert {use.source_locator for use in context.asset_uses} == {
        "custom_sdk:prompt:normalized",
        f"custom_sdk:policies:{anonymous_id}",
        "custom_sdk:policies:policy",
    }
    assert [error.error_type for error in context.errors] == [
        "TypeError",
        "ValueError",
        "TypeError",
    ]


def test_unnamed_multi_asset_identity_is_stable_across_collection_order() -> None:
    class Client:
        def run(self, policies: list[dict[str, bool]]) -> int:
            return len(policies)

    handle = instrument_method(
        Client,
        "run",
        assets=[
            InstrumentAssetSpec(
                kind="policy",
                local_id="policies",
                many=True,
                value_path="kwargs.policies",
            )
        ],
    )
    first = {"enabled": True}
    second = {"enabled": False}
    context = _context()
    token = set_active_run_context(context)
    try:
        client = Client()
        assert client.run(policies=[first, second]) == 2
        assert client.run(policies=[second, first]) == 2
    finally:
        reset_active_run_context(token)
        handle.close()

    prefix = f"python:{Client.__module__}.{Client.__qualname__}.run:policy:policies:"
    source_locators = {use.source_locator for use in context.asset_uses}
    assert source_locators == {
        f"{prefix}{canonical_asset_hash(canonical_asset_content(first))[:12]}",
        f"{prefix}{canonical_asset_hash(canonical_asset_content(second))[:12]}",
    }
    assert len(context.asset_uses) == 2


def test_multi_asset_mapping_keys_provide_stable_identity() -> None:
    class Client:
        def run(self, policies: dict[str, dict[str, bool]]) -> int:
            return len(policies)

    handle = instrument_method(
        Client,
        "run",
        assets=[
            InstrumentAssetSpec(
                kind="policy",
                local_id="policies",
                source_locator="custom_sdk:policies",
                many=True,
                value_path="kwargs.policies",
            )
        ],
    )
    context = _context()
    token = set_active_run_context(context)
    try:
        assert (
            Client().run(
                policies={
                    "fallback": {"enabled": False},
                    "primary": {"enabled": True},
                }
            )
            == 2
        )
    finally:
        reset_active_run_context(token)
        handle.close()

    assert {use.source_locator for use in context.asset_uses} == {
        "custom_sdk:policies:fallback",
        "custom_sdk:policies:primary",
    }
    assert all(
        version.metadata.get("identity_confidence") is None for version in context.asset_versions
    )
