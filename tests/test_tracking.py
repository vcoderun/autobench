from __future__ import annotations as _annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, assert_type, cast

import pytest
from pydantic import BaseModel, Field

from autobench import AssetVersion, FieldAsset, ToolAsset, TrackingRegistry, TypeAsset
from autobench.io import load_yaml
from autobench.tracking import (
    ParamAsset,
    ParamSchema,
    TrackedAsset,
    TypeDecorator,
    asset_to_yaml_view,
)
from autobench.tracking.history import (
    _asset_version_changes,
    _asset_version_payload,
    _asset_version_snapshot,
    _asset_yaml_view,
    _changed_field_paths,
    _collect_changed_paths,
    _existing_asset_current_snapshot,
    _field_asset_yaml_view,
    _version_entry_snapshot,
)
from autobench.tracking.introspection import (
    _annotation_label,
    _callable_name,
    _dataclass_field_values,
    _field_assets_for_type,
    _literal_choices,
    _normalize_value,
    _parameter_kind,
    _plain_class_snapshot,
    _pydantic_field_assets,
    _safe_type_hints,
    _schema_constraints,
    _source_hash,
    _source_path,
    _source_text,
    _structured_type_kind,
    _tracked_type_asset_id,
)

MakeAlias = Literal["audi", "bmw"]


def test_tracking_decorators_support_direct_calls_and_manual_source_hash() -> None:
    registry = TrackingRegistry()

    class ReturnPayload:
        value: int

    decorated_type = registry.type(ReturnPayload)

    @registry.tool(name="decorated_tool")
    def decorated_name(value: int) -> int:
        return value

    @registry.type(name="named_type")
    class NamedType:
        enabled: bool

    def variadic(
        first: int,
        *items: str,
        enabled: bool = True,
        **labels: str,
    ) -> ReturnPayload:
        return ReturnPayload()

    decorated_tool = registry.tool(variadic)
    manual_target = object()
    registry.asset(
        kind="config",
        name="manual_hash",
        hash="content123",
        source_hash="source123",
    )(manual_target)

    tool_asset = registry.asset_of(decorated_tool)
    assert isinstance(tool_asset, ToolAsset)
    assert tool_asset.kind == "tool"
    assert (
        registry.asset_version_of(decorated_tool).hash
        == registry.asset_version_of(decorated_tool).content_hash
    )
    assert registry.asset_version_of(manual_target).source_hash == "source123"
    assert registry.assets["ReturnPayload"] == registry.asset_of(decorated_type)
    assert registry.asset_of(decorated_name).id == "tool.decorated_tool"
    assert registry.asset_of(NamedType).id == "type.named_type"
    assert tool_asset.param_schema.params == (
        ParamAsset(name="first", annotation="int", required=True, kind="positional"),
        ParamAsset(name="items", annotation="str", required=True, kind="var_positional"),
        ParamAsset(name="enabled", annotation="bool", required=False, default=True, kind="keyword"),
        ParamAsset(name="labels", annotation="str", required=True, kind="var_keyword"),
    )


def test_tracking_prompt_supports_text_source_and_raw(tmp_path: Path) -> None:
    registry = TrackingRegistry()
    prompt_path = tmp_path / "system_prompt.md"
    prompt_path.write_text("Use the refund policy.\n", encoding="utf-8")

    prompt_from_text = registry.prompt(name="inline_prompt", text="Hello")
    prompt_from_source = registry.prompt(name="file_prompt", source=prompt_path)

    assert prompt_from_text.raw == "Hello"
    assert str(prompt_from_text) == "Hello"
    assert prompt_from_source.raw == "Use the refund policy.\n"
    prompt_version = registry.asset_version_of(prompt_from_source)
    assert prompt_version.source_path == str(prompt_path.resolve())
    assert prompt_version.source_hash == prompt_version.content_hash

    with pytest.raises(ValueError, match="exactly one"):
        registry.prompt(name="invalid_none")

    with pytest.raises(ValueError, match="exactly one"):
        registry.prompt(name="invalid_both", text="Hello", source=prompt_path)


def test_tracking_registry_writes_asset_index_versions_and_diffs(tmp_path: Path) -> None:
    registry = TrackingRegistry()
    first_prompt = registry.prompt(name="system", text="Respond briefly.\n")
    asset_dir = tmp_path / "assets"

    registry.write_assets(asset_dir)
    second_prompt = registry.prompt(name="system", text="Respond briefly and cite evidence.\n")
    registry.write_assets(asset_dir)

    index = load_yaml(asset_dir / "index.yaml")
    asset = load_yaml(asset_dir / "prompt_system.yaml")

    assert index["record"]["type"] == "asset_index"
    assert index["assets"]["prompt.system"]["file"] == "prompt_system.yaml"
    assert index["assets"]["prompt.system"]["kind"] == "prompt"
    assert index["assets"]["prompt.system"]["current_version"] == registry.version_of(second_prompt)
    assert asset["record"]["type"] == "asset"
    assert asset["asset"]["kind"] == "prompt"
    assert asset["asset"]["current_version"] == registry.version_of(second_prompt)
    assert asset["asset"]["raw"] == "Respond briefly and cite evidence.\n"
    assert registry.version_of(first_prompt) != registry.version_of(second_prompt)
    assert len(asset["versions"]) == 2
    assert asset["versions"][0]["changes"] == {"fields": ["initial"], "diff": None}
    assert asset["versions"][1]["changes"]["fields"] == ["raw"]
    assert isinstance(asset["versions"][1]["changes"]["diff"], str)


def test_tracking_asset_yaml_view_renders_structured_tool_and_type_payloads(
    tmp_path: Path,
) -> None:
    registry = TrackingRegistry()

    class Car(BaseModel):
        make: MakeAlias
        model: str = Field(..., examples=["a3", "320i"])
        year: int = Field(..., gt=0)

    registry.type(Car)

    @registry.tool
    def create_car(make: MakeAlias, model: str, year: int) -> Car:
        """Create a new car instance."""
        return Car(make=make, model=model, year=year)

    asset_dir = tmp_path / "assets"
    registry.write_assets(asset_dir)

    type_asset = load_yaml(asset_dir / "type_Car.yaml")
    tool_asset = load_yaml(asset_dir / "tool_create_car.yaml")

    assert type_asset["asset"]["kind"] == "pydantic_model"
    assert type_asset["asset"]["fields"]["model"]["examples"] == ["a3", "320i"]
    assert type_asset["asset"]["fields"]["year"]["constraints"] == {"exclusiveMinimum": 0}
    assert type_asset["versions"][0]["changes"]["fields"] == ["initial"]
    assert tool_asset["asset"]["kind"] == "tool"
    assert tool_asset["asset"]["doc"] == "Create a new car instance."
    assert tool_asset["asset"]["params"]["make"]["choices"] == ["audi", "bmw"]
    assert tool_asset["asset"]["returns"] == {"type": "Car", "asset_id": "type.Car"}
    assert tool_asset["versions"][0]["state"]["params"]["make"]["type"] == "Literal['audi', 'bmw']"


def test_tracking_registry_handles_missing_and_malformed_asset_history(tmp_path: Path) -> None:
    registry = TrackingRegistry()
    registry.prompt(name="system", text="A\n")
    asset_dir = tmp_path / "assets"

    registry.write_assets(asset_dir)
    (asset_dir / "prompt_system.yaml").write_text("versions: malformed\n", encoding="utf-8")

    registry.prompt(name="system", text="B\n")
    registry.write_assets(asset_dir)
    asset = load_yaml(asset_dir / "prompt_system.yaml")

    assert len(registry.versions) == 2
    assert len(asset["versions"]) == 1
    with pytest.raises(KeyError, match="Asset version is missing"):
        registry._version_for_asset_id("prompt.missing")


def test_tracking_yaml_helper_views_cover_optional_branches() -> None:
    tool_asset = ToolAsset(
        id="tool.lookup",
        kind="tool",
        name="lookup",
        semantic_type="agent.tool",
        metadata={"owner": "tests"},
        qualname="pkg.lookup",
        doc="Lookup data",
        param_schema=ParamSchema(
            params=(
                ParamAsset(
                    name="mode",
                    annotation="str",
                    required=False,
                    default="fast",
                    kind="keyword",
                    literal_choices=("fast", "slow"),
                ),
            )
        ),
    )
    field_asset = FieldAsset(
        name="answer",
        annotation="str",
        required=False,
        default="ok",
        default_factory="factory",
        description="Response text",
        alias="answer_text",
        metadata={"owner": "tests"},
        init=False,
        kw_only=True,
        compare=False,
        repr=False,
    )
    type_asset = TypeAsset(
        id="type.Result",
        kind="type",
        name="Result",
        semantic_type="structured.output",
        metadata={"owner": "tests"},
        qualname="pkg.Result",
        doc="Result payload",
        type_kind="typed_class",
        field_assets=(field_asset,),
    )
    prompt_asset = TrackedAsset(
        id="prompt.system",
        kind="prompt",
        name="system",
        semantic_type="prompt.version",
    )
    config_asset = TrackedAsset(
        id="config.runtime",
        kind="config",
        name="runtime",
        metadata={"owner": "tests"},
    )

    tool_view = _asset_yaml_view(tool_asset, current_version="v1")
    type_view = _asset_yaml_view(type_asset, current_version="v2")
    prompt_view = _asset_yaml_view(prompt_asset, current_version="v3")
    config_view = _asset_yaml_view(config_asset, current_version="v4")

    assert tool_view["semantic"] == "agent.tool"
    assert tool_view["metadata"] == {"owner": "tests"}
    assert tool_view["params"]["mode"] == {
        "required": False,
        "type": "str",
        "default": "fast",
        "kind": "keyword",
        "choices": ["fast", "slow"],
    }
    assert "returns" not in tool_view
    assert type_view["semantic"] == "structured.output"
    assert type_view["metadata"] == {"owner": "tests"}
    assert type_view["fields"]["answer"] == {
        "required": False,
        "type": "str",
        "default": "ok",
        "default_factory": "factory",
        "description": "Response text",
        "alias": "answer_text",
        "metadata": {"owner": "tests"},
        "init": False,
        "kw_only": True,
        "compare": False,
        "repr": False,
    }
    assert prompt_view["kind"] == "prompt"
    assert prompt_view["semantic"] == "prompt.version"
    assert "raw" not in prompt_view
    assert config_view["metadata"] == {"owner": "tests"}


def test_tracking_version_and_change_helpers_cover_snapshot_branches() -> None:
    tool_asset = ToolAsset(
        id="tool.lookup",
        kind="tool",
        name="lookup",
        metadata={"owner": "tests"},
        qualname="pkg.lookup",
        doc="Lookup data",
        param_schema=ParamSchema(
            params=(ParamAsset(name="query", required=True, kind="positional"),)
        ),
        return_type_asset_id="type.Result",
    )
    type_asset = TypeAsset(
        id="type.Result",
        kind="type",
        name="Result",
        metadata={"owner": "tests"},
        qualname="pkg.Result",
        doc="Result payload",
        type_kind="typed_class",
        field_assets=(
            FieldAsset(
                name="choices",
                annotation="list[str]",
                required=True,
                examples=("a", "b"),
                constraints={"minItems": 1},
                literal_choices=("a", "b"),
            ),
        ),
    )
    prompt_asset = TrackedAsset(
        id="prompt.system",
        kind="prompt",
        name="system",
        metadata={"raw": "Hello", "owner": "tests"},
    )
    config_asset = TrackedAsset(
        id="config.runtime",
        kind="config",
        name="runtime",
        metadata={"owner": "tests"},
    )
    previous_snapshot = {
        "items": [1, {"name": "before"}],
        "missing": "old",
        "same": [1, 2],
    }
    current_snapshot = {
        "items": [1, {"name": "after"}],
        "same": [1, 2, 3],
        "added": True,
    }
    version = AssetVersion(
        asset_id="tool.lookup",
        version="v2",
        content_hash="content",
        source_hash="source",
        source_path="tracking.py",
        git_commit="abc123",
        parent_version="v1",
        metadata={"owner": "tests"},
    )

    assert _asset_version_snapshot(tool_asset)["returns"] == {"asset_id": "type.Result"}
    assert _asset_version_snapshot(type_asset)["fields"]["choices"] == {
        "required": True,
        "type": "list[str]",
        "examples": ["a", "b"],
        "constraints": {"minItems": 1},
        "choices": ["a", "b"],
    }
    assert _asset_version_snapshot(prompt_asset)["metadata"] == {"owner": "tests"}
    assert _asset_version_snapshot(config_asset) == {
        "kind": "config",
        "name": "runtime",
        "metadata": {"owner": "tests"},
    }
    assert _collect_changed_paths(previous_snapshot, current_snapshot) == [
        "added",
        "items[1].name",
        "missing",
        "same",
    ]
    assert _collect_changed_paths([1, 2], [1, 2]) == []
    assert _changed_field_paths(previous_snapshot, current_snapshot) == [
        "added",
        "items[1].name",
        "missing",
        "same",
    ]
    assert _asset_version_changes(None, current_snapshot) == {
        "fields": ["initial"],
        "diff": None,
    }

    version_payload = _asset_version_payload(
        version,
        current_snapshot,
        previous_snapshot=previous_snapshot,
    )

    assert version_payload["parent"] == "v1"
    assert version_payload["hashes"] == {"content": "content", "source": "source"}
    assert version_payload["source"] == {"path": "tracking.py", "git_commit": "abc123"}
    assert version_payload["metadata"] == {"owner": "tests"}
    assert version_payload["state"] == current_snapshot
    assert version_payload["changes"]["fields"] == [
        "added",
        "items[1].name",
        "missing",
        "same",
    ]
    assert isinstance(version_payload["changes"]["diff"], str)
    assert _version_entry_snapshot({"state": {"answer": 1}}) == {"answer": 1}
    assert _version_entry_snapshot({"snapshot": {"answer": 1}}) == {"answer": 1}
    assert _version_entry_snapshot({"snapshot": "bad"}) is None
    assert _existing_asset_current_snapshot(
        {
            "asset": {
                "id": "prompt.system",
                "kind": "prompt",
                "name": "system",
                "current_version": "v2",
                "semantic": "prompt.version",
                "raw": "Hello",
                "metadata": {"owner": "tests"},
            }
        }
    ) == {
        "kind": "prompt",
        "name": "system",
        "raw": "Hello",
        "metadata": {"owner": "tests"},
    }
    assert _existing_asset_current_snapshot({"asset": "bad"}) is None


def test_asset_to_yaml_view_falls_back_to_last_version_state_when_current_asset_is_missing() -> (
    None
):
    prompt_asset = TrackedAsset(
        id="prompt.system",
        kind="prompt",
        name="system",
        metadata={"raw": "next"},
    )
    version = AssetVersion(
        asset_id="prompt.system",
        version="v2",
        content_hash="content",
        parent_version="v1",
    )

    payload = asset_to_yaml_view(
        prompt_asset,
        version,
        existing={
            "versions": [
                {
                    "version": "v1",
                    "state": {
                        "kind": "prompt",
                        "name": "system",
                        "raw": "before",
                    },
                }
            ]
        },
    )

    assert payload["versions"][-1]["changes"]["fields"] == ["raw"]


def test_tracking_yaml_helpers_cover_absent_optional_branches() -> None:
    tool_without_optional_fields = ToolAsset(
        id="tool.lookup",
        kind="tool",
        name="lookup",
        param_schema=ParamSchema(
            params=(ParamAsset(name="query", required=True, kind="positional"),)
        ),
    )
    tool_with_return_annotation_only = ToolAsset(
        id="tool.render",
        kind="tool",
        name="render",
        param_schema=ParamSchema(),
        return_annotation="str",
    )
    tool_with_return_asset_only = ToolAsset(
        id="tool.decode",
        kind="tool",
        name="decode",
        param_schema=ParamSchema(),
        return_type_asset_id="type.Result",
    )
    type_without_optional_fields = TypeAsset(
        id="type.Empty",
        kind="type",
        name="Empty",
        type_kind="typed_class",
    )
    field_without_optional_fields = FieldAsset(name="value", required=True)
    prompt_without_raw = TrackedAsset(id="prompt.empty", kind="prompt", name="empty")
    prompt_with_only_raw = TrackedAsset(
        id="prompt.raw",
        kind="prompt",
        name="raw",
        metadata={"raw": "hello"},
    )
    config_without_metadata = TrackedAsset(id="config.runtime", kind="config", name="runtime")

    assert _asset_yaml_view(tool_without_optional_fields, current_version="v1")["params"] == {
        "query": {"required": True}
    }
    assert _asset_yaml_view(tool_with_return_annotation_only, current_version="v2")["returns"] == {
        "type": "str"
    }
    assert _asset_yaml_view(tool_with_return_asset_only, current_version="v3")["returns"] == {
        "asset_id": "type.Result"
    }
    assert _asset_yaml_view(type_without_optional_fields, current_version="v4")["fields"] == {}
    assert _field_assets_for_type(type("NoAnnotations", (), {}), "typed_class") == []
    assert _asset_version_snapshot(tool_without_optional_fields) == {
        "kind": "tool",
        "name": "lookup",
        "params": {"query": {"required": True}},
    }
    assert _asset_version_snapshot(tool_with_return_annotation_only)["returns"] == {"type": "str"}
    assert _asset_version_snapshot(type_without_optional_fields) == {
        "kind": "typed_class",
        "name": "Empty",
        "fields": {},
    }
    assert _asset_version_snapshot(prompt_without_raw) == {"kind": "prompt", "name": "empty"}
    assert _asset_version_snapshot(prompt_with_only_raw) == {
        "kind": "prompt",
        "name": "raw",
        "raw": "hello",
    }
    assert _asset_version_snapshot(config_without_metadata) == {
        "kind": "config",
        "name": "runtime",
    }
    assert _field_asset_yaml_view(field_without_optional_fields) == {"required": True}


def test_tracking_invalid_decorators_raise() -> None:
    registry = TrackingRegistry()
    invalid_tool_decorator = cast(Callable[[object], object], registry.tool())
    invalid_type_decorator = cast(Callable[[object], object], registry.type())
    invalid_decorated_type = registry.decorate_type(cast(TypeDecorator[[]], lambda value: 1))

    class Demo:
        pass

    with pytest.raises(TypeError, match="callables"):
        invalid_tool_decorator(123)

    with pytest.raises(TypeError, match="classes"):
        invalid_type_decorator("nope")

    with pytest.raises(TypeError, match="returns a class"):
        invalid_decorated_type(Demo)


def test_tracking_decorators_preserve_exact_types_for_type_checkers() -> None:
    registry = TrackingRegistry()

    @registry.type
    class Payload(BaseModel):
        value: int

    class DataclassPayload:
        value: int

    class BuilderSig(Protocol):
        def __call__(self, value: int, *, enabled: bool = True) -> Payload: ...

    @registry.tool
    def build_payload(value: int, *, enabled: bool = True) -> Payload:
        assert enabled
        return Payload(value=value)

    named_tool = registry.tool(name="named_builder")(build_payload)
    named_type = registry.type(name="NamedPayload")(Payload)
    dataclass_type = registry.dataclass(frozen=True, slots=True)(DataclassPayload)
    builder_sig: BuilderSig = build_payload
    named_builder_sig: BuilderSig = named_tool

    assert builder_sig is build_payload
    assert named_builder_sig is named_tool
    assert_type(build_payload(1), Payload)
    assert_type(named_tool(1), Payload)
    payload_type: type[Payload] = Payload
    named_payload_type: type[Payload] = named_type
    dataclass_payload_type: type[DataclassPayload] = dataclass_type
    assert payload_type is named_payload_type
    assert dataclass_payload_type is dataclass_type


def test_tracking_decorate_type_applies_decorator_and_captures_metadata() -> None:
    registry = TrackingRegistry()

    @registry.decorate_type(dataclass, frozen=True, slots=True)
    class CarRequest:
        make: str
        year: int

    asset = registry.asset_of(CarRequest)
    version = registry.asset_version_of(CarRequest)

    assert isinstance(asset, TypeAsset)
    assert asset.type_kind == "dataclass"
    assert version.metadata["decorator"] == {
        "name": "dataclass",
        "module": "dataclasses",
        "args": [],
        "kwargs": {"frozen": True, "slots": True},
    }
    assert hasattr(CarRequest, "__dataclass_fields__")


def test_tracking_dataclass_applies_transform_and_captures_metadata() -> None:
    registry = TrackingRegistry()

    @registry.dataclass(frozen=True, slots=True)
    class Invoice:
        id: str
        total: int

    class DirectInvoice:
        id: str
        total: int

    asset = registry.asset_of(Invoice)
    version = registry.asset_version_of(Invoice)
    direct_type = registry.dataclass(DirectInvoice)

    assert isinstance(asset, TypeAsset)
    assert asset.type_kind == "dataclass"
    assert version.metadata["decorator"] == {
        "name": "dataclass",
        "module": "dataclasses",
        "args": [],
        "kwargs": {
            "init": True,
            "repr": True,
            "eq": True,
            "order": False,
            "unsafe_hash": False,
            "frozen": True,
            "match_args": True,
            "kw_only": False,
            "slots": True,
            "weakref_slot": False,
        },
    }
    assert hasattr(Invoice, "__dataclass_fields__")
    assert hasattr(direct_type, "__dataclass_fields__")


def test_tracking_private_helpers_cover_fallback_paths() -> None:
    registry = TrackingRegistry()

    class Car(BaseModel):
        make: Literal["audi", "bmw"]
        year: int = Field(..., ge=2000)

    registry.type(Car)

    @dataclass
    class WithFactory:
        values: list[int] = field(default_factory=list)

    @dataclass
    class WithAlias:
        make: MakeAlias

    class Plain:
        flag: bool
        note: str = "ok"

    class PlainAlias:
        make: MakeAlias

    broken_init_type = cast(type[Any], type("BrokenInit", (), {"__init__": 1}))

    class Holder:
        def __getattr__(self, name: str) -> str:
            if name == "__qualname__":
                return "named-via-qualname"
            raise AttributeError(name)

    pydantic_assets = _field_assets_for_type(Car, "pydantic_model")
    dataclass_assets = _field_assets_for_type(WithFactory, "dataclass")
    dataclass_alias_assets = _field_assets_for_type(WithAlias, "dataclass")
    typed_assets = _field_assets_for_type(Plain, "typed_class")
    typed_alias_assets = _field_assets_for_type(PlainAlias, "typed_class")

    assert isinstance(registry.asset_of(Car), TypeAsset)
    assert _structured_type_kind(Car) == "pydantic_model"
    assert _structured_type_kind(WithFactory) == "dataclass"
    assert _structured_type_kind(Plain) == "typed_class"
    assert _pydantic_field_assets(object) == []
    assert pydantic_assets[0].literal_choices == ("audi", "bmw")
    assert pydantic_assets[1].constraints["minimum"] == 2000
    assert dataclass_assets[0].default_factory == "list"
    assert dataclass_alias_assets[0].annotation == "Literal['audi', 'bmw']"
    assert dataclass_alias_assets[0].literal_choices == ("audi", "bmw")
    assert typed_assets[1].default == "ok"
    assert typed_alias_assets[0].annotation == "Literal['audi', 'bmw']"
    assert typed_alias_assets[0].literal_choices == ("audi", "bmw")
    assert _tracked_type_asset_id("missing", registry) is None
    assert _tracked_type_asset_id(Car, registry) == "type.Car"
    assert _tracked_type_asset_id("Car", registry) == "type.Car"
    assert _tracked_type_asset_id(int, registry) is None
    assert _tracked_type_asset_id(inspect.Signature.empty, registry) is None
    registry.prompt(name="Car", text="shadowed")
    assert _tracked_type_asset_id("Car", registry) is None
    assert _safe_type_hints(len) == {}
    assert _safe_type_hints(42) == {}
    assert _dataclass_field_values(Plain) == ()
    fake_dataclass = cast(
        type[Any], type("FakeDataclass", (), {"__dataclass_fields__": {"bad": "value"}})
    )
    assert _dataclass_field_values(fake_dataclass) == ()
    assert _plain_class_snapshot(broken_init_type)["init_signature"] is None
    assert _annotation_label(inspect.Signature.empty) is None
    annotation_value = _annotation_label(object())
    assert isinstance(annotation_value, str)
    assert annotation_value.startswith("<object object at")
    assert _literal_choices(str, {"enum": ["x", "y"]}) == ["x", "y"]
    assert _schema_constraints({"description": "ignored", "pattern": "^ok$"}) == {"pattern": "^ok$"}
    assert _normalize_value(Path("demo.yaml")) == "demo.yaml"
    assert _normalize_value({"z": 1, "a": [1, 2]}) == {"a": [1, 2], "z": 1}
    assert _normalize_value((1, 2)) == [1, 2]
    plain_value = _normalize_value(Plain)
    assert isinstance(plain_value, str)
    assert plain_value.endswith(".Plain")
    object_value = _normalize_value(object())
    assert isinstance(object_value, str)
    assert object_value.startswith("<object object at")
    assert _parameter_kind(inspect.Parameter.KEYWORD_ONLY) == "keyword"
    assert _parameter_kind(inspect.Parameter.VAR_POSITIONAL) == "var_positional"
    assert _parameter_kind(inspect.Parameter.VAR_KEYWORD) == "var_keyword"
    assert _callable_name(Holder()) == "named-via-qualname"
    assert _callable_name(object()).startswith("<object object at")
    assert _source_text(len) is None
    assert _source_hash(len) is None
    assert _source_path(len) is None
