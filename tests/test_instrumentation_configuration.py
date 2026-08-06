from __future__ import annotations as _annotations

import json
import sys
from collections.abc import Collection, Sequence
from importlib.machinery import ModuleSpec
from io import StringIO
from pathlib import Path

import pytest
from click.testing import CliRunner
from pydantic import ValidationError
from rich.console import Console

import autobench.instrumentation.registry as registry_module
import autobench.runtime.pipeline as pipeline_module
from autobench import (
    AssetDiscoverySettings,
    AssetRepresentation,
    AutoInstrumentation,
    Benchmark,
    CaptureLevel,
    CapturePolicy,
    Case,
    Compatibility,
    CompatibilityStatus,
    HTTPXCaptureSettings,
    HTTPXInstrumentation,
    InstrumentationConfig,
    InstrumentationError,
    InstrumentationHandle,
    InstrumentationManager,
    InstrumentationRuntime,
    Instrumentor,
    InstrumentorInfo,
    OpenAIAgentsInstrumentation,
    OpenAIInstrumentation,
    PydanticAIInstrumentation,
    RunContext,
    SpecValidationError,
    __version__,
    benchmark_spec_to_yaml_view,
    load_benchmark_spec,
    record_experiment,
    run_benchmark_path,
)
from autobench.cli import cli
from autobench.instrumentation.config import BuiltinInstrumentationConfig
from autobench.instrumentation.httpx import HTTPX
from autobench.instrumentation.registry import InstrumentorStatus
from autobench.io import benchmark_schema
from autobench.protocol.signals import AbstractionLayer, CaptureMechanism
from autobench.reports.rich import render_instrumentor_statuses, render_trace_summary

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class RecordingInstrumentor:
    def __init__(self, instrumentor_id: str = "test.instrumentor") -> None:
        self._info = InstrumentorInfo(
            id=instrumentor_id,
            version="1",
            mechanism=CaptureMechanism.HOOK,
            layer=AbstractionLayer.APPLICATION,
        )
        self.installs = 0
        self.closes = 0

    @property
    def info(self) -> InstrumentorInfo:
        return self._info

    def check(self) -> Compatibility:
        return Compatibility.compatible()

    def install(self, runtime: InstrumentationRuntime) -> InstrumentationHandle:
        assert isinstance(runtime, InstrumentationRuntime)
        self.installs += 1
        return InstrumentationHandle(self._close, info=self.info)

    def _close(self) -> None:
        self.closes += 1


def successful_task(ctx: RunContext, case: Case) -> str:
    assert ctx.case == case
    return "ok"


def test_fluent_and_yaml_instrumentation_have_the_same_serializable_spec(
    tmp_path: Path,
) -> None:
    capture = HTTPXCaptureSettings(
        path="full",
        request_headers=("X-Request-ID", "x-request-id"),
        response_headers=("X-Trace",),
        max_body_bytes=2048,
    )
    fluent = (
        Benchmark("instrumented")
        .description("typed instrumentation")
        .instrument_all(exclude={"httpx"}, strict=True)
        .instrument(
            PydanticAIInstrumentation(enabled=False),
            OpenAIInstrumentation(),
            OpenAIAgentsInstrumentation(enabled=False),
            HTTPXInstrumentation(capture=capture),
        )
        .to_spec()
    )
    path = tmp_path / "autobench.yaml"
    path.write_text(
        """benchmark:
  instrumented:
    description: typed instrumentation
    dataset: {}
    instrumentation:
      all:
        exclude: [httpx]
        strict: true
      pydantic_ai: false
      openai: {}
      openai_agents:
        enabled: false
      httpx:
        capture:
          path: full
          request_headers: [X-Request-ID, x-request-id]
          response_headers: [X-Trace]
          max_body_bytes: 2048
""",
        encoding="utf-8",
    )

    loaded = load_benchmark_spec(path)

    assert loaded == fluent
    http_config = loaded.instrumentation[4]
    assert isinstance(http_config, HTTPXInstrumentation)
    assert http_config.capture.request_headers == ("x-request-id",)
    assert benchmark_spec_to_yaml_view(loaded)["benchmark"]["instrumented"]["instrumentation"] == {
        "all": {"exclude": ["httpx"], "strict": True},
        "pydantic_ai": False,
        "openai": {},
        "openai_agents": False,
        "httpx": {
            "capture": {
                "path": "full",
                "request_headers": ["x-request-id"],
                "response_headers": ["x-trace"],
                "max_body_bytes": 2048,
            }
        },
    }


def test_instrument_all_replaces_its_previous_configuration() -> None:
    benchmark = (
        Benchmark("automatic")
        .instrument_all(exclude={"openai"})
        .instrument(OpenAIInstrumentation(enabled=False))
        .instrument_all(exclude={"httpx", "pydantic_ai"}, strict=True)
    )

    assert benchmark.to_spec().instrumentation == [
        AutoInstrumentation(exclude=("httpx", "pydantic_ai"), strict=True),
        OpenAIInstrumentation(enabled=False),
    ]


def test_asset_discovery_settings_normalize_filter_and_serialize() -> None:
    settings = AssetDiscoverySettings(
        representations=(
            AssetRepresentation.EFFECTIVE,
            AssetRepresentation.DEFINITION,
            AssetRepresentation.EFFECTIVE,
        ),
        include=(" tool ", "prompt", "tool"),
    )
    benchmark = Benchmark("assets").instrument_all(assets=settings)

    assert settings.representations == (
        AssetRepresentation.EFFECTIVE,
        AssetRepresentation.DEFINITION,
    )
    assert settings.include == ("prompt", "tool")
    assert settings.allows("tool", AssetRepresentation.EFFECTIVE)
    assert not settings.allows("output_schema", AssetRepresentation.EFFECTIVE)
    assert not AssetDiscoverySettings(discover=False).allows("tool", AssetRepresentation.DEFINITION)
    assert not AssetDiscoverySettings(representations=(AssetRepresentation.DEFINITION,)).allows(
        "tool", AssetRepresentation.EFFECTIVE
    )
    assert benchmark_spec_to_yaml_view(benchmark.to_spec())["benchmark"]["assets"][
        "instrumentation"
    ] == {
        "all": {
            "assets": {
                "representations": ["effective", "definition"],
                "include": ["prompt", "tool"],
            }
        }
    }
    with pytest.raises(ValidationError, match="asset kind names cannot be empty"):
        AssetDiscoverySettings(include=(" ",))


def test_fluent_and_yaml_capture_policies_share_one_typed_contract(
    tmp_path: Path,
) -> None:
    policy = CapturePolicy.hashed(
        semantic_overrides={"tool": CaptureLevel.FULL},
        deny_paths=("prompt.secret",),
    )
    fluent = Benchmark("private-assets").capture(policy).to_spec()
    mapped = (
        Benchmark("mapped-assets")
        .capture(
            {
                "default_level": "hash",
                "use_semantic_defaults": False,
                "semantic_overrides": {"tool": "full"},
                "deny_paths": ["prompt.secret"],
            }
        )
        .to_spec()
    )
    path = tmp_path / "autobench.yaml"
    path.write_text(
        """benchmark:
  private-assets:
    capture:
      default_level: hash
      use_semantic_defaults: false
      semantic_overrides:
        tool: full
      deny_paths: [prompt.secret]
""",
        encoding="utf-8",
    )

    loaded = load_benchmark_spec(path)

    assert loaded.capture == policy
    assert mapped.capture == policy
    assert loaded == fluent
    assert benchmark_spec_to_yaml_view(loaded)["benchmark"]["private-assets"]["capture"] == {
        "default_level": "hash",
        "use_semantic_defaults": False,
        "semantic_overrides": {"tool": "full"},
        "deny_paths": ["prompt.secret"],
    }


@pytest.mark.parametrize(
    "instrumentation",
    [
        "unknown: {}",
        "httpx:\n        unexpected: true",
        "httpx: 42",
        "all:\n        exclude: [unknown]",
        "all:\n        strict: sometimes",
    ],
)
def test_yaml_rejects_unknown_instrumentors_and_invalid_settings(
    instrumentation: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "benchmark:\n  invalid:\n    instrumentation:\n      " + instrumentation + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError):
        load_benchmark_spec(path)


def test_http_capture_settings_validate_headers_and_body_limit() -> None:
    with pytest.raises(ValidationError, match="cannot be empty"):
        HTTPXCaptureSettings(request_headers=(" ",))
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        HTTPXCaptureSettings(max_body_bytes=0)


def test_yaml_accepts_internal_instrumentation_lists_and_null_defaults(
    tmp_path: Path,
) -> None:
    list_path = tmp_path / "list.yaml"
    list_path.write_text(
        """benchmark:
  list-config:
    instrumentation:
      - kind: openai
        enabled: false
""",
        encoding="utf-8",
    )
    null_path = tmp_path / "null.yaml"
    null_path.write_text(
        """benchmark:
  null-config:
    instrumentation:
      openai:
""",
        encoding="utf-8",
    )

    list_spec = load_benchmark_spec(list_path)
    null_spec = load_benchmark_spec(null_path)

    assert list_spec.instrumentation == [OpenAIInstrumentation(enabled=False)]
    assert null_spec.instrumentation == [OpenAIInstrumentation()]


@pytest.mark.parametrize(
    "instrumentation",
    [
        "- invalid",
        "invalid",
        "1: {}",
    ],
)
def test_yaml_rejects_invalid_instrumentation_container_shapes(
    instrumentation: str,
    tmp_path: Path,
) -> None:
    path = tmp_path / "invalid-container.yaml"
    path.write_text(
        "benchmark:\n  invalid-container:\n    instrumentation:\n      " + instrumentation + "\n",
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError):
        load_benchmark_spec(path)


def test_benchmark_installs_runtime_instrumentors_once_and_closes_them() -> None:
    instrumentor = RecordingInstrumentor()

    result = Benchmark("runtime-instrumentor").instrument(instrumentor).run()

    assert result.total_count == 0
    assert instrumentor.installs == 1
    assert instrumentor.closes == 1


def test_disabled_settings_are_not_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_resolved(config: BuiltinInstrumentationConfig) -> RecordingInstrumentor:
        raise AssertionError(config)

    monkeypatch.setattr(registry_module, "resolve_instrumentor", fail_if_resolved)
    result = Benchmark("disabled").instrument(PydanticAIInstrumentation(enabled=False)).run()

    assert result.total_count == 0


def test_auto_instrumentation_selects_compatible_integrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumentor = RecordingInstrumentor("autobench.openai")
    status = InstrumentorStatus(
        name="openai",
        extra="openai",
        info=instrumentor.info,
        compatibility=Compatibility.compatible(),
        capture_mode="metadata",
    )

    def inspect(
        config: BuiltinInstrumentationConfig,
        *,
        manager: InstrumentationManager,
    ) -> tuple[Instrumentor, InstrumentorStatus]:
        assert config.kind == "openai"
        assert isinstance(manager, InstrumentationManager)
        return instrumentor, status

    monkeypatch.setattr(registry_module, "_inspect_instrumentor", inspect)

    selected, skipped = registry_module.resolve_instrumentors(
        [AutoInstrumentation(exclude=("pydantic_ai", "openai_agents", "httpx"))]
    )

    assert selected == (instrumentor,)
    assert skipped == ()


def test_auto_instrumentation_honors_explicit_excluded_and_reserved_integrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, "find_spec", lambda module: None)

    selected, skipped = registry_module.resolve_instrumentors(
        [
            AutoInstrumentation(exclude=("httpx",)),
            OpenAIInstrumentation(enabled=False),
        ],
        reserved_ids={"autobench.pydantic_ai"},
    )

    assert selected == ()
    assert [status.name for status in skipped] == ["openai_agents"]


def test_explicit_instrumentation_replaces_automatic_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumentor = RecordingInstrumentor("autobench.openai")

    def resolve_explicit(config: BuiltinInstrumentationConfig) -> Instrumentor:
        assert config == OpenAIInstrumentation()
        return instrumentor

    monkeypatch.setattr(registry_module, "resolve_instrumentor", resolve_explicit)
    selected, skipped = registry_module.resolve_instrumentors(
        [
            AutoInstrumentation(exclude=("pydantic_ai", "openai_agents", "httpx")),
            OpenAIInstrumentation(),
        ]
    )

    assert selected == (instrumentor,)
    assert skipped == ()


def test_disabled_auto_instrumentation_does_not_discover_integrations() -> None:
    selected, skipped = registry_module.resolve_instrumentors([AutoInstrumentation(enabled=False)])

    assert selected == ()
    assert skipped == ()


def test_auto_instrumentation_propagates_asset_discovery_to_semantic_integrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = AssetDiscoverySettings(include=("prompt",))
    observed: list[BuiltinInstrumentationConfig] = []

    def inspect(
        config: BuiltinInstrumentationConfig,
        *,
        manager: InstrumentationManager,
    ) -> tuple[Instrumentor | None, InstrumentorStatus]:
        assert isinstance(manager, InstrumentationManager)
        observed.append(config)
        status = InstrumentorStatus(
            name=config.kind,
            extra=config.kind.replace("_", "-"),
            info=registry_module._INFO[config.kind],
            compatibility=Compatibility(status=CompatibilityStatus.UNAVAILABLE),
            capture_mode="test",
        )
        return None, status

    monkeypatch.setattr(registry_module, "_inspect_instrumentor", inspect)
    selected, skipped = registry_module.resolve_instrumentors(
        [AutoInstrumentation(assets=settings)]
    )

    assert selected == ()
    assert len(skipped) == 4
    assert [
        config.assets
        for config in observed
        if isinstance(
            config,
            (
                PydanticAIInstrumentation,
                OpenAIInstrumentation,
                OpenAIAgentsInstrumentation,
            ),
        )
    ] == [settings, settings, settings]
    assert isinstance(observed[-1], HTTPXInstrumentation)


def test_automatically_selected_instrumentor_uses_the_benchmark_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    instrumentor = RecordingInstrumentor("autobench.openai")

    def resolve_all(
        configs: Sequence[InstrumentationConfig],
        *,
        reserved_ids: Collection[str] = (),
    ) -> tuple[tuple[Instrumentor, ...], tuple[InstrumentorStatus, ...]]:
        assert configs == [AutoInstrumentation()]
        assert not reserved_ids
        return (instrumentor,), ()

    monkeypatch.setattr(pipeline_module, "resolve_instrumentors", resolve_all)

    result = Benchmark("automatic-lifecycle").instrument_all().run()

    assert result.total_count == 0
    assert instrumentor.installs == 1
    assert instrumentor.closes == 1


def test_strict_auto_instrumentation_rejects_an_unavailable_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, "find_spec", lambda module: None)

    with pytest.raises(
        InstrumentationError,
        match="automatic instrumentation 'pydantic_ai' is not installable",
    ):
        registry_module.resolve_instrumentors([AutoInstrumentation(strict=True)])


def test_auto_discovery_diagnostics_are_recorded_on_each_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = InstrumentorStatus(
        name="openai",
        extra="openai",
        info=InstrumentorInfo(
            id="autobench.openai",
            version="1",
            mechanism=CaptureMechanism.PATCH,
            layer=AbstractionLayer.CLIENT,
        ),
        compatibility=Compatibility(
            status=CompatibilityStatus.UNAVAILABLE,
            diagnostics=("distribution is not installed",),
        ),
        capture_mode="metadata",
    )

    def resolve_all(
        configs: Sequence[InstrumentationConfig],
        *,
        reserved_ids: Collection[str] = (),
    ) -> tuple[tuple[Instrumentor, ...], tuple[InstrumentorStatus, ...]]:
        assert configs == [AutoInstrumentation()]
        assert not reserved_ids
        return (), (status,)

    monkeypatch.setattr(pipeline_module, "resolve_instrumentors", resolve_all)
    result = (
        Benchmark("auto-diagnostics")
        .dataset([{"id": "one"}])
        .variants([{"id": "default"}])
        .task(f"{__name__}:successful_task")
        .instrument_all()
        .run()
    )

    diagnostic = next(
        observation
        for observation in result.runs[0].task_result.observations
        if observation.name == "instrumentation.skipped"
    )
    assert diagnostic.value == "unavailable"
    assert diagnostic.source == "instrumentation"
    assert diagnostic.tags == {
        "instrumentor": "openai",
        "extra": "openai",
        "diagnostics": ["distribution is not installed"],
    }


def test_pipeline_rejects_duplicate_runtime_instrumentor_ids() -> None:
    benchmark = Benchmark("duplicates").instrument(
        RecordingInstrumentor("duplicate"),
        RecordingInstrumentor("duplicate"),
    )

    with pytest.raises(InstrumentationError, match="duplicate instrumentors configured: duplicate"):
        benchmark.run()


def test_registry_resolves_each_installed_builtin_and_http_capture() -> None:
    resolved = (
        registry_module.resolve_instrumentor(PydanticAIInstrumentation()),
        registry_module.resolve_instrumentor(OpenAIInstrumentation()),
        registry_module.resolve_instrumentor(OpenAIAgentsInstrumentation()),
        registry_module.resolve_instrumentor(
            HTTPXInstrumentation(capture=HTTPXCaptureSettings(path="omit"))
        ),
    )

    assert [instrumentor.info.id for instrumentor in resolved] == [
        "autobench.pydantic_ai",
        "autobench.openai",
        "autobench.openai_agents",
        "autobench.httpx",
    ]
    http_instrumentor = resolved[-1]
    assert isinstance(http_instrumentor, HTTPX)
    assert http_instrumentor.capture.path == "omit"


def test_registry_reports_missing_extras_without_importing_integrations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry_module, "find_spec", lambda module: None)

    statuses = registry_module.instrumentor_statuses()

    assert [status.name for status in statuses] == [
        "pydantic_ai",
        "openai",
        "openai_agents",
        "httpx",
    ]
    assert all(
        status.compatibility.status is CompatibilityStatus.UNAVAILABLE for status in statuses
    )
    assert all("install autobench[" in status.compatibility.diagnostics[0] for status in statuses)
    with pytest.raises(InstrumentationError, match=r"autobench\[openai\]"):
        registry_module.resolve_instrumentor(OpenAIInstrumentation())


def test_registry_turns_an_available_but_broken_import_into_a_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken_resolver(config: BuiltinInstrumentationConfig) -> Instrumentor:
        raise InstrumentationError(config.kind)

    monkeypatch.setattr(
        registry_module,
        "find_spec",
        lambda module: ModuleSpec(module, loader=None),
    )
    monkeypatch.setattr(registry_module, "resolve_instrumentor", broken_resolver)

    statuses = registry_module.instrumentor_statuses()

    assert all(
        status.compatibility.status is CompatibilityStatus.UNAVAILABLE for status in statuses
    )
    assert all(
        "integration import failed" in status.compatibility.diagnostics[0] for status in statuses
    )


def test_resolver_wraps_broken_optional_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "autobench.instrumentation.openai", None)

    with pytest.raises(InstrumentationError, match="could not be imported"):
        registry_module.resolve_instrumentor(OpenAIInstrumentation())


def test_spec_rejects_duplicate_instrumentation_before_runtime(
    tmp_path: Path,
) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text(
        """benchmark:
  duplicate:
    instrumentation:
      - kind: openai
      - kind: openai
""",
        encoding="utf-8",
    )

    with pytest.raises(SpecValidationError, match="Duplicate instrumentation ids: openai"):
        load_benchmark_spec(path)


def test_benchmark_schema_exposes_typed_instrumentation_completion() -> None:
    generated = benchmark_schema()
    checked_in = json.loads(
        (PROJECT_ROOT / "schemas" / __version__ / "benchmark_schema.json").read_text(
            encoding="utf-8"
        )
    )
    generated_instrumentation = generated["properties"]["benchmark"]["additionalProperties"][
        "properties"
    ]["instrumentation"]
    checked_in_instrumentation = checked_in["properties"]["benchmark"]["additionalProperties"][
        "properties"
    ]["instrumentation"]
    checked_in_capture = checked_in["properties"]["benchmark"]["additionalProperties"][
        "properties"
    ]["capture"]

    assert checked_in == generated
    assert generated_instrumentation["additionalProperties"] is False
    assert set(generated_instrumentation["properties"]) == {
        "all",
        "pydantic_ai",
        "openai",
        "openai_agents",
        "httpx",
    }
    assert checked_in_instrumentation["properties"]["httpx"]["oneOf"][1]["properties"]["capture"][
        "properties"
    ]["path"]["enum"] == ["omit", "hash", "full"]
    all_properties = checked_in_instrumentation["properties"]["all"]["oneOf"][1]["properties"]
    assert all_properties["exclude"]["items"]["enum"] == [
        "pydantic_ai",
        "openai",
        "openai_agents",
        "httpx",
    ]
    assert all_properties["strict"]["default"] is False
    assert all_properties["assets"]["properties"]["representations"]["items"]["enum"] == [
        "definition",
        "effective",
    ]
    assert (
        checked_in_instrumentation["properties"]["pydantic_ai"]["oneOf"][1]["properties"]["assets"][
            "properties"
        ]["discover"]["default"]
        is True
    )
    assert (
        "assets" not in checked_in_instrumentation["properties"]["httpx"]["oneOf"][1]["properties"]
    )
    assert checked_in_capture["properties"]["default_level"]["enum"] == [
        "none",
        "metadata",
        "hash",
        "redacted",
        "full",
    ]
    assert checked_in_capture["properties"]["semantic_overrides"]["additionalProperties"][
        "enum"
    ] == ["none", "metadata", "hash", "redacted", "full"]


def test_cli_doctor_and_trace_render_professional_tables(
    tmp_path: Path,
) -> None:
    status = InstrumentorStatus(
        name="openai",
        extra="openai",
        info=InstrumentorInfo(
            id="autobench.openai",
            version="1",
            target_distribution="openai",
            mechanism=CaptureMechanism.PATCH,
            layer=AbstractionLayer.CLIENT,
            span_kinds=("llm",),
            semantic_families=("llm", "stream"),
        ),
        compatibility=Compatibility.compatible(target_version="2.52.0"),
        capture_mode="metadata, lifecycle, usage",
    )
    output = StringIO()
    console = Console(file=output, force_terminal=False, color_system=None, width=180)
    render_instrumentor_statuses(console, (status,))

    result = run_benchmark_path(
        PROJECT_ROOT / "examples" / "basic" / "autobench.yaml",
        experiment_id="exp_instrumentation_cli",
    )
    first_trace = result.runs[0].trace
    assert first_trace is not None
    partial_trace = first_trace.model_copy(update={"partial": True})
    traced_result = result.model_copy(
        update={
            "runs": [
                result.runs[0].model_copy(update={"trace": partial_trace}),
                result.runs[1].model_copy(update={"trace": None}),
            ]
        }
    )
    render_trace_summary(console, traced_result)
    rendered = output.getvalue()
    assert "ABP Instrumentation Doctor" in rendered
    assert "Compatibility Diagnostics" in rendered
    assert "ABP Trace Summary" in rendered
    assert "Trace Composition" in rendered
    assert "yes" in rendered

    record_dir = tmp_path / "recorded"
    record_experiment(result, record_dir)
    runner = CliRunner()
    doctor = runner.invoke(cli, ["instrumentation", "doctor"])
    trace = runner.invoke(cli, ["instrumentation", "trace", str(record_dir)])

    assert doctor.exit_code == 0
    assert "ABP Instrumentation Doctor" in doctor.output
    assert "Capture Defaults" in doctor.output
    assert trace.exit_code == 0
    assert "ABP Trace Summary" in trace.output
    assert "autobench.manual" in trace.output


def test_cli_reports_missing_configured_instrumentation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "missing-extra.yaml"
    path.write_text(
        """benchmark:
  missing-extra:
    instrumentation:
      openai: {}
""",
        encoding="utf-8",
    )
    monkeypatch.setattr(registry_module, "find_spec", lambda module: None)
    result = CliRunner().invoke(cli, ["run", str(path), "--no-record"])

    assert result.exit_code == 1
    assert "Instrumentation failed" in result.output
    assert "autobench[openai]" in result.output
