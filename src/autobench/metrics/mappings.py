from __future__ import annotations

from collections.abc import Callable, Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, model_validator

from autobench.metrics.semantics import (
    DEFAULT_SEMANTIC_REGISTRY,
    Semantic,
    SemanticRegistry,
    SemanticType,
)
from autobench.protocol.capture import CaptureSession
from autobench.protocol.signals import CaptureLevel, KnownSpanKind, SourceProvenance
from autobench.protocol.traces import Diagnostic, DiagnosticSeverity
from autobench.protocol.values import EvidenceRef, ReferenceKind, SerializedValue

PathSegment: TypeAlias = str | int


class SourceSelector(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    key: str = Field(min_length=1)
    path: tuple[PathSegment, ...] = ()
    deprecated: bool = False


class RenameRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["rename"] = "rename"
    sources: tuple[SourceSelector, ...] = Field(min_length=1)
    semantic_type: SemanticType
    capture: CaptureLevel | None = None
    authority: float = Field(default=1.0, ge=0.0, le=1.0)


class SplitOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    path: tuple[PathSegment, ...]
    semantic_type: SemanticType
    capture: CaptureLevel | None = None
    authority: float = Field(default=1.0, ge=0.0, le=1.0)


class SplitRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["split"] = "split"
    source: SourceSelector
    outputs: tuple[SplitOutput, ...] = Field(min_length=1)


class SpanClassification(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: str = Field(min_length=1)
    kind: str = Field(min_length=1)
    sources: tuple[SourceProvenance, ...] = ()


class ClassificationRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["classify"] = "classify"
    source: SourceSelector
    cases: dict[str, SpanClassification] = Field(min_length=1)


class ReferenceRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reference"] = "reference"
    source: SourceSelector
    semantic_type: SemanticType
    reference_kind: ReferenceKind
    id_path: tuple[PathSegment, ...] = ()
    version_path: tuple[PathSegment, ...] | None = None
    media_type: str | None = Field(default=None, min_length=1)
    authority: float = Field(default=1.0, ge=0.0, le=1.0)


class UnitConversionRule(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["convert_unit"] = "convert_unit"
    source: SourceSelector
    semantic_type: SemanticType
    source_unit: str = Field(min_length=1)
    target_unit: str = Field(min_length=1)
    multiplier: float = 1.0
    offset: float = 0.0
    authority: float = Field(default=1.0, ge=0.0, le=1.0)


MappingRule: TypeAlias = Annotated[
    RenameRule | SplitRule | ClassificationRule | ReferenceRule | UnitConversionRule,
    Field(discriminator="kind"),
]


class SourceMap(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    version: int = Field(ge=1)
    source_system: str = Field(min_length=1)
    convention_version: str = Field(min_length=1)
    instrumentor: str | None = Field(default=None, min_length=1)
    instrumented_library_version: str | None = Field(default=None, min_length=1)
    rules: tuple[MappingRule, ...] = ()


class SourceData(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str = Field(min_length=1)
    convention_version: str = Field(min_length=1)
    values: dict[str, SerializedValue] = Field(default_factory=dict)


class CanonicalFact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    semantic_type: SemanticType
    value: SerializedValue = None
    reference: EvidenceRef | None = None
    unit: str | None = Field(default=None, min_length=1)
    authority: float = Field(default=1.0, ge=0.0, le=1.0)
    sources: tuple[SourceProvenance, ...]


class RetainedSourceFact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    selector: SourceSelector
    value: SerializedValue = None
    reference: EvidenceRef | None = None
    available: bool
    reason: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_availability(self) -> RetainedSourceFact:
        if self.available and self.reason is not None:
            raise ValueError("available source facts cannot have an unavailable reason")
        if not self.available and self.reason is None:
            raise ValueError("unavailable source facts require a reason")
        return self


class SourceSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    system: str = Field(min_length=1)
    convention_version: str = Field(min_length=1)
    source_map_id: str = Field(min_length=1)
    source_map_version: int = Field(ge=1)
    facts: tuple[RetainedSourceFact, ...] = ()


class CanonicalizationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    source_map_id: str = Field(min_length=1)
    source_map_version: int = Field(ge=1)
    facts: tuple[CanonicalFact, ...] = ()
    classification: SpanClassification | None = None
    diagnostics: tuple[Diagnostic, ...] = ()
    source_snapshot: SourceSnapshot
    replayed_from: str | None = Field(default=None, min_length=1)


class MappingStatus(StrEnum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


SourceLookup: TypeAlias = Callable[[SourceSelector], tuple[MappingStatus, SerializedValue]]


def canonicalize(
    data: SourceData,
    source_map: SourceMap,
    *,
    capture: CaptureSession | None = None,
    registry: SemanticRegistry | None = None,
) -> CanonicalizationResult:
    session = CaptureSession() if capture is None else capture
    active_registry = DEFAULT_SEMANTIC_REGISTRY if registry is None else registry
    diagnostics: list[Diagnostic] = []
    retained: dict[SourceSelector, RetainedSourceFact] = {}

    if data.system != source_map.source_system:
        diagnostics.append(
            Diagnostic(
                code="source_system_mismatch",
                message="source data system does not match the source map",
                severity=DiagnosticSeverity.ERROR,
                details={"actual": data.system, "expected": source_map.source_system},
            )
        )
    if data.convention_version != source_map.convention_version:
        diagnostics.append(
            Diagnostic(
                code="source_version_mismatch",
                message="source convention version does not match the source map",
                severity=DiagnosticSeverity.ERROR,
                details={
                    "actual": data.convention_version,
                    "expected": source_map.convention_version,
                },
            )
        )
    if diagnostics:
        snapshot = SourceSnapshot(
            system=data.system,
            convention_version=data.convention_version,
            source_map_id=source_map.id,
            source_map_version=source_map.version,
        )
        return CanonicalizationResult(
            source_map_id=source_map.id,
            source_map_version=source_map.version,
            diagnostics=tuple(diagnostics),
            source_snapshot=snapshot,
        )

    def lookup(selector: SourceSelector) -> tuple[MappingStatus, SerializedValue]:
        available, value = resolve_source_value(data.values, selector)
        if available:
            return MappingStatus.AVAILABLE, value
        return MappingStatus.UNAVAILABLE, None

    facts, classification = _apply_source_map(
        source_map,
        lookup,
        session,
        active_registry,
        diagnostics,
        retained,
    )
    snapshot = SourceSnapshot(
        system=data.system,
        convention_version=data.convention_version,
        source_map_id=source_map.id,
        source_map_version=source_map.version,
        facts=tuple(retained.values()),
    )
    return CanonicalizationResult(
        source_map_id=source_map.id,
        source_map_version=source_map.version,
        facts=tuple(facts),
        classification=classification,
        diagnostics=tuple(diagnostics),
        source_snapshot=snapshot,
    )


def recanonicalize(
    snapshot: SourceSnapshot,
    source_map: SourceMap,
    *,
    capture: CaptureSession | None = None,
    registry: SemanticRegistry | None = None,
) -> CanonicalizationResult:
    session = CaptureSession() if capture is None else capture
    active_registry = DEFAULT_SEMANTIC_REGISTRY if registry is None else registry
    diagnostics: list[Diagnostic] = []
    retained = {fact.selector: fact for fact in snapshot.facts}

    def lookup(selector: SourceSelector) -> tuple[MappingStatus, SerializedValue]:
        fact = retained.get(selector)
        if fact is not None:
            if fact.available:
                return MappingStatus.AVAILABLE, fact.value
            diagnostics.append(
                Diagnostic(
                    code="source_fact_unavailable",
                    message="source fact was not retained in replayable form",
                    path=source_selector_label(selector),
                    details={"reason": fact.reason or "unavailable"},
                )
            )
            return MappingStatus.UNAVAILABLE, None
        for candidate in snapshot.facts:
            if (
                candidate.available
                and candidate.selector.key == selector.key
                and selector.path[: len(candidate.selector.path)] == candidate.selector.path
            ):
                suffix = selector.path[len(candidate.selector.path) :]
                available, value = resolve_nested_value(candidate.value, suffix)
                if available:
                    return MappingStatus.AVAILABLE, value
        diagnostics.append(
            Diagnostic(
                code="source_fact_unavailable",
                message="source fact was not present in the retained snapshot",
                path=source_selector_label(selector),
                details={"reason": "not_retained"},
            )
        )
        return MappingStatus.UNAVAILABLE, None

    if snapshot.system != source_map.source_system:
        diagnostics.append(
            Diagnostic(
                code="source_system_mismatch",
                message="retained source system does not match the source map",
                severity=DiagnosticSeverity.ERROR,
                details={"actual": snapshot.system, "expected": source_map.source_system},
            )
        )
    if snapshot.convention_version != source_map.convention_version:
        diagnostics.append(
            Diagnostic(
                code="source_version_mismatch",
                message="retained convention version does not match the source map",
                severity=DiagnosticSeverity.ERROR,
                details={
                    "actual": snapshot.convention_version,
                    "expected": source_map.convention_version,
                },
            )
        )
    if any(diagnostic.severity is DiagnosticSeverity.ERROR for diagnostic in diagnostics):
        return CanonicalizationResult(
            source_map_id=source_map.id,
            source_map_version=source_map.version,
            diagnostics=tuple(diagnostics),
            source_snapshot=snapshot,
            replayed_from=f"{snapshot.source_map_id}@{snapshot.source_map_version}",
        )

    facts, classification = _apply_source_map(
        source_map,
        lookup,
        session,
        active_registry,
        diagnostics,
        {},
        retain=False,
    )
    return CanonicalizationResult(
        source_map_id=source_map.id,
        source_map_version=source_map.version,
        facts=tuple(facts),
        classification=classification,
        diagnostics=tuple(diagnostics),
        source_snapshot=snapshot,
        replayed_from=f"{snapshot.source_map_id}@{snapshot.source_map_version}",
    )


def resolve_source_value(
    values: Mapping[str, SerializedValue],
    selector: SourceSelector,
) -> tuple[bool, SerializedValue]:
    if selector.key not in values:
        return False, None
    return resolve_nested_value(values[selector.key], selector.path)


def resolve_nested_value(
    value: SerializedValue,
    path: tuple[PathSegment, ...],
) -> tuple[bool, SerializedValue]:
    current = value
    for segment in path:
        if isinstance(segment, str) and isinstance(current, dict) and segment in current:
            current = current[segment]
            continue
        if isinstance(segment, int) and isinstance(current, list) and 0 <= segment < len(current):
            current = current[segment]
            continue
        return False, None
    return True, current


def source_selector_label(selector: SourceSelector) -> str:
    label = selector.key
    for segment in selector.path:
        label += f"[{segment}]" if isinstance(segment, int) else f".{segment}"
    return label


def source_map_to_yaml_view(source_map: SourceMap) -> dict[str, Any]:
    source: dict[str, Any] = {
        "system": source_map.source_system,
        "convention": source_map.convention_version,
    }
    if source_map.instrumentor is not None:
        source["instrumentor"] = source_map.instrumentor
    if source_map.instrumented_library_version is not None:
        source["library_version"] = source_map.instrumented_library_version
    return {
        "record": {"type": "source_map", "version": 1},
        "source_map": {
            "id": source_map.id,
            "version": source_map.version,
            "source": source,
            "rules": source_map.model_dump(mode="json")["rules"],
        },
    }


def source_map_payload_from_yaml_view(raw: Any) -> dict[str, Any]:
    payload = raw
    if isinstance(raw, dict):
        header = raw.get("record")
        if isinstance(header, dict) and header.get("type") == "source_map":
            payload = raw.get("source_map")
    if not isinstance(payload, dict):
        raise TypeError("source_map must be a mapping")
    normalized = dict(payload)
    source = normalized.pop("source", None)
    if source is None:
        return normalized
    if not isinstance(source, dict):
        raise TypeError("source_map.source must be a mapping")
    normalized["source_system"] = source.get("system")
    normalized["convention_version"] = source.get("convention")
    if "instrumentor" in source:
        normalized["instrumentor"] = source["instrumentor"]
    if "library_version" in source:
        normalized["instrumented_library_version"] = source["library_version"]
    return normalized


def _apply_source_map(
    source_map: SourceMap,
    lookup: SourceLookup,
    capture: CaptureSession,
    registry: SemanticRegistry,
    diagnostics: list[Diagnostic],
    retained: dict[SourceSelector, RetainedSourceFact],
    *,
    retain: bool = True,
) -> tuple[list[CanonicalFact], SpanClassification | None]:
    facts: list[CanonicalFact] = []
    classifications: list[SpanClassification] = []
    for rule in source_map.rules:
        if isinstance(rule, RenameRule):
            resolved = [
                (selector, value)
                for selector in rule.sources
                for status, value in (lookup(selector),)
                if status is MappingStatus.AVAILABLE
            ]
            if not resolved:
                continue
            for selector, value in resolved:
                if retain:
                    _retain_source(
                        retained,
                        selector,
                        value,
                        rule.semantic_type,
                        capture,
                        source_map,
                    )
                if selector.deprecated:
                    diagnostics.append(
                        Diagnostic(
                            code="deprecated_source_key",
                            message="a deprecated source key supplied canonical evidence",
                            path=source_selector_label(selector),
                            semantic_type=rule.semantic_type,
                        )
                    )
            if any(value != resolved[0][1] for _, value in resolved[1:]):
                diagnostics.append(
                    Diagnostic(
                        code="ambiguous_source_aliases",
                        message="source aliases supplied conflicting values",
                        severity=DiagnosticSeverity.ERROR,
                        semantic_type=rule.semantic_type,
                        details={
                            "sources": [source_selector_label(selector) for selector, _ in resolved]
                        },
                    )
                )
                continue
            fact = _capture_fact(
                resolved[0][1],
                rule.semantic_type,
                tuple(_provenance(source_map, selector) for selector, _ in resolved),
                rule.authority,
                rule.capture,
                None,
                capture,
                registry,
                diagnostics,
            )
            if fact is not None:
                _append_fact(facts, fact, diagnostics)
            continue

        status, raw_value = lookup(rule.source)
        if status is MappingStatus.UNAVAILABLE:
            continue
        if isinstance(rule, SplitRule):
            for output in rule.outputs:
                available, value = resolve_nested_value(raw_value, output.path)
                if not available:
                    diagnostics.append(
                        Diagnostic(
                            code="source_path_unavailable",
                            message="split mapping source path was unavailable",
                            path=source_selector_label(rule.source)
                            + "".join(
                                f"[{segment}]" if isinstance(segment, int) else f".{segment}"
                                for segment in output.path
                            ),
                            semantic_type=output.semantic_type,
                        )
                    )
                    continue
                selector = rule.source.model_copy(update={"path": rule.source.path + output.path})
                if retain:
                    _retain_source(
                        retained,
                        selector,
                        value,
                        output.semantic_type,
                        capture,
                        source_map,
                    )
                fact = _capture_fact(
                    value,
                    output.semantic_type,
                    (_provenance(source_map, selector),),
                    output.authority,
                    output.capture,
                    None,
                    capture,
                    registry,
                    diagnostics,
                )
                if fact is not None:
                    _append_fact(facts, fact, diagnostics)
            continue
        if isinstance(rule, ClassificationRule):
            if retain:
                _retain_source(
                    retained,
                    rule.source,
                    raw_value,
                    Semantic.OPERATION_NAME,
                    capture,
                    source_map,
                )
            if not isinstance(raw_value, str) or raw_value not in rule.cases:
                diagnostics.append(
                    Diagnostic(
                        code="classification_unavailable",
                        message="source operation could not be classified confidently",
                        path=source_selector_label(rule.source),
                    )
                )
                continue
            classification = rule.cases[raw_value].model_copy(
                update={"sources": (_provenance(source_map, rule.source),)}
            )
            classifications.append(classification)
            continue
        if isinstance(rule, ReferenceRule):
            available, reference_id = resolve_nested_value(raw_value, rule.id_path)
            if not available or not isinstance(reference_id, str):
                diagnostics.append(
                    Diagnostic(
                        code="reference_mapping_failed",
                        message="reference mapping did not resolve a string identifier",
                        severity=DiagnosticSeverity.ERROR,
                        path=source_selector_label(rule.source),
                        semantic_type=rule.semantic_type,
                    )
                )
                continue
            version: str | None = None
            if rule.version_path is not None:
                version_available, raw_version = resolve_nested_value(raw_value, rule.version_path)
                if not version_available or not isinstance(raw_version, str):
                    diagnostics.append(
                        Diagnostic(
                            code="reference_mapping_failed",
                            message="reference mapping did not resolve a string version",
                            severity=DiagnosticSeverity.ERROR,
                            path=source_selector_label(rule.source),
                            semantic_type=rule.semantic_type,
                        )
                    )
                    continue
                version = raw_version
            if retain:
                _retain_source(
                    retained,
                    rule.source,
                    raw_value,
                    rule.semantic_type,
                    capture,
                    source_map,
                )
            fact = CanonicalFact(
                semantic_type=_normalize_semantic(rule.semantic_type, registry, diagnostics),
                reference=EvidenceRef(
                    kind=rule.reference_kind,
                    id=reference_id,
                    version=version,
                    media_type=rule.media_type,
                ),
                authority=rule.authority,
                sources=(_provenance(source_map, rule.source),),
            )
            _append_fact(facts, fact, diagnostics)
            continue
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            diagnostics.append(
                Diagnostic(
                    code="unit_conversion_failed",
                    message="unit conversion requires a numeric source value",
                    severity=DiagnosticSeverity.ERROR,
                    path=source_selector_label(rule.source),
                    semantic_type=rule.semantic_type,
                )
            )
            continue
        if retain:
            _retain_source(
                retained,
                rule.source,
                raw_value,
                rule.semantic_type,
                capture,
                source_map,
            )
        converted = raw_value * rule.multiplier + rule.offset
        fact = _capture_fact(
            converted,
            rule.semantic_type,
            (_provenance(source_map, rule.source),),
            rule.authority,
            None,
            rule.target_unit,
            capture,
            registry,
            diagnostics,
        )
        if fact is not None:
            _append_fact(facts, fact, diagnostics)

    classification: SpanClassification | None = None
    if classifications:
        classification = classifications[0]
        if any(
            item.operation != classification.operation or item.kind != classification.kind
            for item in classifications[1:]
        ):
            diagnostics.append(
                Diagnostic(
                    code="ambiguous_classification",
                    message="source mappings produced conflicting span classifications",
                    severity=DiagnosticSeverity.ERROR,
                )
            )
            classification = None
        else:
            classification = classification.model_copy(
                update={
                    "sources": tuple(source for item in classifications for source in item.sources)
                }
            )
    return facts, classification


def _capture_fact(
    value: SerializedValue,
    semantic_type: SemanticType,
    sources: tuple[SourceProvenance, ...],
    authority: float,
    level: CaptureLevel | None,
    unit: str | None,
    capture: CaptureSession,
    registry: SemanticRegistry,
    diagnostics: list[Diagnostic],
) -> CanonicalFact | None:
    normalized_semantic = _normalize_semantic(semantic_type, registry, diagnostics)
    result = capture.capture(
        value,
        semantic_type=normalized_semantic,
        path=tuple(normalized_semantic.split(".")),
        level=level,
    )
    diagnostics.extend(result.diagnostics)
    if result.omitted and result.reference is None:
        return None
    return CanonicalFact(
        semantic_type=normalized_semantic,
        value=result.value,
        reference=result.reference,
        unit=unit,
        authority=authority,
        sources=sources,
    )


def _append_fact(
    facts: list[CanonicalFact],
    fact: CanonicalFact,
    diagnostics: list[Diagnostic],
) -> None:
    competing = [item for item in facts if item.semantic_type == fact.semantic_type]
    for index, item in enumerate(facts):
        if (
            item.semantic_type == fact.semantic_type
            and item.value == fact.value
            and item.reference == fact.reference
            and item.unit == fact.unit
        ):
            facts[index] = item.model_copy(
                update={
                    "authority": max(item.authority, fact.authority),
                    "sources": tuple(dict.fromkeys(item.sources + fact.sources)),
                }
            )
            return
    if any(item.value != fact.value or item.reference != fact.reference for item in competing):
        diagnostics.append(
            Diagnostic(
                code="ambiguous_canonical_value",
                message="multiple source mappings produced conflicting canonical values",
                severity=DiagnosticSeverity.ERROR,
                semantic_type=fact.semantic_type,
            )
        )
    facts.append(fact)


def _normalize_semantic(
    semantic_type: SemanticType,
    registry: SemanticRegistry,
    diagnostics: list[Diagnostic],
) -> str:
    normalized = registry.normalize(semantic_type) or semantic_type
    if normalized != semantic_type:
        diagnostics.append(
            Diagnostic(
                code="deprecated_semantic",
                message="source map target was normalized to its canonical semantic",
                semantic_type=normalized,
                details={"source_semantic": semantic_type},
            )
        )
    return normalized


def _retain_source(
    retained: dict[SourceSelector, RetainedSourceFact],
    selector: SourceSelector,
    value: SerializedValue,
    semantic_type: SemanticType,
    capture: CaptureSession,
    source_map: SourceMap,
) -> None:
    if selector in retained:
        return
    if not capture.policy.retain_source_attributes:
        retained[selector] = RetainedSourceFact(
            selector=selector,
            available=False,
            reason="source_retention_disabled",
        )
        return
    result = capture.capture(
        value,
        semantic_type=semantic_type,
        path=(
            "source",
            source_map.source_system,
            *selector.key.split("."),
            *(str(segment) for segment in selector.path),
        ),
    )
    replayable = (
        not result.omitted
        and not result.truncated
        and result.reference is None
        and result.level in {CaptureLevel.METADATA, CaptureLevel.FULL}
        and result.value == value
    )
    unavailable_reason: str | None = None
    if not replayable:
        if result.reference is not None:
            unavailable_reason = "artifact_reference"
        elif result.truncated:
            unavailable_reason = "truncated"
        elif result.diagnostics:
            unavailable_reason = result.diagnostics[0].code
        else:
            unavailable_reason = f"capture_{result.level}"
    retained[selector] = RetainedSourceFact(
        selector=selector,
        value=result.value,
        reference=result.reference,
        available=replayable,
        reason=unavailable_reason,
    )


def _provenance(source_map: SourceMap, selector: SourceSelector) -> SourceProvenance:
    return SourceProvenance(
        system=source_map.source_system,
        key=selector.key,
        path=selector.path,
        convention_version=source_map.convention_version,
        source_map_id=source_map.id,
        source_map_version=source_map.version,
        instrumentor=source_map.instrumentor,
        instrumented_library_version=source_map.instrumented_library_version,
    )


def _selector(key: str, *path: PathSegment, deprecated: bool = False) -> SourceSelector:
    return SourceSelector(key=key, path=path, deprecated=deprecated)


OTEL_GENAI_SOURCE_MAP = SourceMap(
    id="otel.genai",
    version=1,
    source_system="otel.genai",
    convention_version="1.43.0",
    rules=(
        RenameRule(
            sources=(_selector("gen_ai.request.model"),), semantic_type=Semantic.LLM_MODEL_REQUESTED
        ),
        RenameRule(
            sources=(_selector("gen_ai.response.model"),), semantic_type=Semantic.LLM_MODEL_RESPONSE
        ),
        RenameRule(
            sources=(
                _selector("gen_ai.provider.name"),
                _selector("gen_ai.system", deprecated=True),
            ),
            semantic_type=Semantic.LLM_PROVIDER_NAME,
        ),
        RenameRule(
            sources=(_selector("gen_ai.request.temperature"),),
            semantic_type=Semantic.LLM_TEMPERATURE,
        ),
        RenameRule(
            sources=(
                _selector("gen_ai.usage.input_tokens"),
                _selector("gen_ai.usage.prompt_tokens", deprecated=True),
            ),
            semantic_type=Semantic.LLM_TOKENS_INPUT,
        ),
        RenameRule(
            sources=(
                _selector("gen_ai.usage.output_tokens"),
                _selector("gen_ai.usage.completion_tokens", deprecated=True),
            ),
            semantic_type=Semantic.LLM_TOKENS_OUTPUT,
        ),
        RenameRule(
            sources=(_selector("gen_ai.usage.cache_read.input_tokens"),),
            semantic_type=Semantic.LLM_TOKENS_CACHED_INPUT,
        ),
        RenameRule(
            sources=(_selector("gen_ai.usage.cache_creation.input_tokens"),),
            semantic_type=Semantic.LLM_TOKENS_CACHE_WRITE,
        ),
        RenameRule(
            sources=(_selector("gen_ai.usage.reasoning.output_tokens"),),
            semantic_type=Semantic.LLM_TOKENS_REASONING_OUTPUT,
        ),
        RenameRule(
            sources=(
                _selector("gen_ai.response.time_to_first_chunk"),
                _selector("gen_ai.client.operation.time_to_first_chunk"),
            ),
            semantic_type=Semantic.TIME_FIRST_CHUNK,
        ),
        RenameRule(sources=(_selector("gen_ai.agent.id"),), semantic_type=Semantic.AGENT_ID),
        RenameRule(sources=(_selector("gen_ai.agent.name"),), semantic_type=Semantic.AGENT_NAME),
        RenameRule(
            sources=(_selector("gen_ai.agent.version"),), semantic_type=Semantic.AGENT_VERSION
        ),
        RenameRule(
            sources=(_selector("gen_ai.workflow.name"),), semantic_type=Semantic.WORKFLOW_NAME
        ),
        RenameRule(sources=(_selector("gen_ai.tool.name"),), semantic_type=Semantic.TOOL_NAME),
        RenameRule(sources=(_selector("gen_ai.tool.type"),), semantic_type=Semantic.TOOL_TYPE),
        RenameRule(
            sources=(_selector("gen_ai.tool.definitions"),), semantic_type=Semantic.TOOL_DEFINITIONS
        ),
        RenameRule(
            sources=(_selector("gen_ai.tool.call.id"),), semantic_type=Semantic.TOOL_CALL_ID
        ),
        RenameRule(
            sources=(_selector("gen_ai.tool.call.arguments"),),
            semantic_type=Semantic.TOOL_CALL_ARGUMENTS,
        ),
        RenameRule(
            sources=(_selector("gen_ai.tool.call.result"),), semantic_type=Semantic.TOOL_CALL_RESULT
        ),
        RenameRule(
            sources=(_selector("gen_ai.conversation.id"),), semantic_type=Semantic.CONVERSATION_ID
        ),
        RenameRule(
            sources=(_selector("gen_ai.input.messages"),), semantic_type=Semantic.MESSAGE_INPUT
        ),
        RenameRule(
            sources=(_selector("gen_ai.output.messages"),), semantic_type=Semantic.MESSAGE_OUTPUT
        ),
        RenameRule(
            sources=(_selector("gen_ai.system_instructions"),), semantic_type=Semantic.PROMPT_SYSTEM
        ),
        RenameRule(
            sources=(_selector("gen_ai.retrieval.query.text"),),
            semantic_type=Semantic.RETRIEVAL_QUERY,
        ),
        RenameRule(
            sources=(_selector("gen_ai.retrieval.documents"),),
            semantic_type=Semantic.RETRIEVAL_DOCUMENTS,
        ),
        RenameRule(
            sources=(_selector("gen_ai.evaluation.name"),), semantic_type=Semantic.EVALUATION_NAME
        ),
        RenameRule(
            sources=(_selector("gen_ai.evaluation.score.value"),),
            semantic_type=Semantic.EVALUATION_SCORE,
        ),
        RenameRule(
            sources=(_selector("gen_ai.evaluation.score.label"),),
            semantic_type=Semantic.EVALUATION_LABEL,
        ),
        RenameRule(
            sources=(_selector("gen_ai.evaluation.explanation"),),
            semantic_type=Semantic.EVALUATION_EXPLANATION,
        ),
        ClassificationRule(
            source=_selector("gen_ai.operation.name"),
            cases={
                "chat": SpanClassification(operation="chat", kind=KnownSpanKind.LLM),
                "text_completion": SpanClassification(
                    operation="text_completion", kind=KnownSpanKind.LLM
                ),
                "generate_content": SpanClassification(
                    operation="generate_content", kind=KnownSpanKind.LLM
                ),
                "embeddings": SpanClassification(
                    operation="embeddings", kind=KnownSpanKind.EMBEDDING
                ),
                "execute_tool": SpanClassification(
                    operation="execute_tool", kind=KnownSpanKind.TOOL
                ),
                "invoke_agent": SpanClassification(
                    operation="invoke_agent", kind=KnownSpanKind.AGENT
                ),
                "invoke_workflow": SpanClassification(
                    operation="invoke_workflow", kind=KnownSpanKind.WORKFLOW
                ),
                "retrieval": SpanClassification(
                    operation="retrieval", kind=KnownSpanKind.RETRIEVER
                ),
            },
        ),
    ),
)


OPENINFERENCE_SOURCE_MAP = SourceMap(
    id="openinference",
    version=1,
    source_system="openinference",
    convention_version="1.0",
    rules=(
        RenameRule(
            sources=(_selector("llm.model_name"),), semantic_type=Semantic.LLM_MODEL_RESPONSE
        ),
        RenameRule(
            sources=(_selector("llm.token_count.prompt"),), semantic_type=Semantic.LLM_TOKENS_INPUT
        ),
        RenameRule(
            sources=(_selector("llm.token_count.completion"),),
            semantic_type=Semantic.LLM_TOKENS_OUTPUT,
        ),
        RenameRule(
            sources=(_selector("llm.token_count.total"),), semantic_type=Semantic.LLM_TOKENS_TOTAL
        ),
        RenameRule(
            sources=(_selector("llm.input_messages"),), semantic_type=Semantic.MESSAGE_INPUT
        ),
        RenameRule(
            sources=(_selector("llm.output_messages"),), semantic_type=Semantic.MESSAGE_OUTPUT
        ),
        RenameRule(sources=(_selector("tool.name"),), semantic_type=Semantic.TOOL_NAME),
        RenameRule(
            sources=(_selector("tool.parameters"),), semantic_type=Semantic.TOOL_CALL_ARGUMENTS
        ),
    ),
)


OPENAI_RESPONSES_SOURCE_MAP = SourceMap(
    id="openai.responses",
    version=1,
    source_system="openai.responses",
    convention_version="v1",
    instrumentor="autobench.openai",
    rules=(
        RenameRule(
            sources=(_selector("request", "model"),), semantic_type=Semantic.LLM_MODEL_REQUESTED
        ),
        RenameRule(
            sources=(_selector("response", "model"),), semantic_type=Semantic.LLM_MODEL_RESPONSE
        ),
        RenameRule(
            sources=(_selector("response", "usage", "input_tokens"),),
            semantic_type=Semantic.LLM_TOKENS_INPUT,
        ),
        RenameRule(
            sources=(_selector("response", "usage", "output_tokens"),),
            semantic_type=Semantic.LLM_TOKENS_OUTPUT,
        ),
        RenameRule(
            sources=(_selector("response", "usage", "input_tokens_details", "cached_tokens"),),
            semantic_type=Semantic.LLM_TOKENS_CACHED_INPUT,
        ),
        RenameRule(
            sources=(_selector("response", "usage", "output_tokens_details", "reasoning_tokens"),),
            semantic_type=Semantic.LLM_TOKENS_REASONING_OUTPUT,
        ),
    ),
)


__all__ = (
    "CanonicalFact",
    "CanonicalizationResult",
    "ClassificationRule",
    "MappingRule",
    "MappingStatus",
    "OPENAI_RESPONSES_SOURCE_MAP",
    "OPENINFERENCE_SOURCE_MAP",
    "OTEL_GENAI_SOURCE_MAP",
    "PathSegment",
    "ReferenceRule",
    "RenameRule",
    "RetainedSourceFact",
    "SourceData",
    "SourceMap",
    "SourceSelector",
    "SourceSnapshot",
    "SpanClassification",
    "SplitOutput",
    "SplitRule",
    "UnitConversionRule",
    "canonicalize",
    "recanonicalize",
    "resolve_nested_value",
    "resolve_source_value",
    "source_map_payload_from_yaml_view",
    "source_map_to_yaml_view",
    "source_selector_label",
)
