from __future__ import annotations as _annotations

import pytest

from autobench import (
    DEFAULT_SEMANTIC_REGISTRY,
    Direction,
    Observation,
    ObservationKind,
    ObservationQuery,
    ObservationRole,
    ObservationSource,
    Semantic,
    filter_observations,
    project_observations,
)


def _make_observation(
    *,
    observation_id: str,
    name: str,
    kind: ObservationKind = ObservationKind.METRIC,
    semantic_type: str | None = None,
    value: object = 1,
    role: ObservationRole | None = None,
    source: ObservationSource | str | None = None,
    span_id: str | None = None,
    case_id: str | None = "case-1",
    variant_id: str | None = "variant-1",
) -> Observation:
    return Observation(
        id=observation_id,
        name=name,
        kind=kind,
        semantic_type=semantic_type,
        value=value,
        role=role,
        source=source,
        span_id=span_id,
        case_id=case_id,
        variant_id=variant_id,
    )


def test_known_semantic_types_validate() -> None:
    observation = _make_observation(
        observation_id="obs-1",
        name="coverage",
        semantic_type=Semantic.COVERAGE_RATIO,
        value=0.92,
        role=ObservationRole.OBJECTIVE,
    )

    assert observation.semantic_type == Semantic.COVERAGE_RATIO


def test_custom_semantic_types_validate() -> None:
    observation = _make_observation(
        observation_id="obs-2",
        name="custom_chunks",
        semantic_type="myapp.retrieval.chunk_count",
        value=7,
    )

    assert observation.semantic_type == "myapp.retrieval.chunk_count"


def test_factor_observations_cannot_declare_direction() -> None:
    with pytest.raises(ValueError):
        Observation(
            id="obs-3",
            name="model",
            kind=ObservationKind.FACTOR,
            semantic_type=Semantic.LLM_MODEL_NAME,
            value="gpt-x",
            direction=Direction.MAXIMIZE,
        )


@pytest.mark.parametrize("kind", [ObservationKind.ARTIFACT, ObservationKind.EVENT])
def test_artifact_and_event_observations_cannot_declare_direction(
    kind: ObservationKind,
) -> None:
    with pytest.raises(ValueError):
        Observation(
            id="obs-direction",
            name="bad_direction",
            kind=kind,
            value=True,
            direction=Direction.MAXIMIZE,
        )


def test_filter_observations_supports_parent_semantic_type() -> None:
    observations = [
        _make_observation(
            observation_id="obs-4",
            name="spec_model",
            kind=ObservationKind.FACTOR,
            semantic_type="ai.codegen.spec_model",
            value="model-a",
        ),
        _make_observation(
            observation_id="obs-5",
            name="cost",
            semantic_type=Semantic.MONEY_COST,
            value=0.1,
        ),
    ]

    filtered = filter_observations(
        observations,
        parent_semantic_type=Semantic.LLM_MODEL_NAME,
        registry=DEFAULT_SEMANTIC_REGISTRY,
    )

    assert [item.id for item in filtered] == ["obs-4"]


def test_filter_observations_supports_role_source_and_span_filters() -> None:
    observations = [
        _make_observation(
            observation_id="obs-6",
            name="coverage",
            semantic_type=Semantic.COVERAGE_RATIO,
            role=ObservationRole.OBJECTIVE,
            source=ObservationSource.SCORE,
            span_id="span-1",
        ),
        _make_observation(
            observation_id="obs-7",
            name="coverage",
            semantic_type=Semantic.COVERAGE_RATIO,
            role=ObservationRole.DIAGNOSTIC,
            source=ObservationSource.TASK_OBSERVATION,
            span_id="span-2",
        ),
    ]

    filtered = filter_observations(
        observations,
        role=ObservationRole.OBJECTIVE,
        source=ObservationSource.SCORE,
        span_id="span-1",
    )

    assert [item.id for item in filtered] == ["obs-6"]


def test_filter_observations_rejects_each_non_matching_selector() -> None:
    observation = _make_observation(
        observation_id="obs-filter",
        name="coverage",
        kind=ObservationKind.METRIC,
        semantic_type=Semantic.COVERAGE_RATIO,
        role=ObservationRole.OBJECTIVE,
        source=ObservationSource.SCORE,
        span_id="span-1",
    )

    assert filter_observations([observation], name="other") == []
    assert filter_observations([observation], kind=ObservationKind.FACTOR) == []
    assert filter_observations([observation], role=ObservationRole.DIAGNOSTIC) == []
    assert filter_observations([observation], source=ObservationSource.IMPORTED) == []
    assert filter_observations([observation], span_id="span-2") == []
    assert filter_observations([observation], semantic_type=Semantic.MONEY_COST) == []
    assert filter_observations([observation], parent_semantic_type=Semantic.LLM_MODEL_NAME) == []


def test_observation_query_supports_raw_projected_and_source_filtered_values() -> None:
    raw = _make_observation(
        observation_id="obs-query-raw",
        name="quality",
        semantic_type=Semantic.QUALITY_SCORE,
        value=0.4,
        source=ObservationSource.TASK_OBSERVATION,
    )
    score = _make_observation(
        observation_id="obs-query-score",
        name="quality",
        semantic_type=Semantic.QUALITY_SCORE,
        value=0.9,
        source=ObservationSource.SCORE,
    )
    query = ObservationQuery(observations=[raw, score])

    assert query.all() == [raw, score]
    assert query.first_exact(Semantic.MONEY_COST) is None
    assert query.first_related(Semantic.MONEY_COST) is None
    assert query.values(Semantic.QUALITY_SCORE, source=ObservationSource.SCORE) == [0.9]
    assert query.values(Semantic.QUALITY_SCORE, related=True) == [0.9]
    assert query.exact(
        Semantic.QUALITY_SCORE,
        kind=ObservationKind.METRIC,
        projected=False,
    ) == [raw, score]
    assert query.exact(
        Semantic.QUALITY_SCORE,
        kind=(ObservationKind.METRIC, ObservationKind.FACTOR),
        source=ObservationSource.SCORE,
        projected=False,
    ) == [score]


def test_projection_prefers_score_source_for_duplicate_metric() -> None:
    raw_observations = [
        _make_observation(
            observation_id="obs-8",
            name="coverage",
            semantic_type=Semantic.COVERAGE_RATIO,
            value=0.75,
            role=ObservationRole.OBJECTIVE,
            source=ObservationSource.TASK_OBSERVATION,
        ),
        _make_observation(
            observation_id="obs-9",
            name="coverage",
            semantic_type=Semantic.COVERAGE_RATIO,
            value=0.8,
            role=ObservationRole.OBJECTIVE,
            source=ObservationSource.SCORE,
        ),
    ]

    projected = project_observations(raw_observations)

    assert len(projected) == 1
    assert projected[0].observation.id == "obs-9"
    assert not projected[0].ambiguous
    assert [item.id for item in raw_observations] == ["obs-8", "obs-9"]


def test_projection_marks_equal_priority_duplicates_ambiguous() -> None:
    projected = project_observations(
        [
            _make_observation(
                observation_id="obs-10",
                name="coverage",
                semantic_type=Semantic.COVERAGE_RATIO,
                value=0.8,
                role=ObservationRole.OBJECTIVE,
                source=ObservationSource.SCORE,
            ),
            _make_observation(
                observation_id="obs-11",
                name="coverage",
                semantic_type=Semantic.COVERAGE_RATIO,
                value=0.81,
                role=ObservationRole.OBJECTIVE,
                source=ObservationSource.SCORE,
            ),
        ]
    )

    assert len(projected) == 1
    assert projected[0].ambiguous
    assert projected[0].observation.id == "obs-10"


def test_projection_keeps_direct_spans_separate_and_correlates_logical_operations() -> None:
    first = _make_observation(
        observation_id="direct-1",
        name="tokens",
        semantic_type=Semantic.LLM_TOKENS_INPUT,
        value=10,
        source=ObservationSource.DERIVED,
        span_id="span-1",
    ).model_copy(update={"tags": {"abp.measurement_scope": "direct"}})
    second = first.model_copy(update={"id": "direct-2", "span_id": "span-2", "value": 20})

    separate = project_observations([first, second])

    assert len(separate) == 2
    assert {item.key.span_id for item in separate} == {"span-1", "span-2"}

    correlated = project_observations(
        [
            first.model_copy(
                update={"tags": first.tags | {"abp.logical_operation_id": "request-1"}}
            ),
            second.model_copy(
                update={"tags": second.tags | {"abp.logical_operation_id": "request-1"}}
            ),
        ]
    )

    assert len(correlated) == 1
    assert correlated[0].key.span_id is None
    assert correlated[0].key.logical_operation_id == "request-1"
    assert correlated[0].ambiguous


def test_query_prefers_accounting_summary_over_direct_evidence_at_same_source() -> None:
    direct = _make_observation(
        observation_id="direct",
        name="tokens.direct",
        semantic_type=Semantic.LLM_TOKENS_INPUT,
        value=10,
        source=ObservationSource.DERIVED,
        span_id="span-1",
    ).model_copy(update={"tags": {"abp.measurement_scope": "direct"}})
    aggregate = direct.model_copy(
        update={
            "id": "aggregate",
            "name": "tokens.total",
            "value": 30,
            "span_id": None,
            "tags": {"abp.measurement_scope": "aggregate", "abp.summary": True},
        }
    )

    query = ObservationQuery(observations=[direct, aggregate])

    assert query.first_exact(Semantic.LLM_TOKENS_INPUT) == aggregate
    assert query.values(Semantic.LLM_TOKENS_INPUT) == [10, 30]
