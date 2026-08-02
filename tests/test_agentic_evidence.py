from __future__ import annotations as _annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from autobench import (
    ArtifactRef,
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    Direction,
    ErrorRecord,
    ExpectedActionScorer,
    FeedbackRecord,
    MetricPack,
    OutputMetricScorer,
    PydanticAIUsage,
    RunContext,
    RunRecord,
    SampleReason,
    SamplingPolicy,
    ScoreRecord,
    Semantic,
    SemanticRegistry,
    SpanKind,
    SpanRecord,
    SpanSelector,
    SpecValidationError,
    TaskSpec,
    TraceEnvelope,
    Variant,
    attach_trace,
    benchmark_spec_payload_from_yaml_view,
    benchmark_spec_to_yaml_view,
    build_feedback_records,
    build_optimization_feedback_input,
    builtin_metric_pack_registry,
    expected_actions_from_case,
    generated_batch_from_cases,
    mark_generated_case,
    progress_event,
    record_pydantic_ai_usage,
    sample_to_case,
    samples_to_cases,
    select_spans,
)
from autobench.data.ingestion import ProductionSample
from autobench.evaluation.actions import (
    ExpectedAction,
    action_metric_score,
    match_expected_actions,
    observed_action_spans,
)
from autobench.evaluation.scoring import (
    ScoringCall,
    ScoringSpec,
    evaluate_scoring_specs,
    score_records_to_observations,
)
from autobench.metrics.observations import ObservationRole
from autobench.records.recording import RECORD_VERSION
from autobench.runtime.pipeline import EvaluationStatus, RunStatus
from autobench.runtime.progress import ProgressEventKind
from autobench.runtime.tasks import TaskResult, TaskStatus


def test_agentic_semantics_have_parents_and_aliases() -> None:
    registry = SemanticRegistry.with_defaults()

    assert registry.is_a(Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS, Semantic.QUALITY_SCORE)
    assert registry.is_a(Semantic.AGENT_TASK_COMPLETION, Semantic.RESULT_SUCCESS)
    assert registry.normalize("agent.tool.args.correctness") == (
        Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS
    )


def test_trace_envelope_attaches_spans_and_usage_observations() -> None:
    ctx = _ctx(
        expected={
            "actions": [
                {"tool": "lookup_user", "args": {"user_id": "u1"}},
            ]
        }
    )
    started_at = datetime(2026, 1, 1, tzinfo=UTC)
    trace = TraceEnvelope(
        trace_id="trace-1",
        name="support-agent",
        spans=(
            SpanRecord(
                id="span-llm",
                name="call_model",
                kind=SpanKind.LLM,
                started_at=started_at,
                ended_at=started_at,
                duration_seconds=0.5,
                attributes={"model": "demo-model", "provider": "demo-provider"},
                usage={"input_tokens": 12, "output_tokens": 5, "total_tokens": 17},
            ),
            SpanRecord(
                id="span-tool",
                name="lookup_user",
                kind=SpanKind.TOOL,
                started_at=started_at,
                input={"user_id": "u1", "include_orders": True},
            ),
        ),
    )

    observations = attach_trace(ctx, trace)

    assert ctx.spans == list(trace.spans)
    assert {observation.semantic_type for observation in observations} >= {
        Semantic.LLM_TOKENS_INPUT,
        Semantic.LLM_TOKENS_OUTPUT,
        Semantic.LLM_MODEL_NAME,
        Semantic.TIME_LATENCY,
    }
    assert all(observation.tags["trace_id"] == "trace-1" for observation in observations)


@pytest.mark.asyncio
async def test_expected_action_scorer_and_span_selector_score_tool_components() -> None:
    ctx = _ctx(
        expected={
            "actions": [
                {"tool": "lookup_user", "args": {"user_id": "u1"}, "order": 1},
                {"tool": "refund_order", "args": {"order_id": "o1"}, "order": 2},
            ]
        }
    )
    with ctx.span(
        "lookup_user",
        kind=SpanKind.TOOL,
        input={"user_id": "u1"},
        tags={"component": "tools"},
    ):
        pass
    with ctx.span("refund_order", kind=SpanKind.TOOL, input={"order_id": "o1"}):
        pass
    task_result = TaskResult(
        status=TaskStatus.PASSED,
        output={"ok": True},
        observations=list(ctx.observations),
        spans=list(ctx.spans),
    )
    specs: list[ScoringSpec] = [
        ExpectedActionScorer(
            name="tool_arguments",
            semantic_type=Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS,
            metric="arguments",
            span=SpanSelector(kind="tool"),
        )
    ]

    scores = await evaluate_scoring_specs(specs, ctx=ctx, task_result=task_result)
    observations = score_records_to_observations(scores, ctx=ctx)

    assert scores[0].value == 1.0
    assert scores[0].actual_value is not None
    assert observations[0].semantic_type == Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS
    selected = select_spans(
        SpanSelector(kind="tool", tag={"component": "tools"}),
        spans=ctx.spans,
        observations=ctx.observations,
    )
    assert [span.id for span in selected] == ["span_1"]


def test_expected_action_utilities_score_selection_arguments_and_sequence() -> None:
    case = Case(
        id="case-1",
        expected={"tool_calls": [{"tool": "lookup", "args": {"id": 1}}]},
    )
    actions = expected_actions_from_case(case)
    span = SpanRecord(
        id="span-1",
        name="lookup",
        kind=SpanKind.TOOL,
        started_at=datetime.now(UTC),
        input={"id": 1, "extra": True},
    )
    observed = observed_action_spans([span])

    assert action_metric_score(actions, observed, metric="selection") == 1.0
    assert action_metric_score(actions, observed, metric="arguments") == 1.0
    assert action_metric_score(actions, observed, metric="sequence") == 1.0


def test_expected_action_edge_cases_are_explicit() -> None:
    assert expected_actions_from_case(Case(id="scalar", expected="nope")) == []
    assert expected_actions_from_case(Case(id="bad-actions", expected={"actions": "nope"})) == []
    assert expected_actions_from_case(Case(id="skip", expected={"actions": ["bad"]})) == []
    assert (
        expected_actions_from_case(
            Case(id="direct", expected={"actions": [{"id": "direct", "target": "lookup"}]})
        )[0].target
        == "lookup"
    )

    action = ExpectedAction(id="a1", target="missing", input={"items": [1, 2]}, required=False)
    observed = [
        SpanRecord(
            id="span-1",
            name="other",
            kind=SpanKind.TOOL,
            started_at=datetime.now(UTC),
            input={"items": [1]},
        )
    ]
    matches = match_expected_actions([action], observed)

    assert matches[0].matched is False
    assert action_metric_score([action], observed, metric="selection") == 1.0
    required = action.model_copy(update={"required": True})
    assert action_metric_score([required], observed, metric="selection") == 0.0
    assert action_metric_score([required], observed, metric="arguments") == 0.0
    assert action_metric_score([required], observed, metric="sequence") == 0.0

    list_action = ExpectedAction(id="a2", target="other", input={"items": [1, 2]})
    shorter_list_action = ExpectedAction(id="a4", target="other", input={"items": [1]})
    missing_key = ExpectedAction(id="a3", target="other", input={"missing": True})
    assert action_metric_score([list_action], observed, metric="arguments") == 0.0
    assert action_metric_score([shorter_list_action], observed, metric="arguments") == 1.0
    assert action_metric_score([missing_key], observed, metric="arguments") == 0.0


def test_metric_pack_registry_exposes_minimal_builtin_packs() -> None:
    registry = builtin_metric_pack_registry()
    agentic = registry.require("agentic")
    merged = registry.semantic_registry_for(["agentic"])

    assert isinstance(agentic, MetricPack)
    assert "agentic" in registry.names()
    assert merged.types[Semantic.AGENT_TOOL_SELECTION_CORRECTNESS].parent == (
        Semantic.QUALITY_CORRECTNESS
    )
    with pytest.raises(KeyError):
        registry.require("missing")


def test_span_selector_supports_path_semantic_and_negative_filters() -> None:
    ctx = _ctx(expected={})
    with ctx.span("agent", kind=SpanKind.AGENT), ctx.span("lookup", kind=SpanKind.TOOL) as span:
        span.metric(
            "tool_score",
            1.0,
            semantic_type=Semantic.AGENT_TOOL_SELECTION_CORRECTNESS,
        )
    all_spans = select_spans(None, spans=ctx.spans, observations=ctx.observations)

    assert all_spans == ctx.spans
    assert select_spans(SpanSelector(kind="llm"), spans=ctx.spans, observations=[]) == []
    assert select_spans(SpanSelector(name="missing"), spans=ctx.spans, observations=[]) == []
    assert select_spans(SpanSelector(path="agent.missing"), spans=ctx.spans, observations=[]) == []
    assert select_spans(
        SpanSelector(path="agent.lookup"),
        spans=ctx.spans,
        observations=ctx.observations,
    ) == [ctx.spans[1]]
    assert select_spans(
        SpanSelector(semantic_type=Semantic.AGENT_TOOL_SELECTION_CORRECTNESS),
        spans=ctx.spans,
        observations=ctx.observations,
    ) == [ctx.spans[1]]
    assert (
        select_spans(
            SpanSelector(semantic_type=Semantic.MONEY_COST),
            spans=ctx.spans,
            observations=ctx.observations,
        )
        == []
    )


def test_feedback_records_use_none_for_non_failures_and_capture_errors() -> None:
    ctx = _ctx(expected={})
    ctx.check("policy", False, reason="constraint missed", span_id="span-1")
    error = ErrorRecord(error_type="ToolError", message="tool failed", span_id="span-1")
    span = SpanRecord(
        id="span-1",
        name="tool",
        kind=SpanKind.TOOL,
        started_at=datetime.now(UTC),
        error=error,
    )
    record = RunRecord(
        record_version=RECORD_VERSION,
        run_id="run-1",
        experiment_id="exp-1",
        benchmark_id="bench",
        case_id=ctx.case.id,
        variant_id=ctx.variant.id,
        status=RunStatus.FAILED,
        evaluation_status=EvaluationStatus.FAILED,
        task_status=TaskStatus.PASSED,
        case=ctx.case,
        observations=tuple(ctx.observations),
        scores=(
            ScoreRecord(
                name="success",
                semantic_type=Semantic.RESULT_SUCCESS,
                value=True,
            ),
            failed_score_record(
                name="tool_selection",
                semantic_type=Semantic.AGENT_TOOL_SELECTION_CORRECTNESS,
            ),
            ScoreRecord(
                name="errored_score",
                semantic_type=Semantic.QUALITY_SCORE,
                value="bad",
                error=error,
                tags={"reason": 123},
            ),
        ),
        spans=(span,),
        errors=(error,),
    )

    feedback = build_feedback_records(record)
    summary = build_optimization_feedback_input(record)

    assert feedback[0] == FeedbackRecord(
        score_name="tool_selection",
        semantic_type=Semantic.AGENT_TOOL_SELECTION_CORRECTNESS,
        score=0.0,
        passed=False,
        failure_category="low_score",
    )
    assert feedback[1].failure_category == "error"
    assert feedback[2].failure_category == "error"
    assert feedback[3].failure_category == "constraint"
    assert summary.feedback == feedback
    assert summary.trace_excerpt[0]["error"] == "tool failed"
    passing_record = record.model_copy(
        update={
            "scores": (
                ScoreRecord(
                    name="quality",
                    semantic_type=Semantic.QUALITY_SCORE,
                    value=1.0,
                ),
                ScoreRecord(
                    name="diagnostic_text",
                    semantic_type=Semantic.QUALITY_SCORE,
                    value="not numeric",
                ),
            ),
            "errors": (),
            "observations": score_records_to_observations(
                [
                    ScoreRecord(
                        name="passing_constraint",
                        semantic_type=Semantic.RESULT_SUCCESS,
                        value=True,
                        role=ObservationRole.CONSTRAINT,
                    )
                ],
                ctx=ctx,
            ),
        }
    )
    neutral_feedback = build_feedback_records(passing_record)
    assert len(neutral_feedback) == 1
    assert neutral_feedback[0].passed is None
    assert neutral_feedback[0].failure_category is None


def test_pydantic_ai_usage_bridge_records_semantic_observations() -> None:
    ctx = _ctx(expected={})
    usage = PydanticAIUsage(
        requests=1,
        input_tokens=100,
        output_tokens=20,
        total_tokens=120,
        model_name="demo-model",
        provider="demo",
    )

    observations = record_pydantic_ai_usage(ctx, usage)

    assert {observation.semantic_type for observation in observations} >= {
        Semantic.LLM_TOKENS_INPUT,
        Semantic.LLM_TOKENS_OUTPUT,
        Semantic.LLM_TOKENS_TOTAL,
        Semantic.LLM_MODEL_NAME,
        Semantic.LLM_PROVIDER,
    }
    empty_observations = record_pydantic_ai_usage(ctx, PydanticAIUsage())
    assert empty_observations == ()


def test_production_and_synthetic_helpers_preserve_provenance() -> None:
    trace = TraceEnvelope(trace_id="trace-1", name="prod")
    sample = ProductionSample(
        id="sample-1",
        input={"question": "hi"},
        expected={"answer": "hello"},
        trace=trace,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        reason=SampleReason.FAILURE_ONLY,
        privacy_tags=("pii",),
    )

    case = sample_to_case(sample)
    filtered = samples_to_cases(
        [sample],
        policy=SamplingPolicy(reasons=(SampleReason.FAILURE_ONLY,), max_samples=1),
    )
    generated = generated_batch_from_cases(
        filtered,
        generator_asset_version="prompt:v1",
        model_provider="demo",
        model_name="model",
    )

    assert case.metadata["source"] == "production"
    assert case.metadata["timestamp"] == "2026-01-01T00:00:00+00:00"
    assert case.metadata["trace_id"] == "trace-1"
    assert filtered == [case]
    assert generated.cases[0].metadata["source"] == "synthetic"
    assert generated.cases[0].metadata["generator_asset_version"] == "prompt:v1"
    assert samples_to_cases([sample]) == []
    minimal_generated = mark_generated_case(Case(id="minimal"))
    assert minimal_generated.metadata == {
        "source": "synthetic",
        "review_status": "candidate",
    }
    minimal_case = sample_to_case(ProductionSample(id="minimal", reason=SampleReason.RANDOM))
    assert minimal_case.metadata == {
        "source": "production",
        "sample_reason": "random",
        "review_status": "candidate",
    }


def test_trace_attachment_keeps_existing_artifacts_and_records_errors() -> None:
    ctx = _ctx(expected={})
    existing = SpanRecord(id="span-1", name="existing", started_at=datetime.now(UTC))
    artifact = ArtifactRef(id="artifact-1", name="trace", value={"raw": True})
    error = ErrorRecord(error_type="TraceError", message="trace failed", span_id="span-1")
    ctx.spans.append(existing)
    ctx.artifacts.append(artifact)

    observations = attach_trace(
        ctx,
        TraceEnvelope(
            trace_id="trace-2",
            name="trace",
            spans=(existing,),
            errors=(error,),
            raw_artifact=artifact,
        ),
    )

    assert ctx.spans == [existing]
    assert ctx.artifacts == [artifact]
    assert ctx.errors == [error]
    assert any(observation.name == "trace_error" for observation in observations)
    new_artifact = ArtifactRef(id="artifact-2", name="trace-raw", value={"raw": False})
    attach_trace(
        ctx,
        TraceEnvelope(trace_id="trace-raw", name="trace", raw_artifact=new_artifact),
    )
    assert ctx.artifacts[-1] == new_artifact


def test_span_error_trace_and_span_mutators_are_recorded() -> None:
    ctx = _ctx(expected={})
    with ctx.span("llm", kind=SpanKind.LLM) as span:
        span.set_attribute("model", "demo")
        span.set_usage("input_tokens", 1)
        span.set_output("ok")
    errored = SpanRecord(
        id="span-error",
        name="tool",
        kind=SpanKind.TOOL,
        started_at=datetime.now(UTC),
        error=ErrorRecord(error_type="ToolError", message="boom"),
    )

    observations = attach_trace(
        ctx,
        TraceEnvelope(trace_id="trace-3", name="trace", spans=(errored,)),
    )

    assert ctx.spans[0].attributes["model"] == "demo"
    assert ctx.spans[0].usage["input_tokens"] == 1
    assert ctx.spans[0].output == "ok"
    assert any(observation.name == "span_error" for observation in observations)


def test_scoring_call_exposes_selected_spans_alias() -> None:
    span = SpanRecord(id="span-1", name="tool", started_at=datetime.now(UTC))
    call = ScoringCall(
        ctx=_ctx(expected={}),
        task_result=TaskResult(status=TaskStatus.PASSED, output={}),
        selected_spans=[span],
    )

    assert call.spans == [span]


def test_progress_event_separates_known_fields_from_payload() -> None:
    event = progress_event(
        ProgressEventKind.RUN_STARTED,
        "running",
        benchmark_id="bench",
        custom="value",
    )

    assert event.benchmark_id == "bench"
    assert event.data == {"custom": "value"}


def test_expected_action_yaml_dsl_round_trips_and_rejects_bad_shapes() -> None:
    raw = {
        "benchmark": {
            "agentic": {
                "dataset": {"cases": [{"id": "case-1"}]},
                "run": "tests.fake:run",
                "score": {
                    "tool_sequence": {
                        "expected_action": "sequence",
                        "semantic": Semantic.AGENT_TOOL_SEQUENCE_CORRECTNESS,
                    },
                    "tool_arguments": {
                        "expected_action": {"metric": "arguments", "observed_kind": "tool"},
                        "span": {"kind": "tool"},
                        "semantic": Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS,
                    },
                },
            }
        }
    }

    payload = benchmark_spec_payload_from_yaml_view(raw)
    rendered = benchmark_spec_to_yaml_view(
        BenchmarkSpec(
            benchmark=BenchmarkInfo(id="agentic"),
            dataset=DatasetSpec(cases=[Case(id="case-1")]),
            task=TaskSpec(kind="python", target="tests.fake:run"),
            scoring=[
                ExpectedActionScorer(
                    name="tool_arguments",
                    semantic_type=Semantic.AGENT_TOOL_ARGUMENT_CORRECTNESS,
                    metric="arguments",
                    span=SpanSelector(kind="tool"),
                ),
                OutputMetricScorer(
                    name="answer",
                    path="output.answer",
                    semantic_type=Semantic.QUALITY_SCORE,
                    direction=Direction.MAXIMIZE,
                ),
            ],
        )
    )

    assert payload["scoring"][0]["kind"] == "expected_action"
    assert payload["scoring"][0]["metric"] == "sequence"
    assert payload["scoring"][1]["span"]["kind"] == "tool"
    assert rendered["benchmark"]["agentic"]["score"]["tool_arguments"]["span"] == {"kind": "tool"}
    with pytest.raises(SpecValidationError):
        benchmark_spec_payload_from_yaml_view(
            {
                "benchmark": {
                    "bad": {
                        "dataset": {"cases": [{"id": "case-1"}]},
                        "score": {
                            "bad": {
                                "expected_action": 123,
                                "semantic": Semantic.QUALITY_SCORE,
                            }
                        },
                    }
                }
            }
        )


def failed_score_record(*, name: str, semantic_type: str) -> ScoreRecord:
    return ScoreRecord(
        name=name,
        semantic_type=semantic_type,
        value=0.0,
        role=ObservationRole.CONSTRAINT,
    )


def _ctx(*, expected: Any) -> RunContext:
    return RunContext(
        benchmark_id="bench",
        case=Case(id="case-1", expected=expected),
        variant=Variant(id="variant-1"),
    )
