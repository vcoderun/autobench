from __future__ import annotations as _annotations

from collections.abc import Callable
from pathlib import Path
from textwrap import dedent

import pytest

from autobench import (
    BenchmarkInfo,
    BenchmarkSpec,
    Case,
    DatasetSpec,
    Direction,
    ErrorRecord,
    ObservationRole,
    ObservationSource,
    RunContext,
    RunStatus,
    Semantic,
    TaskResolutionError,
    TaskResult,
    TaskSpec,
    TaskStatus,
    Variant,
    get_active_run_context,
    instrument_method,
    project_observations,
    run_benchmark_spec,
)
from autobench.evaluation.scoring import (
    ExactScorer,
    OutputMetricScorer,
    PassFailScorer,
    PythonScorer,
    SchemaScorer,
    ScoreRecord,
    ScoringCall,
    evaluate_scoring_specs,
    resolve_dotted_path,
    resolve_python_scorer,
    score_records_to_observations,
    validate_schema_value,
)
from autobench.runtime.instrumentation import (
    InstrumentFactorSpec,
    InstrumentMetricSpec,
    reset_active_run_context,
    set_active_run_context,
)


async def test_output_metric_and_pass_fail_scoring_emit_score_observations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "score_tasks.py",
        """
        def run(ctx, case):
            return {'success': True, 'coverage': 0.75}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="score-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="score_tasks:run"),
        variants=[Variant(id="variant_1")],
        scoring=[
            PassFailScorer(
                name="success",
                path="output.success",
                semantic_type=Semantic.RESULT_SUCCESS,
                role=ObservationRole.OBJECTIVE,
            ),
            OutputMetricScorer(
                name="coverage",
                path="output.coverage",
                semantic_type=Semantic.COVERAGE_RATIO,
                direction=Direction.MAXIMIZE,
                role=ObservationRole.OBJECTIVE,
            ),
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    observations = result.runs[0].task_result.observations
    assert any(
        observation.name == "success"
        and observation.semantic_type == Semantic.RESULT_SUCCESS
        and observation.source == ObservationSource.SCORE
        and observation.value is True
        for observation in observations
    )
    assert any(
        observation.name == "coverage"
        and observation.semantic_type == Semantic.COVERAGE_RATIO
        and observation.source == ObservationSource.SCORE
        and observation.value == 0.75
        for observation in observations
    )


async def test_exact_and_python_scorers_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "python_score_tasks.py",
        """
        def run(ctx, case):
            return {'answer': 'ok', 'coverage': 0.8}

        def custom_score(call):
            return len(call.output['answer'])
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="python-score-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1", expected={"answer": "ok"})]),
        task=TaskSpec(kind="python", target="python_score_tasks:run"),
        variants=[Variant(id="variant_1")],
        scoring=[
            ExactScorer(
                name="answer_match",
                actual="output.answer",
                expected="case.expected.answer",
                semantic_type=Semantic.QUALITY_CORRECTNESS,
            ),
            PythonScorer(
                name="answer_length",
                target="python_score_tasks:custom_score",
                semantic_type="text.output.length",
            ),
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert [score.value for score in result.runs[0].scores] == [1.0, 2]
    assert result.runs[0].status is RunStatus.PASSED


async def test_schema_scorer_and_structured_scorer_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "schema_score_tasks.py",
        """
        def run(ctx, case):
            return {'answer': 'ok'}

        def broken_score(call):
            raise RuntimeError('scorer exploded')
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="schema-score-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="schema_score_tasks:run"),
        variants=[Variant(id="variant_1")],
        scoring=[
            SchemaScorer(
                name="response_schema",
                path="output",
                schema={"type": "object", "required": ["answer"]},
                semantic_type="output.schema.valid",
            ),
            PythonScorer(
                name="broken",
                target="schema_score_tasks:broken_score",
                semantic_type=Semantic.QUALITY_SCORE,
            ),
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert result.runs[0].scores[0].value is True
    assert result.runs[0].scores[1].error is not None
    assert result.runs[0].scores[1].error.error_type == "RuntimeError"
    assert result.runs[0].status is RunStatus.ERRORED


async def test_optional_scorer_errors_do_not_fail_the_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "optional_score_tasks.py",
        """
        def run(ctx, case):
            return {"answer": "ok"}

        def broken_score(call):
            raise RuntimeError("optional exploded")
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="optional-score-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="optional_score_tasks:run"),
        variants=[Variant(id="variant_1")],
        scoring=[
            PythonScorer(
                name="optional_broken",
                target="optional_score_tasks:broken_score",
                semantic_type=Semantic.QUALITY_SCORE,
                optional=True,
            )
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert result.runs[0].scores[0].error is not None
    assert result.runs[0].status is RunStatus.PASSED


async def test_async_python_scorer_can_return_score_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "async_score_tasks.py",
        """
        from autobench.evaluation.scoring import ScoreRecord

        def run(ctx, case):
            return {"score": 0.77}

        async def custom_score(call):
            return ScoreRecord(
                name="async_quality",
                semantic_type="quality.score",
                value=call.output["score"],
            )
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="async-score-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="async_score_tasks:run"),
        variants=[Variant(id="variant_1")],
        scoring=[
            PythonScorer(
                name="ignored_when_record_is_returned",
                target="async_score_tasks:custom_score",
                semantic_type=Semantic.QUALITY_SCORE,
            )
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    assert result.runs[0].scores[0].name == "async_quality"
    assert result.runs[0].scores[0].value == 0.77


async def test_scoring_call_properties_and_score_observation_conversion() -> None:
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)
    task_result = TaskResult(
        output={"answer": "ok"},
        status=TaskStatus.PASSED,
        observations=[ctx.metric("raw_quality", 0.1, semantic_type=Semantic.QUALITY_SCORE)],
    )
    call = ScoringCall(ctx=ctx, task_result=task_result)
    records = await evaluate_scoring_specs(
        [
            OutputMetricScorer(
                name="answer",
                path="output.answer",
                semantic_type="text.answer",
            )
        ],
        ctx=ctx,
        task_result=task_result,
    )
    records.append(
        ScoreRecord(
            name="broken",
            semantic_type=Semantic.QUALITY_SCORE,
            error=ErrorRecord(error_type="RuntimeError", message="broken"),
        )
    )

    observations = score_records_to_observations(records, ctx=ctx)

    assert call.case is case
    assert call.variant is variant
    assert call.observations == task_result.observations
    assert records[0].value == "ok"
    assert [observation.name for observation in observations] == ["answer"]


def test_resolve_dotted_path_and_schema_validation_error_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Payload:
        def nested(self) -> dict[str, int]:
            return {"value": 42}

    _write_module(
        tmp_path,
        "resolver_targets.py",
        """
        not_callable = 42
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    assert (
        resolve_dotted_path({"output": {"payload": Payload()}}, "output.payload.nested.value") == 42
    )
    with pytest.raises(KeyError, match="not found in mapping"):
        resolve_dotted_path({"output": {}}, "output.missing")
    with pytest.raises(KeyError, match="not found on object"):
        resolve_dotted_path({"output": Payload()}, "output.missing")
    assert validate_schema_value("not an object", {"type": "object"}) is False
    with pytest.raises(ValueError, match="Only schema.type"):
        validate_schema_value({}, {"type": "array"})
    with pytest.raises(ValueError, match="schema.required"):
        validate_schema_value({}, {"type": "object", "required": "answer"})
    with pytest.raises(TaskResolutionError, match="module:function"):
        resolve_python_scorer("resolver_targets")
    with pytest.raises(TaskResolutionError, match="does not define"):
        resolve_python_scorer("resolver_targets:missing")
    with pytest.raises(TaskResolutionError, match="not callable"):
        resolve_python_scorer("resolver_targets:not_callable")


def test_resolve_python_scorer_uses_explicit_search_paths(tmp_path: Path) -> None:
    package_dir = tmp_path / "local_scorers"
    package_dir.mkdir()
    _write_module(package_dir, "__init__.py", "")
    _write_module(
        package_dir,
        "helpers.py",
        """
        def score(call):
            return 1.0
        """,
    )

    scorer = resolve_python_scorer(
        "local_scorers.helpers:score",
        search_paths=(str(tmp_path),),
    )

    assert callable(scorer)


def test_resolve_python_scorer_reports_unimportable_module() -> None:
    with pytest.raises(TaskResolutionError, match="Could not import scorer module"):
        resolve_python_scorer(
            "missing_local_scorers.helpers:score",
            search_paths=("/definitely/missing/path",),
        )


async def test_score_source_wins_over_task_observation_in_projection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "projection_tasks.py",
        """
        def run(ctx, case):
            with ctx.span('task') as span:
                span.metric(
                    'coverage',
                    0.1,
                    semantic_type='coverage.ratio',
                    direction='maximize',
                    role='objective',
                )
            return {'coverage': 0.9}
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="projection-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1")]),
        task=TaskSpec(kind="python", target="projection_tasks:run"),
        variants=[Variant(id="variant_1")],
        scoring=[
            OutputMetricScorer(
                name="coverage",
                path="output.coverage",
                semantic_type=Semantic.COVERAGE_RATIO,
                direction=Direction.MAXIMIZE,
                role=ObservationRole.OBJECTIVE,
            )
        ],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")
    projected = project_observations(result.runs[0].task_result.observations)

    coverage = next(item for item in projected if item.key.name == "coverage")
    assert coverage.observation.source == ObservationSource.SCORE
    assert coverage.observation.value == 0.9


async def test_instrument_method_captures_return_metric(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_module(
        tmp_path,
        "instrument_tasks.py",
        """
        class Worker:
            def execute(self, value):
                return {'coverage': value}

        def run(ctx, case):
            worker = Worker()
            return worker.execute(case.input['coverage'])
        """,
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = __import__("instrument_tasks")
    instrument_method(
        module.Worker,
        "execute",
        span="worker.execute",
        metrics=[
            InstrumentMetricSpec(
                name="coverage_observed",
                semantic_type=Semantic.COVERAGE_RATIO,
                value_path="result.coverage",
            )
        ],
    )
    spec = BenchmarkSpec(
        benchmark=BenchmarkInfo(id="instrument-demo"),
        dataset=DatasetSpec(cases=[Case(id="case_1", input={"coverage": 0.55})]),
        task=TaskSpec(kind="python", target="instrument_tasks:run"),
        variants=[Variant(id="variant_1")],
    )

    result = await run_benchmark_spec(spec, experiment_id="exp_fixed")

    observation = next(
        item for item in result.runs[0].task_result.observations if item.name == "coverage_observed"
    )
    assert observation.source == ObservationSource.INSTRUMENTATION
    assert observation.value == 0.55


def test_instrumentation_handle_restores_original_method_and_supports_context_manager() -> None:
    class Worker:
        def execute(self) -> dict[str, int]:
            return {"count": 1}

    original = Worker.execute
    handle = instrument_method(
        Worker,
        "execute",
        metrics=[InstrumentMetricSpec(name="count", value_path="result.count")],
    )

    assert Worker.execute is not original
    handle.close()
    handle.close()
    assert Worker.execute is original

    first = instrument_method(
        Worker,
        "execute",
        metrics=[InstrumentMetricSpec(name="count", value_path="result.count")],
    )
    second = instrument_method(
        Worker,
        "execute",
        metrics=[InstrumentMetricSpec(name="count_again", value_path="result.count")],
    )
    first.close()
    assert Worker.execute is not original
    second.close()
    assert Worker.execute is original
    assert Worker().execute() == {"count": 1}


def test_instrumentation_supports_static_and_class_methods_and_rejects_invalid_targets() -> None:
    class Worker:
        shared = 3

        @staticmethod
        def compute(value: int) -> dict[str, int]:
            return {"count": value}

        @classmethod
        def size(cls) -> dict[str, int]:
            return {"count": cls.shared}

        @property
        def answer(self) -> int:
            return 42

        not_callable = 1

    static_handle = instrument_method(
        Worker,
        "compute",
        metrics=[InstrumentMetricSpec(name="count", value_path="result.count")],
    )
    class_handle = instrument_method(
        Worker,
        "size",
        metrics=[InstrumentMetricSpec(name="count", value_path="result.count")],
    )
    ctx = RunContext(benchmark_id="demo", case=Case(id="case_1"), variant=Variant(id="variant_1"))
    token = set_active_run_context(ctx)
    try:
        assert Worker.compute(2) == {"count": 2}
        assert Worker.size() == {"count": 3}
        assert Worker().answer == 42
    finally:
        reset_active_run_context(token)
        static_handle.close()
        class_handle.close()

    assert [observation.value for observation in ctx.observations] == [2, 3]
    with pytest.raises(TypeError, match="property instrumentation is not supported"):
        instrument_method(Worker, "answer")
    with pytest.raises(TypeError, match="is not callable"):
        instrument_method(Worker, "not_callable")


def test_instrument_specs_require_explicit_extractors() -> None:
    with pytest.raises(ValueError, match="instrument metrics require value_path or value_factory"):
        InstrumentMetricSpec(name="missing")
    with pytest.raises(ValueError, match="instrument factors require value_path or value_factory"):
        InstrumentFactorSpec(name="missing")


def test_instrumentation_noops_without_active_context_and_restores_context() -> None:
    class Worker:
        def execute(self) -> dict[str, int]:
            return {"count": 1}

    instrument_method(
        Worker,
        "execute",
        metrics=[InstrumentMetricSpec(name="count", value_path="result.count")],
    )
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)

    assert Worker().execute() == {"count": 1}
    assert ctx.observations == []

    token = set_active_run_context(ctx)
    try:
        assert get_active_run_context() is ctx
        assert Worker().execute() == {"count": 1}
    finally:
        reset_active_run_context(token)

    assert get_active_run_context() is None
    assert ctx.observations[0].source == ObservationSource.INSTRUMENTATION


def test_instrument_method_can_append_metrics_factors_and_record_extraction_errors() -> None:
    class Usage:
        def __init__(self, value: int) -> None:
            self.value = value

        def payload(self) -> dict[str, int]:
            return {"tokens": self.value}

        def delayed_payload(self) -> Callable[[], dict[str, int]]:
            return lambda: {"tokens": self.value}

    class Worker:
        def execute(self, tokens: int) -> Usage:
            return Usage(tokens)

    instrument_method(
        Worker,
        "execute",
        span="worker.execute",
        metrics=[
            InstrumentMetricSpec(name="tokens", value_path="result.payload.tokens"),
            InstrumentMetricSpec(name="delayed_tokens", value_path="result.delayed_payload.tokens"),
            InstrumentMetricSpec(
                name="double_tokens",
                value_factory=lambda call: call.result.payload()["tokens"] * 2,
            ),
            InstrumentMetricSpec(name="missing_dict_metric", value_path="result.payload.missing"),
            InstrumentMetricSpec(name="missing_metric", value_path="result.missing"),
        ],
    )
    instrument_method(
        Worker,
        "execute",
        factors=[
            InstrumentFactorSpec(name="call_tokens", value_factory=lambda call: call.args[0]),
            InstrumentFactorSpec(name="missing_factor", value_path="result.nope"),
        ],
    )
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)
    token = set_active_run_context(ctx)
    try:
        result = Worker().execute(7)
    finally:
        reset_active_run_context(token)

    assert result.payload() == {"tokens": 7}
    assert [observation.name for observation in ctx.observations] == [
        "tokens",
        "delayed_tokens",
        "double_tokens",
        "call_tokens",
    ]
    assert [error.error_type for error in ctx.errors] == ["KeyError", "KeyError", "KeyError"]
    assert ctx.spans[0].name == "worker.execute"
    assert ctx.spans[0].error is not None


def test_sync_instrumentation_records_errors_before_reraising() -> None:
    class Worker:
        def execute(self) -> None:
            raise RuntimeError("boom")

    instrument_method(
        Worker,
        "execute",
        span="worker.execute",
        metrics=[
            InstrumentMetricSpec(
                name="error_type",
                value_factory=lambda call: type(call.error).__name__,
            )
        ],
    )
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)
    token = set_active_run_context(ctx)
    try:
        with pytest.raises(RuntimeError, match="boom"):
            Worker().execute()
    finally:
        reset_active_run_context(token)

    assert ctx.observations[0].name == "error_type"
    assert ctx.observations[0].value == "RuntimeError"


async def test_async_instrumentation_records_success_and_errors() -> None:
    class Worker:
        async def execute(self, *, fail: bool) -> dict[str, int]:
            if fail:
                raise RuntimeError("async boom")
            return {"count": 3}

    instrument_method(
        Worker,
        "execute",
        metrics=[InstrumentMetricSpec(name="count", value_path="result.count")],
    )
    instrument_method(
        Worker,
        "execute",
        metrics=[
            InstrumentMetricSpec(
                name="error_type",
                value_factory=lambda call: type(call.error).__name__ if call.error else None,
            )
        ],
    )
    case = Case(id="case_1")
    variant = Variant(id="variant_1")
    ctx = RunContext(benchmark_id="demo", case=case, variant=variant)
    token = set_active_run_context(ctx)
    try:
        assert await Worker().execute(fail=False) == {"count": 3}
        with pytest.raises(RuntimeError, match="async boom"):
            await Worker().execute(fail=True)
    finally:
        reset_active_run_context(token)

    assert [observation.name for observation in ctx.observations] == [
        "count",
        "error_type",
        "error_type",
    ]
    assert ctx.errors[0].error_type == "KeyError"


def _write_module(tmp_path: Path, filename: str, source: str) -> None:
    (tmp_path / filename).write_text(dedent(source).strip() + "\n", encoding="utf-8")
