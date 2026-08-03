from __future__ import annotations as _annotations

from pathlib import Path

from agents import custom_span, function_span, set_trace_processors, trace

from autobench import (
    Benchmark,
    Case,
    OpenAIAgentsInstrumentation,
    Variant,
    record_experiment,
)
from autobench.runtime.context import RunContext


def run(ctx: RunContext, case: Case) -> dict[str, str]:
    with trace("offline-routing-workflow", group_id=case.id):
        with function_span("normalize", input=str(case.input), output="normalized"):
            pass
        with custom_span("route", data={"queue": "support"}):
            pass
    return {"queue": "support"}


def main() -> None:
    set_trace_processors([])
    result = (
        Benchmark("openai-agents")
        .dataset([Case(id="ticket", input="I need help")])
        .variants([Variant(id="default")])
        .task("run_openai_agents:run")
        .instrument(OpenAIAgentsInstrumentation())
        .run()
    )
    output_dir = Path(".autobench/examples/openai-agents") / result.experiment_id
    record_experiment(result, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
