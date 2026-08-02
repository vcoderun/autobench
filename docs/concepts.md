# Concepts

## BenchmarkSpec

The YAML source of truth. It defines:

- benchmark metadata
- dataset and cases
- task target
- variants and factors
- scoring
- derivation and post-derivation
- report configuration

## Case

A single benchmark input with optional expected output, metadata, tags, and attachments.

Cases are the unit of replay and comparison.

## Variant

A concrete factor set for a run. Variants represent the changing parts of the system:

- prompt version
- model
- provider
- policy
- tool version

## Observation

The atomic evidence unit recorded during execution. Observations can be:

- metrics
- factors
- events
- artifacts

Observations may carry a semantic type such as `quality.correctness`, `money.cost`, or `llm.tokens.input`.

## Trace Envelope

A structured trace attached to a run. Trace envelopes preserve agent/workflow spans without making OpenTelemetry or any hosted tracing platform a core dependency.

Trace spans can be typed as:

- `agent`
- `llm`
- `tool`
- `retriever`
- `parser`
- `workflow`
- `custom`

Autobench converts useful trace fields into semantic observations, such as token usage, model/provider factors, duration, and span errors. Large raw trace payloads should be stored as artifacts instead of being embedded directly into `run.yaml`.

## Expected Action

An expected action describes behavior a system should perform during a run. It is generic enough for agent tools, retrievers, workflow steps, or other component calls.

Expected actions support:

- target matching
- input subset matching
- output matching
- ordered sequence checks
- optional vs required actions

The built-in `expected_action` scorer can emit `agent.tool.selection.correctness`, `agent.tool.argument.correctness`, and `agent.tool.sequence.correctness` style metrics without requiring an LLM judge.

## Metric Pack

A metric pack is an optional bundle of semantic registry entries, scorer defaults, report metrics, and feedback extractors.

Metric packs keep Autobench from becoming a large class-per-metric catalog. Core owns evidence, records, and execution; packs provide domain defaults such as `agentic`, `structured_output`, `llm_usage`, and `performance`.

## Score

A score is a structured evaluation result produced by the scoring layer and projected into observations with score precedence.

## Derived Metric

A metric computed from observed inputs. Example: token usage plus model/provider factors become `money.cost`.

## Post-Derivation

Cross-run derivation that needs the full experiment result. Example: paired-baseline latency speedup.

## Run Record

An immutable YAML snapshot of one case x variant execution, including:

- task output
- observations
- spans
- scores
- factors
- tracked asset versions
- artifacts
- errors

## Report

A replay-time view over recorded runs. Reports aggregate semantic metrics into:

- leaderboards
- case matrices
- comparisons
- metric distributions

## Optimization Feedback

Autobench can compact failed scores, task errors, span errors, policy violations, factors, and asset versions into feedback records. These records are designed for optimization systems such as pydantic-gepa and autoptimize, so they do not need to scrape raw reports or infer failure categories from terminal output.
