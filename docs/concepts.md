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
