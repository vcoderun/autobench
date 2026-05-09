# Scoring And Derivation

## Scoring Modes

Autobench 0.1.0 supports:

- `output`
- `pass_fail`
- `exact`
- `schema`
- `python`

Python scorers receive a structured `ScoringCall` instead of raw framework internals.

## Token Cost

`token_cost` derives `money.cost` from:

- `llm.tokens.input`
- `llm.tokens.output`
- `llm.provider`
- `llm.model.name`

The pricing table stays external and YAML-backed.

## Paired Baseline

`paired_baseline` is a post-derivation step that compares runs after the experiment finishes.

Typical use:

- latency speedup vs baseline
- cost delta vs baseline
- quality delta vs baseline

It matches runs by `case_id` or factor keys and can emit diagnostics when the comparison is not available.

## Policies

Policies are release-style requirements over semantic metrics, such as:

- `must_less_equal`
- `must_greater_equal`
- `must_between`
- `must_in`

Policy results are appended as derived observations and can influence final run status.
