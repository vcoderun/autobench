# Datasets And Variants

Autobench expands a dataset against variants to create a deterministic run matrix. Cases describe
what is evaluated; variants describe what changes.

## Cases

Each `Case` has a stable ID and may carry arbitrary input, expected output, metadata, tags, and
attachments:

```python
from autobench import Case

case = Case(
    id="refund-request",
    input={"message": "I need a refund for order 42"},
    expected={"route": "billing", "priority": "normal"},
    metadata={"tenant": "demo"},
    tags=["routing", "smoke"],
)
```

Inputs and expected values are intentionally generic. They can be strings, mappings, Pydantic
models serialized by the task, structured multimodal references, or domain-specific payloads.
Attachments use `ArtifactRef` values when a case depends on external material.

## Dataset Sources

Cases may be authored inline:

```yaml
dataset:
  version: v1
  cases:
    - id: refund-request
      input:
        message: I need a refund
      expected:
        route: billing
```

Or loaded relative to the benchmark file:

```yaml
dataset:
  source: file://datasets/cases.yaml
  version: v1
```

File-backed datasets use the same DSL representation. Glob-backed sources can combine separate
case files while duplicate case IDs remain validation errors.

## Case Defaults

Defaults reduce repeated metadata without hiding the final case payload:

```yaml
dataset:
  defaults:
    metadata:
      locale: en-US
    tags: [regression]
  cases:
    - id: ticket-1
      input: {message: Reset my password}
      tags: [authentication]
```

Mapping values are merged, tags are deduplicated, and explicit scalar case values override
defaults.

## Variants And Factors

A variant is one concrete factor set:

```yaml
variants:
  baseline:
    label: Current production route
    factors:
      model:
        value: openrouter:openai/gpt-5.6-luna
        semantic: llm.model.name
        optimize: true
      prompt_version:
        value: route-v3
        semantic: prompt.version
        optimize: true
      temperature: 0
```

`value` is the runtime value. `semantic` tells downstream consumers what the factor means.
`optimize` is a hint that the factor is a candidate optimization axis; Autobench records it but
does not choose search strategies.

The Python form is equivalent:

```python
from autobench import FactorValue, Semantic, Variant

variant = Variant(
    id="baseline",
    label="Current production route",
    factors=[
        FactorValue(
            name="model",
            value="openrouter:openai/gpt-5.6-luna",
            semantic_type=Semantic.LLM_MODEL_NAME,
            optimize=True,
        ),
        FactorValue(name="temperature", value=0),
    ],
)
```

Tasks read factors through `ctx.factor(name)`. Factors are also copied into RunRecords and report
variant-configuration tables.

## Generated And Production Cases

The data helpers preserve where generated examples came from:

- `ProductionSample` models a source sample and review state.
- `sample_to_case` and `samples_to_cases` convert samples without losing provenance.
- `mark_generated_case` records generation metadata.
- `generated_batch_from_cases` creates a `GeneratedCaseBatch` with generator, model, and source
  details.

This layer is intentionally not a synthetic-data generator. It defines the evidence contract so a
generator, production sampler, or review system can supply cases consistently.

## Identity And Reproducibility

- Case IDs and variant IDs must be unique.
- Dataset content hashes depend on normalized content rather than filesystem location.
- Matrix order is deterministic.
- Run IDs are stable for a given plan position, case, and variant.
- Dataset version may also be emitted as `dataset.version` semantic evidence.
