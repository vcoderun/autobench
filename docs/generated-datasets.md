# Generated Datasets

Autobench can prepare generated cases before a benchmark begins, preserve how they were produced,
and publish them as an ordinary reviewable dataset. Generation is deliberately separate from the
case x variant execution loop: every variant sees the same frozen cases.

## Lifecycle

```text
generation request
  -> sync or async CaseGenerator
  -> GeneratedCaseBatch
  -> review/provenance normalization
  -> complete dataset YAML + generation manifest
  -> normal benchmark dataset.source
```

This separation prevents a model, random seed, provider retry, or partially completed generation
job from changing matrix identity while variants are already running.

## Generator Contract

A generator receives one `CaseGeneratorInput` and returns `GeneratedCaseBatch`. It may be
synchronous or asynchronous:

```python
from autobench import (
    Case,
    CaseGeneratorInput,
    GeneratedCaseBatch,
    GeneratedCaseReview,
    GenerationCost,
    GenerationDeterminism,
    GenerationUsage,
    ReviewStatus,
)


def generate_cases(request: CaseGeneratorInput) -> GeneratedCaseBatch:
    route = str(request.settings.get("route", "billing"))
    cases = (
        Case(
            id="generated-refund",
            input={"message": "Refund a duplicate charge"},
            expected={"route": route},
        ),
    )
    return GeneratedCaseBatch(
        cases=cases,
        generator_asset_version=request.prompt_asset_version,
        model_provider="openrouter",
        model_name="openai/gpt-5.6-luna",
        determinism=GenerationDeterminism.NOT_GUARANTEED,
        usage=GenerationUsage(input_tokens=120, output_tokens=45, requests=1),
        cost=GenerationCost(amount=0.0012, currency="usd"),
        reviews=(
            GeneratedCaseReview(
                case_id="generated-refund",
                status=ReviewStatus.ACCEPTED,
            ),
        ),
    )
```

Autobench does not prescribe the model, sampler, prompt framework, or review service. The callable
is the application-owned adapter; its typed output is the portable evidence boundary.

## Request YAML

The CLI can pass a human-authored request to the generator:

```yaml
# yaml-language-server: $schema=schemas/0.3.0/generation_request_schema.json
generation:
  request:
    seed: 17
    prompt:
      content: Generate privacy-safe support routing cases.
      asset_version: prompt.routing-generator@v1
    settings:
      route: billing
      count: 20
    metadata:
      owner: evaluation
    seed_cases:
      - id: reviewed-refund
        input:
          message: Refund for a duplicate charge
        expected:
          route: billing
```

`settings` and `metadata` accept portable serialized values. `seed_cases` provide reviewed examples
without requiring the generator to load a Pydantic Evals dataset or an Autobench benchmark spec.

## Generate From The CLI

```bash
autobench dataset generate generator:generate_cases \
  --request generation-request.yaml \
  --output datasets/generated-routing.yaml \
  --id generated-routing \
  --version v1
```

A complete invocation writes:

- `datasets/generated-routing.yaml`: normal dataset DSL consumed by `dataset.source`;
- `datasets/generated-routing.generation.yaml`: provider, model, prompt version, seed, settings,
  usage, cost, review state, timestamps, and content hashes.

Existing outputs are protected. Use `--force` only when intentionally replacing both artifacts.

## Generate From Python

```python
from pathlib import Path

from autobench import (
    CaseGeneratorInput,
    generate_dataset_sync,
    write_generation_result,
)

request = CaseGeneratorInput(
    seed=17,
    prompt="Generate privacy-safe support routing cases.",
    prompt_asset_version="prompt.routing-generator@v1",
    settings={"route": "billing", "count": 20},
)
result = generate_dataset_sync(
    generate_cases,
    request,
    generator_id="generator:generate_cases",
    dataset_id="generated-routing",
    version="v1",
)
written = write_generation_result(result, Path("datasets/generated-routing.yaml"))
```

Use `await generate_dataset(...)` inside an async application. `GenerationResult` validates the
request hash, case records, review projection, dataset content, timestamps, and dataset hash before
publication.

## Review Semantics

Every generated case has one of three states:

| State | Published in complete dataset? | Meaning |
| --- | --- | --- |
| `candidate` | Yes | Generated and not yet explicitly accepted or rejected |
| `accepted` | Yes | Explicitly reviewed and accepted |
| `rejected` | No | Excluded; a nonempty rejection reason is required |

The generation manifest retains all three states, including the complete rejected case and its
reason. The dataset retains `source`, review status, generator/model provenance, and its own content
hash in each included case's metadata. Teams that permit only accepted data can review or filter
candidates in their generator before returning a complete batch.

## Incomplete Generation

A generator that reaches a budget, provider, or review boundary can return:

```python
return GeneratedCaseBatch(
    complete=False,
    incomplete_reason="manual review required",
    cases=partial_cases,
)
```

Autobench writes only `<output-stem>.incomplete.yaml` and exits the CLI with status `2`. It does not
write or replace the requested dataset, even with `--force`. The sidecar preserves partial cases for
inspection without presenting them as benchmark-ready truth. An exception is a failed generation
operation, not a partial result; return an explicit incomplete batch when partial evidence exists.

## Determinism And Hashes

`GenerationDeterminism` is an evidence claim made by the generator adapter:

- `guaranteed`: the provider and adapter contract guarantee the same generated content;
- `not_guaranteed`: repeated requests may differ;
- `unknown`: no reliable guarantee is available.

Autobench never upgrades this claim based only on a seed. The request hash covers seed cases,
prompt, prompt version, settings, and metadata. Each generated case and the frozen dataset have
separate SHA-256 identities. For a guaranteed generator, identical normalized output produces
byte-identical dataset YAML; the provenance manifest still has new execution timestamps.

The manifest stores the prompt hash and asset version rather than copying prompt content. Keep the
request file or tracked prompt asset when the exact historical text must be reconstructed.

## Use In A Benchmark

After generation, consume the output like any other file-backed dataset:

```yaml
benchmark:
  routing:
    dataset:
      source: file://datasets/generated-routing.yaml
      version: v1
    run:
      python: benchmark_task:run
    variants:
      current: {}
      candidate: {}
```

Generation never runs implicitly during `autobench run`. Regenerate deliberately, review the diff,
then start a new experiment so every run has one immutable dataset identity.
