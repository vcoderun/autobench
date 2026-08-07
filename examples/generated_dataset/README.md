# Generated Dataset Preparation

Run generation before creating or executing a benchmark matrix:

```bash
autobench dataset generate generator:generate_routing_cases \
  --request request.yaml \
  --output generated-cases.yaml \
  --id routing-generated \
  --version v1
```

Run the command from this directory. A complete invocation writes `generated-cases.yaml` plus
`generated-cases.generation.yaml`. An incomplete generator result writes only
`generated-cases.incomplete.yaml` and exits nonzero.
