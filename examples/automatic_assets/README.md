# Automatic Asset Discovery

These offline examples prove both automatic discovery paths without credentials or network access.

```bash
uv run python examples/automatic_assets/pydantic_ai_discovery.py \
  --record /tmp/autobench-pydantic-assets

uv run python examples/automatic_assets/custom_sdk_discovery.py \
  --record /tmp/autobench-custom-assets
```

`pydantic_ai_discovery.py` uses a real Pydantic AI `Agent`, `AbstractCapability`, tool, and Pydantic
output type. None is explicitly tracked. `Benchmark.instrument_all()` discovers their definitions,
effective model-facing representations, scopes, versions, and source links.

`custom_sdk_discovery.py` adds the same lineage to an arbitrary SDK method through
`InstrumentAssetSpec`. It demonstrates declarative value paths, multiple tool definitions, span
binding, persistence, and replay without changing the SDK class.
