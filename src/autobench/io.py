from __future__ import annotations as _annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from autobench._version import __version__
from autobench.errors import SpecLoadError

SchemaDocument = dict[str, Any]
_SCHEMA_TOP_LEVEL_KEYS: dict[str, tuple[str, ...]] = {
    "artifact": ("record", "artifact"),
    "artifact_payload": ("record", "artifact", "payload"),
    "asset": ("record", "asset", "versions"),
    "asset_index": ("record", "assets"),
    "benchmark": (
        "benchmark",
        "capture",
        "execution",
        "dataset",
        "run",
        "variants",
        "score",
        "instrumentation",
        "derive",
        "post_derive",
        "policies",
        "report",
        "semantic_registry",
    ),
    "dataset": ("record", "dataset"),
    "generation": ("record", "generation"),
    "generation_request": ("generation",),
    "experiment": ("record", "experiment", "benchmark", "runs", "files", "environment"),
    "manifest": ("record", "experiment", "files"),
    "pricing": ("record", "pricing"),
    "report": ("record", "report", "leaderboard", "cases", "comparison", "distributions"),
    "run_record": (
        "record",
        "protocol",
        "run",
        "case",
        "variant",
        "scores",
        "metrics",
        "trace",
        "spans",
        "artifacts",
        "assets",
        "asset_uses",
        "errors",
        "canonicalization",
        "extraction",
        "lineage",
        "extensions",
        "output",
    ),
    "semantic_registry": ("record", "semantic_registry"),
    "summary": ("record", "summary"),
    "staging": (
        "staging",
        "experiment",
        "plan",
        "runs",
        "post_processing",
        "environment",
        "semantic_registry",
    ),
    "staging_manifest": ("staging", "experiment", "runs", "checkpoints"),
    "checkpoint": ("checkpoint", "run", "evidence"),
    "trace": ("record", "trace"),
}


class _DslDumper(yaml.SafeDumper):
    def increase_indent(self, flow: bool = False, indentless: bool = False) -> None:
        super().increase_indent(flow=flow, indentless=False)


def load_yaml(path: Path) -> Any:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        raise SpecLoadError(
            f"Failed to parse YAML in {path}",
            path=path,
            line=(mark.line + 1) if mark is not None else None,
            column=(mark.column + 1) if mark is not None else None,
        ) from exc


def dump_yaml(
    value: Any,
    path: Path | None = None,
    *,
    schema_name: str | None = None,
    schema: SchemaDocument | None = None,
) -> str:
    rendered = yaml.dump(
        value,
        Dumper=_DslDumper,
        sort_keys=False,
        allow_unicode=False,
        default_flow_style=False,
    )
    if schema_name is not None:
        rendered = (
            f"# yaml-language-server: $schema={ensure_yaml_schema(schema_name, schema)}\n{rendered}"
        )
    if path is not None:
        path.write_text(rendered, encoding="utf-8")
    return rendered


def schema_cache_dir(*, version: str = __version__) -> Path:
    root = Path(os.environ.get("AUTOBENCH_HOME", Path.home() / ".autobench"))
    return root / version / "schemas"


def schema_path(schema_name: str, *, version: str = __version__) -> Path:
    return schema_cache_dir(version=version) / f"{schema_name}_schema.json"


def ensure_yaml_schema(
    schema_name: str,
    schema: SchemaDocument | None = None,
    *,
    version: str = __version__,
) -> Path:
    target = schema_path(schema_name, version=version)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = schema or yaml_schema(schema_name)
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if not target.exists() or target.read_text(encoding="utf-8") != rendered:
        target.write_text(rendered, encoding="utf-8")
    return target


def loose_yaml_schema(title: str) -> SchemaDocument:
    return yaml_schema(title)


def yaml_schema(schema_name: str) -> SchemaDocument:
    if schema_name == "benchmark":
        return benchmark_schema()
    if schema_name == "dataset":
        return dataset_schema()
    if schema_name == "generation":
        return generation_schema()
    if schema_name == "generation_request":
        return generation_request_schema()
    if schema_name in {"experiment", "manifest", "run_record", "summary"}:
        return record_schema(schema_name)
    if schema_name in {"staging", "staging_manifest", "checkpoint"}:
        return staging_schema(schema_name)
    title = schema_name.replace("_", " ").title()
    properties = {
        key: {
            "description": f"Autobench {schema_name} '{key}' section.",
        }
        for key in _SCHEMA_TOP_LEVEL_KEYS.get(schema_name, ())
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"Autobench {title} YAML",
        "type": "object",
        "properties": properties,
        "additionalProperties": True,
    }


def record_schema(schema_name: str) -> SchemaDocument:
    status = {"type": "string", "enum": ["passed", "failed", "errored", "skipped", "cancelled"]}
    record_type = "run" if schema_name == "run_record" else schema_name
    properties: SchemaDocument = {
        "record": {
            "type": "object",
            "required": ["type", "version"],
            "properties": {
                "type": {"const": record_type},
                "version": {"type": "integer", "minimum": 1},
            },
            "additionalProperties": False,
        }
    }
    required = ["record"]
    if schema_name == "manifest":
        properties.update(
            {
                "experiment": {
                    "type": "object",
                    "required": ["id"],
                    "properties": {"id": {"type": "string", "minLength": 1}},
                    "additionalProperties": False,
                },
                "files": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["path", "sha256", "bytes", "kind", "identity"],
                        "properties": {
                            "path": {"type": "string", "minLength": 1},
                            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                            "bytes": {"type": "integer", "minimum": 0},
                            "kind": {
                                "type": "string",
                                "enum": [
                                    "experiment",
                                    "summary",
                                    "run",
                                    "trace",
                                    "artifact",
                                    "asset",
                                    "source",
                                    "other",
                                ],
                            },
                            "identity": {"type": "string", "minLength": 1},
                        },
                        "additionalProperties": False,
                    },
                },
            }
        )
        required.extend(("experiment", "files"))
    elif schema_name == "experiment":
        properties.update(
            {
                "experiment": {
                    "type": "object",
                    "required": ["id", "benchmark", "termination"],
                    "properties": {
                        "id": {"type": "string", "minLength": 1},
                        "benchmark": {"type": "string", "minLength": 1},
                        "correlation": correlation_schema(),
                        "termination": {
                            "type": "object",
                            "required": ["status", "partial", "post_processing"],
                            "properties": {
                                "status": {
                                    "type": "string",
                                    "enum": ["completed", "cancelled", "aborted"],
                                },
                                "partial": {"type": "boolean"},
                                "post_processing": {
                                    "type": "object",
                                    "required": ["cross_run_derivation", "policies"],
                                    "properties": {
                                        "cross_run_derivation": {"type": "boolean"},
                                        "policies": {"type": "boolean"},
                                    },
                                    "additionalProperties": False,
                                },
                                "planned_runs": {"type": "array", "items": {"type": "string"}},
                                "recorded_runs": {"type": "array", "items": {"type": "string"}},
                                "missing_runs": {"type": "array", "items": {"type": "string"}},
                                "error": {"type": "object"},
                            },
                            "additionalProperties": False,
                        },
                    },
                    "additionalProperties": False,
                },
                "benchmark": {"type": "object"},
                "runs": {"type": "object"},
                "environment": {"type": "object"},
                "manifest": {"type": "string"},
            }
        )
        required.extend(("experiment", "benchmark", "runs", "environment"))
    elif schema_name == "run_record":
        properties["run"] = {
            "type": "object",
            "required": ["id", "experiment", "benchmark", "case", "variant", "status"],
            "properties": {
                "id": {"type": "string", "minLength": 1},
                "parent": {"type": "string"},
                "experiment": {"type": "string", "minLength": 1},
                "benchmark": {"type": "string", "minLength": 1},
                "case": {"type": "string", "minLength": 1},
                "variant": {"type": "string", "minLength": 1},
                "status": status,
                "correlation": correlation_schema(),
                "partial": {"type": "boolean"},
                "end_reason": {
                    "type": "string",
                    "enum": [
                        "completed",
                        "failed",
                        "cancelled",
                        "deferred",
                        "timeout",
                        "abandoned",
                    ],
                },
                "outcome": {"type": "object"},
            },
            "additionalProperties": False,
        }
        required.append("run")
    else:
        properties.update(
            {
                "summary": {
                    "type": "object",
                    "properties": {"correlation": correlation_schema()},
                    "additionalProperties": True,
                },
                "runs": {"type": "object"},
            }
        )
        required.extend(("summary", "runs"))
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"Autobench {schema_name.replace('_', ' ').title()} YAML",
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": True,
    }


def staging_schema(schema_name: str) -> SchemaDocument:
    properties: SchemaDocument
    if schema_name == "staging":
        required = [
            "staging",
            "experiment",
            "plan",
            "runs",
            "post_processing",
            "environment",
            "semantic_registry",
        ]
        properties = {
            "staging": {"type": "object"},
            "experiment": {
                "type": "object",
                "properties": {"correlation": correlation_schema()},
                "additionalProperties": True,
            },
            "plan": {"type": "object"},
            "runs": {"type": "array", "items": {"type": "object"}},
            "post_processing": {"type": "object"},
            "environment": {"type": "object"},
            "semantic_registry": {"type": "object"},
            "report": {"type": ["object", "null"]},
            "benchmark_spec": {"type": ["object", "null"]},
            "spec_hash": {"type": ["string", "null"]},
            "source_files": {"type": "array", "items": {"type": "object"}},
        }
    elif schema_name == "staging_manifest":
        required = ["staging", "experiment", "runs", "checkpoints"]
        properties = {
            "staging": {"type": "object"},
            "experiment": {"type": "object"},
            "runs": {"type": "array", "items": {"type": "object"}},
            "checkpoints": {"type": "array", "items": {"type": "object"}},
            "payloads": {"type": "array", "items": {"type": "object"}},
        }
    else:
        required = ["checkpoint", "run", "evidence"]
        properties = {
            "checkpoint": {"type": "object"},
            "run": {
                "type": "object",
                "properties": {"correlation": correlation_schema()},
                "additionalProperties": True,
            },
            "evidence": {"type": "object"},
        }
    properties[required[0]] = {
        "type": "object",
        "required": ["version"],
        "properties": {"version": {"const": 1}},
        "additionalProperties": True,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"Autobench {schema_name.replace('_', ' ').title()} YAML",
        "type": "object",
        "required": required,
        "properties": properties,
        "additionalProperties": False,
    }


def benchmark_schema() -> SchemaDocument:
    """Return the JSON Schema for the human-facing benchmark DSL."""

    asset_discovery = {
        "type": "object",
        "description": "Automatic behavioral asset discovery at supported SDK boundaries.",
        "properties": {
            "discover": {"type": "boolean", "default": True},
            "representations": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["definition", "effective"],
                },
                "uniqueItems": True,
                "default": ["definition", "effective"],
            },
            "include": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
            },
        },
        "additionalProperties": False,
    }
    semantic_switch = {
        "oneOf": [
            {"type": "boolean"},
            {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "default": True},
                    "assets": asset_discovery,
                },
                "additionalProperties": False,
            },
        ]
    }
    instrumentation = {
        "type": "object",
        "description": "Native ABP instrumentors installed for benchmark execution.",
        "properties": {
            "all": {
                "description": "Every installed and compatible built-in instrumentor.",
                "oneOf": [
                    {"type": "boolean"},
                    {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "default": True},
                            "exclude": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                    "enum": [
                                        "pydantic_ai",
                                        "openai",
                                        "openai_agents",
                                        "httpx",
                                    ],
                                },
                                "uniqueItems": True,
                            },
                            "strict": {"type": "boolean", "default": False},
                            "assets": asset_discovery,
                        },
                        "additionalProperties": False,
                    },
                ],
            },
            "pydantic_ai": {
                **semantic_switch,
                "description": "Pydantic AI agent, model, tool, and validation capture.",
            },
            "openai": {
                **semantic_switch,
                "description": "Official OpenAI Python client and streaming capture.",
            },
            "openai_agents": {
                **semantic_switch,
                "description": "OpenAI Agents native trace-processor capture.",
            },
            "httpx": {
                "description": "HTTPX public transport capture.",
                "oneOf": [
                    {"type": "boolean"},
                    {
                        "type": "object",
                        "properties": {
                            "enabled": {"type": "boolean", "default": True},
                            "capture": {
                                "type": "object",
                                "properties": {
                                    "path": {
                                        "type": "string",
                                        "enum": ["omit", "hash", "full"],
                                        "default": "hash",
                                    },
                                    "request_headers": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "response_headers": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "request_body": {"type": "boolean", "default": False},
                                    "response_body": {"type": "boolean", "default": False},
                                    "max_body_bytes": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "default": 65536,
                                    },
                                },
                                "additionalProperties": False,
                            },
                        },
                        "additionalProperties": False,
                    },
                ],
            },
        },
        "additionalProperties": False,
    }
    capture_policy = {
        "type": "object",
        "description": "Privacy and retention policy for evidence and discovered assets.",
        "properties": {
            "default_level": {
                "type": "string",
                "enum": ["none", "metadata", "hash", "redacted", "full"],
                "default": "metadata",
            },
            "asset_default_level": {
                "type": "string",
                "enum": ["none", "metadata", "hash", "redacted", "full"],
                "default": "full",
            },
            "use_semantic_defaults": {"type": "boolean", "default": True},
            "semantic_overrides": {
                "type": "object",
                "additionalProperties": {
                    "type": "string",
                    "enum": ["none", "metadata", "hash", "redacted", "full"],
                },
            },
            "allow_semantics": {"type": "array", "items": {"type": "string"}},
            "deny_semantics": {"type": "array", "items": {"type": "string"}},
            "allow_paths": {"type": "array", "items": {"type": "string"}},
            "deny_paths": {"type": "array", "items": {"type": "string"}},
            "secret_names": {"type": "array", "items": {"type": "string"}},
            "max_inline_bytes": {"type": "integer", "minimum": 1},
            "max_artifact_bytes": {"type": "integer", "minimum": 1},
            "max_collection_items": {"type": "integer", "minimum": 1},
            "max_string_length": {"type": "integer", "minimum": 1},
            "max_depth": {"type": "integer", "minimum": 1},
            "store_binary": {"type": "boolean"},
            "retain_source_attributes": {"type": "boolean"},
        },
        "additionalProperties": False,
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": f"Autobench {__version__} Benchmark DSL",
        "type": "object",
        "required": ["benchmark"],
        "properties": {
            "benchmark": {
                "type": "object",
                "minProperties": 1,
                "maxProperties": 1,
                "additionalProperties": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "capture": capture_policy,
                        "execution": {
                            "type": "object",
                            "properties": {"correlation": correlation_schema()},
                            "additionalProperties": False,
                        },
                        "cases": {
                            "oneOf": [
                                {"type": "array"},
                                {"type": "object"},
                                {"type": "string"},
                            ]
                        },
                        "dataset": {"type": "object"},
                        "run": {"oneOf": [{"type": "string"}, {"type": "object"}]},
                        "variants": {"type": "object"},
                        "score": {"type": "object"},
                        "derive": {"type": "array"},
                        "post_derive": {"type": "array"},
                        "policies": {"type": "array"},
                        "report": {"type": "object"},
                        "semantic_registry": {"type": "object"},
                        "instrumentation": instrumentation,
                    },
                    "additionalProperties": False,
                },
            }
        },
        "additionalProperties": False,
    }


def correlation_schema() -> SchemaDocument:
    scalar = {"type": ["string", "integer", "number", "boolean"]}
    return {
        "type": ["object", "null"],
        "properties": {
            "group_id": {"type": ["string", "null"], "minLength": 1},
            "attempt": {"type": ["integer", "null"], "minimum": 1},
            "phase": {"type": ["string", "null"], "minLength": 1},
            "parent_experiment_id": {"type": ["string", "null"], "minLength": 1},
            "resumed_from_experiment_id": {
                "type": ["string", "null"],
                "minLength": 1,
            },
            "labels": {"type": "object", "additionalProperties": scalar},
        },
        "additionalProperties": False,
    }


def dataset_schema() -> SchemaDocument:
    body = dataset_body_schema()
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Autobench Dataset YAML",
        "type": "object",
        "required": ["dataset"],
        "properties": {
            "record": {
                "type": "object",
                "required": ["type", "version"],
                "properties": {
                    "type": {"const": "dataset"},
                    "version": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
            "dataset": {
                "oneOf": [
                    body,
                    {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 1,
                        "additionalProperties": body,
                    },
                ]
            },
        },
        "additionalProperties": False,
    }


def dataset_body_schema() -> SchemaDocument:
    return {
        "type": "object",
        "properties": {
            "id": {"type": ["string", "null"], "minLength": 1},
            "source": {"type": ["string", "null"], "minLength": 1},
            "version": {"type": ["string", "null"]},
            "metadata": {"type": "object"},
            "defaults": {"type": "object"},
            "cases": {"type": "array", "items": case_schema()},
        },
        "additionalProperties": False,
    }


def case_schema() -> SchemaDocument:
    return {
        "type": "object",
        "required": ["id"],
        "properties": {
            "id": {"type": "string", "minLength": 1},
            "input": {},
            "expected": {},
            "metadata": {"type": "object"},
            "tags": {"type": "array", "items": {"type": "string"}},
            "attachments": {"type": "array", "items": {"type": "object"}},
        },
        "additionalProperties": False,
    }


def generation_request_schema() -> SchemaDocument:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Autobench Dataset Generation Request YAML",
        "type": "object",
        "required": ["generation"],
        "properties": {
            "generation": {
                "type": "object",
                "required": ["request"],
                "properties": {"request": generation_request_body_schema()},
                "additionalProperties": False,
            }
        },
        "additionalProperties": False,
    }


def generation_request_body_schema() -> SchemaDocument:
    return {
        "type": "object",
        "properties": {
            "seed": {"type": ["integer", "string", "null"]},
            "prompt": {
                "type": ["object", "null"],
                "properties": {
                    "content": {"type": ["string", "null"]},
                    "asset_version": {"type": ["string", "null"]},
                },
                "additionalProperties": False,
            },
            "settings": {"type": "object"},
            "metadata": {"type": "object"},
            "seed_cases": {"type": "array", "items": case_schema()},
        },
        "additionalProperties": False,
    }


def generation_schema() -> SchemaDocument:
    nullable_string = {"type": ["string", "null"]}
    sha256_string = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
    nonnegative_integer = {"type": "integer", "minimum": 0}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Autobench Dataset Generation Manifest YAML",
        "type": "object",
        "required": ["record", "generation"],
        "properties": {
            "record": {
                "type": "object",
                "required": ["type", "version"],
                "properties": {
                    "type": {"const": "generation"},
                    "version": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
            "generation": {
                "type": "object",
                "required": [
                    "status",
                    "started_at",
                    "completed_at",
                    "determinism",
                    "generator",
                    "request",
                    "usage",
                    "output",
                    "cases",
                ],
                "properties": {
                    "status": {"enum": ["complete", "incomplete"]},
                    "reason": nullable_string,
                    "started_at": {"type": "string", "format": "date-time"},
                    "completed_at": {"type": "string", "format": "date-time"},
                    "determinism": {"enum": ["guaranteed", "not_guaranteed", "unknown"]},
                    "generator": {
                        "type": "object",
                        "required": ["id"],
                        "properties": {
                            "id": {"type": "string", "minLength": 1},
                            "asset_version": nullable_string,
                            "provider": nullable_string,
                            "model": nullable_string,
                        },
                        "additionalProperties": False,
                    },
                    "request": {
                        "type": "object",
                        "required": ["sha256", "seed_cases"],
                        "properties": {
                            "sha256": sha256_string,
                            "seed": {"type": ["integer", "string", "null"]},
                            "prompt": {
                                "type": ["object", "null"],
                                "properties": {
                                    "sha256": {"oneOf": [sha256_string, {"type": "null"}]},
                                    "asset_version": nullable_string,
                                },
                                "additionalProperties": False,
                            },
                            "settings": {"type": "object"},
                            "metadata": {"type": "object"},
                            "seed_cases": {"type": "array", "items": case_schema()},
                        },
                        "additionalProperties": False,
                    },
                    "usage": {
                        "type": "object",
                        "required": [
                            "input_tokens",
                            "output_tokens",
                            "cached_input_tokens",
                            "requests",
                            "metadata",
                        ],
                        "properties": {
                            "input_tokens": nonnegative_integer,
                            "output_tokens": nonnegative_integer,
                            "cached_input_tokens": nonnegative_integer,
                            "requests": nonnegative_integer,
                            "metadata": {"type": "object"},
                        },
                        "additionalProperties": False,
                    },
                    "cost": {
                        "type": ["object", "null"],
                        "required": ["amount", "currency"],
                        "properties": {
                            "amount": {"type": "number", "minimum": 0},
                            "currency": {"type": "string", "minLength": 1},
                        },
                        "additionalProperties": False,
                    },
                    "output": {
                        "type": "object",
                        "required": ["generated", "included", "rejected"],
                        "properties": {
                            "dataset": {
                                "type": ["object", "null"],
                                "required": ["id", "sha256"],
                                "properties": {
                                    "id": {"type": ["string", "null"], "minLength": 1},
                                    "version": nullable_string,
                                    "sha256": {"oneOf": [sha256_string, {"type": "null"}]},
                                    "path": nullable_string,
                                },
                                "additionalProperties": False,
                            },
                            "generated": nonnegative_integer,
                            "included": nonnegative_integer,
                            "rejected": nonnegative_integer,
                        },
                        "additionalProperties": False,
                    },
                    "cases": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["id", "status", "sha256", "case"],
                            "properties": {
                                "id": {"type": "string", "minLength": 1},
                                "status": {"enum": ["candidate", "accepted", "rejected"]},
                                "rejection_reason": nullable_string,
                                "sha256": sha256_string,
                                "case": case_schema(),
                            },
                            "additionalProperties": False,
                        },
                    },
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }


def resolve_file_ref(ref: str, *, base_path: Path) -> Path:
    parsed = urlparse(ref)
    if parsed.scheme and parsed.scheme != "file":
        raise SpecLoadError(
            f"Unsupported remote reference scheme '{parsed.scheme}' in '{ref}'",
            path=base_path,
        )
    if parsed.scheme == "file":
        location = "/".join(part for part in (parsed.netloc, parsed.path.lstrip("/")) if part)
        relative = Path(location)
    else:
        relative = Path(ref)
    return (base_path.parent / relative).resolve()


__all__ = (
    "SchemaDocument",
    "benchmark_schema",
    "dataset_schema",
    "dump_yaml",
    "ensure_yaml_schema",
    "load_yaml",
    "loose_yaml_schema",
    "resolve_file_ref",
    "generation_request_schema",
    "generation_schema",
    "schema_cache_dir",
    "schema_path",
    "yaml_schema",
)
