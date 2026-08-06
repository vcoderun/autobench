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
    "experiment": ("record", "experiment", "benchmark", "runs", "files", "environment"),
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
    "dump_yaml",
    "ensure_yaml_schema",
    "load_yaml",
    "loose_yaml_schema",
    "resolve_file_ref",
    "schema_cache_dir",
    "schema_path",
    "yaml_schema",
)
